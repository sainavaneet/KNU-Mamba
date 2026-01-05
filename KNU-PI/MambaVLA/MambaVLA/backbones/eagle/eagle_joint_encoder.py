"""
Eagle-based joint encoder that processes both images and text together.
This encoder uses Eagle backbone to encode images and text jointly, creating unified embeddings.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from transformers.feature_extraction_utils import BatchFeature
from .eagle_backbone import EagleBackbone, DEFAULT_EAGLE_MODEL_NAME
from .eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues, build_transform
import torchvision.transforms.functional as TF


class EagleJointEncoder(nn.Module):
    """
    Eagle-based joint encoder that processes both images and text together.
    This creates unified embeddings where image and text information are jointly encoded.
    """
    
    def __init__(
        self,
        camera_names: List[str],
        latent_dim: int = 256,
        model_name: str = DEFAULT_EAGLE_MODEL_NAME,
        tune_llm: bool = True,  # Allow tuning language model
        tune_visual: bool = True,  # Allow tuning visual features
        reproject_vision: bool = False,
        scale_image_resolution: int = 1,
        processor_cfg: Optional[dict] = None,
        projector_dim: int = -1,
        allow_reshape_visual: bool = True,
        use_local_eagle_hg_model: bool = True,
        input_size: int = 224,
        norm_type: str = "siglip",
    ):
        super().__init__()
        
        self.camera_names = camera_names
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.norm_type = norm_type
        
        # Initialize Eagle backbone
        self.eagle_backbone = EagleBackbone(
            select_layer=12,
            model_name=model_name,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            reproject_vision=reproject_vision,
            scale_image_resolution=scale_image_resolution,
            processor_cfg=processor_cfg,
            projector_dim=projector_dim,
            allow_reshape_visual=allow_reshape_visual,
            use_local_eagle_hg_model=use_local_eagle_hg_model,
        )
        
        # Initialize processor for image and text preprocessing
        if processor_cfg is None:
            processor_cfg = {
                "model_path": model_name,
                "max_input_tiles": 1,
                "model_spec": {
                    "template": "qwen2-chat",
                    "num_image_token": 64
                }
            }
        
        self.processor = EagleProcessor(
            model_path=processor_cfg["model_path"],
            max_input_tiles=processor_cfg["max_input_tiles"],
            model_spec=ModelSpecificValues(**processor_cfg["model_spec"]),
            use_local_eagle_hg_model=use_local_eagle_hg_model,
        )
        
        # Get image context token ID
        self.img_context_token_id = self.processor.get_img_context_token()
        
        # Build transform for image preprocessing
        self.transform = build_transform(input_size=input_size, norm_type=norm_type)
        
        # Projection layer to match expected latent dimension
        eagle_output_dim = 1536  # Eagle's default output dimension
        if projector_dim != -1:
            eagle_output_dim = projector_dim
            
        self.projection = nn.Linear(eagle_output_dim, latent_dim)
        
        # Dummy variable for device compatibility
        self._dummy_variable = nn.Parameter(torch.zeros(0))
    
    def _preprocess_images(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess images from multiple cameras into the format expected by Eagle.
        
        Args:
            obs_dict: Dictionary containing image tensors for each camera
            
        Returns:
            torch.Tensor: Preprocessed images [batch_size, channels, height, width]
        """
        batch_size = None
        processed_images = []
        
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            if image_key not in obs_dict:
                raise KeyError(f"Expected key '{image_key}' not found in obs_dict")
            
            images = obs_dict[image_key]  # [batch_size, channels, height, width]
            
            if batch_size is None:
                batch_size = images.shape[0]
            else:
                assert batch_size == images.shape[0], "Batch size mismatch across cameras"
            
            # Process each image in the batch
            camera_images = []
            for i in range(batch_size):
                img_tensor = images[i].cpu()
                
                # Ensure values are in [0, 1] range
                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1) / 2
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                # Convert to PIL image
                img_pil = TF.to_pil_image(img_tensor)
                
                # Apply transform (resize to input_size and normalize)
                img_transformed = self.transform(img_pil)
                camera_images.append(img_transformed)
            
            # Stack processed images for this camera
            camera_images = torch.stack(camera_images).to(images.device)
            processed_images.append(camera_images)
        
        # For now, we'll use the first camera's images
        # TODO: Consider how to handle multiple cameras with Eagle
        return processed_images[0]
    
    def _preprocess_text(self, text_input: Union[List[str], str]) -> tuple:
        """
        Preprocess text inputs for Eagle backbone.
        
        Args:
            text_input: Text string or list of text strings
            
        Returns:
            tuple: (input_ids, attention_mask) for Eagle backbone
        """
        # Convert to list if single string
        if isinstance(text_input, str):
            text_list = [text_input]
        else:
            text_list = text_input
        
        # Process text using Eagle processor
        processed = self.processor.process_text(text_list)
        
        input_ids = processed["input_ids"]
        attention_mask = processed["attention_mask"]
        
        return input_ids, attention_mask
    
    def _create_joint_eagle_inputs(
        self, 
        images: torch.Tensor, 
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        batch_size: int
    ) -> BatchFeature:
        """
        Create inputs for Eagle backbone with both images and text.
        
        Args:
            images: Preprocessed images [batch_size, channels, height, width]
            text_input_ids: Tokenized text input [batch_size, text_seq_length]
            text_attention_mask: Text attention mask [batch_size, text_seq_length]
            batch_size: Batch size
            
        Returns:
            BatchFeature: Inputs for Eagle backbone
        """
        device = images.device
        
        # Number of image tokens (from processor config)
        num_img_tokens = self.processor.model_spec.num_image_token  # Usually 64
        
        # Get text sequence length
        text_seq_len = text_input_ids.shape[1]
        
        # Total sequence length: image tokens + text tokens
        total_seq_len = num_img_tokens + text_seq_len
        
        # Create combined input_ids: [image_tokens, text_tokens]
        combined_input_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device=device)
        
        # Fill first part with image context tokens
        combined_input_ids[:, :num_img_tokens] = self.img_context_token_id
        
        # Fill second part with text tokens
        combined_input_ids[:, num_img_tokens:] = text_input_ids
        
        # Create combined attention mask
        combined_attention_mask = torch.ones(batch_size, total_seq_len, device=device)
        # Text attention mask (0s for padding) should be applied to text portion
        combined_attention_mask[:, num_img_tokens:] = text_attention_mask
        
        return BatchFeature(data={
            "pixel_values": images,
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask
        })
    
    def forward(
        self, 
        obs_dict: Dict[str, torch.Tensor], 
        lang_input: Optional[Union[str, List[str], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Eagle joint encoder.
        
        Args:
            obs_dict: Dictionary containing image tensors for each camera
            lang_input: Language input - can be:
                - Text string or list of strings (will be tokenized)
                - Pre-computed lang_emb tensor (will be ignored, use text instead)
                
        Returns:
            torch.Tensor: Joint encoded features [batch_size, num_cameras, latent_dim]
        """
        batch_size = None
        features = []
        
        # Preprocess text if provided
        if lang_input is not None:
            if isinstance(lang_input, torch.Tensor):
                # If it's a tensor, we can't use it directly with Eagle
                # Convert to string representation or skip
                raise ValueError("For joint encoding, lang_input must be text (str or List[str]), not a tensor. "
                               "Use the text directly for joint encoding.")
            
            # Preprocess text
            text_input_ids, text_attention_mask = self._preprocess_text(lang_input)
            text_input_ids = text_input_ids.to(self._dummy_variable.device)
            text_attention_mask = text_attention_mask.to(self._dummy_variable.device)
        else:
            # If no text provided, create empty text tokens
            # This will just encode images (fallback behavior)
            raise ValueError("lang_input is required for joint encoding. Provide text string or list of strings.")
        
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            if image_key not in obs_dict:
                raise KeyError(f"Expected key '{image_key}' not found in obs_dict")
            
            images = obs_dict[image_key]  # [batch_size, channels, height, width]
            
            if batch_size is None:
                batch_size = images.shape[0]
            else:
                assert batch_size == images.shape[0], "Batch size mismatch across cameras"
            
            # Process images for this camera
            processed_images = self._preprocess_images({image_key: images})
            
            # Create joint Eagle inputs (images + text)
            eagle_inputs = self._create_joint_eagle_inputs(
                processed_images, 
                text_input_ids, 
                text_attention_mask,
                batch_size
            )
            
            # Forward through Eagle backbone
            with torch.no_grad() if not self.training else torch.enable_grad():
                eagle_outputs = self.eagle_backbone(eagle_inputs)
            
            # Extract features and project to latent dimension
            backbone_features = eagle_outputs["backbone_features"]  # [batch_size, seq_length, hidden_dim]
            
            # For joint encoding, we can use different strategies:
            # 1. Use the first token (image token)
            # 2. Pool over all tokens (mean/max)
            # 3. Use specific tokens (e.g., last text token)
            # Here we'll use mean pooling over the entire sequence to capture both image and text info
            pooled_features = torch.mean(backbone_features, dim=1)  # [batch_size, hidden_dim]
            
            # Project to latent dimension
            projected_features = self.projection(pooled_features)  # [batch_size, latent_dim]
            
            features.append(projected_features)
        
        # Stack features from all cameras [batch_size, num_cameras, latent_dim]
        features = torch.stack(features, dim=1)
        
        return features
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class MultiImageEagleJointEncoder(nn.Module):
    """
    Multi-image Eagle joint encoder that maintains compatibility with existing code.
    This is a wrapper around EagleJointEncoder to match the expected interface.
    """
    
    def __init__(
        self,
        shape_meta: dict,
        latent_dim: int = 256,
        model_name: str = DEFAULT_EAGLE_MODEL_NAME,
        tune_llm: bool = True,
        tune_visual: bool = True,
        reproject_vision: bool = False,
        scale_image_resolution: int = 1,
        processor_cfg: Optional[dict] = None,
        projector_dim: int = -1,
        allow_reshape_visual: bool = True,
        use_local_eagle_hg_model: bool = True,
        input_size: int = 224,
        norm_type: str = "siglip",
    ):
        super().__init__()
        
        # Extract camera names from shape_meta
        obs_shape_meta = shape_meta['obs']
        camera_names = []
        for key, attr in obs_shape_meta.items():
            if attr.get('type') == 'rgb':
                # Extract camera name from key (e.g., "agentview_image" -> "agentview")
                camera_name = key.replace('_image', '')
                camera_names.append(camera_name)
        
        self.camera_names = camera_names
        self.shape_meta = shape_meta
        
        # Initialize Eagle joint encoder
        self.eagle_encoder = EagleJointEncoder(
            camera_names=camera_names,
            latent_dim=latent_dim,
            model_name=model_name,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            reproject_vision=reproject_vision,
            scale_image_resolution=scale_image_resolution,
            processor_cfg=processor_cfg,
            projector_dim=projector_dim,
            allow_reshape_visual=allow_reshape_visual,
            use_local_eagle_hg_model=use_local_eagle_hg_model,
            input_size=input_size,
            norm_type=norm_type,
        )
        
        # Dummy variable for device compatibility
        self._dummy_variable = nn.Parameter(torch.zeros(0))
    
    def forward(
        self, 
        obs_dict: Dict[str, torch.Tensor], 
        lang_input: Optional[Union[str, List[str]]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the multi-image Eagle joint encoder.
        
        Args:
            obs_dict: Dictionary containing image tensors for each camera
            lang_input: Language input (text string or list of strings)
            
        Returns:
            torch.Tensor: Joint encoded features [batch_size, num_cameras, latent_dim]
        """
        return self.eagle_encoder(obs_dict, lang_input)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    @torch.no_grad()
    def output_shape(self):
        """Get the output shape of the encoder."""
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape,
                dtype=self.dtype,
                device=self.device
            )
            example_obs_dict[key] = this_obs
        
        # Use dummy text for output shape
        example_output = self.forward(example_obs_dict, lang_input="dummy text")
        output_shape = example_output.shape[1:]
        return output_shape

