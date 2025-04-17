import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Get the parent directory to access other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.ijepa_model import ContextEncoder, Predictor, TargetEncoder
from utils.masking import create_multi_block_mask, apply_mask_to_image

class IJEPADemo:
    def __init__(self, model_path=None):
        """
        Initialize the I-JEPA demonstration

        Args:
            model_path: Path to pretrained model weights (if available)
        """
        # Initialize model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.embed_dim = 768  # Embedding dimension
        self.num_heads = 12   # Number of attention heads
        self.depth = 12       # Transformer depth

        # Initialize model components
        self.context_encoder = ContextEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads
        ).to(self.device)

        self.predictor = Predictor(
            embed_dim=self.embed_dim,
            depth=4,  # Narrower predictor
            num_heads=8
        ).to(self.device)

        self.target_encoder = TargetEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads
        ).to(self.device)

        # Load pretrained weights if available
        if model_path:
            self._load_weights(model_path)

        # Set target encoder to evaluation mode (no gradient updates)
        self.target_encoder.eval()

        # For demonstration, we'll use the same encoder for both context and target
        # In a real implementation, the target encoder would be an EMA of the context encoder
        self._copy_weights_to_target()

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Inverse transform for visualization
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        ])

    def _load_weights(self, model_path):
        """
        Load pretrained weights

        Args:
            model_path: Path to pretrained weights
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.context_encoder.load_state_dict(checkpoint['context_encoder'])
            self.predictor.load_state_dict(checkpoint['predictor'])
            self.target_encoder.load_state_dict(checkpoint['target_encoder'])
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Failed to load weights: {e}")

    def _copy_weights_to_target(self):
        """
        Copy weights from context encoder to target encoder
        """
        # Instead of copying parameters directly, use state_dict to ensure compatibility
        target_state_dict = self.target_encoder.state_dict()
        context_state_dict = self.context_encoder.state_dict()

        # Only copy parameters with matching shapes
        for k in target_state_dict.keys():
            if k in context_state_dict and target_state_dict[k].shape == context_state_dict[k].shape:
                target_state_dict[k].copy_(context_state_dict[k])

        # Load the updated state dict
        self.target_encoder.load_state_dict(target_state_dict)

        # Set requires_grad to False for all parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def process_image(self, image_path, masking_ratio=0.75, block_size=32, num_target_blocks=4):
        """
        Process an image through the I-JEPA model

        Args:
            image_path: Path to the image file
            masking_ratio: Ratio of blocks to mask (0.0-1.0)
            block_size: Size of masking blocks
            num_target_blocks: Number of target blocks

        Returns:
            original_image: Original image (numpy array)
            masked_image: Masked image (numpy array)
            context_mask: Context mask (numpy array)
            target_mask: Target mask (numpy array)
            predicted_embeddings: Predicted embeddings for target regions
            target_embeddings: Target embeddings
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image: {e}")
            # Create a synthetic image for demonstration
            image_tensor = torch.randn(1, 3, 224, 224).to(self.device)
            image_tensor = torch.sigmoid(image_tensor)  # Normalize to [0, 1]

        # Create masks
        context_mask_np, target_mask_np = create_multi_block_mask(
            height=224,
            width=224,
            masking_ratio=masking_ratio,
            block_size=block_size,
            num_target_blocks=num_target_blocks
        )

        # Convert masks to tensors
        context_mask = torch.from_numpy(context_mask_np).to(self.device)
        target_mask = torch.from_numpy(target_mask_np).to(self.device)

        # Apply mask to image
        masked_image_tensor = apply_mask_to_image(image_tensor, context_mask)

        # Process through model
        with torch.no_grad():
            # Get context embeddings
            context_embeddings = self.context_encoder(masked_image_tensor, context_mask.unsqueeze(0))

            # Get target embeddings
            target_embeddings = self.target_encoder(image_tensor)

            # Predict target embeddings
            predicted_embeddings = self.predictor(context_embeddings)

        # Convert tensors to numpy for visualization
        original_image = self.inverse_transform(image_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        original_image = np.clip(original_image, 0, 1)

        masked_image = self.inverse_transform(masked_image_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        masked_image = np.clip(masked_image, 0, 1)

        return original_image, masked_image, context_mask_np, target_mask_np, predicted_embeddings, target_embeddings

    def generate_synthetic_image(self):
        """
        Generate a synthetic image for demonstration

        Returns:
            image: Synthetic image as a PIL Image
        """
        # Create a synthetic image with simple shapes
        img = np.zeros((224, 224, 3), dtype=np.float32)

        # Background
        img[:, :, :] = 0.1

        # Add some shapes
        # Circle
        center_x, center_y = 112, 112
        radius = 50
        for h in range(224):
            for w in range(224):
                dist = np.sqrt((h - center_y)**2 + (w - center_x)**2)
                if dist < radius:
                    img[h, w, 0] = 0.8  # R
                    img[h, w, 1] = 0.2  # G
                    img[h, w, 2] = 0.2  # B

        # Square
        x1, y1 = 30, 30
        size = 40
        img[y1:y1+size, x1:x1+size, 0] = 0.2  # R
        img[y1:y1+size, x1:x1+size, 1] = 0.8  # G
        img[y1:y1+size, x1:x1+size, 2] = 0.2  # B

        # Triangle
        x1, y1 = 160, 40
        x2, y2 = 190, 40
        x3, y3 = 175, 80

        # Simple triangle rasterization
        min_x, max_x = min(x1, x2, x3), max(x1, x2, x3)
        min_y, max_y = min(y1, y2, y3), max(y1, y2, y3)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Barycentric coordinates
                d1 = (y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)
                d2 = (y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)
                d3 = (y1 - y2) * (x - x1) + (x2 - x1) * (y - y1)

                if (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0):
                    img[y, x, 0] = 0.2  # R
                    img[y, x, 1] = 0.2  # G
                    img[y, x, 2] = 0.8  # B

        # Convert to PIL Image
        img = (img * 255).astype(np.uint8)
        image = Image.fromarray(img)

        return image
