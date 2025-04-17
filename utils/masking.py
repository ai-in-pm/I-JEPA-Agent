import numpy as np
import torch

def create_multi_block_mask(height, width, masking_ratio=0.75, block_size=32, num_target_blocks=4):
    """
    Create a multi-block masking pattern for I-JEPA
    
    Args:
        height: Height of the image in pixels
        width: Width of the image in pixels
        masking_ratio: Ratio of pixels to mask (0.0-1.0)
        block_size: Size of masking blocks in pixels
        num_target_blocks: Number of target blocks to create
        
    Returns:
        context_mask: Binary mask for context region (1 = visible, 0 = masked)
        target_mask: Binary mask for target regions (1 = target, 0 = non-target)
    """
    # Calculate grid dimensions
    h_blocks = height // block_size
    w_blocks = width // block_size
    total_blocks = h_blocks * w_blocks
    
    # Initialize masks
    context_mask = np.ones((height, width), dtype=np.float32)
    target_mask = np.zeros((height, width), dtype=np.float32)
    
    # Calculate number of blocks to mask
    num_blocks_to_mask = int(total_blocks * masking_ratio)
    
    # Create a grid of block indices
    block_indices = np.arange(total_blocks)
    np.random.shuffle(block_indices)
    
    # Select target blocks from the masked blocks
    target_block_indices = block_indices[:num_target_blocks]
    
    # Select remaining blocks to mask
    other_masked_block_indices = block_indices[num_target_blocks:num_blocks_to_mask]
    
    # Apply target blocks to masks
    for idx in target_block_indices:
        h_idx = idx // w_blocks
        w_idx = idx % w_blocks
        
        h_start = h_idx * block_size
        h_end = min((h_idx + 1) * block_size, height)
        w_start = w_idx * block_size
        w_end = min((w_idx + 1) * block_size, width)
        
        context_mask[h_start:h_end, w_start:w_end] = 0  # Masked in context
        target_mask[h_start:h_end, w_start:w_end] = 1   # Target region
    
    # Apply other masked blocks to context mask
    for idx in other_masked_block_indices:
        h_idx = idx // w_blocks
        w_idx = idx % w_blocks
        
        h_start = h_idx * block_size
        h_end = min((h_idx + 1) * block_size, height)
        w_start = w_idx * block_size
        w_end = min((w_idx + 1) * block_size, width)
        
        context_mask[h_start:h_end, w_start:w_end] = 0  # Masked in context
    
    return context_mask, target_mask

def apply_mask_to_image(image, mask):
    """
    Apply a mask to an image
    
    Args:
        image: Image tensor [C, H, W] or [B, C, H, W]
        mask: Mask tensor [H, W] or [B, H, W] (1 = visible, 0 = masked)
        
    Returns:
        masked_image: Masked image
    """
    if len(image.shape) == 3:  # [C, H, W]
        C, H, W = image.shape
        masked_image = image.clone()
        
        # Expand mask to match image channels
        expanded_mask = mask.unsqueeze(0).expand(C, -1, -1)
        
        # Apply mask (set masked regions to gray)
        masked_image = masked_image * expanded_mask + 0.5 * (1 - expanded_mask)
        
    elif len(image.shape) == 4:  # [B, C, H, W]
        B, C, H, W = image.shape
        masked_image = image.clone()
        
        # Expand mask to match image channels
        if len(mask.shape) == 2:  # [H, W]
            expanded_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        else:  # [B, H, W]
            expanded_mask = mask.unsqueeze(1).expand(-1, C, -1, -1)
        
        # Apply mask (set masked regions to gray)
        masked_image = masked_image * expanded_mask + 0.5 * (1 - expanded_mask)
    
    return masked_image
