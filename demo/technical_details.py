import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def show_technical_details():
    """
    Display technical details about I-JEPA implementation
    """
    st.header("Technical Details of I-JEPA")
    
    # Model Architecture
    st.subheader("Model Architecture")
    
    st.markdown("""
    I-JEPA consists of three main components:
    
    1. **Context Encoder**
       - Architecture: Vision Transformer (ViT)
       - Input: Masked image with mask tokens
       - Output: Context embeddings
       - Parameters:
         - Embedding dimension: 768
         - Depth: 12 transformer blocks
         - Number of attention heads: 12
         - MLP ratio: 4.0
         - Patch size: 16x16
    
    2. **Predictor**
       - Architecture: Narrow Vision Transformer (ViT)
       - Input: Context embeddings
       - Output: Predicted target embeddings
       - Parameters:
         - Embedding dimension: 768
         - Depth: 4 transformer blocks (shallower than encoder)
         - Number of attention heads: 8
         - MLP ratio: 4.0
    
    3. **Target Encoder**
       - Architecture: Same as Context Encoder
       - Input: Unmasked image
       - Output: Target embeddings
       - Parameters:
         - Same as Context Encoder
         - Weights updated via EMA of Context Encoder
    """)
    
    # Masking Strategy
    st.subheader("Masking Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Multi-Block Masking**
        
        I-JEPA uses a multi-block masking strategy:
        
        - **Target Blocks**: Multiple, relatively large blocks (typically 4)
        - **Context Block**: Remaining visible regions
        - **Masking Ratio**: ~75% of the image is masked
        - **Block Size**: Typically 32x32 pixels
        
        This strategy is crucial for guiding the model towards learning semantic representations.
        """)
        
        # Create a simple masking visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        mask = np.ones((14, 14))  # 14x14 grid (224/16 patches)
        
        # Create context region (visible)
        mask[3:7, 3:7] = 0
        mask[8:11, 8:11] = 0
        
        # Create target regions (to predict)
        mask[3:7, 8:11] = 2
        mask[8:11, 3:7] = 2
        
        # Plot mask
        cmap = plt.cm.get_cmap('viridis', 3)
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=2)
        
        # Add colorbar
        cbar = plt.colorbar(im, ticks=[0.33, 1, 1.67])
        cbar.ax.set_yticklabels(['Context', 'Masked', 'Target'])
        
        ax.set_title('I-JEPA Masking Strategy')
        ax.axis('off')
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        **Masking Implementation**
        
        The masking process involves:
        
        1. Dividing the image into a grid of blocks
        2. Randomly selecting blocks for masking
        3. Designating a subset of masked blocks as target blocks
        4. Replacing masked regions with a learnable mask token in the context encoder
        5. Processing the full image through the target encoder
        6. Predicting representations only for the target blocks
        
        The mask is applied at the patch level, with each patch being either fully visible or fully masked.
        """)
        
        st.markdown("""
        **Importance of Masking Strategy**
        
        Ablation studies show that the masking strategy significantly impacts performance:
        
        - **Target Size**: Larger target blocks lead to better performance
        - **Context Size**: More informative context regions improve results
        - **Number of Targets**: Multiple target blocks are better than a single large target
        
        The optimal strategy forces the model to learn long-range dependencies and semantic features.
        """)
    
    # Loss Function
    st.subheader("Loss Function")
    
    st.markdown("""
    I-JEPA uses a simple L2 loss between predicted and target embeddings:
    
    $$L = \\frac{1}{N} \\sum_{i=1}^{N} \\| f_\\theta(x_i^{ctx}) - g_\\xi(x_i) \\|_2^2$$
    
    Where:
    - $f_\\theta$ is the predictor applied to context embeddings
    - $g_\\xi$ is the target encoder
    - $x_i^{ctx}$ is the masked input with only context regions visible
    - $x_i$ is the original unmasked input
    - $N$ is the number of target patches
    
    This loss encourages the model to predict the target embeddings from the context embeddings.
    """)
    
    # Training Details
    st.subheader("Training Details")
    
    st.markdown("""
    **Pretraining Dataset**
    
    I-JEPA is pretrained on ImageNet-1K:
    - 1.28 million training images
    - 1000 classes
    
    **Optimization**
    
    - Optimizer: AdamW
    - Learning rate: 1.5e-4 with cosine decay
    - Weight decay: 0.05
    - Batch size: 2048
    - Training epochs: 300-800
    - Hardware: 32-64 GPUs
    
    **Data Augmentation**
    
    Unlike most self-supervised methods, I-JEPA does not rely on data augmentations:
    - No random crops
    - No color jitter
    - No flips
    - No rotations
    
    Only basic preprocessing is applied:
    - Resizing to 224x224
    - Normalization
    """)
    
    # Evaluation Protocol
    st.subheader("Evaluation Protocol")
    
    st.markdown("""
    I-JEPA is evaluated using several protocols:
    
    1. **Linear Probing**
       - Freeze the pretrained encoder
       - Train a linear classifier on top
       - Evaluate on ImageNet-1K validation set
    
    2. **Low-Shot Classification**
       - Use 1%, 5%, or 10% of labeled data
       - Freeze the pretrained encoder
       - Train a linear classifier on top
       - Evaluate on ImageNet-1K validation set
    
    3. **Fine-Tuning**
       - Initialize with pretrained weights
       - Fine-tune the entire model
       - Evaluate on ImageNet-1K validation set
    
    4. **Transfer Learning**
       - Initialize with pretrained weights
       - Fine-tune on downstream tasks:
         - Object detection (COCO)
         - Instance segmentation (COCO)
         - Semantic segmentation (ADE20K)
         - Depth estimation (NYU Depth V2)
         - Object counting (FSC-147)
    """)
    
    # Implementation Details
    st.subheader("Implementation Details")
    
    st.markdown("""
    **Context Encoder**
    
    ```python
    class ContextEncoder(nn.Module):
        def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12):
            # Initialize patch embedding
            self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
            
            # Position embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            
            # Mask token
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
            # Transformer blocks
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads) for _ in range(depth)
            ])
            
            self.norm = nn.LayerNorm(embed_dim)
    ```
    
    **Predictor**
    
    ```python
    class Predictor(nn.Module):
        def __init__(self, embed_dim=768, depth=4, num_heads=8):
            # Transformer blocks (narrower than encoder)
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads) for _ in range(depth)
            ])
            
            self.norm = nn.LayerNorm(embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)
    ```
    
    **Target Encoder Update**
    
    ```python
    # EMA update of target encoder
    for param_q, param_k in zip(context_encoder.parameters(), target_encoder.parameters()):
        param_k.data = param_k.data * ema_decay + param_q.data * (1 - ema_decay)
    ```
    """)
    
    # Results
    st.subheader("Results")
    
    # Create a simple bar chart comparing methods
    methods = ['I-JEPA', 'MAE', 'DINO', 'iBOT', 'SimCLR']
    linear_probe = [76.1, 68.0, 78.2, 79.5, 69.3]  # Example values
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, linear_probe, color='skyblue')
    
    # Add a horizontal line for supervised baseline
    ax.axhline(y=76.5, color='red', linestyle='--', label='Supervised')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}%', ha='center', va='bottom')
    
    ax.set_ylabel('ImageNet Linear Probe Accuracy (%)')
    ax.set_title('Comparison of Self-Supervised Methods')
    ax.legend()
    
    st.pyplot(fig)
    
    st.markdown("""
    **Key Results**
    
    - I-JEPA outperforms other augmentation-free methods like MAE in linear probing
    - It achieves competitive results with augmentation-based methods like DINO and iBOT
    - I-JEPA excels in low-shot learning scenarios (1%, 5%, 10% of labels)
    - It shows strong transfer learning performance, especially on tasks requiring semantic understanding
    - The method is more computationally efficient than MAE and augmentation-based methods
    """)
    
    # Limitations
    st.subheader("Limitations and Future Work")
    
    st.markdown("""
    **Current Limitations**
    
    - Fine-tuning performance slightly below state-of-the-art MAE
    - Sensitive to masking strategy parameters
    - Requires careful tuning of predictor architecture
    - Trade-off between linear evaluation and fine-tuning performance
    
    **Future Directions**
    
    - Extending to video data (V-JEPA)
    - Combining with language models for multimodal learning
    - Exploring different masking strategies
    - Scaling to larger models and datasets
    - Applying to specialized domains (medical imaging, satellite imagery, etc.)
    """)
    
    # Conclusion
    st.subheader("Conclusion")
    
    st.markdown("""
    I-JEPA represents a significant advancement in self-supervised learning for computer vision:
    
    - It demonstrates that high-quality representations can be learned without relying on hand-crafted augmentations
    - The predictive approach in latent space encourages the model to focus on semantic content
    - The multi-block masking strategy is crucial for guiding the model towards learning useful representations
    - The method is computationally efficient and scales well with model and dataset size
    - It achieves strong performance across a wide range of downstream tasks
    
    This demonstration provides an interactive way to explore the key concepts of I-JEPA and understand how it processes image data.
    """)
