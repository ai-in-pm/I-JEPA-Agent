import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def create_architecture_diagram():
    """
    Create a simple architecture diagram for I-JEPA
    
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Define components
    components = {
        'image': (0.1, 0.5, 0.15, 0.3),
        'masked_image': (0.1, 0.1, 0.15, 0.3),
        'context_encoder': (0.35, 0.1, 0.15, 0.3),
        'target_encoder': (0.35, 0.5, 0.15, 0.3),
        'predictor': (0.6, 0.1, 0.15, 0.3),
        'target_embeddings': (0.6, 0.5, 0.15, 0.3),
        'loss': (0.85, 0.3, 0.1, 0.1)
    }
    
    # Draw components
    for name, (x, y, w, h) in components.items():
        if name == 'loss':
            circle = plt.Circle((x + w/2, y + h/2), 0.05, fill=True, color='red', alpha=0.7)
            plt.gca().add_patch(circle)
            plt.text(x + w/2, y + h/2, 'L2', ha='center', va='center', fontsize=10, color='white')
        else:
            rect = plt.Rectangle((x, y), w, h, fill=True, alpha=0.7, 
                                color='skyblue' if 'encoder' in name or 'predictor' in name else 'lightgray')
            plt.gca().add_patch(rect)
            plt.text(x + w/2, y + h/2, name.replace('_', '\n'), ha='center', va='center', fontsize=10)
    
    # Draw arrows
    arrows = [
        ('image', 'target_encoder'),
        ('image', 'masked_image'),
        ('masked_image', 'context_encoder'),
        ('context_encoder', 'predictor'),
        ('predictor', 'loss'),
        ('target_encoder', 'target_embeddings'),
        ('target_embeddings', 'loss')
    ]
    
    for start, end in arrows:
        start_x = components[start][0] + components[start][2]
        start_y = components[start][1] + components[start][3]/2
        
        end_x = components[end][0]
        end_y = components[end][1] + components[end][3]/2
        
        # Special case for the loss arrows
        if end == 'loss':
            end_x = components[end][0] + components[end][2]/2
            end_y = components[end][1] + components[end][3]/2
        
        plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                 head_width=0.02, head_length=0.02, fc='black', ec='black', length_includes_head=True)
    
    # Set limits and remove axes
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    return fig

def show_educational_content():
    """
    Display educational content about I-JEPA
    """
    st.header("Understanding I-JEPA")
    
    # Introduction
    st.subheader("Introduction")
    st.write("""
    I-JEPA (Image-based Joint-Embedding Predictive Architecture) is a self-supervised learning method 
    for generating semantic image representations without relying on hand-crafted data augmentations 
    often used in invariance-based methods.
    
    I-JEPA operates on the principle of predicting the representations of target image blocks from the 
    representation of a single context block within the same image. It is a non-generative model, 
    differentiating itself from methods like Masked Autoencoders (MAE) by making predictions in an 
    abstract representation space rather than pixel or token space.
    
    This abstract prediction is intended to encourage the model to learn more semantic features by 
    ignoring low-level pixel details.
    """)
    
    # Key Innovations
    st.subheader("Key Innovations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Augmentation-Free Learning**
        
        I-JEPA learns semantic representations without relying on hand-crafted data augmentations like:
        - Random crops
        - Color jitter
        - Flips
        - Rotations
        
        This makes it more generalizable and less biased by human-designed invariances.
        """)
        
        st.markdown("""
        **Non-Generative Approach**
        
        Unlike methods like MAE that reconstruct pixels, I-JEPA:
        - Predicts in a learned representation space
        - Focuses on semantic content rather than pixel details
        - Is more computationally efficient
        """)
    
    with col2:
        st.markdown("""
        **Multi-Block Masking Strategy**
        
        I-JEPA uses a carefully designed masking strategy:
        - Multiple, relatively large target blocks
        - An informative, spatially distributed context block
        - ~75% masking ratio
        
        This guides the model towards learning semantic representations.
        """)
        
        st.markdown("""
        **EMA-Updated Target Encoder**
        
        The target encoder is updated via Exponential Moving Average (EMA) of the context encoder:
        - Provides stable targets for prediction
        - Prevents representation collapse
        - Allows for smoother optimization
        """)
    
    # Architecture Diagram
    st.subheader("I-JEPA Architecture")
    
    # Create a simple architecture diagram
    fig = create_architecture_diagram()
    st.pyplot(fig)
    
    # Training Process
    st.subheader("Training Process")
    st.write("""
    1. **Input Processing**: An image is divided into patches and processed through the model.
    
    2. **Masking**: A multi-block mask is applied, hiding ~75% of the image patches.
    
    3. **Context Encoding**: The masked image is processed by the context encoder, replacing
       masked regions with learnable mask tokens.
    
    4. **Target Encoding**: The original, unmasked image is processed by the target encoder
       to generate target representations.
    
    5. **Prediction**: The predictor takes the context embeddings and predicts representations
       for the masked regions.
    
    6. **Loss Calculation**: The L2 distance between predicted and target representations is
       calculated and minimized.
    
    7. **Parameter Updates**: The context encoder and predictor are updated via gradient descent,
       while the target encoder is updated via EMA.
    """)
    
    # Performance and Results
    st.subheader("Performance and Results")
    st.write("""
    I-JEPA demonstrates strong performance and scalability:
    
    - **Linear Probing**: Outperforms augmentation-free generative methods like MAE and data2vec
    
    - **Low-Shot Classification**: Shows strong performance in few-shot learning scenarios
    
    - **Computational Efficiency**: More efficient than MAE and augmentation-based methods like iBOT
    
    - **Scalability**: Benefits from scaling both model size and dataset size
    
    - **Semantic Tasks**: Competitive with augmentation-based invariance methods on semantic tasks
    
    - **Low-Level Vision**: Surpasses augmentation-based methods on tasks like object counting and depth prediction
    """)
    
    # Comparison with Other Methods
    st.subheader("Comparison with Other Methods")
    
    comparison_data = {
        "Method": ["I-JEPA", "MAE", "DINO", "iBOT", "SimCLR"],
        "Approach": ["Predictive", "Generative", "Invariance", "Invariance+Generative", "Invariance"],
        "Augmentations": ["No", "No", "Yes", "Yes", "Yes"],
        "Prediction Space": ["Latent", "Pixel", "N/A", "Pixel+Latent", "N/A"],
        "Compute Efficiency": ["High", "Medium", "Medium", "Low", "Medium"]
    }
    
    st.table(comparison_data)
    
    # Conclusion
    st.subheader("Conclusion")
    st.write("""
    I-JEPA represents a significant advancement in self-supervised learning for computer vision:
    
    - It demonstrates that high-quality representations can be learned without relying on hand-crafted augmentations
    
    - The predictive approach in latent space encourages the model to focus on semantic content
    
    - The multi-block masking strategy is crucial for guiding the model towards learning useful representations
    
    - The method is computationally efficient and scales well with model and dataset size
    
    - It achieves strong performance across a wide range of downstream tasks
    """)
    
    # References
    st.subheader("References")
    st.markdown("""
    1. Assran, M., Duval, F., Bose, J., Misra, I., Bojanowski, P., Joulin, A., Brock, A., Rabbat, M., & Synnaeve, G. (2023). 
       *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. 
       arXiv:2301.08243.
    
    2. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). 
       *Masked Autoencoders Are Scalable Vision Learners*. 
       CVPR 2022.
    
    3. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). 
       *Emerging Properties in Self-Supervised Vision Transformers*. 
       ICCV 2021.
    """)
