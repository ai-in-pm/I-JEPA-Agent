import os
import sys
import streamlit as st
import numpy as np
from PIL import Image
import tempfile

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the script directory to the Python path
sys.path.append(script_dir)

from demo.ijepa_demo import IJEPADemo
from demo.educational import show_educational_content
from demo.technical_details import show_technical_details
from utils.visualization import visualize_masking, visualize_predictions, visualize_embeddings

# Set page configuration
st.set_page_config(
    page_title="I-JEPA Demonstration",
    page_icon="🖼️",
    layout="wide"
)

def main():
    st.title("I-JEPA: Image-based Joint-Embedding Predictive Architecture")
    st.subheader("An Interactive Demonstration")

    # Sidebar with explanation and navigation
    with st.sidebar:
        st.header("About I-JEPA")
        st.write("""
        I-JEPA is a self-supervised learning method for generating semantic image representations 
        without relying on hand-crafted data augmentations often used in invariance-based methods.
        
        It operates on the principle of predicting the representations of target image blocks from the 
        representation of a single context block within the same image, all in a learned latent space.
        """)

        st.header("Key Components")
        st.markdown("""
        - **Context Encoder**: ViT processing masked image
        - **Predictor**: Narrow ViT predicting masked region representations
        - **Target Encoder**: EMA-updated ViT processing unmasked image
        - **Multi-block Masking**: ~75% masking ratio
        - **Latent Prediction**: Prediction in representation space, not pixel space
        """)

        st.header("Navigation")
        page = st.radio(
            "Select a page:",
            ["Interactive Demo", "Educational Content", "Technical Details"]
        )

    # Display the selected page
    if page == "Interactive Demo":
        show_interactive_demo()
    elif page == "Educational Content":
        show_educational_content()
    elif page == "Technical Details":
        show_technical_details()

def show_interactive_demo():
    """
    Display the interactive I-JEPA demonstration
    """
    st.header("Interactive I-JEPA Demo")
    
    st.write("""
    This interactive demonstration shows how I-JEPA processes images and predicts representations
    for masked regions. You can upload your own image or use a sample image, adjust the masking
    parameters, and see how the model processes the image.
    """)
    
    # Initialize the demo
    demo = IJEPADemo()
    
    # Image selection
    st.subheader("Image Selection")
    
    image_source = st.radio(
        "Select image source:",
        ["Upload Image", "Sample Image", "Generate Synthetic Image"]
    )
    
    image_path = None
    
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_path = tmp_file.name
    
    elif image_source == "Sample Image":
        # Use a sample image
        sample_images = {
            "Sample 1": "https://images.unsplash.com/photo-1533450718592-29d45635f0a9?q=80&w=1000&auto=format&fit=crop",
            "Sample 2": "https://images.unsplash.com/photo-1543877087-ebf71fde2be1?q=80&w=1000&auto=format&fit=crop",
            "Sample 3": "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?q=80&w=1000&auto=format&fit=crop"
        }
        
        selected_sample = st.selectbox("Select a sample image:", list(sample_images.keys()))
        
        # Download the sample image
        import requests
        from io import BytesIO
        
        response = requests.get(sample_images[selected_sample])
        img = Image.open(BytesIO(response.content))
        
        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img.save(tmp_file, format="JPEG")
            image_path = tmp_file.name
    
    elif image_source == "Generate Synthetic Image":
        # Generate a synthetic image
        img = demo.generate_synthetic_image()
        
        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img.save(tmp_file, format="JPEG")
            image_path = tmp_file.name
    
    # Process parameters
    st.subheader("Masking Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        masking_ratio = st.slider("Masking Ratio (%)", 50, 95, 75)
    with col2:
        block_size = st.slider("Block Size", 16, 64, 32, step=8)
    with col3:
        num_target_blocks = st.slider("Number of Target Blocks", 1, 8, 4)
    
    # Process button
    if image_path and st.button("Process Image"):
        with st.spinner("Processing image with I-JEPA..."):
            # Process the image
            original_image, masked_image, context_mask, target_mask, predicted_embeddings, target_embeddings = demo.process_image(
                image_path, masking_ratio=masking_ratio/100, block_size=block_size, num_target_blocks=num_target_blocks
            )
            
            # Display results
            st.subheader("Results")
            
            # Masking visualization
            st.write("**Masking Visualization**")
            fig = visualize_masking(original_image, context_mask, target_mask)
            st.pyplot(fig)
            
            # Prediction visualization
            st.write("**Prediction Results**")
            fig = visualize_predictions(original_image, context_mask, target_mask, predicted_embeddings, target_embeddings)
            st.pyplot(fig)
            
            # Embedding visualization
            st.write("**Embedding Visualization**")
            fig = visualize_embeddings(predicted_embeddings.detach().cpu().numpy())
            st.pyplot(fig)
    
    # Quick explanation
    with st.expander("How does this demonstration work?"):
        st.write("""
        This interactive demonstration shows how I-JEPA processes image data:
        
        1. **Input**: An image is selected and processed
        2. **Masking**: Regions of the image are masked according to the specified parameters
        3. **Processing**: The masked image is processed through the I-JEPA model
        4. **Prediction**: The model predicts representations for the masked regions
        5. **Visualization**: The results are visualized, showing the original image, masked image, and predictions
        
        Note that this is a simplified demonstration. A full I-JEPA model would be trained on millions of images
        and would learn more sophisticated representations.
        
        The key insight of I-JEPA is that by predicting in a learned representation space rather than pixel space,
        the model focuses on semantic content rather than low-level details, resulting in more useful representations
        for downstream tasks.
        """)

if __name__ == "__main__":
    main()
