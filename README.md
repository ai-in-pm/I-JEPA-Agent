# I-JEPA Demonstration

This project provides an interactive demonstration of I-JEPA (Image-based Joint-Embedding Predictive Architecture), a PhD-level Artificial Machine Intelligence (AMI) that showcases how I-JEPA works for learning visual representations from image data.

## Overview

I-JEPA is a self-supervised learning method for generating semantic image representations without relying on hand-crafted data augmentations often used in invariance-based methods. It operates on the principle of predicting the representations of target image blocks from the representation of a single context block within the same image. It is a non-generative model, differentiating itself from methods like Masked Autoencoders (MAE) by making predictions in an abstract representation space rather than pixel or token space.

This demonstration provides:
1. An interactive interface to explore I-JEPA's masking and prediction capabilities
2. Educational content explaining the key concepts and innovations
3. Technical details about the implementation and architecture

## Key Components

- **Context Encoder**: ViT processing masked image
- **Predictor**: Narrow ViT predicting masked region representations
- **Target Encoder**: EMA-updated ViT processing unmasked image
- **Multi-block Masking**: ~75% masking ratio
- **Latent Prediction**: Prediction in representation space, not pixel space

## Features of this Demonstration

- **Interactive Masking**: Adjust masking ratio, block size, and number of target blocks
- **Real-time Processing**: Process images through the I-JEPA model in real-time
- **Visualization**: Visualize masking strategy, predictions, and embeddings
- **Educational Content**: Learn about the key concepts and innovations of I-JEPA
- **Technical Details**: Explore the implementation details and architecture

## Installation

```bash
# Clone the repository (if applicable)
# git clone https://github.com/your-username/ijepa-demo.git
# cd ijepa-agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

To run the demonstration:

```bash
python run_demo.py
```

This will:
1. Start the Streamlit app
2. Open the demonstration in your browser

Additional options:

```bash
# Run on a specific port
python run_demo.py --port 8502

# Run in debug mode
python run_demo.py --debug
```

## Project Structure

```
IJEPA-Agent/
├── app.py                 # Main Streamlit application
├── run_demo.py            # Script to run the demonstration
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── demo/                  # Demo components
│   ├── ijepa_demo.py      # Core demonstration logic
│   ├── educational.py     # Educational content
│   └── technical_details.py # Technical details
├── models/                # Model architecture
│   └── ijepa_model.py     # I-JEPA model implementation
├── utils/                 # Utility functions
│   ├── masking.py         # Masking strategies
│   └── visualization.py   # Visualization utilities
```

## References

1. Assran, M., Duval, F., Bose, J., Misra, I., Bojanowski, P., Joulin, A., Brock, A., Rabbat, M., & Synnaeve, G. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. arXiv:2301.08243.

2. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022.

3. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). *Emerging Properties in Self-Supervised Vision Transformers*. ICCV 2021.
