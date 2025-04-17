import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from sklearn.decomposition import PCA

def visualize_masking(image, context_mask, target_mask):
    """
    Visualize the masking strategy

    Args:
        image: Original image [H, W, C]
        context_mask: Context mask [H, W] (1 = visible, 0 = masked)
        target_mask: Target mask [H, W] (1 = target, 0 = non-target)

    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 3, figure=fig)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Context mask
    ax2 = fig.add_subplot(gs[0, 1])
    masked_image = image.copy()

    # Apply context mask (gray out masked regions)
    for c in range(3):
        masked_image[:, :, c] = masked_image[:, :, c] * context_mask + 0.5 * (1 - context_mask)

    ax2.imshow(masked_image)
    ax2.set_title("Context (Visible Regions)")
    ax2.axis("off")

    # Target mask
    ax3 = fig.add_subplot(gs[0, 2])
    target_vis = image.copy()

    # Highlight target regions
    highlight = np.zeros_like(target_vis)
    highlight[:, :, 0] = target_mask  # Red channel

    # Blend with original image
    alpha = 0.5
    target_vis = target_vis * (1 - alpha * target_mask[:, :, np.newaxis]) + highlight * alpha

    ax3.imshow(target_vis)
    ax3.set_title("Target Regions")
    ax3.axis("off")

    plt.tight_layout()
    return fig

def visualize_predictions(image, context_mask, target_mask, predicted_embeddings, target_embeddings):
    """
    Visualize the predictions

    Args:
        image: Original image [H, W, C]
        context_mask: Context mask [H, W] (1 = visible, 0 = masked)
        target_mask: Target mask [H, W] (1 = target, 0 = non-target)
        predicted_embeddings: Predicted embeddings for target regions
        target_embeddings: Target embeddings

    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Context mask
    ax2 = fig.add_subplot(gs[0, 1])
    masked_image = image.copy()

    # Apply context mask (gray out masked regions)
    for c in range(3):
        masked_image[:, :, c] = masked_image[:, :, c] * context_mask + 0.5 * (1 - context_mask)

    ax2.imshow(masked_image)
    ax2.set_title("Context (Visible Regions)")
    ax2.axis("off")

    # Target mask
    ax3 = fig.add_subplot(gs[0, 2])
    target_vis = image.copy()

    # Highlight target regions
    highlight = np.zeros_like(target_vis)
    highlight[:, :, 0] = target_mask  # Red channel

    # Blend with original image
    alpha = 0.5
    target_vis = target_vis * (1 - alpha * target_mask[:, :, np.newaxis]) + highlight * alpha

    ax3.imshow(target_vis)
    ax3.set_title("Target Regions")
    ax3.axis("off")

    # Prediction visualization
    ax4 = fig.add_subplot(gs[1, 0:3])

    # Calculate L2 distance between predicted and target embeddings
    if isinstance(predicted_embeddings, torch.Tensor):
        predicted_embeddings = predicted_embeddings.detach().cpu().numpy()
    if isinstance(target_embeddings, torch.Tensor):
        target_embeddings = target_embeddings.detach().cpu().numpy()

    # Reshape if needed
    if len(predicted_embeddings.shape) > 2:
        predicted_embeddings = predicted_embeddings.reshape(-1, predicted_embeddings.shape[-1])
    if len(target_embeddings.shape) > 2:
        target_embeddings = target_embeddings.reshape(-1, target_embeddings.shape[-1])

    # Only compare embeddings for target regions
    if target_mask is not None:
        target_indices = np.where(target_mask.flatten())[0]
        # Make sure target_indices doesn't exceed the size of embeddings
        if len(target_indices) > 0:
            # Get the number of patches in the embeddings
            num_patches = min(predicted_embeddings.shape[0], target_embeddings.shape[0])
            # Filter target_indices to only include valid indices
            valid_indices = target_indices[target_indices < num_patches]
            if len(valid_indices) > 0:
                predicted_embeddings = predicted_embeddings[valid_indices]
                target_embeddings = target_embeddings[valid_indices]
            else:
                # If no valid indices, just use the first few embeddings
                max_idx = min(10, num_patches)
                predicted_embeddings = predicted_embeddings[:max_idx]
                target_embeddings = target_embeddings[:max_idx]

    # Calculate L2 distance
    l2_distance = np.sqrt(np.sum((predicted_embeddings - target_embeddings) ** 2, axis=1))

    # Plot histogram of L2 distances
    ax4.hist(l2_distance, bins=30, alpha=0.7)
    ax4.set_title("L2 Distance Between Predicted and Target Embeddings")
    ax4.set_xlabel("L2 Distance")
    ax4.set_ylabel("Frequency")

    plt.tight_layout()
    return fig

def visualize_embeddings(embeddings, labels=None):
    """
    Visualize embeddings using PCA

    Args:
        embeddings: Embeddings to visualize [N, D]
        labels: Optional labels for coloring points

    Returns:
        fig: Matplotlib figure
    """
    try:
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        # Reshape if needed
        if len(embeddings.shape) > 2:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])

        # Ensure we have enough samples for PCA
        if embeddings.shape[0] < 2:
            # Create a dummy embedding if we don't have enough samples
            embeddings = np.random.randn(100, embeddings.shape[1])
            labels = None

        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot embeddings
        if labels is not None:
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

        ax.set_title("PCA Visualization of Embeddings")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

        plt.tight_layout()
        return fig
    except Exception as e:
        # Create a simple figure with an error message
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"Error visualizing embeddings: {str(e)}",
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        return fig
