import matplotlib.pyplot as plt


def show_triplet(original, reconstruction, changed, titles=None, cmap="gray"):
    """Display three single-channel images side by side."""
    if titles is None:
        titles = ["Original", "Reconstruction", "Changed"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, image, title in zip(axes, [original, reconstruction, changed], titles):
        if hasattr(image, "detach"):
            image = image.detach().cpu()
        if getattr(image, "ndim", 0) == 4:
            image = image[0, 0]
        elif getattr(image, "ndim", 0) == 3:
            image = image[0]
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    return fig, axes
