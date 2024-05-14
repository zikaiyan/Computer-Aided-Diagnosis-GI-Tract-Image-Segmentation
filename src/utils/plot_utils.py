import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np

def plot_masks(X, y, title=""):

    if X.shape[0] == 3:  # Assuming 3 channels
        X = np.transpose(X, (1, 2, 0))  # Convert X from (3, 128, 128) to (128, 128, 3)
    
    # Similarly adjust y if it is in the shape (3, 128, 128)
    if y.shape[0] == 3:
        y = np.transpose(y, (1, 2, 0)) 

    # Create custom colormaps for the masks
    cmap1 = ListedColormap(['none', 'red'])  # Mask 1 in red
    cmap2 = ListedColormap(['none', 'green'])  # Mask 2 in green
    cmap3 = ListedColormap(['none', 'blue'])  # Mask 3 in blue

    fig, ax = plt.subplots()
    # Display the grayscale image
    ax.imshow((X/255.)[:,:,0], cmap='gray')

    # Display the first mask
    ax.imshow(y[:,:,0], cmap=cmap1, alpha=0.5)

    # Display the second mask
    ax.imshow(y[:,:,1], cmap=cmap2, alpha=0.5)

    # Display the third mask
    ax.imshow(y[:,:,2], cmap=cmap3, alpha=0.5)

    # Create a legend for the masks
    red_patch = mpatches.Patch(color='red', label='Small Bowel')
    green_patch = mpatches.Patch(color='green', label='Large Bowel')
    blue_patch = mpatches.Patch(color='blue', label='Stomach')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.title(title)
    plt.show()

def display_comparison(img_s, pred_s, mask_s):
    num_images = len(img_s)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i in range(num_images):
        plt.subplot(num_images, 2, 2*i + 1)
        X, y = img_s[i], mask_s[i]
        if X.shape[0] == 3:  # Assuming 3 channels
            X = np.transpose(X, (1, 2, 0))  # Convert X from (3, 128, 128) to (128, 128, 3)
        
        # Similarly adjust y if it is in the shape (3, 128, 128)
        if y.shape[0] == 3:
            y = np.transpose(y, (1, 2, 0)) 

        cmap1 = ListedColormap(['none', 'red'])  # Mask 1 in red
        cmap2 = ListedColormap(['none', 'green'])  # Mask 2 in green
        cmap3 = ListedColormap(['none', 'blue'])  # Mask 3 in blue
    
        # Display the grayscale image
        plt.imshow((X/255.)[:,:,0], cmap='gray')
    
        # Display the masks
        plt.imshow(y[:, :, 0], cmap=cmap1, alpha=0.5)  # Display the first mask
        plt.imshow(y[:, :, 1], cmap=cmap2, alpha=0.5)  # Display the second mask
        plt.imshow(y[:, :, 2], cmap=cmap3, alpha=0.5)  # Display the third mask
    
        plt.title(f"Image {i+1} - Actual Mask")

        plt.subplot(num_images, 2, 2*i + 2)
        X, y = img_s[i], pred_s[i]
        if X.shape[0] == 3:  # Assuming 3 channels
            X = np.transpose(X, (1, 2, 0))  # Convert X from (3, 128, 128) to (128, 128, 3)
        
        # Similarly adjust y if it is in the shape (3, 128, 128)
        if y.shape[0] == 3:
            y = np.transpose(y, (1, 2, 0)) 
    
        # Display the grayscale image
        plt.imshow((X/255.)[:,:,0], cmap='gray')
    
        # Display the masks
        plt.imshow(y[:, :, 0], cmap=cmap1, alpha=0.5)  # Display the first mask
        plt.imshow(y[:, :, 1], cmap=cmap2, alpha=0.5)  # Display the second mask
        plt.imshow(y[:, :, 2], cmap=cmap3, alpha=0.5)  # Display the third mask
    
        plt.title(f"Image {i+1} - Prediction")

    plt.tight_layout()
    plt.show()
