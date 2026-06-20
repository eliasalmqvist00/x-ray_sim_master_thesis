import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import os
import matplotlib.pyplot as plt
from numpy.typing import NDArray

plt.rcParams["font.family"] = "Segoe UI"
 
def get_org_projection(path, scaling = 1, flip = False):
    img = np.array(Image.open(path), dtype=np.single) 

    if scaling > 1:
        h = img.shape[0] // scaling
        w = img.shape[1] // scaling

        img = np.array(
            Image.fromarray(img).resize((int(w), int(h))), dtype=np.single)
    if flip: 
        img = np.fliplr(img)    
    return img 

# Normilises the projection based on the background intensity
def normalize_with_row(img):
    white_value = img[:10,:].mean()
    return img / white_value

def get_histograms(img1, img2):
    max_val = max(float(np.max(img1)), float(np.max(img2)))
    min_val = min(float(np.min(img1)), float(np.min(img2)))

    range = (min_val, max_val)
    bins = 100
    
    hist_img1, _ = np.histogram(img1.flatten(), bins=bins, range=range)
    hist_img2, edges = np.histogram(img2.flatten(), bins=bins, range=range)
    hist_img1 = hist_img1 / np.sum(hist_img1)
    hist_img2 = hist_img2 / np.sum(hist_img2)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return hist_img1, hist_img2, bin_centers

# Histogram comparison of simulated and experimental projection
def histogram_plot(img_sim, img_org, save=False, title=None, path=None):
    hist_sim, hist_org, bin_centers = get_histograms(img_sim, img_org)
    HI = hi(img_sim, img_org)

    colors = ["#1c9d08ff", "#df9227ff"]
    plt.figure(figsize=(10,5))
    plt.plot(bin_centers, hist_org, label="Experimental Projection", color=colors[0])
    plt.plot(bin_centers, hist_sim, label="Simulated Projection", color=colors[1])
    plt.title(f"Histogram Comparison, HI = {HI}")
    plt.xlabel("Normalised Pixel Intensity", fontsize=18)
    plt.ylabel("Normalised Frequency", fontsize=18)
    plt.legend()
    if save: 
        plt.savefig(path / f"Histogram-{title}.png")
    plt.grid(True)
    plt.show()

# Line intensity plot comparison of simulated and experimental projcetion for a given detector row "row"
def two_line_plot(img_sim, img_org, row, save=False, title=None, path = None):

    colors = ["#1c9d08ff", "#df9227ff"]

    row_sim = img_sim[row, :]
    row_org = img_org[row, :]
    
    plt.figure(figsize=(10,5))

    plt.plot(row_org, label="Experimental Projection", color = colors[0])
    plt.plot(row_sim, label="Simulated Projection", color=colors[1])
    plt.title(f"Intensity profile")
    plt.xlabel("Pixel position", fontsize=24)
    plt.ylabel("Normalised Pixel Intensity", fontsize=24)
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(path / f"Intensity-profile-row{row}-{title}.png")
    plt.show()

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_org, cmap='gray')
    plt.axis('off')
    plt.axhline(row, color=colors[0], linewidth=1)
    plt.title("Experimental Projection")
    
    plt.subplot(1,2,2)
    plt.imshow(img_sim, cmap='gray')
    plt.axis('off')
    plt.axhline(row, color=colors[1], linewidth=1)
    plt.title("Simulated Projection")
    if save:
        plt.savefig(path / f"Images-profile-row{row}-{title}.png")
    plt.show()

# Root Mean Square Error metric
def rmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))

# Structural Similarity metric
def ssim(img1, img2):
    
    max_val = max(float(np.max(img1)), float(np.max(img2)))
    min_val = min(float(np.min(img1)), float(np.min(img2)))
    data_range = max(1e-12, max_val - min_val)

    return structural_similarity(im1=img1, im2=img2, data_range=data_range, gaussian_weights=True, use_sample_covariance = False)

# Histogram Intersection metric
def hi(img1, img2):
    hist_img1, hist_img2, _ = get_histograms(img1, img2)
    return np.sum(np.minimum(hist_img1, hist_img2))

# Print all of the obove metrics for img1 and img2
def get_metrics(img1, img2, label=None):
    if label != None:
        print(label)
    print("RMSE:" , rmse(img1, img2))
    print("SSIM:" , ssim(img1, img2))
    print("HI:"   , hi(img1, img2))