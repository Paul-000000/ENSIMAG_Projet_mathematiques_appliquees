import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import chan_vese
from skimage.exposure import rescale_intensity
from manipulation_matrices_tif import recuperer_matrices
import numpy as np

image = recuperer_matrices()[0][1]
image_gray = image.astype(float)


image_filtered = gaussian_filter(image, sigma=2)

image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image),np.max(image)),out_range=(0,1))

segmentation_result = chan_vese(image_normalisee, mu=0.5, lambda1=1, lambda2=1, max_num_iter=200, tol=1e-3)





# Affichage de l'image d'entrée et de la segmentation
fig, axes = plt.subplots(1, 4, figsize=(12, 6))
ax = axes[0]
ax.imshow(image_gray, cmap="gray")
ax.set_title("Image en niveaux de gris")
ax.axis('off')

ax = axes[1]
ax.imshow(image_filtered, cmap="gray")
ax.set_title("Segmentation de Chan-Vese")
ax.axis('off')

ax = axes[2]
ax.imshow(image_normalisee, cmap="gray")
ax.set_title("Segmentation de Chan-Vese")
ax.axis('off')

ax = axes[3]
ax.imshow(segmentation_result, cmap="gray")
ax.set_title("Segmentation de Chan-Vese")
ax.axis('off')

plt.tight_layout()
plt.show()
