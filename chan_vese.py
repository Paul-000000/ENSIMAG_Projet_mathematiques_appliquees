from scipy.ndimage import gaussian_filter
from skimage.segmentation import chan_vese
from skimage.exposure import rescale_intensity
from manipulation_matrices_tif import *
import numpy as np


image = recuperer_matrices()[0][1]
image = np.nan_to_num(image, nan=0)

image_gray = image.astype(float)

image_filtered = gaussian_filter(image_gray, sigma=2)

image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

segmentation_result = chan_vese(image_normalisee, mu=0.25, lambda1=1, lambda2=1, max_num_iter=200, tol=5e-4)
print(type(segmentation_result))

#afficher_segmentation(image)
#afficher_segmentation(image_gray)
#afficher_segmentation(image_filtered)
#afficher_segmentation(image_normalisee)
afficher_oasis("",image)
afficher_segmentation(segmentation_result)
