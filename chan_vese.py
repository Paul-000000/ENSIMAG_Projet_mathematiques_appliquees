from scipy.ndimage import gaussian_filter
from skimage.segmentation import chan_vese
from skimage.exposure import rescale_intensity
from manipulation_matrices_tif import *
import numpy as np


image = recuperer_matrices()[0][1]
image = image.filled(np.nan)

valid_mean = np.nanmean(image)
image_no_nan = np.nan_to_num(image, nan=valid_mean)

image_gray = image_no_nan.astype(float)

image_filtered = gaussian_filter(image_gray, sigma=2)

image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

segmentation_result = chan_vese(image_normalisee, mu=0.07, lambda1=1, lambda2=1, max_num_iter=200, tol=5e-4)

comparaison_OASIS_segmentation(image, segmentation_result)
