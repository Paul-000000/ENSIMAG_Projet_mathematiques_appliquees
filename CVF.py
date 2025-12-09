from scipy.ndimage import gaussian_filter
from skimage.segmentation import chan_vese
from skimage.exposure import rescale_intensity
from manipulation_images_tif import *
from numpy import ndarray
import numpy as np


def segmentation_chan_vese(image: MaskedArray, sigma=2, mu=0.07, lambda1=1, lambda2=1, max_num_iter=200, tol=5e-4) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=np.nanmean(image_no_nan))

    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=sigma)

    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

    segmentation_result = chan_vese(image_normalisee, mu=mu, lambda1=lambda1, lambda2=lambda2, max_num_iter=max_num_iter, tol=tol)

    return segmentation_result


if __name__ == "__main__": # tests

    image = recuperer_images(False, 5, ['20210816'])[0]
    comparaison_OASIS_segmentation(image, segmentation_chan_vese(image))
