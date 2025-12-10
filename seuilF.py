from scipy.ndimage import gaussian_filter
import numpy as np
from manipulation_images_tif import *


def segmentation_seuillage_fixe(image: MaskedArray, seuil: float = 0.2, sigma: int = 2) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=np.nanmean(image_no_nan))

    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=sigma)

    segmentation_result_boolean = image_filtered > seuil

    segmentation_result = segmentation_result_boolean.astype(int)

    return segmentation_result


if __name__ == "__main__": # tests

    image = recuperer_images(False, 2, ['20210816'])[0]
    test_segmentation(image, segmentation_seuillage_fixe, "Image OASIS du 16/08/2021","Segmentation par Seuillage Fixe")

    tests_segmentation(segmentation_seuillage_fixe, annee=2021, mean_monthly=False)
