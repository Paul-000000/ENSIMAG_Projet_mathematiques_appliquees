from scipy.ndimage import gaussian_filter
from skimage.segmentation import chan_vese
from manipulation_images_tif import *
from numpy import ndarray
import numpy as np


def segmentation_chan_vese_moyenne(image: MaskedArray, sigma=1.2, mu=0.15, lambda1=1, lambda2=1, max_num_iter=300, tol=1e-4) -> ndarray:
    
    # 1. Convertir l'image et gérer les NaN
    img = np.nan_to_num(image.filled(np.nan), nan=np.nanmean(image))

    # 2. Normalisation robuste (évite domination du bruit)
    vmin, vmax = np.nanpercentile(img, [3, 97])
    img_norm = (img - vmin) / (vmax - vmin)
    img_norm = np.clip(img_norm, 0, 1)

    # 3. Lissage pour réduire le bruit radar
    img_filt = gaussian_filter(img_norm, sigma=sigma)

    # 4. Initialisation intelligente : un échiquier (plus stable)
    init_ls = "checkerboard"

    # 5. Chan–Vese
    seg = chan_vese(
        img_filt,
        mu=mu,
        lambda1=lambda1,
        lambda2=lambda2,
        tol=tol,
        max_num_iter=max_num_iter,
        dt=0.5,
        init_level_set=init_ls,
        extended_output=False
    )

    # Sortie binaire
    seg_binary = seg.astype(np.uint8)

    #  Correction automatique : le lac doit être BLANC
    # Règle : l’eau a toujours une intensité plus élevée dans OASIS
    # donc on compare la moyenne des pixels segmentés vs non segmentés

    mean_inside = np.nanmean(image[seg_binary == 1])
    mean_outside = np.nanmean(image[seg_binary == 0])

    # Si le "1" ne correspond PAS au lac → on inverse
    if mean_inside < mean_outside:
        seg_binary = 1 - seg_binary  # inversion : éxterieur devient intérieur

    return seg_binary


if __name__ == "__main__": # tests
    
    #image = recuperer_images(False, 5, ['20210816'])[0]
    #test_segmentation(image, segmentatiosegmentation_chan_vese_moyennen_chan_vese, "image OASIS du 16/08/2021","Segmentation Chan-Vese")
    #tests_segmentation(segmentation_chan_vese_moyenne, annee=2021, mean_monthly=False)

    tests_segmentation(segmentation_chan_vese_moyenne)
    moyenne_scores_annees(segmentation_chan_vese_moyenne)