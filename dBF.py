import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from skimage.exposure import rescale_intensity
from manipulation_images_tif import *


def segmentation_seuillage_fixe(image: MaskedArray, seuil: float = 0.2, sigma: int = 2) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=np.nanmean(image_no_nan))

    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=sigma)

    segmentation_result_boolean = image_filtered < seuil

    segmentation_result = segmentation_result_boolean.astype(int)

    return segmentation_result


if __name__ == "__main__": # tests

    image = recuperer_images(False, 2, ['20210816'])[0]
    test_segmentation(image, segmentation_seuillage_fixe, "Image OASIS du 16/08/2021","Segmentation par Seuillage Fixe")

    tests_segmentation(segmentation_seuillage_fixe, annee=2021, mean_monthly=False)
"""
# ==================== PARAMÈTRE DU SEUIL À CALIBRER ====================
# Le seuil pour l'eau (sur images SAR en dB) doit être très bas.
# -18 dB est un seuil typique pour séparer l'eau du terrain en bande C.
# VOUS DEVEZ AJUSTER CETTE VALEUR EN FONCTION DE VOS DONNÉES.
SEUIL_FIXE_DB = 0.2
# ======================================================================

image = recuperer_images(False, 2, ['20210816'])[0]
image_gray = image.astype(float)
image_normalisee = None # Sera calculée plus tard pour l'affichage

# --- 2. Prétraitement : Filtrage Gaussien (Similaire à 2dBF) ---
# On conserve un sigma faible (ex: 2) pour lisser le speckle tout en préservant les contours.
sigma_filtre = 2 
image_filtered = gaussian_filter(image_gray, sigma=sigma_filtre)

# --- 3. Segmentation : Application du Seuillage Fixe ---
# La segmentation binaire se fait par simple comparaison.
# Si la rétrodiffusion (dB) est INFÉRIEURE au SEUIL_FIXE_DB, on détecte l'eau (True ou 1).
# Note : Nous travaillons ici sur l'image filtrée.
segmentation_result_boolean = image_filtered < SEUIL_FIXE_DB

# Convertir le résultat en entiers (0 ou 1) pour l'affichage en niveaux de gris standard
segmentation_result = segmentation_result_boolean.astype(int)

# --- Préparation des images pour l'affichage (normalisation si nécessaire) ---
# La normalisation n'est pas essentielle pour le seuillage, mais elle est conservée pour la 3e figure.
image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_gray), np.max(image_gray)), out_range=(0, 1))

# --- 4. Affichage de l'image d'entrée et de la segmentation ---
fig, axes = plt.subplots(1, 4, figsize=(16, 5)) # Agrandissement de la figure pour meilleure clarté

# Figure 1: Image d'entrée originale (dB)
ax = axes[0]
im0 = ax.imshow(image_gray, cmap="gray")
ax.set_title("1. Image originale (dB)")
ax.axis('off')
plt.colorbar(im0, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)

# Figure 2: Image filtrée (Prétraitement)
ax = axes[1]
im1 = ax.imshow(image_filtered, cmap="gray")
ax.set_title(f"2. Image Filtrée (Sigma={sigma_filtre})")
ax.axis('off')

# Figure 3: Image normalisée (pour comparaison)
ax = axes[2]
im2 = ax.imshow(image_normalisee, cmap="gray")
ax.set_title("3. Image Normalisée")
ax.axis('off')

# Figure 4: Résultat de la Segmentation (Masque binaire)
ax = axes[3]
im3 = ax.imshow(segmentation_result, cmap="gray")
ax.set_title(f"4. Seuillage Fixe (T={SEUIL_FIXE_DB} dB)")
ax.axis('off')

plt.tight_layout()
plt.show()
"""