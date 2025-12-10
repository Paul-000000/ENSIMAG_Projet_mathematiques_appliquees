import os
import time
import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from typing import Callable
from numpy import ndarray
from numpy.ma import MaskedArray


# ============================================================
# 1) Charger une seule image .tif
# ============================================================

def recuperer_image(path: str) -> MaskedArray:
    """Charge une image raster et gère les valeurs NoData."""

    with rasterio.open(path) as src:
        band = src.read(1)
        nodata = src.nodata

        # Gestion des NoData / NaN
        if nodata is not None:
            mask = band == nodata
        else:
            mask = np.isnan(band)

        data = np.ma.masked_where(mask, band)

    return data


# ============================================================
# 2) Charger plusieurs images selon zone + dates
# ============================================================

def recuperer_images(mean_monthly: bool = False,
                     zone: int = 2,
                     selected_dates: list[str] = ['20210816', '20210828']) -> list[MaskedArray]:
    
    base = "STATS/MeanMonthly" if mean_monthly else "OASIS"
    directory = f'./Data/Test_zone{zone}/{base}/'

    images = []

    if not os.path.exists(directory):
        print(f"Dossier introuvable : {directory}")
        return images

    for date in selected_dates:
        found = False

        for file in os.listdir(directory):
            if date in file and file.endswith(".tif"):
                images.append(recuperer_image(os.path.join(directory, file)))
                found = True
                break
        
        if not found:
            print(f"⚠️ Aucun fichier trouvé pour la date {date} dans Test_zone{zone}")

    return images


# ============================================================
# 3) Afficher le résultat d'une méthode de segmentation
# ============================================================

def test_segmentation(image: MaskedArray,
                      fonction_segmentation: Callable[[MaskedArray], ndarray],
                      titre_oasis: str = "Image OASIS",
                      titre_segmente: str = "Segmentation") -> None:

    start = time.time()
    image_segmentee = fonction_segmentation(image)
    end = time.time()

    print(f"Segmentation terminée en : {round(end - start, 3)} secondes")

    _, plots = plt.subplots(1, 2, figsize=(11, 6))

    # Image OASIS
    im = plots[0].imshow(
        image,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            'mycmap', ['white','gray','blue','magenta','red']
        ),
        origin='upper'
    )
    plt.colorbar(im, ax=plots[0])
    plots[0].set_title(titre_oasis)
    plots[0].axis("off")

    # Segmentation
    plots[1].imshow(image_segmentee, cmap='gray', origin='upper')
    plots[1].set_title(titre_segmente)
    plots[1].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Tester AUTOMATIQUEMENT Chan–Vese sur TOUTES les images
# ============================================================

def tests_segmentation_chanvese(annee: int = 2021, mean_monthly: bool = False):

    from chan_ves import segmentation_chan_vese
    import matplotlib.pyplot as plt
    import matplotlib

    base = "STATS/MeanMonthly" if mean_monthly else "OASIS"

    fig, plots = plt.subplots(16, 12, figsize=(18, 14))

    for zone in range(1, 9):

        directory = f'./Data/Test_zone{zone}/{base}/'
        if not os.path.exists(directory):
            print(f"⚠️ Dossier manquant : {directory}")
            continue

        print(f"\n=== ZONE {zone} ===")

        for mois in range(1, 13):

            date = f"{annee}{mois:02d}"
            image = None

            for file in os.listdir(directory):
                if date in file and file.endswith(".tif"):
                    image = recuperer_image(os.path.join(directory, file))
                    break

            row_oasis = (zone - 1) * 2
            row_seg = row_oasis + 1
            col = mois - 1

            if image is None:
                plots[row_oasis, col].axis("off")
                plots[row_seg, col].axis("off")
                continue

            # 🔥 REDUIRE LA TAILLE DE L'IMAGE POUR VITE CALCULER
            image_small = image[::2, ::2]

            # 🔥 Chan-Vese accéléré
            seg = segmentation_chan_vese(image_small, max_num_iter=50)

            # Affichage OASIS
            plots[row_oasis, col].imshow(
                image_small,
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                    'mycmap', ['white','gray','blue','magenta','red']
                ),
                origin='upper'
            )
            plots[row_oasis, col].set_title(f"Z{zone} M{mois}")
            plots[row_oasis, col].axis("off")

            # Affichage segmentation
            plots[row_seg, col].imshow(seg, cmap="gray", origin="upper")
            plots[row_seg, col].axis("off")

    plt.tight_layout()
    plt.show()
