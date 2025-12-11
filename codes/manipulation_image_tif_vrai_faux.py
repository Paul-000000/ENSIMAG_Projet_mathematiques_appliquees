import os, rasterio, matplotlib, time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Callable
from numpy import ndarray
from numpy.ma import MaskedArray


INDICATEUR = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', ['white','gray','blue', 'magenta','red'])
INDICATEUR_BINAIRE = ListedColormap(['white', 'blue'])


# récupération d'images au format tif
def recuperer_images(mean_monthly : bool = False, zone : int = 2, selected_dates : list[str] = ['20210816', '20210828']) -> list[MaskedArray]:
    
    dir = f'./Data/Test_zone{zone}/{"STATS/MeanMonthly" if mean_monthly else "OASIS"}/'
    images = []

    if not (8 >= zone >=1):
        return images

    for date in selected_dates:

        for file in os.listdir(dir):

            if date in file:

                full_path = os.path.join(dir, file)
                images.append(recuperer_image(full_path))
                break

    return images

def recuperer_image(path : str) -> MaskedArray:
    
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


# affichage des tests
def test_segmentation(image : MaskedArray, fonction_segmentation : Callable[[MaskedArray], ndarray], titre_oasis : str = "Image au format OASIS", titre_segmente : str = "Segmentation de l'image") -> None:

    start = time.time()
    image_segmentee = fonction_segmentation(image)
    end = time.time()

    print(f"fonction {fonction_segmentation.__name__} terminée en : {round(end - start,3)} secondes")

    _, plots = plt.subplots(1, 2, figsize=(10, 6))

    plot_oasis = plots[0]
    plot_segmente = plots[1]

    im = plot_oasis.imshow(image, cmap=INDICATEUR, origin='upper')
    plt.colorbar(im, ax=plot_oasis)
    plot_oasis.set_title(titre_oasis)
    plot_oasis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plot_segmente.imshow(image_segmentee,cmap=INDICATEUR_BINAIRE , origin='upper')
    plot_segmente.set_title(titre_segmente)
    plot_segmente.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.show()

def tests_segmentation(fonction_segmentation : Callable[[MaskedArray], ndarray], annee : int = 2021, mean_monthly : bool = False, resolution : int = 500) -> None:

    fig, plots = plt.subplots(16,12,figsize=(12, 8))
    mois_annee = [
        "Janvier",
        "Février",
        "Mars",
        "Avril",
        "Mai",
        "Juin",
        "Juillet",
        "Août",
        "Septembre",
        "Octobre",
        "Novembre",
        "Décembre"
    ]

    plt.suptitle(f"Segmentation des Images pour l'Année {annee}", fontsize=16, fontweight='bold')
    temps_execution = []

    for zone in range(1,9):

        print(f"segmentation Zone {zone}/8 ", end="")
        dir = f'./Data/Test_zone{zone}/{"STATS/MeanMonthly" if mean_monthly else "OASIS"}/'

        for mois in range(1,13):

            date = f"{annee}{mois:02d}"
            print(f".", end="")

            for file in os.listdir(dir):

                if date in file:

                    full_path = os.path.join(dir, file)
                    image = recuperer_image(full_path)

                    start = time.time()
                    image_segmentee = fonction_segmentation(image)
                    end = time.time()
                    temps_execution.append(end-start)

                    plot_oasis = plots[(zone-1) * 2, mois-1]
                    plot_segmente = plots[(zone-1) * 2 + 1, mois-1]

                    plot_oasis.imshow(image, cmap=INDICATEUR, origin='upper')
                    plot_oasis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
                    plot_segmente.imshow(image_segmentee, cmap=INDICATEUR_BINAIRE, origin='upper')
                    plot_segmente.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

                    break
        
        print()

    for zone in range(1,9):    
        fig.text(0.02, 0.97 - (zone / 8) * 0.9, f"Zone {zone}", ha='center', va='center', rotation='vertical', fontsize=9, fontweight='bold')
    
    for mois in range(1,13):
        fig.text((mois / 12) * 0.98 -0.025, 0.93, f"{mois_annee[mois - 1]}", ha='center', va='center', fontsize=9, fontweight='bold')

    print(f"temps d'éxécution moyen de {fonction_segmentation.__name__} : {round(sum(temps_execution) / len(temps_execution),3)}")
    print(f"temps d'éxécution total de {fonction_segmentation.__name__} : {round(sum(temps_execution),3)}")
    print("affichage et sauvegarde du graphique")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"resultats_tests/{fonction_segmentation.__name__}_{annee}.png", dpi=resolution)
    plt.show()


# tests et indicateurs
def distance_hamming(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 0 signifie parfait
    
    return np.sum(image_segmentee_1 != image_segmentee_2) / (image_segmentee_1.shape[0] * image_segmentee_2.shape[1])

def test_correlation(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait

    arr1 = np.asarray(image_segmentee_1).ravel()
    arr2 = np.asarray(image_segmentee_2).ravel()

    mask = ~np.isnan(arr1) & ~np.isnan(arr2)

    arr1 = arr1[mask]
    arr2 = arr2[mask]

    if len(arr1) == 0:
        return float('nan')

    return np.corrcoef(arr1, arr2)[0, 1]

def difference(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait
    pass

def similarite_structurelle(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait
    pass

def vraie_detection(image_segmentee_1: ndarray, image_segmentee_2: ndarray) -> float:
   
    ref = np.asarray(image_segmentee_1).ravel()
    seg = np.asarray(image_segmentee_2).ravel()

    if ref.shape != seg.shape:
        raise ValueError("Les deux images doivent avoir la même taille")

    mask = ~np.isnan(ref) & ~np.isnan(seg)
    ref = ref[mask]
    seg = seg[mask]

    if ref.size == 0:
        return float('nan')

    ref_pos = ref > 0
    seg_pos = seg > 0

    TP = np.sum(ref_pos & seg_pos)   # vrais positifs
    FN = np.sum(ref_pos & ~seg_pos)  # faux négatifs

    denom = TP + FN
    if denom == 0:
        return float('nan')

    return float(TP / denom)


def fausse_detection(image_segmentee_1: ndarray, image_segmentee_2: ndarray) -> float:
   
    ref = np.asarray(image_segmentee_1).ravel()
    seg = np.asarray(image_segmentee_2).ravel()

    if ref.shape != seg.shape:
        raise ValueError("Les deux images doivent avoir la même taille")

    mask = ~np.isnan(ref) & ~np.isnan(seg)
    ref = ref[mask]
    seg = seg[mask]

    if ref.size == 0:
        return float('nan')

    ref_pos = ref > 0
    seg_pos = seg > 0

    TN = np.sum(~ref_pos & ~seg_pos)  # vrais négatifs
    FP = np.sum(~ref_pos & seg_pos)   # faux positifs

    denom = TN + FP
    if denom == 0:
        return float('nan')

    # Spécificité = TN / (TN + FP), 1 = aucune fausse détection
    return float(TN / denom)



if __name__ == "__main__": # tests

    def segmentation_test(image : MaskedArray) -> ndarray:

        return np.asarray(image)

    #images = recuperer_images()
    image = recuperer_image("./Data/Test_zone6/OASIS/s1a_fusion_ASC_161_20210118_oasis_VV_Offset55_Test_zone6.tif")
    
    print(test_correlation(image, image))

    test_segmentation(image, segmentation_test)
    tests_segmentation(segmentation_test)