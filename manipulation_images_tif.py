import os, rasterio, matplotlib, time
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from numpy import ndarray
from numpy.ma import MaskedArray



def recuperer_images(mean_monthly : bool = False, zone : int = 2, selected_dates : list[str] = ['20210816', '20210828']) -> list[MaskedArray]:
    
    if not (8 >= zone >=1):
        return images
    
    dir = f'./Data/Test_zone{zone}/{"STATS/MeanMonthly" if mean_monthly else "OASIS"}/'
    images = []

    if mean_monthly:

        date_months=set()
        for date in selected_dates:
            date_months.add(date[:6])

        selected_dates=date_months
        
    for date in selected_dates:

        for file in os.listdir(dir):

            if (date[:6] if mean_monthly else date) in file:

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

def test_segmentation(image : MaskedArray, fonction_segmentation : Callable[[MaskedArray], ndarray], titre_oasis : str = "Image au format OASIS", titre_segmente : str = "Segmentation de l'image") -> None:

    start = time.time()
    image_segmentee = fonction_segmentation(image)
    end = time.time()

    print(f"segmentation terminée en : {round(end - start,3)} secondes")

    _, plots = plt.subplots(1, 2, figsize=(10, 6))

    plot_oasis = plots[0]
    plot_segmente = plots[1]

    im = plot_oasis.imshow(image, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', ['white','gray','blue', 'magenta','red']), origin='upper')
    plt.colorbar(im, ax=plot_oasis)
    plot_oasis.set_title(titre_oasis)
    plot_oasis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plot_segmente.imshow(image_segmentee, cmap='gray', origin='upper')
    plot_segmente.set_title(titre_segmente)
    plot_segmente.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.show()

def tests_segmentation(fonction_segmentation : Callable[[MaskedArray], ndarray], annee : int = 2021, mean_monthly : bool = False) -> None:

    fig, plots = plt.subplots(16,12,figsize=(12, 8))
    
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

                    plot_oasis.imshow(image, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', ['white','gray','blue', 'magenta','red']), origin='upper')
                    plot_oasis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
                    plot_segmente.imshow(image_segmentee, cmap='gray', origin='upper')
                    plot_segmente.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

                    break
        
        print()

    fig.text(0.02, 0.5, 'Zone', ha='center', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.95, 'Mois', ha='center', va='center', fontsize=12, fontweight='bold')

    print(f"temps d'éxécution moyen de {fonction_segmentation.__name__} : {round(sum(temps_execution) / len(temps_execution),3)}")
    print(f"temps d'éxécution total de {fonction_segmentation.__name__} : {round(sum(temps_execution),3)}")

    plt.tight_layout()
    plt.savefig(f"{fonction_segmentation.__name__}_{annee}.png", dpi=500)
    plt.show()


if __name__ == "__main__": # tests

    images = recuperer_images()

    image = recuperer_image("./Data/Test_zone2/OASIS/s1a_fusion_ASC_161_20161228_oasis_VV_Offset55_Test_zone2.tif")
    
    test_segmentation(image, lambda image : image)
    tests_segmentation(lambda image : image)



