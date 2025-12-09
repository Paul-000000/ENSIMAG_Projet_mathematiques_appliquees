import os, rasterio, matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import MaskedArray


def recuperer_matrices(mean_monthly : bool = False, zone : int = 2, selected_dates : list[str] = ['20210816', '20210828']) -> list[tuple[str, MaskedArray]]:
    
    dir = f'./Data/Test_zone{zone}/{"STATS/MeanMonthly" if mean_monthly else "OASIS"}/'
    dates_matrices = []

    for date in selected_dates:

        for file in os.listdir(dir):

            if date in file:

                full_path = os.path.join(dir, file)
                dates_matrices.append((date, recuperer_matrice(full_path)))
                break

    # Trie les résultats par ordre de date
    dates_matrices.sort(key=lambda x: x[0])

    return dates_matrices

def recuperer_matrice(path : str) -> MaskedArray:
    
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

def comparaison_OASIS_segmentation(image_oasis : MaskedArray, matrice_segmentee) -> None:
    
    _, plots = plt.subplots(1, 2, figsize=(12, 6))

    plot_oasis = plots[0]
    plot_segmente = plots[1]

    im = plot_oasis.imshow(image_oasis, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', ['white','gray','blue', 'magenta','red']), origin='upper') # type: ignore
    plt.colorbar(im, ax=plot_oasis)
    plot_oasis.set_title("Image au format OASIS")
    plot_oasis.set_xlabel('Longitude')
    plot_oasis.set_ylabel('Latitude')

    plot_segmente.imshow(matrice_segmentee, cmap='gray', origin='upper')
    plot_segmente.set_title("segmentation effectuée")
    plot_segmente.set_xlabel('Longitude')
    plot_segmente.set_ylabel('Latitude')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    date, matrice = recuperer_matrices()[1]

    matrice = recuperer_matrice("./Data/Test_zone2/OASIS/s1a_fusion_ASC_161_20161228_oasis_VV_Offset55_Test_zone2.tif")
    
    comparaison_OASIS_segmentation(matrice, matrice)



