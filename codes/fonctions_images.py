import rasterio, glob
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import numpy as np
from numpy import ndarray
from numpy.ma import MaskedArray


DOSSIER_SORTIE = "resultats_tests"
INDICATEUR_OASIS = LinearSegmentedColormap.from_list('mycmap', ['white','gray','blue', 'magenta','red'])
INDICATEUR_BINAIRE = ListedColormap(['white', 'blue'])


# récupération d'images au format tif
def recuperer_images(mean_monthly : bool = True, zone : int = 2, selected_dates : list[str] = ['20210816', '20210828']) -> list[tuple[MaskedArray, MaskedArray | None]]:
    
    dir = f'./Data/Test_zone{zone}/{"STATS/MeanMonthly" if mean_monthly else "OASIS"}/'
    dir_mean_monthly = f"./GroundTruth_DYN/Test_zone{zone}/"

    images_refs = []

    if not (8 >= zone >=1):
        return images_refs
    
    if mean_monthly :
        
        for i in range(len(selected_dates)) :
            selected_dates[i] = selected_dates[i][:6]

    for date in selected_dates:

        chemin_image = premier_fichier_dossier(f"{dir}*{date}*.tif")

        if mean_monthly :

            chemin_image_ref = premier_fichier_dossier(f"{dir_mean_monthly}*{date}*.tif")

            image_reference = image_reference_binaire(recuperer_image(chemin_image_ref))

            images_refs.append((recuperer_image(chemin_image),image_reference))
            
        else :
            images_refs.append((recuperer_image(chemin_image),None))
    
    return images_refs

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

def image_reference_binaire(image_ref : MaskedArray) -> MaskedArray:

    image_ref[image_ref < 1] = 0
    return image_ref


# autre fonction intermédiaire
def premier_fichier_dossier(path : str) -> str | None :

    l = glob.glob(path)

    if l == [] :
        return None

    return l[0]


if __name__ == "__main__": # tests

    image_ref = recuperer_images(zone = 2, selected_dates=['20210816'])[0]
    image_ref = recuperer_image("./Data/Test_zone6/OASIS/s1a_fusion_ASC_161_20210118_oasis_VV_Offset55_Test_zone6.tif")
    