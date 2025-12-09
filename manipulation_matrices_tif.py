import os, glob, rasterio, matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import MaskedArray

# chemin vers moyennes mensuelles: Test_zone1/STATS/MeanMonthly/
# chemin vers oasis non moyennés (tous les 6 ou 12 jours): Test_zone1/OASIS

data_dir = './Data/Test_zone2/OASIS/'
shp_dir = './Shapefiles/'                                
var_name = 'oasis'                            
cmap4 = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', ['white','gray','blue', 'magenta','red'])

def recuperer_matrices(year : int = 2021, selected_dates : list[str] = ['20210816', '20210828'], data_dir : str = './Data/Test_zone2/OASIS/',zone : str = 'Test_zone2.shp') -> list[tuple[str, MaskedArray]]:
    
    file_pattern = f'*_{year}*.tif' 
    tif_files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))

    # Dictionnaire : date (str AAAAMMJJ) → données + métadonnées
    data_dict = {}
    meta = None

    for tif_file in tif_files:

        filename = os.path.basename(tif_file)
        # Extraction de la date au format AAAAMMJJ

        parts = filename.split('_')
        date_str = None

        for p in parts:
            if len(p) == 8 and p.isdigit() and p.startswith(str(year)):
                date_str = p
                break
        if date_str is None:
            print(f"Date non trouvée dans {filename} → ignoré")
            continue

        with rasterio.open(tif_file) as src:
            band = src.read(1)                     # les fichiers contiennent une seule variable
            nodata = src.nodata
            
            # Gestion des NoData / NaN
            if nodata is not None:
                mask = band == nodata
            else:
                mask = np.isnan(band)
            data = np.ma.masked_where(mask, band)

            data_dict[date_str] = {
                'data': data,
                'transform': src.transform,
                'crs': src.crs
            }
            if meta is None:
                meta = src.meta.copy()

    nom_matrice = []

    for date_str in selected_dates:

        if date_str in data_dict:
            nom_matrice.append((date_str,data_dict[date_str]['data']))

    return nom_matrice

def afficher_oasis(titre : str, matrice : MaskedArray) -> None:
    
    plt.figure(figsize=(8, 8))
    plt.imshow(matrice, cmap=cmap4, origin='upper')
    plt.colorbar(label='Valeur')
    plt.title(titre)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def afficher_segmentation(matrice) -> None:
    
    plt.figure(figsize=(8, 8))
    plt.imshow(matrice, cmap='gray', origin='upper')
    plt.colorbar(label='Valeur')
    plt.title("segmentation")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

"""
date, matrice = recuperer_matrices()[0]

afficher_oasis(date, matrice)
#afficher_segmentation(matrice)
"""

