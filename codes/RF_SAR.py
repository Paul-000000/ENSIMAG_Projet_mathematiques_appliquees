import joblib
import numpy as np
from os import listdir
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.exposure import rescale_intensity
from seuilF import segmentation_seuillage_fixe
from sklearn.ensemble import RandomForestClassifier
from fonctions_images import *
from fonctions_tests import *


DOSSIER_ENTRAINEMENT = "modeles RF"


def extract_features(image : MaskedArray) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=0) # np.nanmean(image_no_nan)
    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=2)
    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

    region_locale = 20
    moyenne_locale = uniform_filter(image_gray, size=region_locale)
    variance_locale = uniform_filter(image_gray**2, size=region_locale) - moyenne_locale**2
    
    seuil = segmentation_seuillage_fixe(image)

    features = np.stack([
        
        image_normalisee,
        variance_locale,
        seuil

    ], axis=-1)

    return features.reshape(-1, features.shape[-1])

def train_random_forest(images : list[MaskedArray], masks : list[MaskedArray],  nb_arbres : int = 20, profondeur_max_arbre : int = 10, pixels_min_feuilles : int = 1) -> RandomForestClassifier:

    x_train = []
    y_train = []

    for img, mask in zip(images, masks):

        x = extract_features(img)

        y = mask.filled(0).astype(int).reshape(-1) # classes 0/1

        x_train.append(x)
        y_train.append(y)
        
    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)

    print(f"Taille du dataset d'entraînement : {x_train.shape[0]}")

    model = RandomForestClassifier(
        n_estimators=nb_arbres,
        max_depth=profondeur_max_arbre,
        min_samples_leaf=pixels_min_feuilles,
        random_state=0,
        n_jobs=10,
        verbose=2 # 2
    )
    model.fit(x_train, y_train)

    return model

def predict_segmentation(model : RandomForestClassifier , image : MaskedArray) -> ndarray:

    x = extract_features(image)
    y_pred = model.predict(x)
    
    return y_pred.reshape(image.shape)

def verify_features(model : RandomForestClassifier) -> None:

    for i, imp in enumerate(model.feature_importances_):

        print(f"Importance feature {i+1} : {round(imp,3)}")

def load_training_data(annees : list[int] = [2021]) -> tuple[list[MaskedArray],list[MaskedArray]]:

    images_x = []
    images_y = []

    for annee in annees:

        for zone in range(1, 9):

            dir_y = f'./GroundTruth_DYN/Test_zone{zone}/'
            dir_x = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'

            for mois in range(1,13):

                date = f"{annee}{mois:02d}"
                chemin_image_x = premier_fichier_dossier(f"{dir_x}*{date}*.tif")
                chemin_image_y = premier_fichier_dossier(f"{dir_y}*{date}*.tif")
                
                if chemin_image_x is not None and chemin_image_y is not None :
                    
                    images_x.append(recuperer_image(chemin_image_x))
                    images_y.append(image_reference_binaire(recuperer_image(chemin_image_y)))

    return images_x, images_y

def save_model(model : RandomForestClassifier, filemane : str) -> None:

    joblib.dump(model, f"{DOSSIER_SORTIE}/{DOSSIER_ENTRAINEMENT}/{filemane}.pkl")

def load_model(filemane : str) -> RandomForestClassifier:

    return joblib.load(f"{DOSSIER_SORTIE}/{DOSSIER_ENTRAINEMENT}/{filemane}.pkl")

def nb_elements(liste : list[list[list[MaskedArray]]]) -> int :
    
    n = 0
    for l1 in liste :
        for l2 in l1 :
            n += len(l2)

    return n


if __name__ == "__main__":

    entrainement = True
    nom_entrainement = "modele RF SAR 2023-2024"

    if entrainement :

        images, masks = load_training_data(annees=[2023,2024])

        print(f"images d'entraînement : {nb_elements(images)}")

        start = time.time()
        model = train_random_forest(images, masks, nb_arbres=20, profondeur_max_arbre=10, pixels_min_feuilles=1)
        end=time.time()

        print(f"temps d'entrainement {round(end-start,3)} secondes")
    
        save_model(model, nom_entrainement)


    model = load_model(nom_entrainement)
    verify_features(model)

    def segmentation_random_forest_SAR(image : np.ma.MaskedArray) -> np.ndarray:

        return predict_segmentation(model, image)
    

    tests_segmentation(segmentation_random_forest_SAR,annee=2021)
    moyenne_scores_annees(segmentation_random_forest_SAR, annees=[2021,2022])
    graphe_scores(segmentation_random_forest_SAR, annees=[2021,2022])