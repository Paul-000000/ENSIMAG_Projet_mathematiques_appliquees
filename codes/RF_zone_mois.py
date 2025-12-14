import joblib,os
import numpy as np
from os import listdir
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.exposure import rescale_intensity
from seuilF import segmentation_seuillage_fixe
from sklearn.ensemble import RandomForestClassifier
from fonctions_images import *
from fonctions_tests import *


DOSSIER_ENTRAINEMENT = "entrainements RF"


def extract_features(image : MaskedArray, colocated : MaskedArray, mois : int) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=0) # np.nanmean(image_no_nan)
    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=2)
    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

    region_locale = 20
    moyenne_locale = uniform_filter(image_gray, size=region_locale)
    variance_locale = uniform_filter(image_gray**2, size=region_locale) - moyenne_locale**2
    
    seuil = segmentation_seuillage_fixe(image)

    image_mois = np.full(image.shape, mois / 12)

    features = np.stack([
        
        colocated,
        image_mois,
        image_normalisee,
        variance_locale,
        seuil

    ], axis=-1)

    return features.reshape(-1, features.shape[-1])

def train_random_forest(images : list[list[list[MaskedArray]]], masks : list[list[list[MaskedArray]]], colocated : list[MaskedArray], nb_arbres : int = 20, profondeur_max_arbre : int = 10, pixels_min_feuilles : int = 1) -> RandomForestClassifier:

    x_train = []
    y_train = []

    for zone in range(1,9):

        for mois in range(1,13):

            for img, mask in zip(images[zone -1][mois -1], masks[zone -1][mois -1]):

                x = extract_features(img, colocated[zone -1], mois)
                y = mask.filled(0).astype(int).reshape(-1) # classes 0/1

                x_train.append(x)
                y_train.append(y)
        
    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)

    print(f"Taille dataset entraînement : {x_train.shape[0]}/{y_train.shape[0]}")

    model = RandomForestClassifier(
        n_estimators=nb_arbres,
        max_depth=profondeur_max_arbre,
        min_samples_leaf=pixels_min_feuilles,
        random_state=0,
        n_jobs=-1,
        verbose=2 # 2
    )
    model.fit(x_train, y_train)

    return model

def predict_segmentation(model : RandomForestClassifier , image : MaskedArray, zone : MaskedArray, mois : int) -> ndarray:

    x = extract_features(image, zone, mois)
    y_pred = model.predict(x)
    
    return y_pred.reshape(image.shape)

def verify_features(model : RandomForestClassifier) -> None:

    for i, imp in enumerate(model.feature_importances_):

        print(f"Importance feature {i+1} : {round(imp,3)}")

def load_training_data(annees : list[int] = [2021]) -> tuple[list[list[list[MaskedArray]]], list[list[list[MaskedArray]]]]:

    images_x = []
    images_y = []

    for zone in range(1, 9):

        dir_y = f'./GroundTruth_DYN/Test_zone{zone}/'
        dir_x = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'
        images_zone_x = []
        images_zone_y = []

        for mois in range(1,13):

            images_mois_x = []
            images_mois_y = []

            for annee in annees:

                date = f"{annee}{mois:02d}"
                chemin_image_x = premier_fichier_dossier(f"{dir_x}*{date}*.tif")
                chemin_image_y = premier_fichier_dossier(f"{dir_y}*{date}*.tif")
                
                if chemin_image_x is not None and chemin_image_y is not None :
                    
                    images_mois_x.append(recuperer_image(chemin_image_x))
                    images_mois_y.append(image_reference_binaire(recuperer_image(chemin_image_y)))
            
            images_zone_x.append(images_mois_x)
            images_zone_y.append(images_mois_y)

        images_x.append(images_zone_x)
        images_y.append(images_zone_y)

    return images_x, images_y

def load_colocated_data() -> list[MaskedArray]:

    images = []

    for zone in range(1, 9):
        
        dir = f"./NDWI_colocalise_avec_sar/Colocated_Images/Zone{zone}"
        #image_zone = []
        images.append(recuperer_image(os.path.join(dir,listdir(dir)[0])))

        #for chemin in listdir(dir):
        #    print(chemin)
        #    image_zone.append(recuperer_image(chemin))

        #images.append(image_zone)

    return images

def save_model(model : RandomForestClassifier, filemane : str) -> None:

    joblib.dump(model, f"{DOSSIER_SORTIE}/{DOSSIER_ENTRAINEMENT}/{filemane}.pkl")

def load_model(filemane : str) -> RandomForestClassifier:

    return joblib.load(f"{DOSSIER_SORTIE}/{DOSSIER_ENTRAINEMENT}/{filemane}.pkl")



if __name__ == "__main__":

    entrainement = True
    nom_entrainement = "entrainement RF 2021-2022 zone mois"
    colocated = load_colocated_data()

    if entrainement :

        images, masks = load_training_data(annees=[2021,2022])
        
        print(f"nombre d'images : {len(images)}/{len(masks)}")

        start = time.time()
        model = train_random_forest(images, masks, colocated, nb_arbres=20, profondeur_max_arbre=10, pixels_min_feuilles=1)
        end=time.time()

        print(f"temps d'entrainement {round(end-start,3)} secondes")
    
        save_model(model, nom_entrainement)


    model = load_model(nom_entrainement)
    verify_features(model)


    def segmentation_random_forest(image : MaskedArray, colocated : MaskedArray, mois : int) -> ndarray:

        return predict_segmentation(model, image, colocated, mois)
    

    def moyenne_scores_annees_zone_mois(fonction_segmentation: Callable[[MaskedArray, MaskedArray, int], ndarray], colocated : list[MaskedArray], annees: list[int] = [2021, 2022, 2023, 2024]) -> None:

        temps_execution = []

        hamming_vals = []
        diff_aire_vals = []
        fausse_vals = []

        vraie_vals = []
        accuracy_vals = []
        corr_vals = []
        ssim_vals = []

        for annee in annees:

            print(f"Année {annee} ", end="")

            for zone in range(1, 9):

                print(f".", end="")

                dir_oasis = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'
                dir_gt = f'./GroundTruth_DYN/Test_zone{zone}/'

                for mois in range(1, 13):

                    date = f"{annee}{mois:02d}"
                    img_path = premier_fichier_dossier(f"{dir_oasis}*{date}*.tif")

                    if img_path is None:
                        continue

                    gt_path = premier_fichier_dossier(f"{dir_gt}*{date}*.tif")

                    if gt_path is None:
                        continue
                    

                    image_oasis = recuperer_image(img_path)
                    image_gt = recuperer_image(gt_path).astype(int)

                    start = time.time()
                    image_seg = fonction_segmentation(image_oasis, colocated[zone - 1], mois)
                    end = time.time()

                    temps_execution.append(end - start)

                    hamming_vals.append(distance_hamming(image_seg, image_gt))
                    diff_aire_vals.append(difference_aire(image_seg, image_gt))
                    fausse_vals.append(fausse_detection(image_seg, image_gt))

                    vraie_vals.append(vraie_detection(image_seg, image_gt))
                    accuracy_vals.append(score_precision(image_seg, image_gt))
                    corr_vals.append(score_correlation(image_seg, image_gt))
                    ssim_vals.append(similarite_structurelle(image_seg, image_gt))

            print()

        temps_execution_moyen = round(np.mean(temps_execution),3)

        noms_scores_moyens = [
            "Distance de Hamming\nmoyenne",
            "Différence d'aire\nmoyenne",
            "Fausse détection\nmoyenne",
            "Vraie détection\nmoyenne",
            "Score de précision\nmoyen",
            "Corrélation\nmoyenne",
            "Similarité structurelle\nmoyenne"
        ]

        scores_moyens = [
            round(np.nanmean(hamming_vals), 3),
            round(np.nanmean(diff_aire_vals), 3),
            round(np.nanmean(fausse_vals), 3),
            round(np.nanmean(vraie_vals), 3),
            round(np.nanmean(accuracy_vals), 3),
            round(np.nanmean(corr_vals), 3),
            round(np.nanmean(ssim_vals), 3)
        ]

        for i in range(len(noms_scores_moyens)):
            noms_scores_moyens[i] = noms_scores_moyens[i] + "\n" + str(scores_moyens[i])  

        plt.figure(figsize=(18, 7))
        plt.bar(noms_scores_moyens, scores_moyens, color=['red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue'])
        plt.ylabel("Score moyen")
        plt.title(f"Scores moyens sur les années {annees}\nTemps d'éxécution moyen : {temps_execution_moyen} secondes")
        plt.savefig(f"{DOSSIER_SORTIE}/{DOSSIER_SCORES}/score {fonction_segmentation.__name__}.png", dpi=150)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()


    #image_test = recuperer_images(mean_monthly=True,zone=5,selected_dates=['202407'])[0]
    #test_segmentation(image_test, segmentation_random_forest)

    # tester la methode des seuils en feature
    # essayer les images colocalisées avec les dates les plus proches 

    #tests_segmentation(segmentation_random_forest,annee=2023)
    moyenne_scores_annees_zone_mois(segmentation_random_forest, colocated, annees=[2023,2024])
    #graphe_scores(segmentation_random_forest, annees=[2023,2024])

