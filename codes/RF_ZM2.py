import joblib
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.exposure import rescale_intensity
from seuilF import segmentation_seuillage_fixe
from sklearn.ensemble import RandomForestClassifier
from fonctions_images import *
from fonctions_tests import *


DOSSIER_ENTRAINEMENT = "modeles RF"


def extract_features(image : MaskedArray, colocated : MaskedArray, mois : int, zone : int) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=0) # np.nanmean(image_no_nan)
    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=2)
    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

    region_locale = 20
    moyenne_locale = uniform_filter(image_gray, size=region_locale)
    variance_locale = uniform_filter(image_gray**2, size=region_locale) - moyenne_locale**2
    
    seuil = segmentation_seuillage_fixe(image)

    image_mois = np.full(image.shape, (mois-1) / 11)
    image_zone = np.full(image.shape, (zone-1) / 7)

    features = np.stack([
        
        colocated,
        image_mois,
        image_zone,
        image_normalisee,
        variance_locale,
        seuil

    ], axis=-1)

    return features.reshape(-1, features.shape[-1])

def train_random_forest(images : list[list[list[MaskedArray]]], masks : list[list[list[MaskedArray]]], colocated : list[list[MaskedArray]], nb_arbres : int = 20, profondeur_max_arbre : int = 10, pixels_min_feuilles : int = 1) -> RandomForestClassifier:

    x_train = []
    y_train = []

    for zone in range(1,9):

        for mois in range(1,13):

            for img, mask in zip(images[zone -1][mois -1], masks[zone -1][mois -1]):

                x = extract_features(img, colocated[zone -1][mois -1], mois, zone)
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

def predict_segmentation(model : RandomForestClassifier , image : MaskedArray, colocated : MaskedArray, mois : int, zone : int) -> ndarray:

    x = extract_features(image, colocated, mois, zone)
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

def load_colocated_data() -> list[list[MaskedArray]]:
    
    annee = 2024
    images = [[None for _ in range(12)] for _ in range(8)]
    
    for zone in range(1, 9):
        
        dir = f"./NDWI_colocalise_avec_sar/Colocated_Images/Zone{zone}"
        for mois in range(1,13):

            chemin = premier_fichier_dossier(f"{dir}/*{annee}-{mois:02d}*")

            if chemin is not None :
                images[zone -1][mois -1] = recuperer_image(chemin)
    
    for zone in range(8):
        mois_valides = [i for i, img in enumerate(images[zone]) if img is not None]

        if not mois_valides:
            continue

        for mois in range(12):
            if images[zone][mois] is None:
                
                plus_proche = min(mois_valides, key=lambda x: abs(x - mois))
                images[zone][mois] = images[zone][plus_proche].copy()

    return images

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
    nom_entrainement = "modele RF ZM2 2023-2024"
    colocated = load_colocated_data()

    if entrainement :

        images, masks = load_training_data(annees=[2023,2024])
        
        print(f"images d'entraînement : {nb_elements(images)}")

        start = time.time()
        model = train_random_forest(images, masks, colocated, nb_arbres=20, profondeur_max_arbre=10, pixels_min_feuilles=1)
        end=time.time()

        print(f"temps d'entrainement {round(end-start,3)} secondes")
    
        save_model(model, nom_entrainement)


    model = load_model(nom_entrainement)
    verify_features(model)


    def segmentation_random_forest_ZM2(image : MaskedArray, zone : int, mois : int) -> ndarray:

        return predict_segmentation(model, image, colocated[zone -1][mois -1], mois, zone)
    

    def moyenne_scores_annees_ZM(fonction_segmentation: Callable[[MaskedArray, int, int], ndarray], annees: list[int] = [2021, 2022, 2023, 2024]) -> None:

        temps_execution = []

        hamming_vals = []
        diff_aire_vals = []
        fausse_vals = []

        vraie_vals = []
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
                    image_seg = fonction_segmentation(image_oasis, zone, mois)
                    end = time.time()

                    temps_execution.append(end - start)

                    hamming_vals.append(distance_hamming(image_seg, image_gt))
                    diff_aire_vals.append(difference_aire(image_seg, image_gt))
                    fausse_vals.append(fausse_detection(image_seg, image_gt))

                    vraie_vals.append(vraie_detection(image_seg, image_gt))
                    corr_vals.append(score_correlation(image_seg, image_gt))
                    ssim_vals.append(similarite_structurelle(image_seg, image_gt))

            print()

        temps_execution_moyen = round(np.mean(temps_execution),3)

        noms_scores_moyens = [
            "Distance de Hamming\nmoyenne",
            "Différence d'aire\nmoyenne",
            "Fausse détection\nmoyenne",
            "Vraie détection\nmoyenne",
            "Corrélation\nmoyenne",
            "Similarité structurelle\nmoyenne"
        ]

        scores_moyens = [
            round(np.nanmean(hamming_vals), 3),
            round(np.nanmean(diff_aire_vals), 3),
            round(np.nanmean(fausse_vals), 3),
            round(np.nanmean(vraie_vals), 3),
            round(np.nanmean(corr_vals), 3),
            round(np.nanmean(ssim_vals), 3)
        ]

        for i in range(len(noms_scores_moyens)):
            noms_scores_moyens[i] = noms_scores_moyens[i] + "\n" + str(scores_moyens[i])  

        plt.figure(figsize=(18, 7))
        plt.bar(noms_scores_moyens, scores_moyens, color=['red', 'red', 'red', 'blue', 'blue', 'blue'])
        plt.ylabel("Score moyen")
        plt.title(f"Scores moyens sur les années {annees}\nTemps d'éxécution moyen : {temps_execution_moyen} secondes")
        plt.savefig(f"{DOSSIER_SORTIE}/{DOSSIER_SCORES}/score {fonction_segmentation.__name__}.png", dpi=150)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def tests_segmentation_ZM(fonction_segmentation : Callable[[MaskedArray, int, int], ndarray], annee : int = 2021, mean_monthly : bool = True, resolution : int = 300) -> None:   

        if mean_monthly :
            fig, plots = plt.subplots(24,12,figsize=(10, 14))

        else :
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

            if mean_monthly :
                dir_mean_monthly = f"./GroundTruth_DYN/Test_zone{zone}/"

            for mois in range(1,13):
                
                print(f".", end="")

                chemin_image = None
                chemin_image_ref = None
                date = f"{annee}{mois:02d}"
                chemin_image = premier_fichier_dossier(f"{dir}*{date}*.tif")
                
                if mean_monthly :

                    chemin_image_ref = premier_fichier_dossier(f"{dir_mean_monthly}*{date}*.tif")
    
                image = recuperer_image(chemin_image)

                start = time.time()
                image_segmentee = fonction_segmentation(image, zone, mois)
                end = time.time()
                temps_execution.append(end-start)
                
                if chemin_image_ref is None :
                    plot_oasis = plots[(zone-1) * 2, mois-1]
                    plot_segmente = plots[(zone-1) * 2 + 1, mois-1]

                else :
                    plot_oasis = plots[(zone-1) * 3, mois-1]
                    plot_segmente = plots[(zone-1) * 3 + 1, mois-1]
                    plot_ref = plots[(zone-1) * 3 + 2, mois-1]

                    plot_ref.imshow(image_reference_binaire(recuperer_image(chemin_image_ref)), cmap=INDICATEUR_BINAIRE, origin='upper')
                    plot_ref.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    
                plot_oasis.imshow(image, cmap=INDICATEUR_OASIS, origin='upper')
                plot_oasis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            
                plot_segmente.imshow(image_segmentee, cmap=INDICATEUR_BINAIRE, origin='upper')
                plot_segmente.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
            print()

        for zone in range(1,9):    
            fig.text(0.02, 0.97 - (zone / 8) * 0.9, f"Zone {zone}", ha='center', va='center', rotation='vertical', fontsize=9, fontweight='bold')
        
        for mois in range(1,13):
            fig.text((mois / 12) * 0.98 -0.025, 0.93, f"{mois_annee[mois - 1]}", ha='center', va='center', fontsize=9, fontweight='bold')

        print(f"temps d'éxécution moyen de {fonction_segmentation.__name__} : {round(np.mean(temps_execution),3)} secondes")
        print("affichage et sauvegarde du graphique")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{DOSSIER_SORTIE}/{DOSSIER_GRAPHES_SEGMENTATION}/{fonction_segmentation.__name__}_{annee}.png", dpi=resolution)
        plt.show()

    def graphe_scores_ZM(fonction_segmentation: Callable[[MaskedArray, int, int], ndarray],
                    annees: list[int] = [2021, 2022, 2023, 2024],
                    mean_monthly: bool = True,
                    resolution: int = 250,
                    figsize: tuple[int, int] = (16, 14)) -> None:
        
        x = np.arange(1, len(annees) * 12 + 1)
        fig, axes = plt.subplots(8, 1, figsize=figsize, sharex=True)

        for i, zone in enumerate(range(1, 9)):
            
            print(f"Zone {zone}/8", end="")

            dir_oasis = f'./Data/Test_zone{zone}/{"STATS/MeanMonthly" if mean_monthly else "OASIS"}/'
            dir_gt = f'./GroundTruth_DYN/Test_zone{zone}/'
            ax = axes[i]
            s1, s0 = [], []

            for annee in annees:

                print(f".", end="")

                for mois in range(1, 13):
                    
                    date = f"{annee}{mois:02d}"
                    img_path = premier_fichier_dossier(f"{dir_oasis}*{date}*.tif")
                    gt_path = premier_fichier_dossier(f"{dir_gt}*{date}*.tif")

                    if img_path is None or gt_path is None:
                        continue

                    image = recuperer_image(img_path)
                    image_gt = image_reference_binaire(recuperer_image(gt_path))
                    image_seg = fonction_segmentation(image, zone, mois)

                    s1.append(scores_1(image_seg, image_gt))
                    s0.append(scores_0(image_seg, image_gt))

            print()

            ax.plot(x, s1, marker="o", color="blue", label="moyenne des scores qui tendent vers 1")
            ax.plot(x, s0, marker="o", color="red", label="moyenne des scores qui tendent vers 0")
            ax.set_ylim(0, 1)
            
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)
            
            annees_str = [str(annee) for annee in annees]
            annees_positions = [i*12 for i in range(len(annees))]

            ax.set_xticks(annees_positions)
            ax.set_xticklabels(annees_str, ha="right")
            ax.set_ylabel(f"Zone {zone}\n", fontsize=9)


        fig.suptitle(f"{fonction_segmentation.__name__} — Scores mensuels par zone et par année", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(f"{DOSSIER_SORTIE}/{DOSSIER_GRAPHES_SCORES}/graphe_scores {fonction_segmentation.__name__}.png", dpi=resolution)
        plt.show()


    tests_segmentation_ZM(segmentation_random_forest_ZM2,annee=2021)
    moyenne_scores_annees_ZM(segmentation_random_forest_ZM2, annees=[2021,2022])
    graphe_scores_ZM(segmentation_random_forest_ZM2, annees=[2021,2022])

