import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from sklearn.ensemble import RandomForestClassifier
from fonctions_images import *
from fonctions_tests import *



def extract_intensity_features(image : MaskedArray) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=0) # np.nanmean(image_no_nan)

    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=2)

    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

    return image_normalisee.reshape(-1, 1)  # Feature = intensité

def train_random_forest(images : list[MaskedArray], masks : list[MaskedArray]) -> RandomForestClassifier:

    x_train = []
    y_train = []

    for img, mask in zip(images, masks):

        x = extract_intensity_features(img)

        y = mask.filled(0).astype(int).reshape(-1) # classes 0/1

        x_train.append(x)
        y_train.append(y)
        
    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)

    print(f"Taille dataset entraînement : {x_train.shape, y_train.shape}")

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        random_state=0,
        n_jobs=-1,
        verbose=0 # 2
    )
    model.fit(x_train, y_train)

    return model

def predict_segmentation(model : RandomForestClassifier , image : MaskedArray) -> ndarray:

    x = extract_intensity_features(image)
    y_pred = model.predict(x)
    
    return y_pred.reshape(image.shape)

def load_training_images() -> tuple[list[MaskedArray],list[MaskedArray]]:

    dates = ['202101', '202102', '202103']
    images = recuperer_images(mean_monthly=True, zone=2, selected_dates=dates)
    masks = []

    for i in range(len(dates)) :

        masks.append(images[i][1])
        images[i] = images[i][0]

    return images, masks

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

    joblib.dump(model, f"{DOSSIER_SORTIE}/{filemane}.pkl")

def load_model(filemane : str) -> RandomForestClassifier:

    return joblib.load(f"{DOSSIER_SORTIE}/{filemane}.pkl")



if __name__ == "__main__":

    """
    images, masks = load_training_data(annees=[2021,2022])
    print(f"nombre d'images : {len(images)}/{len(masks)}")

    start = time.time()
    model = train_random_forest(images, masks)
    end=time.time()

    print(f"temps d'entrainement {round(end-start,3)} secondes")
   
    save_model(model,"entrainement RF 2021-2022")
    """

    model = load_model("entrainement RF 2021-2022")

    def segmentation_random_forest(image : np.ma.MaskedArray) -> np.ndarray:

        return predict_segmentation(model, image)
    
    #image_test = recuperer_images(mean_monthly=True,zone=5,selected_dates=['202407'])[0]
    #test_segmentation(image_test, segmentation_random_forest)

    #tests_segmentation(segmentation_random_forest,annee=2023)
    moyenne_scores_annees(segmentation_random_forest, annees=[2023,2024])
    graphe_scores(segmentation_random_forest)