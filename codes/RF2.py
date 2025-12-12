import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from sklearn.ensemble import RandomForestClassifier
from manipulation_images_tif import *


def extract_intensity_features(image):
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=np.nanmean(image_no_nan))

    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=2)

    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

    return image_normalisee.reshape(-1, 1)  # Feature = intensité

def train_random_forest(images, masks):

    X_train = []
    y_train = []

    for img, mask in zip(images, masks):

        x = extract_intensity_features(img)

        y = mask.filled(0).astype(int).reshape(-1)   # classes 0/1

        X_train.append(x)
        y_train.append(y)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    print(f"Taille dataset entraînement : {X_train.shape, y_train.shape}")

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        random_state=0,
        n_jobs=-1,
        verbose=2
    )
    model.fit(X_train, y_train)

    return model


def predict_segmentation(model, image):

    x = extract_intensity_features(image)
    y_pred = model.predict(x)
    return y_pred.reshape(image.shape)

def load_training_data(annees : list[int] = [2021]):

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

# joblib.dump(model, "mon_modele_random_forest.pkl")

# model = joblib.load("mon_modele_random_forest.pkl")

if __name__ == "__main__":

    """
    images = recuperer_images(
        mean_monthly=True,
        zone=2,
        selected_dates=['202101', '202102', '202103']
    )

    masks = []

    for i in range(len(images)) :
        images[i] = images[i][0]
        masks.append(images[i][1])
        
    
    masks = [
        recuperer_image("./GroundTruth_DYN/Test_zone2/Var_202101.tif"),
        recuperer_image("./GroundTruth_DYN/Test_zone2/Var_202102.tif"),
        recuperer_image("./GroundTruth_DYN/Test_zone2/Var_202103.tif"),
    ]
    """

    images, masks = load_training_data()
    print(f"nombre d'images : {len(images)}/{len(masks)}")

    start = time.time()
    model = train_random_forest(images, masks)
    end=time.time()

    print(f"temps d'entrainement {round(end-start,3)} secondes")
   
    #image_test = recuperer_images(mean_monthly=True,zone=5,selected_dates=['202407'])[0]

    def segmentation_random_forest(image : np.ma.MaskedArray) -> np.ndarray:

        return predict_segmentation(model, image)

    #test_segmentation(image_test, segmentation_random_forest)

    #true_y=recuperer_image("./GroundTruth_DYN/Test_zone5/Var_202407.tif")

    tests_segmentation(segmentation_random_forest,annee=2022)
    moyenne_scores_annees(segmentation_random_forest, annees=[2021,2022,2023,2024])