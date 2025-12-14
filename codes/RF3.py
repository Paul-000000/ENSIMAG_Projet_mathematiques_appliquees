import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from sklearn.ensemble import RandomForestClassifier
from fonctions_images import *
from fonctions_tests import *
from skimage.feature import local_binary_pattern


def extract_intensity_features(image : MaskedArray) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=0) # np.nanmean(image_no_nan)

    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=2)

    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))
    
    return image_normalisee.reshape(-1, 1)  # Feature = intensité


import numpy as np
from numpy.ma import MaskedArray
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.exposure import rescale_intensity
from skimage.filters import sobel_h, sobel_v
from seuilF import segmentation_seuillage_fixe

def extract_intensity_texture_features(image: MaskedArray) -> np.ndarray:
    """
    Pour chaque pixel, on crée un vecteur de features :
    [intensité, gradient_x, gradient_y, variance_locale]
    """
    # Remplacer les masques par NaN et ensuite 0
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=0) # np.nanmean(image_no_nan)

    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=2)

    img_intensity = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))
    seuil=segmentation_seuillage_fixe(image)
    # Gradient horizontal et vertical
    # grad_x = sobel_h(img_float)
    # grad_y = sobel_v(img_float)

    # Variance locale avec filtre moyen
    window_size = 3  # taille de la fenêtre
    mean_local = uniform_filter(image_filtered, size=window_size)
    sq_mean_local = uniform_filter(image_filtered**2, size=window_size)
    var_local = sq_mean_local - mean_local**2
    var_local_norm = rescale_intensity(var_local, in_range='image', out_range=(0, 1))
    # Empiler les features pour chaque pixel
    # features = np.stack([img_intensity, grad_x, grad_y, var_local], axis=-1)
    # lbp = local_binary_pattern(image_gray, P=8, R=1)
    # features = np.stack([img_intensity,  var_local_norm,seuil], axis=-1)
    # features = np.stack([img_intensity, grad_x, grad_y], axis=-1)
    features = np.stack([img_intensity,seuil], axis=-1)
    # Retourner sous forme (nombre_pixels, nombre_features)
    return features.reshape(-1, features.shape[-1])


from sklearn.ensemble import RandomForestClassifier

# Exemple pour entraîner le modèle
def train_random_forest_with_texture(images, masks,n_estimators=20, max_depth=10):
    X_train, y_train = [], []

    for img, mask in zip(images, masks):
        X = extract_intensity_texture_features(img)
        y = mask.filled(0).astype(int).reshape(-1)
        X_train.append(X)
        y_train.append(y)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    model = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                      n_jobs=-1,
                                        random_state=0,
                                        verbose=2)
    model.fit(X_train, y_train)

    return model


def train_random_forest(images : list[MaskedArray], masks : list[MaskedArray],n_estimators=20,
        max_depth=10) -> RandomForestClassifier:

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
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=0,
        n_jobs=-1,
        verbose=2 # 2
    )
    model.fit(x_train, y_train)

    return model

def predict_segmentation(model : RandomForestClassifier , image : MaskedArray) -> ndarray:

    x = extract_intensity_texture_features(image)
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

def load_training_data(annees : list[int] = [2021],
                       zones:list[int]=[1,2,3,4,5,6,7,8]
                       ) -> tuple[list[MaskedArray],list[MaskedArray]]:

    images_x = []
    images_y = []

    for annee in annees:

        for zone in zones:

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

   
    # images, masks = load_training_data(annees=[2021,2022])
    # print(f"nombre d'images : {len(images)}/{len(masks)}")

    # start = time.time()
    # model = train_random_forest_with_texture(images, masks,n_estimators=30)
    # end=time.time()

    # print(f"temps d'entrainement {round(end-start,3)} secondes")
   
    # save_model(model,"RF_2021-2022_with_texture_with_seuil_only_tree30")
    

    model = load_model("RF_2021-2022_with_texture_with_seuil_only_tree30")
    print(model.feature_importances_)
    def segmentation_random_forest(image : np.ma.MaskedArray) -> np.ndarray:

        return predict_segmentation(model, image)
    
    #image_test = recuperer_images(mean_monthly=True,zone=5,selected_dates=['202407'])[0]
    #test_segmentation(image_test, segmentation_random_forest)

    # tests_segmentation(segmentation_random_forest,annee=2023)
    moyenne_scores_annees(segmentation_random_forest, annees=[2023,2024])
    # graphe_scores(segmentation_random_forest)
