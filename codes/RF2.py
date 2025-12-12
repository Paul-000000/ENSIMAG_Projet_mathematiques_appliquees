import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

        X = extract_intensity_features(img)

        y = mask.filled(0).astype(int).reshape(-1)   # classes 0/1

        X_train.append(X)
        y_train.append(y)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    print("Taille dataset entraînement :", X_train.shape, y_train.shape)


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
    X = extract_intensity_features(image)
    y_pred = model.predict(X)
    return y_pred.reshape(image.shape)

def compute_accuracy(pred, mask):
    gt = mask.filled(0).astype(int)

    y_true = gt.reshape(-1)
    y_pred = pred.reshape(-1)

    acc = accuracy_score(y_true, y_pred)
    return acc


import os
from manipulation_images_tif import recuperer_image

def load_training_data():
    images_x = []
    images_y = []

    for zone in range(1, 9):

        dir_y = f'./GroundTruth_DYN/Test_zone{zone}/'
        dir_x = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'

        files_x = sorted(os.listdir(dir_x))
        files_y = sorted(os.listdir(dir_y))

        for file_x in files_x:
            if "MoyenneMensuelle" not in file_x:
                continue
            date_x = file_x[17:23]  
            
            for file_y in files_y:
                if "Var_" not in file_y:
                    continue
                date_y = file_y[4:10]  
                if (date_x == date_y) and "2021" in date_x:
                   
                    full_path_x = os.path.join(dir_x, file_x)
                    full_path_y = os.path.join(dir_y, file_y)
                    images_x.append(recuperer_image(full_path_x))
                    images_y.append(recuperer_image(full_path_y))
                    break  

    return images_x, images_y


if __name__ == "__main__":

   
    images = recuperer_images(
        mean_monthly=True,
        zone=2,
        selected_dates=['202101', '202102', '202103']
    )

    for i in range(len(images)) :
        images[i] = images[i][0]
   
    masks = [
        recuperer_image("./GroundTruth_DYN/Test_zone2/Var_202101.tif"),
        recuperer_image("./GroundTruth_DYN/Test_zone2/Var_202102.tif"),
        recuperer_image("./GroundTruth_DYN/Test_zone2/Var_202103.tif"),
    ]
    #images,masks=load_training_data()
    print(f"nombre d'images y: {len(images)}")
    print(f"nombre d'images x:{len(masks)}")

    start = time.time()
    model = train_random_forest(images, masks)
    end=time.time()
    print(f"temps d'entrainement {end-start}")
   
    image_test = recuperer_images(
        mean_monthly=True,
        zone=5,
        selected_dates=['202407']
    )[0]

    
    #segmentation = predict_segmentation(model, image_test[0])

    def segmentation_random_forest(image : np.ma.MaskedArray) -> np.ndarray:

        return predict_segmentation(model, image)

    #test_segmentation(image_test, segmentation_random_forest)

    #true_y=recuperer_image("./GroundTruth_DYN/Test_zone5/Var_202407.tif")
    #acc=compute_accuracy(segmentation,true_y)
    #print(f"accuracy {acc}")
    moyenne_scores(segmentation_random_forest)