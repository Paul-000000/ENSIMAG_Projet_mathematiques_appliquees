import numpy as np
from skimage.util import view_as_windows
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from manipulation_images_tif import *


def segmentation_kmeans_features(image, patch=5, n_clusters=2):
    img = np.nan_to_num(image, nan=np.nanmean(image))
    img = img.astype(float)
    
    img_filtered = gaussian_filter(img, sigma=2)
 
    patches = view_as_windows(img_filtered, (patch, patch))
    H, W = patches.shape[:2]
    patches_reshaped = patches.reshape(-1, patch*patch)
  
    pca = PCA(n_components=5)
    features = pca.fit_transform(patches_reshaped)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_.reshape(H, W)

    return labels
if __name__ == "__main__": # tests

    image = recuperer_images(False, 5, ['20210106'])[0]
    test_segmentation(image, segmentation_kmeans_features, "image OASIS du 16/08/2021","Segmentation Kmeans")
    #tests_segmentation(fonction_segmentation=segmentation_kmeans_features)