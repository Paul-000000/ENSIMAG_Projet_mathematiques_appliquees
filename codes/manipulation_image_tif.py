import rasterio, matplotlib, time, glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Callable
from numpy import ndarray
from numpy.ma import MaskedArray
from skimage.metrics import structural_similarity
from sklearn.metrics import accuracy_score


DOSSIER_SORTIE = "resultats_tests"
INDICATEUR_OASIS = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', ['white','gray','blue', 'magenta','red'])
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

            image_reference = recuperer_image(chemin_image_ref).astype(int)

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

def premier_fichier_dossier(path : str) -> str | None :

    l = glob.glob(path)

    if l == [] :
        return None

    return l[0]


# affichage des tests
def test_segmentation(image_ref : tuple[MaskedArray, MaskedArray | None], fonction_segmentation : Callable[[MaskedArray], ndarray]) -> None:

    image = image_ref[0]
    image_reference = image_ref[1]

    start = time.time()
    image_segmentee = fonction_segmentation(image)
    end = time.time()

    print(f"fonction {fonction_segmentation.__name__} terminée en : {round(end - start,3)} secondes")
    print_scores(image_segmentee, image_reference)

    if image_reference is not None :
        _, plots = plt.subplots(1, 3, figsize=(14, 6))

    else :
        _, plots = plt.subplots(1, 2, figsize=(10, 6))

    plot_oasis = plots[0]
    plot_segmente = plots[1]

    im = plot_oasis.imshow(image, cmap=INDICATEUR_OASIS, origin='upper')
    plt.colorbar(im, ax=plot_oasis)
    plot_oasis.set_title("Image au format OASIS")
    plot_oasis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plot_segmente.imshow(image_segmentee,cmap=INDICATEUR_BINAIRE , origin='upper')
    plot_segmente.set_title(f"Segmentation de l'image avec\n{fonction_segmentation.__name__}")
    plot_segmente.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if image_reference is not None :
        plot_reference = plots[2]
        plot_reference.imshow(image_reference,cmap=INDICATEUR_BINAIRE , origin='upper')
        plot_reference.set_title(f"Image de référence")
        plot_reference.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.show()

def tests_segmentation(fonction_segmentation : Callable[[MaskedArray], ndarray], annee : int = 2021, mean_monthly : bool = True, resolution : int = 300) -> None:

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
            image_segmentee = fonction_segmentation(image)
            end = time.time()
            temps_execution.append(end-start)
            
            if chemin_image_ref is None :
                plot_oasis = plots[(zone-1) * 2, mois-1]
                plot_segmente = plots[(zone-1) * 2 + 1, mois-1]

            else :
                plot_oasis = plots[(zone-1) * 3, mois-1]
                plot_segmente = plots[(zone-1) * 3 + 1, mois-1]
                plot_ref = plots[(zone-1) * 3 + 2, mois-1]

                plot_ref.imshow(recuperer_image(chemin_image_ref).astype(int), cmap=INDICATEUR_BINAIRE, origin='upper')
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
    plt.savefig(f"{DOSSIER_SORTIE}/{fonction_segmentation.__name__}_{annee}.png", dpi=resolution)
    plt.show()


# tests et scores
def distance_hamming(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 0 signifie parfait
    
    if image_segmentee_1.shape != image_segmentee_2.shape:
        return float('nan')
    
    return np.sum(image_segmentee_1 != image_segmentee_2) / (image_segmentee_1.shape[0] * image_segmentee_2.shape[1])

def difference_aire(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 0 signifie parfait
    
    aire_1 = np.sum(image_segmentee_1)
    aire_2 = np.sum(image_segmentee_2)

    return abs(aire_1 - aire_2) / max(aire_1, aire_2)

def fausse_detection(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 0 signifie parfait
    
    ref = np.asarray(image_segmentee_1).ravel()
    seg = np.asarray(image_segmentee_2).ravel()

    if ref.shape != seg.shape:
        return float('nan')

    ref_pos = ref > 0
    seg_pos = seg > 0

    vrais_negatifs = np.sum(~ref_pos & ~seg_pos) # vrais négatifs
    faux_positifs = np.sum(~ref_pos & seg_pos) # faux positifs

    denom = vrais_negatifs + faux_positifs
    if denom == 0:

        if faux_positifs == 0:
            return 1.0
        
        return float('nan')

    return float(faux_positifs / denom)


def score_correlation(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait

    if image_segmentee_1.shape != image_segmentee_2.shape:
        return float('nan')
    
    arr1 = np.asarray(image_segmentee_1).ravel()
    arr2 = np.asarray(image_segmentee_2).ravel()

    mask = ~np.isnan(arr1) & ~np.isnan(arr2)

    arr1 = arr1[mask]
    arr2 = arr2[mask]

    if len(arr1) == 0:
        return float('nan')

    return np.corrcoef(arr1, arr2)[0, 1]

def similarite_structurelle(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait
    
    if image_segmentee_1.shape != image_segmentee_2.shape:
        return float('nan')

    score, _ = structural_similarity(image_segmentee_1.astype(float), image_segmentee_2.astype(float), data_range=1., full=True)
    return score

def vraie_detection(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait
    
    ref = np.asarray(image_segmentee_1).ravel()
    seg = np.asarray(image_segmentee_2).ravel()

    if ref.shape != seg.shape:
        return float('nan')

    ref_pos = ref > 0
    seg_pos = seg > 0

    vrais_positifs = np.sum(ref_pos & seg_pos) # vrais positifs
    faux_negatifs = np.sum(ref_pos & ~seg_pos) # faux négatifs

    denom = vrais_positifs + faux_negatifs
    if denom == 0:
        return float('nan')

    return float(vrais_positifs / denom)

def score_precision(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait

    if image_segmentee_1.shape != image_segmentee_2.shape:
        return float('nan')
    
    ref = (image_segmentee_1 > 0).astype(int).ravel()
    seg = (image_segmentee_2 > 0).astype(int).ravel()

    return accuracy_score(ref, seg)


def print_scores(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray):

    print("Résultat des scores :")

    print(f"Distance de Hamming : {round(distance_hamming(image_segmentee_1, image_segmentee_2),3)}")
    print(f"Différence d'aire : {round(difference_aire(image_segmentee_1, image_segmentee_2),3)}")
    print(f"Fausse détection : {round(fausse_detection(image_segmentee_1, image_segmentee_2),3)}\n")

    print(f"Vraie détection : {round(vraie_detection(image_segmentee_1, image_segmentee_2),3)}")
    print(f"Score de précision : {round(score_precision(image_segmentee_1, image_segmentee_2),3)}")
    print(f"Corrélation : {round(score_correlation(image_segmentee_1, image_segmentee_2),3)}")
    print(f"Similarité structurelle : {round(similarite_structurelle(image_segmentee_1, image_segmentee_2),3)}")
    
def moyenne_scores(fonction_segmentation : Callable[[MaskedArray], ndarray], annee : int = 2021) -> None:
    
    temps_execution = []
    
    hamming_vals = []
    diff_aire_vals = []
    fausse_vals = []

    vraie_vals = []
    accuracy_vals = []
    corr_vals = []
    ssim_vals = []

    for zone in range(1, 9):

        print(f"segmentation Zone {zone}/8 ", end="")

        dir_oasis = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'
        dir_gt = f'./GroundTruth_DYN/Test_zone{zone}/'

        for mois in range(1, 13):

            print(f".", end="")

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
            image_seg = fonction_segmentation(image_oasis)
            end = time.time()

            temps_execution.append(end-start)

            hamming_vals.append(distance_hamming(image_seg, image_gt))
            diff_aire_vals.append(difference_aire(image_seg, image_gt))
            fausse_vals.append(fausse_detection(image_seg, image_gt))

            vraie_vals.append(vraie_detection(image_seg, image_gt))
            accuracy_vals.append(score_precision(image_seg, image_gt))
            corr_vals.append(score_correlation(image_seg, image_gt))
            ssim_vals.append(similarite_structurelle(image_gt, image_gt))
            
        print()
    
    resultat = f"""
Scores moyens sur l'année {annee} :

Temps d'éxécution moyen          : {round(np.mean(temps_execution),3)}

Distance de Hamming moyenne      : {round(np.nanmean(hamming_vals),3)}
Différence d'aire moyenne        : {round(np.nanmean(diff_aire_vals),3)}
Fausse détection moyenne         : {round(np.nanmean(fausse_vals),3)}
    
Vraie détection moyenne          : {round(np.nanmean(vraie_vals),3)}
Score de précision moyen         : {round(np.nanmean(accuracy_vals),3)}
Corrélation moyenne              : {round(np.nanmean(corr_vals),3)}
Similarité structurelle moyenne  : {round(np.nanmean(ssim_vals),3)}
"""

    print(resultat)

    with open(f"{DOSSIER_SORTIE}/score {fonction_segmentation.__name__}.txt", "w") as f:
        
        f.write(resultat)


def moyenne_scores_anne(fonction_segmentation: Callable[[MaskedArray], ndarray],
                        annees: list[int] = [2021, 2022, 2023, 2024]) -> None:

    temps_execution = []

    hamming_vals = []
    diff_aire_vals = []
    fausse_vals = []

    vraie_vals = []
    accuracy_vals = []
    corr_vals = []
    ssim_vals = []

    for annee in annees:
        for zone in range(1, 9):

            print(f"segmentation Zone {zone}/8 ", end="")

            dir_oasis = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'
            dir_gt = f'./GroundTruth_DYN/Test_zone{zone}/'

            for mois in range(1, 13):

                print(f".", end="")

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
                image_seg = fonction_segmentation(image_oasis)
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

    resultat = f"""
Scores moyens sur toutes les années {annees} :

Temps d'éxécution moyen          : {round(np.mean(temps_execution), 3)}

Distance de Hamming moyenne      : {round(np.nanmean(hamming_vals), 3)}
Différence d'aire moyenne        : {round(np.nanmean(diff_aire_vals), 3)}
Fausse détection moyenne         : {round(np.nanmean(fausse_vals), 3)}

Vraie détection moyenne          : {round(np.nanmean(vraie_vals), 3)}
Score de précision moyen         : {round(np.nanmean(accuracy_vals), 3)}
Corrélation moyenne              : {round(np.nanmean(corr_vals), 3)}
Similarité structurelle moyenne  : {round(np.nanmean(ssim_vals), 3)}
"""

    print(resultat)

    with open(f"{DOSSIER_SORTIE}/score_{fonction_segmentation.__name__}.txt", "w") as f:
        f.write(resultat)



import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable
from numpy import ndarray
from numpy.ma import MaskedArray

from manipulation_image_tif import premier_fichier_dossier, recuperer_image, DOSSIER_SORTIE


def _to_binary(a):
    if hasattr(a, "filled"):
        a = a.filled(0)
    return (a > 0).astype(np.uint8)


def iou_score(pred, gt):
    p = _to_binary(pred).ravel()
    g = _to_binary(gt).ravel()
    inter = np.sum((p == 1) & (g == 1))
    union = np.sum((p == 1) | (g == 1))
    return 1.0 if union == 0 else inter / union


import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable
from numpy import ndarray
from numpy.ma import MaskedArray

from manipulation_image_tif import premier_fichier_dossier, recuperer_image, DOSSIER_SORTIE


def _to_binary(a):
    if hasattr(a, "filled"):
        a = a.filled(0)
    return (a > 0).astype(np.uint8)


def iou_score(pred, gt):
    p = _to_binary(pred).ravel()
    g = _to_binary(gt).ravel()
    inter = np.sum((p == 1) & (g == 1))
    union = np.sum((p == 1) | (g == 1))
    return 1.0 if union == 0 else inter / union


def graphe_scores_mensuels_2courbes_8subplots(fonction_segmentation: Callable[[MaskedArray], ndarray],
                                             annee: int = 2021,
                                             mean_monthly: bool = True,
                                             resolution: int = 250) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from manipulation_image_tif import premier_fichier_dossier, recuperer_image, DOSSIER_SORTIE

    def _to_binary(a):
        if hasattr(a, "filled"):
            a = a.filled(0)
        return (a > 0).astype(np.uint8)

    def iou_score(pred, gt):
        p = _to_binary(pred).ravel()
        g = _to_binary(gt).ravel()
        inter = np.sum((p == 1) & (g == 1))
        union = np.sum((p == 1) | (g == 1))
        return 1.0 if union == 0 else inter / union

    def score_0_meilleur(pred, gt):
        if "distance_hamming" in globals():
            return distance_hamming(pred, gt)
        if "difference_aire" in globals():
            return difference_aire(pred, gt)
        if "fausse_detection" in globals():
            return fausse_detection(pred, gt)
        return np.nan

    mois_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    x = np.arange(1, 13)

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    axes = axes.ravel()

    temps_execution = []

    for zone in range(1, 9):
        ax = axes[zone - 1]

        dir_oasis = f'./Data/Test_zone{zone}/{"STATS/MeanMonthly" if mean_monthly else "OASIS"}/'
        dir_gt = f'./GroundTruth_DYN/Test_zone{zone}/'

        s1 = []
        s0 = []

        for mois in range(1, 13):
            date = f"{annee}{mois:02d}"

            img_path = premier_fichier_dossier(f"{dir_oasis}*{date}*.tif")
            gt_path = premier_fichier_dossier(f"{dir_gt}*{date}*.tif")

            if img_path is None or gt_path is None:
                s1.append(np.nan)
                s0.append(np.nan)
                continue

            image = recuperer_image(img_path)
            image_gt = recuperer_image(gt_path).astype(int)

            start = time.time()
            image_seg = fonction_segmentation(image)
            end = time.time()
            temps_execution.append(end - start)

            s1.append(iou_score(image_seg, image_gt))
            s0.append(score_0_meilleur(image_seg, image_gt))

        ax.plot(x, s1, marker="o", color="blue", label="1 meilleur")
        ax2 = ax.twinx()
        ax2.plot(x, s0, marker="o", color="red", label="0 meilleur")

        ax.set_ylim(0, 1)
        ax.set_title(f"Zone {zone}")
        ax.grid(True, alpha=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(mois_labels, rotation=45, ha="right")

        if zone in (7, 8):
            ax.set_xlabel("Mois")
        ax.set_ylabel("Score (1 meilleur)")
        ax2.set_ylabel("Score (0 meilleur)")

    fig.suptitle(f"{fonction_segmentation.__name__} — Scores mensuels par zone ({annee})", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"{DOSSIER_SORTIE}/scores_8zones_2courbes_{fonction_segmentation.__name__}_{annee}.png",
                dpi=resolution)
    plt.show()

    if len(temps_execution) > 0:
        print(f"temps d'éxécution moyen : {round(float(np.mean(temps_execution)), 3)} s")

if __name__ == "__main__": # tests


    def segmentation_parfaite(image : MaskedArray) -> ndarray:

        image_ref = recuperer_image("GroundTruth_DYN/Test_zone2/Var_202108.tif").astype(int)

        return image_ref


    image_ref = recuperer_images(zone = 2, selected_dates=['20210816'])[0]
    #image_ref = recuperer_image("./Data/Test_zone6/OASIS/s1a_fusion_ASC_161_20210118_oasis_VV_Offset55_Test_zone6.tif")
    #image_segmentee = segmentation_parfaite(image_ref)

    #test_segmentation(image_ref, segmentation_parfaite)
    tests_segmentation(segmentation_parfaite)
    #moyenne_scores(segmentation_parfaite)




