from RF_SAR import *
from RF_Z import *
from RF_ZM import *
from fonctions_RF import *
from fonctions_scores import *
from RF_Z import load_colocated_data as colocated_Z
from RF_ZM import load_colocated_data as colocated_ZM


def graphes_scores(
        zones : list[int] = [3,8],
        mois : list[int] = np.arange(1,13),
        annees: list[int] = [2021, 2022],
        resolution: int = 250,
        figsize: tuple[int, int] = (16, 14)) -> None:
    
    x = np.arange(len(annees) * len(mois))
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)

    fonctions_segmentation = [segmentation_random_forest_SAR,segmentation_random_forest_Z,segmentation_random_forest_ZM]
    modeles = [load_model("modele RF SAR 2023-2024"),load_model("modele RF Z 2023-2024"),load_model("modele RF ZM 2023-2024")]
    colocated = [[],colocated_Z(),colocated_ZM()]
    couleurs = ['lightblue','blue','purple']
    noms = ["RF SAR","RF Z","RF ZM"]

    noms_scores_moyens = [
        "Distance de Hamming\nmoyenne",
        "Différence d'aire\nmoyenne",
        "Fausse détection\nmoyenne",
        "Vraie détection\nmoyenne",
        "Corrélation\nmoyenne",
        "Similarité structurelle\nmoyenne"
    ]

    eval_scores_moyens = [distance_hamming,difference_aire,fausse_detection,vraie_detection,score_correlation,similarite_structurelle]

    
    
    for i in range(6):
    
        ax = axes[i % 3][i // 3]

        for f in range(3):

            val_scores_moyens = []

            for annee in annees:
                for m in mois:
                    
                    scores_zone = []
                    
                    for zone in zones :
                        
                        dir_oasis = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'
                        dir_gt = f'./GroundTruth_DYN/Test_zone{zone}/'
                    
                        date = f"{annee}{m:02d}"
                        img_path = premier_fichier_dossier(f"{dir_oasis}*{date}*.tif")
                        gt_path = premier_fichier_dossier(f"{dir_gt}*{date}*.tif")

                        if img_path is None or gt_path is None:
                            continue

                        image = recuperer_image(img_path)
                        image_gt = image_reference_binaire(recuperer_image(gt_path))
                        
                        image_seg = fonctions_segmentation[f](image, modeles[f], colocated[f], zone, m)
                        scores_zone.append(eval_scores_moyens[i](image_seg, image_gt))

                    val_scores_moyens.append(np.nanmean(scores_zone))
                


            ax.plot(x, val_scores_moyens, marker="o", color=couleurs[f], label=noms[f])

        ax.set_title(noms_scores_moyens[i])
        #ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
        
        x_str = []
        for annee in annees:
        
            for m in mois:
                x_str.append(f"{MOIS_ANNEE[m - 1]} {annee}")

        x_positions = [j for j in range(len(annees) * len(mois))]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_str, ha="right", rotation = 45)


    plt.tight_layout(rect=[0, 0, 1, 0.98])
    #plt.title(f"Scores moyens sur les années {affichage_liste(annees)}\nmois de {affichage_liste([MOIS_ANNEE[m - 1] for m in mois])}\ndans les zones {affichage_liste(zones)}")
    plt.savefig(f"{DOSSIER_SORTIE}/graphe_scores.png", dpi=resolution)
    plt.show()

def segmentation(
        annee : int = 2021,
        zone : int = 8,
        mois : list[int] = [4,8],
        resolution : int = 300) -> None:   
        
    
    fig, plots = plt.subplots(2,4,figsize=(14, 9))

    plt.suptitle(f"Segmentation des Images pour l'Année {annee}\nmois de {affichage_liste([MOIS_ANNEE[m - 1] for m in mois])}\ndans la zone {zone}", fontsize=14, fontweight='bold')
    dir = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'
    dir_mean_monthly = f"./GroundTruth_DYN/Test_zone{zone}/"
    
    fonctions_segmentation = [segmentation_random_forest_SAR,segmentation_random_forest_Z,segmentation_random_forest_ZM]
    modeles = [load_model("modele RF SAR 2023-2024"),load_model("modele RF Z 2023-2024"),load_model("modele RF ZM 2023-2024")]
    colocated = [[],colocated_Z(),colocated_ZM()]
    noms = ["RF SAR","RF Z","RF ZM"]

    for y, m in enumerate(mois):
        
        date = f"{annee}{m:02d}"
        chemin_image = premier_fichier_dossier(f"{dir}/*{date}*.tif")
        chemin_image_ref = premier_fichier_dossier(f"{dir_mean_monthly}/*{date}*.tif")
        image = recuperer_image(chemin_image)

        for f in range(3):

            image_segmentee = fonctions_segmentation[f](image, modeles[f], colocated[f], zone, m)
            plot = plots[y,f]

            plot.imshow(image_segmentee, cmap=INDICATEUR_BINAIRE, origin='upper')
            plot.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plot.set_title(noms[f])

            if f == 0:
                plot.set_title(f"{MOIS_ANNEE[m - 1]} {annee}", loc="left")

        plot_ref = plots[y,3]
        plot_ref.imshow(image_reference_binaire(recuperer_image(chemin_image_ref)), cmap=INDICATEUR_BINAIRE, origin='upper')
        plot_ref.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plot_ref.set_title("image de référence")
            
    

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{DOSSIER_SORTIE}/segmentation.png", dpi=resolution)
    plt.show()


#graphes_scores()

segmentation()