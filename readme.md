# Projet Math-appli 1A — Segmentation d’images SAR
**Équipe 28**

## Description
Ce projet vise la **segmentation automatique de surfaces d’eau** à partir d’images  
**SAR Sentinel-1 (format OASIS)**.

Plusieurs méthodes de segmentation sont implémentées et comparées sur différentes  
**zones géographiques**, **mois** et **années**, à l’aide de métriques quantitatives.

L’objectif est d’évaluer la **robustesse des méthodes** selon les conditions
temporelles et spatiales.

---

## Table des matières
- [Structure du dépôt](#structure-du-dépôt)
- [Description des dossiers](#description-des-dossiers)
- [Méthodes implémentées](#méthodes-implémentées)
- [Données d’entrée / sortie](#données-dentrée--sortie)
- [Évaluation des performances](#évaluation-des-performances)
- [Utilisation](#utilisation)
- [Planning](#planning)
- [Rapport](#rapport)
- [Sujet](#sujet)
- [Auteurs](#auteurs)

---

## Structure du dépôt

### `codes/`
Scripts Python du projet :
- **`manipulation_images_tif.py`**  
  Chargement des images, affichage, calcul des métriques et génération des graphes.
- **`RF.py`**  
  Méthode **Random Forest** : entraînement, prédiction et évaluation.
- **`CVF_seuil.py`**  
  Segmentation **Chan–Vese** avec initialisation par seuillage.
- **`CV_scikit_modifie.py`**  
  Implémentation modifiée de l’algorithme Chan–Vese.
- **`seuilF.py`**  
  Méthode de segmentation par seuillage fixe.
- **`fonctions_images.py`**  
  Fonctions utilitaires pour le traitement des images.
- **`fonctions_tests.py`**  
  Fonctions de calcul des scores et métriques d’évaluation.

---

### `Data/`
- **`Test_zone*/`**  
  Images SAR Sentinel-1 (format OASIS), organisées par zone géographique.

---

### `GroundTruth_DYN/`
- **`Test_zone*/`**  
  Images de référence (vérité terrain) utilisées pour l’apprentissage et l’évaluation.

---

### `resultats/`
Résultats générés automatiquement par les scripts :
- **`entrainements_RF/`** : modèles Random Forest sauvegardés  
- **`graphes_scores/`** : graphes d’évaluation des performances  
- **`graphes_segmentation/`** : visualisations des segmentations  
- **`scores/`** : scores numériques (fichiers texte)

---

### `planning/`
- **`planning.pdf`**  
  Planning / diagramme de Gantt du projet.

---

### Fichiers principaux
- **`rapport.tex`** : rapport final du projet (LaTeX)  
- **`Ref.bib`** : bibliographie  
- **`sujet.pdf`** : sujet officiel du projet  
- **`README.md`** : documentation du projet



---

## Description des dossiers

### `codes/`
Contient l’ensemble des **scripts Python** du projet :
- prétraitement des images OASIS,
- implémentation des méthodes de segmentation,
- calcul des métriques,
- génération automatique des graphes.

Les scripts sont conçus pour fonctionner sur **8 zones** et sur plusieurs
**années (2021–2024)**.

---

### `Data/`
Images SAR Sentinel-1 au format `.tif` (OASIS), organisées par zone (`Test_zone1` à `Test_zone8`).

---

### `GroundTruth_DYN/`
Images de référence (vérité terrain) correspondant aux zones et dates disponibles.
Elles servent à l’apprentissage supervisé et à l’évaluation des performances.

---

### `resultats/`
Regroupe tous les résultats produits automatiquement :
- modèles entraînés,
- graphes de scores,
- visualisations des segmentations,
- fichiers texte récapitulant les scores.

---

### `planning/`
Contient le planning du projet (diagramme de Gantt) illustrant l’organisation
et la progression du travail.

---

## Méthodes implémentées

### Random Forest (supervisé)
- Classification **pixel-par-pixel**.
- Extraction de caractéristiques (intensité, statistiques locales, information temporelle, données colocalisées).
- Apprentissage à partir des **Ground Truth**.
- Bonne capacité de généralisation sur plusieurs zones et années.

### Chan–Vese (non supervisé)
- Méthode variationnelle basée sur la **minimisation d’une énergie**.
- Ne nécessite pas de données d’apprentissage.
- Implémentation modifiée avec **initialisation par seuillage**.
- Robuste aux contours flous, mais plus coûteuse en temps de calcul.

### Seuillage fixe
- Méthode simple utilisée comme **baseline**.
- Sert également de caractéristique pour certaines approches.

---

## Données d’entrée / sortie

### Entrées
- Images SAR Sentinel-1 (`.tif`)
- Images de référence (Ground Truth)
- Paramètres des méthodes :
  - Random Forest : nombre d’arbres, profondeur, taille minimale des feuilles
  - Chan–Vese : `mu`, `lambda1`, `lambda2`, tolérance, nombre d’itérations

### Sorties
- Images segmentées (eau / non-eau)
- Scores quantitatifs
- Graphes d’évolution des performances par zone et par année

---

## Évaluation des performances
Les performances sont évaluées à l’aide de :
- IoU (Intersection over Union),
- distance de Hamming,
- différence d’aire,
- précision,
- corrélation,
- similarité structurelle (SSIM).

Des graphes mensuels sont générés :
- **courbe bleue** : scores proches de 1 (bonne performance),
- **courbe rouge** : scores proches de 0 (mauvaise performance).

---

## Utilisation
Les scripts principaux se trouvent dans le dossier `codes/`.  
Ils permettent :
- d’entraîner les modèles,
- de lancer les segmentations,
- de calculer les scores,
- de générer automatiquement les graphes.

Les résultats sont enregistrés dans `resultats/`.

---

## Planning
Le planning détaillé du projet est disponible dans `planning/planning.pdf`.

---

## Rapport
Le rapport final est rédigé en **LaTeX** (`rapport.tex`) et présente :
- le cadre théorique,
- les méthodes implémentées,
- les résultats obtenus,
- une analyse critique.

---

## Sujet
Le sujet officiel du projet est disponible dans `sujet.pdf`.

---

## Auteurs
Projet réalisé par **l’équipe 28**  
Module **Math-Appli 1A — 2025**
