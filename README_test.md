# Projet Math-Appli 1A — Segmentation d’images

## Équipe 28 — Année 2025

Ce projet s’inscrit dans le cadre du module **Mathématiques Appliquées 1A**.  
L’objectif est d’étudier et de comparer différentes méthodes de **segmentation d’images**, appliquées à des images de type **OASIS / Mean Monthly**, en s’appuyant sur des **vérités terrain (Ground Truth)** et des **indicateurs quantitatifs de performance**.

Les méthodes principales étudiées sont :
- la **segmentation par Random Forest**,
- la **segmentation par Chan–Vese** (active contours),
- l’analyse des performances selon les **zones**, les **mois** et les **années**.

---

## Table des matières

- [Structure du dépôt](#structure-du-dépôt)
- [Description des dossiers](#description-des-dossiers)
- [Méthodes implémentées](#méthodes-implémentées)
- [Évaluation des performances](#évaluation-des-performances)
- [Planning](#planning)
- [Rapport](#rapport)
- [Sujet](#sujet)

---

## Structure du dépôt


---

## Description des dossiers

###  `codes/`

Ce dossier contient **l’ensemble des scripts Python** utilisés pour le projet :
- prétraitement des images OASIS,
- implémentation des méthodes de segmentation (Random Forest, Chan–Vese),
- calcul des scores (IoU, distance de Hamming, précision, etc.),
- génération automatique des graphes de performance.

Les scripts sont conçus pour fonctionner **sur toutes les zones (1 à 8)** et **sur plusieurs années (2021–2024)**.

---

###  `planning/`

Contient les éléments liés à l’organisation du projet :
- `planning.gantt` : planning du projet au format Gantt,
- `planning.png` : version exportée du planning pour une visualisation rapide.

Ce dossier permet de suivre l’avancement du travail et la répartition des tâches.

---

###  `resultats/`

Regroupe **tous les résultats produits par les scripts** :


- **`entrainements_RF/`**  
  Modèles Random Forest entraînés et sauvegardés.

- **`graphes_scores/`**  
  Graphiques des scores mensuels et annuels (par zone, par année, courbes convergeant vers 0 ou 1).

- **`graphes_segmentation/`**  
  Visualisations des segmentations obtenues (comparaison image OASIS / segmentation / ground truth).

- **`scores/`**  
  Fichiers texte récapitulant les scores moyens et détaillés.

---

## Méthodes implémentées

###  Random Forest
- Extraction de caractéristiques à partir des images (intensité, filtrage).
- Apprentissage supervisé à partir des Ground Truth.
- Segmentation binaire eau / non-eau.
- Évaluation multi-annuelle et multi-zones.

###  Chan–Vese
- Méthode de contours actifs sans détection explicite de contours.
- Initialisation par seuillage.
- Étude de la convergence et du temps de calcul.
- Comparaison qualitative et quantitative avec Random Forest.

---

## Évaluation des performances

Les performances sont évaluées à l’aide de plusieurs indicateurs :
- **IoU (Intersection over Union)**,
- **Distance de Hamming**,
- **Différence d’aire**,
- **Précision**,
- **Corrélation**,
- **Similarité structurelle (SSIM)**.

Des **graphes mensuels par zone et par année** sont générés :
- courbe **bleue** : scores proches de 1 (bonnes performances),
- courbe **rouge** : scores proches de 0 (mauvaises performances).

---

## Planning

Le planning détaillé du projet est disponible dans le dossier [`planning/`](planning/).  
Il présente les différentes étapes :
- prise en main du sujet,
- implémentation des méthodes,
- validation des résultats,
- rédaction du rapport.

---

## Rapport

- `rapport.tex` : rapport final rédigé en **LaTeX**,
- `Ref.bib` : bibliographie utilisée pour le projet.

Le rapport présente :
- le contexte théorique,
- les méthodes implémentées,
- les résultats obtenus,
- une analyse critique des performances.

---

## Sujet

- `sujet.pdf` : énoncé officiel du projet,
- `sujet_segmentation_image.md` : version Markdown du sujet pour consultation rapide.

---

## Auteurs

Projet réalisé par **l’équipe 28**  
Module **Math-Appli 1A — 2025**

---

