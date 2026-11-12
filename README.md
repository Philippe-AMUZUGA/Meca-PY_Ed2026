# Méca-Py

**Avertissement :** les jeux de données de ce dépôt ont un sens mécanique donné par transformation : les noms de colonnes, et parfois les valeurs, sont modifiés pour obtenir des ordres de grandeur réalistes. Certains partent de jeux publics connus, d'autres sont générés par l'auteur. Les métamodèles et les lois issus des estimateurs d'apprentissage automatique de ces notebooks ne doivent pas servir en application réelle. Pour refaire l'apprentissage, chargez vos propres données : c'est la méthode qui compte.

Exercices, corrigés et jeux de données qui accompagnent l'ouvrage *Méca-Py* (édition 2026, version 0.1.0 du dépôt).

Python pour l'ingénierie mécanique, industrielle et productique : Python comme calculatrice d'ingénieur, puis NumPy, pandas, SciPy, SymPy et scikit-learn sur des données de production. Chaque chapitre a ses notebooks, sujets et corrigés, qui se rejouent sur les jeux de ce dépôt.

## À qui il sert

- Aux lecteurs du livre Méca-Py : les notebooks, les corrigés, les jeux et les modules de chaque chapitre sont ici, rien d'autre n'est à télécharger.
- À toute personne qui veut apprendre Python sur des données d'ingénierie, sans lire le livre.

Aucun prérequis en programmation. Pour la suite, le livre Méca-IA (machine learning pour l'ingénierie mécanique, industrielle et productique) a son propre dépôt, avec ses défis, ses corrigés et ses jeux : <https://github.com/philippe-amuzuga/Meca-IA_Ed2026>. Il suppose les notions de ce livre.

## Ce que vous y trouvez

- Les notebooks des chapitres 1 à 19, dans `python/`, un dossier par chapitre (`python/P<chapitre>/`) : un sujet et un corrigé pour chaque exercice ou étude. Chaque notebook de chapitre rejoue aussi les appels des fiches de fonction de ce chapitre, et ceux des chapitres 5 et 17 les figures du livre.
- Les jeux de données lus par ces notebooks, au format Excel, dans `donnees/`, présentés dans [`donnees/LISEZMOI.md`](donnees/LISEZMOI.md).
- Les modules du livre, `meca.py`, `meca_donnees.py` et `meca_style.py` (le style des figures), dans `outils/`, avec leurs tests dans `outils/tests/`.
- Les scripts qui fabriquent les jeux simulés `Qualite_soudure` et `Suivi_Production_Machines` (graine 42), dans `generateurs/` : `python generateurs/generer_qualite_soudure.py verifier donnees` (et `generateurs/generer_suivi_production_machines.py verifier donnees`) régénère le jeu, le compare au classeur et contrôle ses propriétés.
- [`requirements.txt`](requirements.txt) : les bibliothèques et leurs versions.
- `manifest.json` : la liste complète des notebooks, des jeux et des modules, d'où sont tirés les tableaux ci-dessous.

## Parcours en une semaine

Le livre propose de lire les dix-neuf chapitres en sept journées, dans l'ordre, à raison de deux ou trois chapitres par jour. Les notebooks suivent le même parcours :

| Journée | Chapitres | Au programme |
|---|---|---|
| 1 | 1 à 3 | Bases du langage, calculs, listes, boucles, fonctions et tests (étude I : bielle) |
| 2 | 4 à 6 | Tirages et Monte-Carlo, figures d'ingénieur, calcul sur tableaux (étude II : ressorts) |
| 3 | 7 à 9 | Lecture et nettoyage, regroupements et jointures, mesures datées (étude III : tableau de bord) |
| 4 | 10 à 12 | Calcul symbolique, systèmes linéaires et modes, équations et optimisation |
| 5 | 13 à 15 | Intégration et équations différentielles, interpolation d'abaques, dispersion et statistique |
| 6 | 16 et 17 | Filtrage et spectres (étude IV : deux masses), exploration des données |
| 7 | 18 et 19 | Préparation des données, du script au premier modèle (étude V : suivi de production) |

Un lecteur qui programme déjà peut survoler le chapitre 1. Les chapitres 17 à 19 préparent le livre Méca-IA : on les lit après les précédents. Les durées de chaque journée sont données dans la section « Parcours » de l'avant-propos du livre.

## Installer et lancer

Il faut Python 3.12. Les notebooks et les modules se lancent depuis le dépôt cloné, dont la racine contient `donnees/` et `outils/` :

```
git clone https://github.com/philippe-amuzuga/Meca-PY_Ed2026.git
cd Meca-PY_Ed2026
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

Sous Linux ou macOS, la quatrième ligne devient `source .venv/bin/activate`. L'environnement virtuel est facultatif ; les versions de [`requirements.txt`](requirements.txt) sont celles avec lesquelles les corrigés ont été exécutés.

Chaque notebook lit ses jeux dans le dossier `donnees/` de la racine, par un chemin relatif résolu dans sa première cellule de code. Les blocs de code des chapitres 5, 17 et 19 du livre importent les modules du dossier `outils/` : ils se lancent depuis le dépôt cloné, après avoir ajouté `outils/` à la liste des dossiers où Python cherche un module, comme le décrivent ces chapitres, puis `import meca_style as ms` ou `import meca`.

## Dans Colab

Cliquer sur le bouton « Ouvrir dans Colab » du tableau des notebooks. Colab ouvre le notebook depuis GitHub, à la référence `main` du dépôt. Les bibliothèques usuelles y sont déjà installées.

Un notebook ouvert seul ne trouve ni les jeux ni les modules du dépôt. Deux cellules en tête de chaque notebook y pourvoient :

- la première cellule de code, pour les notebooks qui lisent des jeux, récupère ces jeux dans le dépôt public ; c'est la seule cellule qui accède au réseau ;
- la cellule « Préparation du dépôt (Colab) » ne télécharge rien : elle cherche `donnees/` et `outils/`, ajoute `outils/` au chemin de recherche de Python et, sur Colab, liste ce qu'il faut téléverser. Pour le faire, ouvrir le panneau des fichiers de Colab, créer les dossiers `donnees` et `outils` à la racine de l'espace de travail et y téléverser les classeurs et les modules cités.

Les notebooks des chapitres 5 et 19 embarquent le texte du module qu'ils importent et l'écrivent dans un dossier temporaire ; ceux des chapitres 17 et 18 reprennent le code des fonctions du module qu'ils emploient. Tous se lancent donc aussi hors du dépôt cloné.

## Correspondance chapitre, notebook, jeu

### Méca-Py : exercices et études par chapitre

Chaque notebook se juge sur des résultats attendus : le corrigé s'exécute deux fois de suite avec les mêmes sorties, et ces sorties sont celles que le livre annonce. Le sujet se dérive du corrigé.

| Notebook | Partie | Chapitre, section | Problème | Jeux lus | Contrôles | Sujet | Corrigé |
|---|---|---|---|---|---|---|---|
| P01-1 | I | 1, 1 | Les bases de Python | aucun | 7 | [sujet](python/P01/P01-1_Premiers_Pas.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P01/P01-1_Premiers_Pas.ipynb) | [corrigé](python/P01/P01-1_Premiers_Pas_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P01/P01-1_Premiers_Pas_CORRIGE.ipynb) |
| P02-1 | I | 2, 1 | Calculs, listes et boucles | aucun | 6 | [sujet](python/P02/P02-1_Calculer_Parcourir.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P02/P02-1_Calculer_Parcourir.ipynb) | [corrigé](python/P02/P02-1_Calculer_Parcourir_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P02/P02-1_Calculer_Parcourir_CORRIGE.ipynb) |
| P03-1 | I | 3, 1 | Fonctions de calcul | aucun | 3 | [sujet](python/P03/P03-1_Fonctions.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P03/P03-1_Fonctions.ipynb) | [corrigé](python/P03/P03-1_Fonctions_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P03/P03-1_Fonctions_CORRIGE.ipynb) |
| P03-2 | I | 3, 2 | Étude I : bielle en traction | aucun | 4 | [sujet](python/P03/P03-2_Etude_Bielle.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P03/P03-2_Etude_Bielle.ipynb) | [corrigé](python/P03/P03-2_Etude_Bielle_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P03/P03-2_Etude_Bielle_CORRIGE.ipynb) |
| P04-1 | II | 4, 1 | Monte-Carlo avec NumPy | aucun | 3 | [sujet](python/P04/P04-1_Monte_Carlo.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P04/P04-1_Monte_Carlo.ipynb) | [corrigé](python/P04/P04-1_Monte_Carlo_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P04/P04-1_Monte_Carlo_CORRIGE.ipynb) |
| P05-1 | II | 5, 1 | Figures d'ingénieur | aucun | 5 | [sujet](python/P05/P05-1_Figures.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P05/P05-1_Figures.ipynb) | [corrigé](python/P05/P05-1_Figures_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P05/P05-1_Figures_CORRIGE.ipynb) |
| P06-1 | II | 6, 1 | Calcul sur tableaux | aucun | 4 | [sujet](python/P06/P06-1_Tableaux.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P06/P06-1_Tableaux.ipynb) | [corrigé](python/P06/P06-1_Tableaux_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P06/P06-1_Tableaux_CORRIGE.ipynb) |
| P06-2 | II | 6, 2 | Étude II : ressorts de compression | aucun | 5 | [sujet](python/P06/P06-2_Etude_Ressort.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P06/P06-2_Etude_Ressort.ipynb) | [corrigé](python/P06/P06-2_Etude_Ressort_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P06/P06-2_Etude_Ressort_CORRIGE.ipynb) |
| P07-1 | III | 7, 1 | Lecture et nettoyage des mesures | [AI4I_maintenance](donnees/MecaIA-AI4I_maintenance.xlsx), [AI4I_maintenance_TypePanne](donnees/MecaIA-AI4I_maintenance_TypePanne.xlsx), [RUL](donnees/MecaIA-RUL.xlsx), [Suivi_Production_Machines](donnees/MecaIA-Suivi_Production_Machines.xlsx) | 4 | [sujet](python/P07/P07-1_Lire_Nettoyer.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P07/P07-1_Lire_Nettoyer.ipynb) | [corrigé](python/P07/P07-1_Lire_Nettoyer_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P07/P07-1_Lire_Nettoyer_CORRIGE.ipynb) |
| P08-1 | III | 8, 1 | Regroupements et jointures | [AI4I_maintenance](donnees/MecaIA-AI4I_maintenance.xlsx), [AI4I_maintenance_TypePanne](donnees/MecaIA-AI4I_maintenance_TypePanne.xlsx), [RUL](donnees/MecaIA-RUL.xlsx), [TimeSeries_RUL](donnees/MecaIA-TimeSeries_RUL.xlsx) | 3 | [sujet](python/P08/P08-1_Regrouper_Joindre.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P08/P08-1_Regrouper_Joindre.ipynb) | [corrigé](python/P08/P08-1_Regrouper_Joindre_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P08/P08-1_Regrouper_Joindre_CORRIGE.ipynb) |
| P09-1 | III | 9, 1 | Mesures datées | [AI4I_maintenance](donnees/MecaIA-AI4I_maintenance.xlsx), [AI4I_maintenance_TypePanne](donnees/MecaIA-AI4I_maintenance_TypePanne.xlsx) | 3 | [sujet](python/P09/P09-1_Mesures_Datees.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P09/P09-1_Mesures_Datees.ipynb) | [corrigé](python/P09/P09-1_Mesures_Datees_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P09/P09-1_Mesures_Datees_CORRIGE.ipynb) |
| P09-2 | III | 9, 2 | Étude III : tableau de bord d'une ligne | [AI4I_maintenance](donnees/MecaIA-AI4I_maintenance.xlsx), [Suivi_Production_Machines](donnees/MecaIA-Suivi_Production_Machines.xlsx) | 4 | [sujet](python/P09/P09-2_Etude_Ligne.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P09/P09-2_Etude_Ligne.ipynb) | [corrigé](python/P09/P09-2_Etude_Ligne_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P09/P09-2_Etude_Ligne_CORRIGE.ipynb) |
| P10-1 | IV | 10, 1 | Calcul symbolique avec SymPy | aucun | 3 | [sujet](python/P10/P10-1_Calcul_Symbolique.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P10/P10-1_Calcul_Symbolique.ipynb) | [corrigé](python/P10/P10-1_Calcul_Symbolique_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P10/P10-1_Calcul_Symbolique_CORRIGE.ipynb) |
| P11-1 | IV | 11, 1 | Algèbre linéaire et structures | aucun | 4 | [sujet](python/P11/P11-1_Structure.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P11/P11-1_Structure.ipynb) | [corrigé](python/P11/P11-1_Structure_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P11/P11-1_Structure_CORRIGE.ipynb) |
| P12-1 | IV | 12, 1 | Résolution numérique et optimisation | aucun | 4 | [sujet](python/P12/P12-1_Optimiser.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P12/P12-1_Optimiser.ipynb) | [corrigé](python/P12/P12-1_Optimiser_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P12/P12-1_Optimiser_CORRIGE.ipynb) |
| P13-1 | IV | 13, 1 | Intégrer et simuler | aucun | 3 | [sujet](python/P13/P13-1_Integrer.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P13/P13-1_Integrer.ipynb) | [corrigé](python/P13/P13-1_Integrer_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P13/P13-1_Integrer_CORRIGE.ipynb) |
| P14-1 | IV | 14, 1 | Interpoler un abaque | aucun | 3 | [sujet](python/P14/P14-1_Interpoler.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P14/P14-1_Interpoler.ipynb) | [corrigé](python/P14/P14-1_Interpoler_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P14/P14-1_Interpoler_CORRIGE.ipynb) |
| P15-1 | IV | 15, 1 | Modéliser une dispersion | [RUL](donnees/MecaIA-RUL.xlsx), [TimeSeries_RUL](donnees/MecaIA-TimeSeries_RUL.xlsx) | 4 | [sujet](python/P15/P15-1_Dispersion.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P15/P15-1_Dispersion.ipynb) | [corrigé](python/P15/P15-1_Dispersion_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P15/P15-1_Dispersion_CORRIGE.ipynb) |
| P16-1 | IV | 16, 1 | Filtrage et analyse spectrale | aucun | 4 | [sujet](python/P16/P16-1_Spectre.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P16/P16-1_Spectre.ipynb) | [corrigé](python/P16/P16-1_Spectre_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P16/P16-1_Spectre_CORRIGE.ipynb) |
| P16-2 | IV | 16, 2 | Étude IV : système à deux masses | aucun | 5 | [sujet](python/P16/P16-2_Etude_DeuxMasses.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P16/P16-2_Etude_DeuxMasses.ipynb) | [corrigé](python/P16/P16-2_Etude_DeuxMasses_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P16/P16-2_Etude_DeuxMasses_CORRIGE.ipynb) |
| P17-1 | V | 17, 1 | Analyse exploratoire des données | [Suivi_Production_Machines](donnees/MecaIA-Suivi_Production_Machines.xlsx), [Surveillance_Vibratoire](donnees/MecaIA-Surveillance_Vibratoire.xlsx), [TimeSeries_RUL](donnees/MecaIA-TimeSeries_RUL.xlsx) | 5 | [sujet](python/P17/P17-1_EDA.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P17/P17-1_EDA.ipynb) | [corrigé](python/P17/P17-1_EDA_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P17/P17-1_EDA_CORRIGE.ipynb) |
| P18-1 | V | 18, 1 | Préparer les données : ENISE | [Qualite_soudure](donnees/MecaIA-Qualite_soudure.xlsx), [Suivi_Production_Machines](donnees/MecaIA-Suivi_Production_Machines.xlsx), [Surveillance_Vibratoire](donnees/MecaIA-Surveillance_Vibratoire.xlsx) | 5 | [sujet](python/P18/P18-1_ENISE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P18/P18-1_ENISE.ipynb) | [corrigé](python/P18/P18-1_ENISE_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P18/P18-1_ENISE_CORRIGE.ipynb) |
| P19-1 | V | 19, 1 | Du script au premier modèle | [Plaque_Trouee_FEA](donnees/MecaIA-Plaque_Trouee_FEA.xlsx), [RUL](donnees/MecaIA-RUL.xlsx), [Suivi_Production_Machines](donnees/MecaIA-Suivi_Production_Machines.xlsx), [Surveillance_Vibratoire](donnees/MecaIA-Surveillance_Vibratoire.xlsx), [TimeSeries_RUL](donnees/MecaIA-TimeSeries_RUL.xlsx), [Traitement_Acier](donnees/MecaIA-Traitement_Acier.xlsx) | 7 | [sujet](python/P19/P19-1_Outil_Modele.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P19/P19-1_Outil_Modele.ipynb) | [corrigé](python/P19/P19-1_Outil_Modele_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P19/P19-1_Outil_Modele_CORRIGE.ipynb) |
| P19-2 | V | 19, 2 | Étude V : suivi de production | [Suivi_Production_Machines](donnees/MecaIA-Suivi_Production_Machines.xlsx) | 5 | [sujet](python/P19/P19-2_Etude_Projet.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P19/P19-2_Etude_Projet.ipynb) | [corrigé](python/P19/P19-2_Etude_Projet_CORRIGE.ipynb) [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippe-amuzuga/Meca-PY_Ed2026/blob/main/python/P19/P19-2_Etude_Projet_CORRIGE.ipynb) |

Défi du livre, P19-1 : le modèle est noté sur une base de test fixée (méthode `aleatoire`, proportion 0,25, graine 42, empreinte dans `manifest.json`). Métrique `r2`, à maximiser, score à battre 0,358.

## Jeux de données

| Jeu | Problème | Cible | Lignes × colonnes | Taille | Origine publique | Licence d'origine | Exercices |
|---|---|---|---|---|---|---|---|
| [Traitement_Acier](donnees/MecaIA-Traitement_Acier.xlsx) | régression | `resistance_traction_MPa` (MPa) | 150 × 6 | 9 ko | Iris (Fisher, 1936), UCI Machine Learning Repository, doi:10.24432/C56C76 | CC BY 4.0 | P19-1 |
| [Plaque_Trouee_FEA](donnees/MecaIA-Plaque_Trouee_FEA.xlsx) | régression, substitut de calcul par éléments finis | `Disp/mm` (déplacement maximal, mm) | 100 × 12 | 12 ko | jeu construit, cent calculs par éléments finis d'une plaque de 400 × 200 × 20 mm à trou centré, diamètre de 10 à 110 mm, traction de 1 000 N | sans objet, jeu construit | P19-1 |
| [Qualite_soudure](donnees/MecaIA-Qualite_soudure.xlsx) | classification binaire | `qualite_soudure` (conforme, non-conforme) | 900 × 19 | 65 ko | jeu construit, simulation de 900 soudures d'une campagne d'essais par le script generateurs/generer_qualite_soudure.py (graine 42), sans jeu public d'origine | sans objet, jeu construit | P18-1 |
| [Suivi_Production_Machines](donnees/MecaIA-Suivi_Production_Machines.xlsx) | classification multiclasse, préparation des données (valeurs manquantes), régression de l’indice de santé | `etat_sante_machine` (RAS, Degradation_legere, Degradation_severe, Panne_imminente) | 20000 × 14 | 1,19 Mo | jeu construit, simulation de 20 000 machines de production par le script generateurs/generer_suivi_production_machines.py (graine 42), sans jeu public d’origine | sans objet, jeu construit | P07-1, P09-2, P17-1, P18-1, P19-1, P19-2 |
| [Surveillance_Vibratoire](donnees/MecaIA-Surveillance_Vibratoire.xlsx) | régression et classification, sélection de variables | `duree_vie_restante_RUL_h` (h) en régression, alarme_criticite (0 ou 1) en classification | 150 × 25 | 23 ko | jeu construit, relevés simulés d'un parc de 150 machines tournantes | sans objet, jeu construit | P17-1, P18-1, P19-1 |
| [RUL](donnees/MecaIA-RUL.xlsx) | régression | `RUL_jours` (jours) | 240 × 6 | 14 ko | Auto MPG (Quinlan, 1993), UCI Machine Learning Repository, doi:10.24432/C5859H | CC BY 4.0 | P07-1, P08-1, P15-1, P19-1 |
| [TimeSeries_RUL](donnees/MecaIA-TimeSeries_RUL.xlsx) | séries temporelles, régression par machine (100 unités) | `Duree_Vie_Restante_Cycles` (cycles) | 18855 × 10 | 2,02 Mo | jeu construit, simulation de 100 machines suivies cycle par cycle jusqu'à la défaillance | sans objet, jeu construit | P08-1, P15-1, P17-1, P19-1 |
| [AI4I_maintenance](donnees/MecaIA-AI4I_maintenance.xlsx) | classification binaire déséquilibrée sur série temporelle | `Panne` (0 ou 1, 339 sur 10 000) | 10000 × 9 | 639 ko | AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020), UCI Machine Learning Repository, doi:10.24432/C5HS5C | CC BY 4.0 | P07-1, P08-1, P09-1, P09-2 |
| [AI4I_maintenance_TypePanne](donnees/MecaIA-AI4I_maintenance_TypePanne.xlsx) | classification multi-étiquette, fusion temporelle avec AI4I_maintenance | `Panne` et cinq modes de défaillance (outil, hydraulique, puissance, surcharge, alimentation), 0 ou 1 | 10000 × 7 | 284 ko | AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020), UCI Machine Learning Repository, doi:10.24432/C5HS5C | CC BY 4.0 | P07-1, P08-1, P09-1 |

## Modules du livre Python

Les fonctions que les chapitres réutilisent sont publiées dans `outils/`, avec leurs tests. Depuis un notebook ou un script, on les importe en ajoutant `outils/` au chemin de Python :

```
import sys
sys.path.insert(0, "outils")        # depuis la racine du dépôt cloné
import meca
```

| Module | Fichier | Bibliothèques | Jeux lus par les tests | Cas de test |
|---|---|---|---|---|
| `meca` | [`outils/meca.py`](outils/meca.py) | matplotlib, numpy, pandas, scikit-learn | [Plaque_Trouee_FEA](donnees/MecaIA-Plaque_Trouee_FEA.xlsx), [RUL](donnees/MecaIA-RUL.xlsx), [Suivi_Production_Machines](donnees/MecaIA-Suivi_Production_Machines.xlsx) | 18 |
| `meca_donnees` | [`outils/meca_donnees.py`](outils/meca_donnees.py) | matplotlib, numpy, pandas, scikit-learn, scipy, seaborn | [RUL](donnees/MecaIA-RUL.xlsx) | 7 |
| `meca_style` | [`outils/meca_style.py`](outils/meca_style.py) | matplotlib, numpy, pandas, seaborn | aucun | 15 |

Licence MIT.

Pour rejouer les tests, depuis la racine du dépôt :

```
python -m unittest discover -s outils/tests
```

## Pourquoi coder soi-même

Des logiciels existent déjà. Coder soi-même ses calculs et ses modèles donne pourtant trois choses :

- la maîtrise de bout en bout, de la donnée brute au résultat ;
- la personnalisation à son propre problème ;
- la compréhension de ce que le calcul fait, et de ses limites.

## Licences

- Le code des notebooks et des modules : licence MIT, fichier [`LICENSE`](LICENSE).
- Les jeux de données transformés et les textes : licence CC BY 4.0, sous réserve de la licence d'origine de chaque jeu, fichier [`LICENSE-DONNEES.md`](LICENSE-DONNEES.md).

## Citer

Les références à citer figurent dans [`CITATION.cff`](CITATION.cff). GitHub les propose dans le menu de citation du dépôt.
