# Licence des données et des textes

## Ce qui est couvert

Les jeux de données transformés du dossier `donnees/` et les textes du dépôt (README, énoncés et commentaires des notebooks, ce fichier) sont mis à disposition sous licence Creative Commons Attribution 4.0 International (CC BY 4.0).

- Résumé de la licence : <https://creativecommons.org/licenses/by/4.0/deed.fr>
- Texte juridique : <https://creativecommons.org/licenses/by/4.0/legalcode.fr>

Vous pouvez copier, redistribuer et adapter ces jeux et ces textes, y compris à des fins commerciales, à trois conditions : mentionner l'auteur, donner le lien vers la licence, indiquer les modifications apportées.

Le code des notebooks relève de la licence MIT, fichier [`LICENSE`](LICENSE).

## Réserve : la licence d'origine de chaque jeu

La plupart des jeux sont des transformations de jeux publics. Cette licence ne donne pas plus de droits que l'auteur du dépôt n'en détient : chaque jeu reste soumis à la licence de son jeu d'origine. Le tableau ci-dessous indique, pour chaque jeu, son origine, sa licence d'origine et les modifications apportées. Les jeux dits construits ont été produits pour ce dépôt, par calcul ou par simulation : ils n'ont pas de jeu d'origine et sont publiés directement sous CC BY 4.0.

`Qualite_soudure` est simulé par le script `generateurs/generer_qualite_soudure.py` (graine 42), qui régénère le jeu, le compare au classeur et contrôle ses propriétés. Les taux de conformité par qualification, lot, procédé ou protection gazeuse sont ceux du simulateur, non des résultats d'essais.

Les images 8 × 8 du chapitre 6 ne sont pas versées à ce dépôt : elles viennent de `load_digits` de scikit-learn, copie du jeu de test « Optical Recognition of Handwritten Digits » (E. Alpaydin et C. Kaynak, 1998, UCI Machine Learning Repository, doi:10.24432/C50P49). Consultez les conditions de réutilisation du dépôt d'origine avant toute redistribution.

## Comment citer

Citer ce dépôt (fichier [`CITATION.cff`](CITATION.cff)), puis la source d'origine du jeu employé, telle qu'elle est écrite dans la colonne « Origine publique ».

## Origines, licences d'origine et modifications

| Jeu | Origine publique | Licence d'origine | Modifications |
|---|---|---|---|
| [Traitement_Acier](donnees/MecaIA-Traitement_Acier.xlsx) | Iris (Fisher, 1936), UCI Machine Learning Repository, doi:10.24432/C56C76 | CC BY 4.0 | Les quatre mesures des fleurs deviennent par transformation affine des paramètres de traitement thermique, les trois espèces trois nuances d'acier, et la résistance à la traction est une colonne construite. |
| [Plaque_Trouee_FEA](donnees/MecaIA-Plaque_Trouee_FEA.xlsx) | jeu construit, cent calculs par éléments finis d'une plaque de 400 × 200 × 20 mm à trou centré, diamètre de 10 à 110 mm, traction de 1 000 N | sans objet, jeu construit | Aucune, sorties du solveur conservées telles quelles (dont une colonne de messages vide et six colonnes constantes). |
| [Qualite_soudure](donnees/MecaIA-Qualite_soudure.xlsx) | jeu construit, simulation de 900 soudures d'une campagne d'essais par le script generateurs/generer_qualite_soudure.py (graine 42), sans jeu public d'origine | sans objet, jeu construit | Aucune, simulation produite par le script du dépôt : la commande `python generateurs/generer_qualite_soudure.py verifier donnees` régénère le jeu, le compare au classeur et contrôle ses propriétés. |
| [Suivi_Production_Machines](donnees/MecaIA-Suivi_Production_Machines.xlsx) | jeu construit, simulation de 20 000 machines de production par le script generateurs/generer_suivi_production_machines.py (graine 42), sans jeu public d’origine | sans objet, jeu construit | Aucune, simulation produite par le script du dépôt : uniformes PCG64 de graine 42, quantiles SciPy, relations explicites d’usure, de rendement et de complexité, trois capteurs avec vides indépendants de probabilité 0,10. L’état recode l’indice de santé aux seuils 32, 42 et 64 (indice à exclure pour la classification), la classe de productivité recode la productivité aux seuils 70 et 100. La commande `python -X utf8 generateurs/generer_suivi_production_machines.py verifier donnees` régénère le jeu, le compare au classeur cellule par cellule et contrôle ses propriétés. |
| [Surveillance_Vibratoire](donnees/MecaIA-Surveillance_Vibratoire.xlsx) | jeu construit, relevés simulés d'un parc de 150 machines tournantes | sans objet, jeu construit | Aucune, simulation produite directement dans ce contexte. |
| [RUL](donnees/MecaIA-RUL.xlsx) | Auto MPG (Quinlan, 1993), UCI Machine Learning Repository, doi:10.24432/C5859H | CC BY 4.0 | 240 des 398 voitures du fichier d'origine, dans son ordre (cylindres, cylindrée, puissance, masse, accélération), renommées en grandeurs d'organes de transmission et mises à l'échelle (masse en kg, puissance en kW, couple ramené à environ 100 à 500 N·m, temps de réponse en ms) avec un bruit de 1 % de graine fixée, nombre de cylindres regroupé en régime thermique à trois niveaux (trois lignes sans régime renseigné), durée de vie restante construite par simulation à graine fixée. |
| [TimeSeries_RUL](donnees/MecaIA-TimeSeries_RUL.xlsx) | jeu construit, simulation de 100 machines suivies cycle par cycle jusqu'à la défaillance | sans objet, jeu construit | Aucune, simulation produite directement dans ce contexte. |
| [AI4I_maintenance](donnees/MecaIA-AI4I_maintenance.xlsx) | AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020), UCI Machine Learning Repository, doi:10.24432/C5HS5C | CC BY 4.0 | Températures converties en degrés Celsius (celle du procédé recalée en température de palier), usure de l'outil convertie en heures, horodatage au pas de 4 heures, puissance et vibration ajoutées, valeurs ajustées et non utilisables comme lois physiques. |
| [AI4I_maintenance_TypePanne](donnees/MecaIA-AI4I_maintenance_TypePanne.xlsx) | AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020), UCI Machine Learning Repository, doi:10.24432/C5HS5C | CC BY 4.0 | Les cinq modes de défaillance d'origine (TWF, HDF, PWF, OSF, RNF) sont renommés et indexés sur le même horodatage que AI4I_maintenance. |

## Licences d'origine non établies

La licence d'origine de tous les jeux est établie.
