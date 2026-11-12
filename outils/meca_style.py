# SPDX-License-Identifier: MIT
"""meca_style : le style des figures du livre, appliqué en une ligne.

Usage, après avoir ajouté `outils/` au chemin de recherche des modules :

    import meca_style as ms
    ms.appliquer()                 # matplotlib : polices de 7 à 7,5 pt, grille horizontale, deux couleurs
    ms.appliquer(seaborn=True)     # seaborn : même feuille, passée à sns.set_theme

Le module fournit aussi :

    ms.virgule(ax)                 # virgule décimale sur les graduations de l'axe vertical
    ms.virgule(ax, "xy", ".2f")    # idem sur les deux axes, avec deux décimales (axe, format)
    ms.virgule(fig, "xy")          # idem sur tous les axes d'une figure (ou d'une grille seaborn)
    ms.graduations(ax, 3)          # au plus 3 graduations par axe (ax, nombre, axe)
    ms.BLEU, ms.ORANGE, ms.GRIS    # les teintes du livre ; ms.PALE : un aplat clair
    ms.FLECHE                      # flèche d'annotation : ax.annotate(texte, xy, xytext, arrowprops=ms.FLECHE)
    ms.CARTE                       # dégradé de bleus pour un champ ; ms.VIRGULE : format d'une barre de couleur
    ms.police_min()                # plus petite taille de graduation en vigueur (7.0)

Tout est lisible en une page : ce fichier est la seule source des réglages que les figures du livre
n'écrivent plus une à une.

>>> import meca_style as ms
>>> ms.appliquer()
>>> import matplotlib.pyplot as plt
>>> plt.rcParams["xtick.labelsize"]
7.0
>>> ms.police_min()
7.0
>>> fig, ax = plt.subplots()
>>> ax.plot([0.5, 1.5], [0.25, 0.75])[0].get_color()
'#0B3D6B'
>>> reperes = ax.set_xticks([0.5, 1.5])
>>> ms.virgule(ax, "xy", ".1f")
>>> fig.canvas.draw()
>>> [t.get_text() for t in ax.get_xticklabels()]
['0,5', '1,5']
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

BLEU, ORANGE, GRIS, PALE = "#0B3D6B", "#D2691E", "#444444", "#E6EEF8"
FLECHE = {"arrowstyle": "->", "color": BLEU}
CARTE = LinearSegmentedColormap.from_list("bleus", ["#D6E4F0", "#2F6FB3", BLEU])
VIRGULE = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

REGLAGES = {
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#E2E2E2", "grid.linewidth": 0.4,
    "axes.prop_cycle": plt.cycler(color=[BLEU, ORANGE]),
    "lines.linewidth": 1.2, "patch.linewidth": 0.7, "lines.markersize": 3, "lines.color": GRIS,
    "hatch.color": GRIS,
    "pdf.fonttype": 42,
}


def appliquer(seaborn=False):
    """Active la feuille de style ; avec seaborn=True, la passe aussi à sns.set_theme."""
    if seaborn:
        import seaborn as sns
        police = plt.rcParams["font.family"]                    # seaborn la remet à zéro sinon
        rc = {**REGLAGES, "font.family": police, "axes.linewidth": 0.6, "axes.edgecolor": "#2B2B2B"}
        sns.set_theme(style="white", palette=[BLEU, ORANGE], rc=rc)
    else:
        plt.rcParams.update(REGLAGES)


def police_min():
    """Plus petite taille de graduation en vigueur, en points (7.0 avec la feuille du livre)."""
    return float(min(plt.rcParams["xtick.labelsize"], plt.rcParams["ytick.labelsize"]))


def _axes(cible):
    """Les axes d'une cible : un axe, une liste ou un tableau d'axes, une figure ou une grille seaborn."""
    if isinstance(cible, Axes):
        return [cible]
    if isinstance(cible, Figure):
        return cible.get_axes()
    if hasattr(cible, "figure"):                       # grille seaborn (FacetGrid, PairGrid)
        return cible.figure.get_axes()
    return [a for a in np.ravel(cible) if a is not None]


def virgule(cible, axe="y", fmt="g"):
    """Virgule décimale sur les graduations : axe vaut "x", "y", "z" ou une combinaison ("xy"), fmt est le format."""
    virg = FuncFormatter(lambda v, _: f"{v:{fmt}}".replace(".", ","))
    for ax in _axes(cible):
        if "x" in axe:
            ax.xaxis.set_major_formatter(virg)
        if "y" in axe:
            ax.yaxis.set_major_formatter(virg)
        if "z" in axe and hasattr(ax, "zaxis"):
            ax.zaxis.set_major_formatter(virg)


def graduations(cible, nombre, axe="xy"):
    """Au plus `nombre` graduations sur l'axe "x", "y" ou "xy" de chaque axe de la cible."""
    for ax in _axes(cible):
        if "x" in axe:
            ax.xaxis.set_major_locator(MaxNLocator(nombre))
        if "y" in axe:
            ax.yaxis.set_major_locator(MaxNLocator(nombre))
