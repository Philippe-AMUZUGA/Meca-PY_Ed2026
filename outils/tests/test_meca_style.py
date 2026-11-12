# SPDX-License-Identifier: MIT
"""Tests unitaires du module meca_style (unittest, rejouables par pytest).

Depuis la racine du dépôt :

    python -m unittest discover -s outils/tests

Trois familles : les exemples du module (doctest), la feuille de style (tailles de police, couleurs, polices
incorporées) et les trois fonctions d'aide (virgule, graduations, cibles diverses).
"""
import doctest
import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "outils"))
import meca_style as ms


class TestFeuille(unittest.TestCase):
    def setUp(self):
        plt.rcdefaults()

    def tearDown(self):
        plt.close("all")
        plt.rcdefaults()

    def test_exemples_du_module(self):
        echecs, _ = doctest.testmod(ms)
        self.assertEqual(echecs, 0)

    def test_tailles_de_police(self):
        ms.appliquer()
        for cle in ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
            self.assertGreaterEqual(plt.rcParams[cle], 7, cle)

    def test_police_min(self):
        ms.appliquer()
        self.assertEqual(ms.police_min(), 7.0)

    def test_polices_incorporees(self):
        ms.appliquer()
        self.assertEqual(plt.rcParams["pdf.fonttype"], 42)

    def test_deux_couleurs_du_cycle(self):
        ms.appliquer()
        _, ax = plt.subplots()
        premier = ax.plot([0, 1], [0, 1])[0].get_color()
        second = ax.plot([0, 1], [1, 0])[0].get_color()
        self.assertEqual((premier, second), (ms.BLEU, ms.ORANGE))

    def test_seaborn_garde_la_police(self):
        plt.rcParams["font.family"] = "serif"
        ms.appliquer(seaborn=True)
        self.assertEqual(plt.rcParams["font.family"], ["serif"])
        self.assertEqual(plt.rcParams["xtick.labelsize"], 7)


class TestAides(unittest.TestCase):
    def tearDown(self):
        plt.close("all")
        plt.rcdefaults()

    def _etiquettes(self, ax, axe="x"):
        ax.figure.canvas.draw()
        axis = ax.xaxis if axe == "x" else ax.yaxis
        return [t.get_text() for t in axis.get_ticklabels()]

    def test_virgule_sur_un_axe(self):
        _, ax = plt.subplots()
        ax.set(xticks=[0.5, 1.5], yticks=[2.5, 3.5])
        ms.virgule(ax, "y")
        self.assertEqual(self._etiquettes(ax, "y"), ["2,5", "3,5"])
        self.assertNotIn(",", "".join(self._etiquettes(ax, "x")))

    def test_virgule_format(self):
        _, ax = plt.subplots()
        ax.set(xticks=[0.5, 1.5])
        ms.virgule(ax, "x", ".2f")
        self.assertEqual(self._etiquettes(ax), ["0,50", "1,50"])

    def test_virgule_sur_une_figure(self):
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.set(yticks=[0.5, 1.5])
        ms.virgule(fig, "y")
        for ax in axes:
            self.assertEqual(self._etiquettes(ax, "y"), ["0,5", "1,5"])

    def test_virgule_sur_une_liste_et_un_tableau(self):
        _, axes = plt.subplots(2, 2)
        ms.virgule(axes, "x")
        ms.virgule(list(axes.flat), "x")
        axes[0, 0].set(xticks=[0.5])
        self.assertEqual(self._etiquettes(axes[0, 0]), ["0,5"])

    def test_virgule_sur_une_grille_seaborn(self):
        table = pd.DataFrame({"x": [0.5, 1.5, 2.5, 3.5], "g": ["a", "a", "b", "b"]})
        grille = sns.FacetGrid(table, col="g")
        ms.virgule(grille, "x")
        for ax in grille.axes.flat:
            ax.set(xticks=[0.5])
        self.assertEqual(self._etiquettes(grille.axes.flat[0]), ["0,5"])

    def test_virgule_sur_l_axe_z(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set(zticks=[0.5, 1.5])
        ms.virgule(ax, "z")
        fig.canvas.draw()
        self.assertEqual([t.get_text() for t in ax.zaxis.get_ticklabels()], ["0,5", "1,5"])

    def test_couleur_des_hachures(self):
        ms.appliquer()
        self.assertEqual(plt.rcParams["hatch.color"], ms.GRIS)

    def test_graduations(self):
        _, ax = plt.subplots()
        ax.plot([0, 100], [0, 100])
        ms.graduations(ax, 3, "x")
        ax.figure.canvas.draw()
        self.assertLessEqual(len(ax.get_xticks()), 5)

    def test_constantes(self):
        self.assertEqual(ms.FLECHE["arrowstyle"], "->")
        self.assertEqual(ms.VIRGULE(0.25), "0,25")
        self.assertEqual(ms.CARTE(0.0)[:3], matplotlib.colors.to_rgb("#D6E4F0"))


if __name__ == "__main__":
    unittest.main()
