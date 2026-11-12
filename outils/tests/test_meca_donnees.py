# SPDX-License-Identifier: MIT
"""Tests unitaires du module meca_donnees (unittest, rejouables par pytest).

Depuis la racine du dépôt :

    python -m unittest discover -s outils/tests

Trois familles : les exemples des docstrings (doctest), les règles des quarante-cinq briques (quinze lignes au plus,
brique amont citée, ordre d'emploi), et la règle du test : un réglage appris ne regarde jamais le test.
"""
import doctest
import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "outils"))
import meca_donnees as md  # noqa: E402


class TestExemples(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_exemples_des_docstrings(self):
        resultat = doctest.testmod(md)
        self.assertEqual(resultat.failed, 0)
        self.assertGreaterEqual(resultat.attempted, 45)


class TestBriques(unittest.TestCase):
    def test_quarante_cinq_briques_conformes(self):
        self.assertEqual(md.verifier_briques(), [])
        self.assertEqual(len(md.ORDRE_EMPLOI), 45)
        self.assertEqual(len({ident for ident, _nom, _section, _amont in md.ORDRE_EMPLOI}), 45)

    def test_chaque_brique_est_une_fonction_du_module(self):
        for _ident, nom, _section, _amont in md.ORDRE_EMPLOI:
            self.assertTrue(callable(getattr(md, nom)), nom)


class TestLecture(unittest.TestCase):
    def test_lecture_du_jeu_rul_et_copie_de_travail(self):
        data, df = md.lire_classeur("MecaIA-RUL.xlsx", dossier=RACINE / "donnees")
        self.assertEqual(data.shape, (240, 6))
        self.assertIsNot(data, df)
        colonne = data.select_dtypes("number").columns[0]
        avant = data[colonne].iloc[0]
        df.loc[df.index[0], colonne] = -1
        self.assertEqual(data[colonne].iloc[0], avant)

    def test_dossier_des_jeux_voisin_de_outils(self):
        self.assertEqual(md.dossier_jeux().resolve(), (RACINE / "donnees").resolve())


class TestRegleDuTest(unittest.TestCase):
    def test_la_mise_a_l_echelle_apprend_sur_l_apprentissage_seul(self):
        X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        X_test = pd.DataFrame({"x": [100.0, 200.0]})
        _echelle, apprentissage, test = md.mise_echelle_standard(X_train, X_test)
        self.assertAlmostEqual(float(apprentissage["x"].mean()), 0.0, places=9)
        self.assertGreater(float(test["x"].mean()), 10.0)       # la moyenne et l'écart-type du test ne servent pas


class TestDistributions(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_l_asymetrie_s_ecrit_sous_chaque_histogramme(self):
        data, _df = md.lire_classeur("MecaIA-RUL.xlsx", dossier=RACINE / "donnees")
        fig = md.tracer_distributions(data)
        libelles = [ax.get_xlabel() for ax in fig.axes if ax.get_visible()]
        self.assertEqual(len(libelles), len(data.select_dtypes("number").columns))
        for libelle in libelles:
            self.assertRegex(libelle, r"^asymétrie -?\d+,\d$")       # une décimale, virgule française


if __name__ == "__main__":
    unittest.main()
