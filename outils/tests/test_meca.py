# SPDX-License-Identifier: MIT
"""Tests unitaires du module meca (unittest, rejouables par pytest).

Depuis la racine du dépôt :

    python -m unittest discover -s outils/tests

Trois familles : les exemples des docstrings (doctest), les fonctions reprises du code de l'auteur (nombre et exemples),
et des valeurs de référence calculées par ailleurs (valeurs exactes du calcul symbolique, jeux du dépôt).
"""
import doctest
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "outils"))
import meca  # noqa: E402

DONNEES = RACINE / "donnees"

# Valeurs exactes du calcul symbolique : (valeur, tolérance absolue qui correspond aux décimales du livre).
ORACLES = {
    "kt_plaque_02": (2.512, 0.005),
    "contrainte_max_MPa": (251.2, 0.05),
    "fleche_mm": (1.8601190476190477, 5e-05),
    "arbre_mm": (34.881591197368635, 0.005),
}


def confronter(nom, calcule):
    """Vérifie qu'un résultat numérique reste dans la tolérance de la valeur exacte `nom`."""
    attendu, tolerance = ORACLES[nom]
    assert abs(float(calcule) - attendu) <= tolerance, "%s : %s au lieu de %s" % (nom, calcule, attendu)


def lire_jeu(jeu):
    """Le classeur MecaIA-<jeu>.xlsx du dossier `donnees` du dépôt."""
    return pd.read_excel(DONNEES / ("MecaIA-%s.xlsx" % jeu))


class TestDoctest(unittest.TestCase):
    def test_exemples_des_docstrings(self):
        resultat = doctest.testmod(meca)
        self.assertEqual(resultat.failed, 0)
        self.assertGreaterEqual(resultat.attempted, 80)


class TestFonctionsAuteur(unittest.TestCase):
    def test_quinze_fonctions_de_l_auteur_presentes_et_testees(self):
        self.assertEqual(len(meca.FONCTIONS_AUTEUR), 15)
        for nom in meca.FONCTIONS_AUTEUR:
            fonction = getattr(meca, nom)
            self.assertTrue(inspect.isfunction(fonction), nom)
            self.assertIn(">>>", fonction.__doc__ or "", "pas d'exemple doctest : " + nom)

    def test_au_moins_dix_fonctions_vertes(self):
        finder = doctest.DocTestFinder()
        runner = doctest.DocTestRunner(verbose=False)
        vertes = 0
        for nom in meca._fonctions_publiques():
            for test in finder.find(getattr(meca, nom), nom, globs=vars(meca).copy()):
                if not test.examples:
                    continue
                resultat = runner.run(test, clear_globs=False)
                if resultat.failed == 0:
                    vertes += 1
        self.assertGreaterEqual(vertes, 10)


class TestCalculsContreValeursExactes(unittest.TestCase):
    def test_kt_et_contrainte_de_la_plaque(self):
        kt = meca.kt_trou(10.0, 50.0)
        confronter("kt_plaque_02", kt)
        confronter("contrainte_max_MPa", kt * 20000.0 / ((50.0 - 10.0) * 5.0))

    def test_fleche_de_la_poutre(self):
        confronter("fleche_mm", meca.fleche_mm(2000.0, 1000.0, 20.0, 40.0, 210000.0))

    def test_arbre_par_dichotomie(self):
        d, tours = meca.dichotomie(lambda d: meca.tau_MPa(d, 500.0) - 60.0, 10.0, 100.0)
        confronter("arbre_mm", d)
        self.assertEqual(tours, 15)

    def test_garde_fous(self):
        with self.assertRaises(ValueError):
            meca.contrainte_MPa(1000.0, -1.0)
        with self.assertRaises(ValueError):
            meca.dichotomie(lambda x: x**2 + 1.0, 0.0, 1.0)
        self.assertTrue(np.isnan(meca.kt_trou(0.0, 50.0)))


class TestChaineDeCotes(unittest.TestCase):
    def test_ecart_type_du_jeu(self):
        cotes = [(100.0, 0.3, 1), (60.0, 0.15, -1), (38.0, 0.15, -1)]
        jeu = meca.jeu_monte_carlo(cotes, 200000, np.random.default_rng(1))
        attendu = np.sqrt(sum((t / 3)**2 for _, t, _ in cotes))
        assert_allclose(jeu.std(ddof=1), attendu, rtol=0.01)
        assert_allclose(jeu.mean(), 2.0, atol=0.002)


class TestPandas(unittest.TestCase):
    def test_horodatage_et_unite_de_la_fiche(self):
        d = pd.DataFrame({"time": [0.0, 1.0], "T": [1.0, 2.0]})
        jours = meca.indexer_horodatage(d, pd.Timestamp("2025-01-01"), unite="D")
        heures = meca.indexer_horodatage(d, pd.Timestamp("2025-01-01"), unite="h")
        self.assertEqual(str(jours.index[1]), "2025-01-02 00:00:00")
        self.assertEqual(str(heures.index[1]), "2025-01-01 01:00:00")

    def test_manquants_sur_le_suivi_de_production(self):
        suivi = lire_jeu("Suivi_Production_Machines")
        cellules, lignes = meca.cellules_et_lignes(suivi)
        self.assertEqual(cellules, 6043)
        self.assertEqual(lignes, 5462)
        self.assertLess(lignes, cellules)
        taux = meca.taux_manquants(suivi)
        self.assertTrue(taux.is_monotonic_decreasing)

    def test_bornes_physiques_du_jeu_rul(self):
        rul = lire_jeu("RUL")
        ecarts = meca.verifier_bornes(rul, {"RUL_jours": (0, 365), "masse_vibrante_kg": (0, 5000)})
        self.assertEqual(sum(ecarts.values()), 0)


class TestPretraitementAuteur(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plaque = lire_jeu("Plaque_Trouee_FEA")

    def test_taille_des_jeux_apprentissage_et_test(self):
        X_train, X_test, y_train, y_test = meca.preprocessing(self.plaque.copy())
        self.assertEqual(len(X_train) + len(X_test), 99)        # 1 % de 100 lignes retirées
        self.assertEqual(len(X_test), 20)
        self.assertEqual(X_train.shape[1], 1)

    def test_ordre_sans_fuite_laisse_le_test_intact(self):
        _, X_test_a, _, _ = meca.preprocessing(self.plaque.copy(), ordre="auteur")
        X_train_s, X_test_s, _, _ = meca.preprocessing(self.plaque.copy(), ordre="sans_fuite")
        self.assertEqual(len(X_test_s), 20)
        self.assertEqual(len(X_train_s), 79)
        self.assertEqual(len(X_test_a), 20)

    def test_kt_de_la_plaque_est_physique(self):
        out = meca.preprocessing_imputation(self.plaque.copy())
        self.assertTrue(out["Kt"].between(1.0, 10.0).all())

    def test_evaluation_et_equation(self):
        from sklearn.linear_model import LinearRegression
        X_train, X_test, y_train, y_test = meca.preprocessing(self.plaque.copy())
        modele = LinearRegression().fit(X_train, y_train)
        score = meca.evaluation_perf(modele, X_train, y_train, X_test, y_test, tol=0.05)
        self.assertTrue(0.0 <= score <= 100.0)
        equation, r2 = meca.model_equation(modele, X_train, y_train, X_test, y_test)
        self.assertTrue(equation.startswith("Kt = "))
        self.assertLessEqual(r2, 1.0)


class TestReseaux(unittest.TestCase):
    def test_fenetres_et_normalisation_sans_fuite(self):
        serie = np.arange(40.0).reshape(20, 2)
        X, y = meca.fenetres_glissantes(serie, 5, cible=np.arange(20))
        self.assertEqual(X.shape, (16, 5, 2))
        self.assertEqual(y[0], 4)
        a, b, moyenne, ecart = meca.normaliser_sans_fuite(serie[:15], serie[15:])
        assert_allclose(a.mean(axis=0), 0.0, atol=1e-12)
        assert_allclose(moyenne, serie[:15].mean(axis=0))      # la moyenne ignore le test

    def test_utilitaires_de_l_auteur(self):
        assert_allclose(meca.minmax_scale_01([0.0, 5.0, 10.0]), [0.0, 0.5, 1.0])
        assert_allclose(meca.safe_div(np.array([1.0]), np.array([0.0])), [0.0])
        m, b = meca.fit_line([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
        assert_allclose([m, b], [2.0, 1.0], atol=1e-9)
        self.assertEqual(meca.eq_label(m, b), "y = 2.00X + 1.00")


class TestFichiers(unittest.TestCase):
    def test_lecture_et_copie_de_securite(self):
        with tempfile.TemporaryDirectory() as dossier:
            chemin = Path(dossier) / "mesures.xlsx"
            pd.DataFrame({"x": [1, 2, 3]}).to_excel(chemin, index=False)
            data, df = meca.lire_classeur(chemin)
            df.loc[0, "x"] = 99
            self.assertEqual(int(data.loc[0, "x"]), 1)


if __name__ == "__main__":
    unittest.main()
