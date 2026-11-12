# SPDX-License-Identifier: MIT
"""meca : le module de calcul de l'ingénieur, fil rouge du livre.

Chaque fonction est courte, documentée, et porte son test dans sa docstring (module doctest).
Pour les rejouer tous :

    python meca.py            # affiche « meca : N fonctions, 0 échec »

Le module se lit dans l'ordre du livre :

    1. calculs de résistance des matériaux (chapitre 3) et Monte-Carlo (chapitre 4) ;
    2. outils de l'auteur pour mettre à l'échelle, borner, tracer une droite (chapitres 5 et 19) ;
    3. lecture, contrôle et horodatage des classeurs de mesures (chapitres 7 à 9) ;
    4. prétraitement et évaluation en quatre temps, noms de l'auteur (chapitre 19) ;
    5. données prêtes pour les réseaux de neurones (chapitre 19).

Les noms anglais `preprocessing_*` et `evaluation_*` sont ceux de l'auteur : ils sont gardés tels quels.
"""
import math

import numpy as np
import pandas as pd

__version__ = "1.0.0"

# Les 15 fonctions reprises du code de l'auteur, sous leur nom.
FONCTIONS_AUTEUR = (
    "preprocessing_imputation", "preprocessing_outliers", "preprocessing_Xy", "preprocessing",
    "evaluation_perf", "evaluation_lc", "model_equation",
    "scale_to_range", "clip_round", "safe_div", "minmax_scale_01", "_format_unique_values",
    "fit_line", "eq_label", "fmt_num",
)


# Partie 1. Résistance des matériaux et Monte-Carlo (chapitres 3 et 4)

def contrainte_MPa(F_N, d_mm):
    """Contrainte normale dans une tige ronde, en MPa, pour un effort F en N et un diamètre d en mm.

    >>> round(contrainte_MPa(12000.0, 10.0), 2)
    152.79
    >>> contrainte_MPa(12000.0, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: le diamètre doit être positif
    """
    if d_mm <= 0:
        raise ValueError("le diamètre doit être positif")
    return F_N / (math.pi * d_mm**2 / 4)


def securite(Re_MPa, F_N, d_mm):
    """Coefficient de sécurité d'une tige : limite d'élasticité sur contrainte de service.

    >>> round(securite(355.0, 12000.0, 10.0), 2)
    2.32
    """
    return Re_MPa / contrainte_MPa(F_N, d_mm)


def kt_trou(d_mm, W_mm):
    """Kt d'une plaque de largeur W percée d'un trou central de diamètre d, en traction.

    Formule de Heywood : Kt = 2 + (1 - d/W)**3, rapportée à la contrainte nette F / ((W - d) e).
    Source : R. B. Heywood, Designing by Photoelasticity, Chapman and Hall, 1952 ; formule reprise
    par W. D. Pilkey et D. F. Pilkey, Peterson's Stress Concentration Factors, Wiley, et par
    Roark's Formulas for Stress and Strain, McGraw-Hill. Domaine : 0 < d/W < 1.

    >>> round(kt_trou(10.0, 50.0), 3)
    2.512
    >>> kt_trou(60.0, 50.0) != kt_trou(60.0, 50.0)     # nan hors du domaine
    True
    """
    rapport = d_mm / W_mm
    if rapport <= 0 or rapport >= 1:
        return float("nan")      # hors du domaine : pas de trou, ou plus de matière
    return 2 + (1 - rapport)**3


def moment_quadratique_mm4(b_mm, h_mm):
    """Moment quadratique d'une section rectangulaire : b h³ / 12, en mm⁴.

    >>> moment_quadratique_mm4(20.0, 40.0)
    106666.66666666667
    """
    return b_mm * h_mm**3 / 12


def fleche_mm(F_N, L_mm, b_mm, h_mm, E_MPa):
    """Flèche au centre, poutre sur deux appuis, charge F au milieu : F L³ / (48 E I).

    >>> round(fleche_mm(2000.0, 1000.0, 20.0, 40.0, 210000.0), 4)
    1.8601
    """
    return F_N * L_mm**3 / (48 * E_MPa * moment_quadratique_mm4(b_mm, h_mm))


def tau_MPa(d_mm, T_Nm):
    """Cisaillement maximal d'un arbre plein en torsion : 16 T / (pi d³), T en N.m, d en mm.

    >>> round(tau_MPa(34.881591, 500.0), 2)
    60.0
    """
    return 16 * T_Nm * 1000 / (math.pi * d_mm**3)


def dichotomie(g, a, b, tolerance=0.005):
    """Zéro de g dans [a, b] par dichotomie : rend (solution, nombre de tours).

    g doit changer de signe entre a et b. La boucle s'arrête quand l'encadrement fait moins que la tolérance.

    >>> x, tours = dichotomie(lambda d: tau_MPa(d, 500.0) - 60.0, 10.0, 100.0)
    >>> round(x, 2), tours
    (34.88, 15)
    >>> dichotomie(lambda x: x**2 + 1.0, 0.0, 1.0)
    Traceback (most recent call last):
        ...
    ValueError: g ne change pas de signe entre a et b
    """
    if g(a) * g(b) > 0:
        raise ValueError("g ne change pas de signe entre a et b")
    tours = 0
    while b - a > tolerance:
        milieu = (a + b) / 2
        if g(a) * g(milieu) <= 0:
            b = milieu           # le signe change dans la moitié gauche
        else:
            a = milieu           # il change dans la moitié droite
        tours = tours + 1
    return (a + b) / 2, tours


def jeu_monte_carlo(cotes, tirages, rng):
    """Jeu fonctionnel d'une chaîne de cotes, par tirage : somme signée de cotes de loi normale.

    `cotes` est une liste de (nominale, tolérance, signe). Chaque cote suit une loi normale d'écart-type
    tolérance / 3 (la tolérance couvre trois écarts-types) ; le signe vaut +1 si la cote agrandit le jeu,
    -1 si elle le réduit.

    >>> rng = np.random.default_rng(0)
    >>> jeu = jeu_monte_carlo([(100.0, 0.3, 1), (60.0, 0.15, -1), (38.0, 0.15, -1)], 20000, rng)
    >>> jeu.shape
    (20000,)
    >>> round(float(jeu.mean()), 2)
    2.0
    """
    jeu = np.zeros(tirages)
    for nominale, tolerance, signe in cotes:
        jeu = jeu + signe * rng.normal(nominale, tolerance / 3, tirages)
    return jeu


# Partie 2. Outils de l'auteur : mise à l'échelle, bornes, droite et légendes (chapitres 5 et 19)

def scale_to_range(series, new_min, new_max):
    """Met une série à l'échelle [new_min, new_max] ; une série constante rend le milieu de l'intervalle.

    >>> scale_to_range(pd.Series([2.0, 4.0, 6.0]), 0, 100).tolist()
    [0.0, 50.0, 100.0]
    >>> scale_to_range(pd.Series([5.0, 5.0]), 0, 10).tolist()
    [5.0, 5.0]
    """
    series = pd.to_numeric(series, errors="coerce")
    xmin = np.nanmin(series.values)
    xmax = np.nanmax(series.values)

    if not np.isfinite(xmin) or not np.isfinite(xmax) or np.isclose(xmin, xmax):
        return pd.Series(np.full(len(series), (new_min + new_max) / 2), index=series.index)

    return (series - xmin) / (xmax - xmin) * (new_max - new_min) + new_min


def clip_round(s: pd.Series, a: float, b: float, ndigits: int) -> pd.Series:
    """Borne une série entre a et b, puis l'arrondit à ndigits décimales.

    >>> clip_round(pd.Series([1.234, 5.678, 9.0]), 2.0, 6.0, 1).tolist()
    [2.0, 5.7, 6.0]
    """
    return s.clip(lower=a, upper=b).round(ndigits)


def safe_div(a, b, default=0.0):
    """Division protégée : un dénominateur nul rend `default` au lieu d'une erreur ou d'un infini.

    >>> safe_div(np.array([1.0, 2.0, 3.0]), np.array([0.0, 4.0, 2.0])).tolist()
    [0.0, 0.5, 1.5]
    """
    b_ = np.where(b == 0, np.nan, b)
    out = a / b_
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default)


def minmax_scale_01(v):
    """Met un vecteur à l'échelle [0, 1] ; un vecteur constant rend 0,5.

    >>> minmax_scale_01([10.0, 15.0, 20.0]).tolist()
    [0.0, 0.5, 1.0]
    >>> minmax_scale_01([3.0, 3.0]).tolist()
    [0.5, 0.5]
    """
    v = np.asarray(v, dtype=float)
    vmin, vmax = np.min(v), np.max(v)
    if vmax - vmin <= 1e-12:
        return np.full_like(v, 0.5)
    return (v - vmin) / (vmax - vmin)


def _format_unique_values(series: pd.Series, max_values: int = 30) -> str:
    """Résumé lisible des valeurs uniques d'une colonne (avec troncature si la liste est longue).

    >>> print(_format_unique_values(pd.Series(["A", "B", "A", None])))
    n=4, n_nan=1, n_unique=2
    uniques (max 30) = [A, B]
    >>> print(_format_unique_values(pd.Series([1, 2, 3, 4]), max_values=2))
    n=4, n_nan=0, n_unique=4
    uniques (max 2) = [1, 2] ... (+2 autres)
    """
    s = series
    n_total = len(s)
    n_nan = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))

    uniques = s.dropna().unique()

    uniques_str = [str(v) for v in uniques[:max_values]]
    suffix = ""
    if len(uniques) > max_values:
        suffix = f" ... (+{len(uniques) - max_values} autres)"

    return (
        f"n={n_total}, n_nan={n_nan}, n_unique={n_unique}\n"
        f"uniques (max {max_values}) = [{', '.join(uniques_str)}]{suffix}"
    )


def fit_line(x, y):
    """Droite des moindres carrés d'un nuage : rend (pente, ordonnée à l'origine).

    >>> m, b = fit_line([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    >>> round(m, 6), round(b, 6)
    (2.0, 1.0)
    """
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def eq_label(m, b, decimals=2):
    """Équation de droite pour une légende de figure.

    >>> eq_label(2.0, 1.0)
    'y = 2.00X + 1.00'
    >>> eq_label(-0.1138, 358.0, decimals=3)
    'y = -0.114X + 358.000'
    """
    sign = "+" if b >= 0 else "-"
    return f"y = {m:.{decimals}f}X {sign} {abs(b):.{decimals}f}"


def fmt_num(v):
    """Format numérique pour une légende : notation scientifique hors de [0,001 ; 10 000[.

    >>> [fmt_num(v) for v in (0, 0.5, 1234.5, 123456.0, 0.0001)]
    ['0', '0.50', '1234.5', '1.23e+05', '1.00e-04']
    """
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e4 or av < 1e-3:
        return f"{v:.2e}"
    if av >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}"


# Partie 3. Lire, contrôler et horodater des mesures (chapitres 7 à 9)

def lire_classeur(chemin, **options):
    """Lit un classeur et garde l'original : rend (data, df), df étant une copie de travail.

    >>> import os, tempfile
    >>> dossier = tempfile.mkdtemp()
    >>> chemin = os.path.join(dossier, "mesures.xlsx")
    >>> pd.DataFrame({"x": [1, 2, 3]}).to_excel(chemin, index=False)
    >>> data, df = lire_classeur(chemin)
    >>> df.loc[0, "x"] = 99
    >>> int(data.loc[0, "x"]), int(df.loc[0, "x"])
    (1, 99)
    """
    data = pd.read_excel(chemin, **options)
    df = data.copy()
    return data, df


def taux_manquants(df):
    """Taux de valeurs manquantes par colonne, en pourcentage, trié par ordre décroissant.

    >>> d = pd.DataFrame({"a": [1.0, None, None, 4.0], "b": [1.0, 2.0, 3.0, None], "c": [1, 2, 3, 4]})
    >>> taux_manquants(d).round(1).to_dict()
    {'a': 50.0, 'b': 25.0, 'c': 0.0}
    """
    return (df.isna().mean() * 100).sort_values(ascending=False)


def cellules_et_lignes(df):
    """Cellules manquantes et lignes touchées : rend (cellules, lignes), deux comptes qui ne se confondent pas.

    >>> d = pd.DataFrame({"a": [1.0, None, None, 4.0], "b": [1.0, 2.0, 3.0, None]})
    >>> cellules_et_lignes(d)
    (3, 3)
    """
    return int(df.isna().sum().sum()), int(df.isna().any(axis=1).sum())


def verifier_bornes(df, bornes, arreter=True):
    """Compte, par colonne, les mesures hors de leur plage physique ; `arreter` lève une erreur au premier écart.

    `bornes` associe un nom de colonne à (minimum, maximum). Les valeurs manquantes ne sont pas des écarts.

    >>> d = pd.DataFrame({"T": [20.0, 25.0, 900.0], "P": [1.0, 2.0, 3.0]})
    >>> verifier_bornes(d, {"T": (0.0, 200.0), "P": (0.0, 10.0)}, arreter=False)
    {'T': 1, 'P': 0}
    >>> verifier_bornes(d, {"T": (0.0, 200.0)})
    Traceback (most recent call last):
        ...
    ValueError: T : 1 mesure(s) hors de [0.0, 200.0]
    """
    ecarts = {}
    for colonne, (mini, maxi) in bornes.items():
        hors = int((~df[colonne].between(mini, maxi) & df[colonne].notna()).sum())
        ecarts[colonne] = hors
        if hors and arreter:
            raise ValueError(f"{colonne} : {hors} mesure(s) hors de [{mini}, {maxi}]")
    return ecarts


def indexer_horodatage(df_in, depart, colonne="time", unite="D", arrondi="h", nom="Horodatage"):
    """Remplace une colonne de temps relatif par un index de dates (démarche de l'auteur).

    La colonne `colonne` compte le temps écoulé depuis `depart`, dans l'unité de la fiche du jeu ;
    l'index obtenu est arrondi à `arrondi` puis nommé `nom`.

    >>> d = pd.DataFrame({"time": [0.0, 0.1667, 0.3334], "T": [20.0, 21.0, 22.0]})
    >>> out = indexer_horodatage(d, pd.Timestamp("2025-01-01 08:00"))
    >>> [str(t) for t in out.index]
    ['2025-01-01 08:00:00', '2025-01-01 12:00:00', '2025-01-01 16:00:00']
    >>> out.index.name
    'Horodatage'
    """
    df = df_in.copy()
    df = df.set_index(colonne)
    df.index = depart + pd.to_timedelta(df.index, unit=unite)
    df.index = df.index.round(arrondi)
    df.index.name = nom
    return df


# Partie 4. Prétraitement et évaluation en quatre temps, noms de l'auteur (chapitre 19)

def preprocessing_imputation(df_in):
    """Prétraitement propre au jeu de la plaque trouée : colonnes utiles, section, contrainte nominale, Kt.

    Le jeu de l'auteur porte les colonnes de la simulation par éléments finis ; on garde d0 (diamètre du trou)
    et Kt = contrainte maximale / contrainte nominale, avec une force de 1000 N.

    >>> d = pd.DataFrame({"Unnamed: 0": [0], "L": [400], "h": [200], "k": [20], "x0": [200], "y0": [100],
    ...                   "d0": [10.0], "Material": ["Acier"], "Stress/MPa": [0.65789474],
    ...                   "Disp/mm": [0.0005], "Msg": [None], "FEA_Runtime": [10.0]})
    >>> out = preprocessing_imputation(d)
    >>> list(out.columns), round(float(out.loc[0, "Kt"]), 3)
    (['d0', 'Kt'], 2.5)
    """
    df = df_in.copy()  # On refait une copie
    # Elimination de colonnes
    df = df.drop(['Disp/mm', 'Msg', 'FEA_Runtime', 'Material', 'Unnamed: 0'], axis=1)

    # Renommer
    df.rename(columns={"k": "t"}, inplace=True)

    # Kt
    df["Sec"] = df["t"] * (df["h"] - df["d0"])
    df["StressNom"] = 1000 / df["Sec"]
    df["Kt"] = df["Stress/MPa"] / df["StressNom"]

    # Final avec d0 et Kt
    key_cols = ['d0', 'Kt']
    df_out = df[key_cols]
    return df_out


def preprocessing_outliers(df_in):
    """Retire les points aberrants repérés par une forêt d'isolement (1 % supposé).

    >>> rng = np.random.default_rng(1)
    >>> d = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.normal(0, 1, 200)})
    >>> d.loc[0] = [50.0, 50.0]
    >>> out = preprocessing_outliers(d)
    >>> len(d) - len(out), 0 in out.index
    (2, False)
    """
    from sklearn.ensemble import IsolationForest

    model_isoF = IsolationForest(contamination=0.01, random_state=42)
    pred = model_isoF.fit_predict(df_in)             # 1 = normal, -1 = anomalie

    df_in = df_in.assign(outlier=(pred == -1))
    df_in = df_in[df_in["outlier"] == False]

    df_out = df_in.drop("outlier", axis=1)

    return df_out


def preprocessing_Xy(df_in):
    """Sépare X (toutes les colonnes sauf la dernière) et y (la dernière colonne).

    >>> X, y = preprocessing_Xy(pd.DataFrame({"a": [1, 2], "b": [3, 4], "cible": [5, 6]}))
    >>> X.shape, y.shape, y.name
    ((2, 2), (2,), 'cible')
    """
    # X = toutes les colonnes sauf la dernière en DataFrame 2D
    X = df_in.iloc[:, :-1]
    # y = la dernière colonne en série 1D
    y = df_in.iloc[:, -1]
    return X, y


def preprocessing(df_in, ordre="auteur"):
    """Rejoue d'un coup tout le prétraitement : rend X_train, X_test, y_train, y_test.

    ordre="auteur" suit l'enchaînement de l'auteur : imputation, aberrants, X et y, découpage.
    ordre="sans_fuite" découpe avant de chercher les aberrants, qui ne sont retirés que de l'apprentissage :
    le test reste intact. Le jeu attendu est celui de preprocessing_imputation.

    >>> rng = np.random.default_rng(2)
    >>> d = pd.DataFrame({"Unnamed: 0": range(50), "L": 400, "h": 200, "k": 20, "x0": 200, "y0": 100,
    ...                   "d0": np.linspace(10, 100, 50), "Material": "Acier",
    ...                   "Stress/MPa": rng.uniform(0.5, 1.5, 50), "Disp/mm": 0.0, "Msg": None,
    ...                   "FEA_Runtime": 1.0})
    >>> X_train, X_test, y_train, y_test = preprocessing(d)
    >>> (len(X_train), len(X_test))
    (39, 10)
    >>> X_train, X_test, y_train, y_test = preprocessing(d, ordre="sans_fuite")
    >>> (len(X_train), len(X_test))
    (39, 10)
    """
    from sklearn.model_selection import train_test_split

    df_in = preprocessing_imputation(df_in)
    if ordre == "auteur":
        df_in = preprocessing_outliers(df_in)
        X, y = preprocessing_Xy(df_in)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    elif ordre == "sans_fuite":
        X, y = preprocessing_Xy(df_in)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, y_train = preprocessing_Xy(preprocessing_outliers(pd.concat([X_train, y_train], axis=1)))
    else:
        raise ValueError("ordre inconnu : " + str(ordre))
    return X_train, X_test, y_train, y_test


def evaluation_perf(model, X_train, y_train, X_test, y_test, tol=0.05):
    """Pourcentage de prédictions de la base de test dans une tolérance relative `tol`.

    >>> from sklearn.linear_model import LinearRegression
    >>> X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    >>> y = pd.Series([2.0, 4.0, 6.0, 8.0])
    >>> reg = LinearRegression().fit(X, y)
    >>> float(evaluation_perf(reg, X, y, X, y, tol=0.01))
    100.0
    """
    y_pred = model.predict(X_test)
    rel_error = np.abs(y_pred - y_test) / np.abs(y_test)
    return 100 * np.mean(rel_error <= tol)


def evaluation_lc(model, X_train, y_train, tracer=True):
    """Courbe d'apprentissage (validation croisée à 5 plis, score R²) : rend les tailles et les deux scores moyens.

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from sklearn.linear_model import LinearRegression
    >>> rng = np.random.default_rng(3)
    >>> X = pd.DataFrame({"x": rng.uniform(0, 10, 60)})
    >>> y = 2.0 * X["x"] + rng.normal(0, 0.5, 60)
    >>> tailles, entrainement, validation = evaluation_lc(LinearRegression(), X, y, tracer=False)
    >>> len(tailles), bool(validation[-1] > 0.9)
    (10, True)
    """
    from sklearn.model_selection import learning_curve

    train_sizes = np.linspace(0.1, 1.0, 10)
    sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, train_sizes=train_sizes, cv=5, scoring="r2"
    )
    if tracer:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(sizes, np.mean(train_scores, axis=1), "o-", label="Train")
        plt.plot(sizes, np.mean(test_scores, axis=1), "o-", label="Validation")
        plt.xlabel("Taille de l'échantillon d'apprentissage")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
    return sizes, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)


def model_equation(model, X_train, y_train, X_test, y_test, cible="Kt"):
    """Équation d'un modèle linéaire sous forme de texte, et son R² sur la base de test.

    >>> from sklearn.linear_model import LinearRegression
    >>> X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    >>> y = pd.Series([3.0, 5.0, 7.0, 9.0])
    >>> reg = LinearRegression().fit(X, y)
    >>> equation, r2 = model_equation(reg, X, y, X, y, cible="y")
    >>> equation, round(float(r2), 3)
    ('y = 1.000 + (2.00000 × x)', 1.0)
    """
    from sklearn.metrics import r2_score

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    equation = f"{cible} = {model.intercept_:.3f}"

    for coef, name in zip(model.coef_, X_train.columns):
        equation += f" + ({coef:.5f} × {name})"

    return equation, r2


# Partie 5. Données prêtes pour les réseaux de neurones (chapitre 19)

def fenetres_glissantes(valeurs, pas, cible=None):
    """Découpe une série (temps, variables) en fenêtres de `pas` instants : tenseur (échantillons, pas, variables).

    Avec `cible`, rend aussi y : la cible lue à la fin de chaque fenêtre.

    >>> serie = np.arange(12.0).reshape(6, 2)
    >>> X = fenetres_glissantes(serie, 3)
    >>> X.shape
    (4, 3, 2)
    >>> X[1].tolist()
    [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    >>> X, y = fenetres_glissantes(serie, 3, cible=np.arange(6))
    >>> y.tolist()
    [2, 3, 4, 5]
    """
    valeurs = np.asarray(valeurs)
    n = len(valeurs) - pas + 1
    if n <= 0:
        raise ValueError("série plus courte que la fenêtre")
    X = np.stack([valeurs[i:i + pas] for i in range(n)])
    if cible is None:
        return X
    return X, np.asarray(cible)[pas - 1:]


def normaliser_sans_fuite(X_train, X_test, axes=0):
    """Centre et réduit avec la moyenne et l'écart-type de l'apprentissage seul : rend X_train, X_test, moyenne, écart.

    `axes` désigne les axes qui portent les échantillons : 0 pour un tableau (lignes, variables), (0, 1) pour
    un tenseur de fenêtres (échantillons, pas de temps, variables) dont on veut une moyenne par variable.

    >>> a = np.array([[1.0], [2.0], [3.0]])
    >>> b = np.array([[4.0]])
    >>> a_n, b_n, moyenne, ecart_type = normaliser_sans_fuite(a, b)
    >>> round(float(a_n.mean()), 6), round(float(b_n[0, 0]), 3)
    (0.0, 2.449)
    >>> fenetres = np.arange(24.0).reshape(2, 3, 4)
    >>> normaliser_sans_fuite(fenetres, fenetres, axes=(0, 1))[2].shape
    (4,)
    """
    moyenne = X_train.mean(axis=axes)
    ecart = X_train.std(axis=axes)
    ecart = np.where(ecart == 0, 1.0, ecart)
    return (X_train - moyenne) / ecart, (X_test - moyenne) / ecart, moyenne, ecart


def _fonctions_publiques():
    """Les fonctions définies dans ce module, dans l'ordre du fichier."""
    import inspect

    return [nom for nom, obj in globals().items()
            if inspect.isfunction(obj) and obj.__module__ == __name__ and not nom.startswith("__")
            and nom != "_fonctions_publiques"]


if __name__ == "__main__":
    import doctest

    resultat = doctest.testmod()
    print(f"meca : {len(_fonctions_publiques())} fonctions, {resultat.failed} échec(s) sur {resultat.attempted} exemples")
    raise SystemExit(1 if resultat.failed else 0)
