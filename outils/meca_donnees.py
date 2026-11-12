# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Briques clé en main pour explorer un jeu de données (EDA) et le préparer (ENISE), avant tout modèle.

Quarante-cinq briques de quinze lignes au plus, rangées dans l'ordre des chapitres (liste ORDRE_EMPLOI), qui est celui du
livre Méca-IA : l'exploration, puis les opérations ENISE dans l'ordre du sigle (encoder, normaliser, imputer, sélectionner,
extraire), puis l'assemblage. Chacune cite sa brique amont dans la première ligne de sa documentation et porte un
exemple testé par doctest :

    python meca_donnees.py        lance les exemples et contrôle les quinze lignes, la brique amont et l'ordre.

Les noms suivent les habitudes de l'auteur et ceux du livre Méca-IA : df et data, df_in, X_train, y_train,
`preprocessing_Xy`, `liens_entre`, `preparer`, et un nom par opération ENISE (transformer_Encoder, transformer_Scaler,
transformer_Imputer, transformer_xSelecter, transformer_xEngineer), quel que soit l'objet scikit-learn employé.
L'ordre dans lequel un projet enchaîne les opérations (imputer d'abord) n'est pas celui du sigle : c'est ORDRE_OPERATIONS.

Règle du test (la même dans tout le livre) : la structure d'un fichier se lit sur le jeu entier ; tout ce qui croise la
cible, et tout réglage appris (moyenne, médiane, modalités, quartiles), se calcule sur l'apprentissage seul. Une brique
qui apprend reçoit donc X_train, puis applique le réglage à X_test : aucune brique n'ajuste quoi que ce soit sur le test.

Le module est importé par les chapitres 17 et 18 ; il ne touche à aucun fichier, sauf lire_classeur.
"""
import doctest
import inspect
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel, SelectKBest, VarianceThreshold, f_classif,
                                       f_regression)
from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (GridSearchCV, KFold, ShuffleSplit, StratifiedKFold, cross_val_score,
                                     learning_curve, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (Binarizer, FunctionTransformer, KBinsDiscretizer, LabelEncoder, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder, PolynomialFeatures, PowerTransformer, RobustScaler,
                                   StandardScaler)

# Ordre des opérations de la fiche de décisions : on exclut, on impute, on encode, on met à l'échelle, on extrait.
ORDRE_OPERATIONS = ["exclure", "imputer", "encoder", "mettre à l'échelle", "transformer", "extraire", "sélectionner"]
ECHELLES = {"standard": StandardScaler, "robuste": RobustScaler, "minmax": MinMaxScaler}

# Tableau minuscule pour les exemples testés : une mesure manquante, une valeur aberrante, une colonne constante.
JOUET = pd.DataFrame({
    "epaisseur_mm": [2.0, 2.5, np.nan, 3.0, 2.5, 40.0],
    "charge_kN": [10.0, 12.0, 11.0, 15.0, 14.0, 16.0],
    "longueur_mm": [100] * 6,
    "materiau": ["acier", "inox", "acier", np.nan, "inox", "acier"],
    "etat": ["bon", "bon", "bon", "mauvais", "mauvais", "mauvais"]})

# Ordre des briques : (identifiant, fonction, section du livre, briques amont). C'est l'ordre des chapitres, et celui
# du livre Méca-IA : l'exploration (profil, constantes, distributions et valeurs atypiques, paires et corrélations,
# tableau croisé, fiche), puis le découpage, puis l'ENISE dans l'ordre du sigle (B04-1 encoder, B04-2 normaliser,
# B04-3 imputer, B04-4 sélectionner, B04-5 extraire), enfin l'assemblage (B04-6). L'ordre dans lequel un projet
# enchaîne les opérations, imputer d'abord, est celui de ORDRE_OPERATIONS et de la fiche de décisions.
ORDRE_EMPLOI = [
    ("EDA-01", "cadrer", "17.1", ()),
    ("EDA-02", "lire_classeur", "17.1", ("EDA-01",)),
    ("EDA-03", "forme_tableau", "17.1", ("EDA-02",)),
    ("EDA-04", "bilan_manquants", "17.1", ("EDA-03",)),
    ("EDA-05", "colonnes_inutiles", "17.1", ("EDA-04",)),
    ("DEC-01", "decouper", "17.1", ("EDA-05",)),
    ("EDA-06", "tracer_distributions", "17.2", ("DEC-01",)),
    ("EDA-07", "resume_numerique", "17.2", ("EDA-06",)),
    ("EDA-09", "correlations", "17.3", ("EDA-07",)),
    ("EDA-10", "relations_paires", "17.3", ("EDA-09",)),
    ("EDA-15", "liens_entre", "17.3", ("EDA-10",)),
    ("EDA-08", "tableau_croise", "17.3", ("EDA-15",)),
    ("EDA-12", "variable_deterministe", "17.3", ("EDA-08",)),
    ("EDA-11", "aberrants_tukey", "17.4", ("EDA-12",)),
    ("EDA-13", "fiche_decisions", "17.5", ("EDA-11",)),
    ("EDA-14", "mesures_datees", "9.1", ("EDA-02",)),
    ("DEC-02", "preprocessing_Xy", "18.1", ("EDA-13", "DEC-01")),
    ("DEC-03", "decouper_stratifie", "18.1", ("DEC-02",)),
    ("DEC-04", "sensibilite_split", "18.1", ("DEC-03",)),
    ("DEC-05", "separateurs_validation", "18.1", ("DEC-04",)),
    ("DEC-06", "evaluation_lc", "18.1", ("DEC-05",)),
    ("E-1", "encoder_cible", "18.2", ("DEC-06",)),
    ("E-2", "encoder_variables", "18.2", ("E-1",)),
    ("E-3", "encoder_manuel", "18.2", ("E-2",)),
    ("N-1", "pourquoi_echelles", "18.2", ("E-3",)),
    ("N-2", "mise_echelle_minmax", "18.2", ("N-1",)),
    ("N-3", "mise_echelle_standard", "18.2", ("N-2",)),
    ("N-4", "choisir_echelle", "18.2", ("N-3", "EDA-11")),
    ("I-1", "imputer_statistique", "18.3", ("N-4",)),
    ("I-2", "imputer_voisins", "18.3", ("I-1",)),
    ("I-3", "indicateur_manquants", "18.3", ("I-2",)),
    ("S-1", "filtrer_variance", "18.4", ("I-3",)),
    ("S-2", "selection_score", "18.4", ("S-1",)),
    ("S-3", "selection_modele", "18.4", ("S-2",)),
    ("S-4", "elimination_recursive", "18.4", ("S-3",)),
    ("S-5", "controle_selection", "18.4", ("S-4",)),
    ("X-5", "preprocessing_grandeurs", "18.4", ("S-5",)),
    ("X-1", "termes_polynomiaux", "18.4", ("X-5",)),
    ("X-2", "transformer_distribution", "18.4", ("X-1",)),
    ("X-3", "decouper_en_classes", "18.4", ("X-2",)),
    ("X-4", "transformer_fonction", "18.4", ("X-3",)),
    ("A-1", "workflow_quatre_temps", "18.5", ("X-4",)),
    ("A-2", "pipeline_simple", "18.5", ("A-1",)),
    ("A-3", "preparer", "18.5", ("A-2",)),
    ("A-4", "decrire_pipeline", "18.5", ("A-3",)),
]


def dossier_jeux():
    """Dossier des classeurs : la variable MECAPY_DONNEES, sinon le dossier `donnees` voisin de `outils/`, sinon le
    dossier de ce fichier."""
    voisin = Path(__file__).resolve().parent.parent / "donnees"
    return Path(os.environ.get("MECAPY_DONNEES", voisin if voisin.is_dir() else Path(__file__).parent))


# EDA : lire, forme, fond, cible, synthèse

def cadrer(terminal, instrumental, critere, verification):
    """Amont : aucune, départ de la chaîne. Cellule de texte qui cadre le projet avant tout code.
    >>> print(cadrer("Prévoir l'état", "Classer", "exactitude", "sur le test")[:21])
    **Objectif terminal**
    """
    etapes = [("Objectif terminal", terminal), ("Objectif instrumental", instrumental),
              ("Critère de résultat", critere), ("Vérification", verification)]
    return "\n".join(f"**{nom}** : {valeur}" for nom, valeur in etapes)


def lire_classeur(source, dossier=None, **options):
    """Amont : EDA-01. Lit un classeur Excel et garde l'original : renvoie (data, df), df est une copie de travail.
    >>> import io; tampon = io.BytesIO(); JOUET.to_excel(tampon, index=False)
    >>> data, df = lire_classeur(tampon); df.shape, df is not data
    ((6, 5), True)
    """
    if isinstance(source, str):
        source = Path(dossier or dossier_jeux()) / source
    data = pd.read_excel(source, **options)
    return data, data.copy()


def forme_tableau(df):
    """Amont : EDA-02. Forme du tableau : lignes, colonnes, types (numériques, booléens, textes).
    >>> f = forme_tableau(JOUET); list(f.values())
    [6, 5, 3, 0, 2]
    """
    forme = {"lignes": df.shape[0], "colonnes": df.shape[1], "numeriques": df.select_dtypes("number").shape[1],
             "booleens": df.select_dtypes("bool").shape[1]}
    forme["textes"] = df.shape[1] - forme["numeriques"] - forme["booleens"]
    return forme


def bilan_manquants(df, ax=None):
    """Amont : EDA-03. Taux de manquants par colonne, cellules et lignes touchées ; carte des manquants si ax est donné.
    >>> t = bilan_manquants(JOUET); t["cellules"], t["lignes"], round(float(t["taux"].iloc[0]), 3)
    (2, 2, 0.167)
    """
    taux = (df.isna().sum() / df.shape[0]).sort_values(ascending=False)
    if ax is not None:
        sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax)
    return {"taux": taux, "cellules": int(df.isna().sum().sum()), "lignes": int(df.isna().any(axis=1).sum())}


def colonnes_inutiles(df, seuil_vide=0.9, seuil_dominante=0.999):
    """Amont : EDA-04. Colonnes à retirer : presque vides, constantes ou presque, identifiants (une valeur par ligne).
    >>> colonnes_inutiles(JOUET)
    {'vides': [], 'constantes': ['longueur_mm'], 'identifiants': []}
    """
    vides = [c for c in df if df[c].isna().mean() > seuil_vide]
    dominante = df.apply(lambda s: s.value_counts(normalize=True).max() if s.notna().any() else 1.0)
    constantes = [c for c in df if c not in vides and dominante[c] >= seuil_dominante]
    identifiants = [c for c in df if c not in vides + constantes and df[c].nunique() == len(df)
                    and not pd.api.types.is_float_dtype(df[c])]
    return {"vides": vides, "constantes": constantes, "identifiants": identifiants}


def decouper(df, test_size=0.2, random_state=42, afficher=False):
    """Amont : EDA-05. Met le test de côté avant tout croisement avec la cible ; contrôle les tailles.
    >>> apprentissage, test = decouper(JOUET, test_size=0.5); len(apprentissage), len(test)
    (3, 3)
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    assert len(df_train) + len(df_test) == len(df) and not set(df_train.index) & set(df_test.index)
    if afficher:
        print(f"{'Train set size ':.<25} {df_train.shape}\n{'Test set size ':.<25} {df_test.shape}")
    return df_train, df_test


def tracer_distributions(df, colonnes=None, par_ligne=4):
    """Amont : DEC-01. Une distribution par variable numérique (histplot de seaborn) ; renvoie la figure.
    >>> fig = tracer_distributions(JOUET); len(fig.axes)
    3
    """
    cols = list(df.select_dtypes("number").columns if colonnes is None else colonnes)
    lignes = -(-len(cols) // par_ligne)
    fig, axes = plt.subplots(lignes, min(par_ligne, len(cols)), figsize=(2.3 * par_ligne, 1.9 * lignes), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        sns.histplot(df[col], ax=ax, color="0.45", edgecolor="white")
        ax.set_xlabel(f"asymétrie {df[col].skew():.1f}".replace(".", ","))
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    return fig


def resume_numerique(df, groupe=None):
    """Amont : EDA-06. Statistiques d'ensemble, ou moyennes et effectifs par groupe (à lire sur l'apprentissage).
    >>> resume_numerique(JOUET)["count"].tolist(), resume_numerique(JOUET, "etat")["effectif"].tolist()
    ([5.0, 6.0, 6.0], [3, 3])
    """
    if groupe is None:
        return df.describe().T
    moyennes = df.groupby(groupe).mean(numeric_only=True)
    return moyennes.assign(effectif=df[groupe].value_counts().reindex(moyennes.index))


def correlations(df, cible=None, ax=None):
    """Amont : EDA-07. Matrice de corrélation, ou corrélations avec la cible triées ; carte annotée si ax est donné.
    >>> correlations(JOUET).shape, correlations(JOUET, "charge_kN").index.tolist()
    ((3, 3), ['epaisseur_mm', 'longueur_mm'])
    """
    corr = df.corr(numeric_only=True)
    if ax is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greys", vmin=-1, vmax=1, ax=ax)
    return corr if cible is None else corr[cible].drop(cible).sort_values()


def relations_paires(df, colonnes, hue=None, n_max=400):
    """Amont : EDA-09. Nuages deux à deux (pairplot) sur un échantillon, la classe en teinte ; renvoie la grille.
    >>> grille = relations_paires(JOUET, ["epaisseur_mm", "charge_kN"], hue="etat"); len(grille.axes.flat)
    4
    """
    echantillon = df.sample(min(n_max, len(df)), random_state=0)
    garde = list(colonnes) + ([hue] if hue else [])
    return sns.pairplot(echantillon[garde], hue=hue, diag_kind="hist", plot_kws={"s": 8}, height=1.6)


def liens_entre(df_num, seuil=0.85):
    """Amont : EDA-10. Paires liées à plus de seuil par Pearson et par Spearman (redondantes), et paires monotones courbes.
    >>> a = pd.DataFrame({"u": [1.0, 2.0, 3.0, 4.0], "v": [2.0, 4.0, 6.0, 8.0], "w": [4.0, 1.0, 3.0, 2.0]})
    >>> redondantes, courbes = liens_entre(a); redondantes[["a", "b"]].values.tolist(), len(courbes)
    ([['u', 'v']], 0)
    """
    liens = df_num.corr().stack().to_frame("r")
    liens["rho"] = df_num.corr(method="spearman").stack()
    liens = liens.reset_index(names=["a", "b"]).query("a < b")
    cond = liens[["r", "rho"]].abs().min(axis=1) >= seuil
    return liens[cond], liens[liens.rho.abs() - liens.r.abs() > 0.2]


def tableau_croise(ligne, colonne, ax=None):
    """Amont : EDA-15. Effectifs avec totaux et parts par ligne ; carte annotée des parts si ax est donné.
    >>> eff, parts = tableau_croise(JOUET["materiau"], JOUET["etat"]); int(eff.loc["Total", "Total"]), parts.shape
    (5, (2, 2))
    """
    effectifs = pd.crosstab(ligne, colonne, margins=True, margins_name="Total")
    parts = pd.crosstab(ligne, colonne, normalize="index").round(3)
    if ax is not None:
        sns.heatmap(parts, annot=True, fmt=".2f", cbar=False, cmap="Greys", vmin=0, vmax=1, ax=ax)
    return effectifs, parts


def variable_deterministe(df, col, cible):
    """Amont : EDA-08. Part des lignes dont la cible se déduit de la variable seule (1,0 : variable à exclure).
    >>> variable_deterministe(JOUET.assign(x=[1, 1, 1, 9, 9, 9]), "x", "etat")
    1.0
    """
    d = df[[col, cible]].dropna()
    if pd.api.types.is_numeric_dtype(d[col]):
        g = d.groupby(cible)[col].agg(["min", "median", "max"]).sort_values("median")
        coupures = np.sort(((g["max"] + g["min"].shift(-1)) / 2).dropna().to_numpy())
        predite = g.index.to_numpy()[np.searchsorted(coupures, d[col].to_numpy())]
        return float((predite == d[cible].to_numpy()).mean())
    return float(d.groupby(col)[cible].agg(lambda v: v.value_counts().iloc[0]).sum() / len(d))


def aberrants_tukey(serie, k=1.5):
    """Amont : EDA-12. Bornes de Tukey (quartiles ± k fois l'écart interquartile) et masque des valeurs hors bornes.
    >>> bas, haut, hors = aberrants_tukey(JOUET["epaisseur_mm"]); (bas, haut, int(hors.sum()))
    (1.75, 3.75, 1)
    """
    q1, q3 = serie.quantile([0.25, 0.75])
    ecart = q3 - q1
    bas, haut = q1 - k * ecart, q3 + k * ecart
    return bas, haut, (serie < bas) | (serie > haut)


def fiche_decisions(df, cible, exclure=()):
    """Amont : EDA-11. Fiche de décisions : variable, constat, opération prévue, rangées dans l'ordre d'emploi.
    >>> fiche_decisions(JOUET, "etat", exclure=["longueur_mm"]).iloc[:, 2].tolist()
    ['exclure', 'imputer', 'imputer', 'encoder', "mettre à l'échelle", 'transformer']
    """
    cols = [c for c in df.columns if c not in (cible, *exclure)]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    regles = ([(c, "inutile ou déterminée", "exclure") for c in exclure]
              + [(c, f"{100 * df[c].isna().mean():.1f} % manquants", "imputer") for c in cols if df[c].isna().any()]
              + [(c, f"{df[c].nunique()} modalités", "encoder") for c in cols if c not in num]
              + [(c, f"asymétrie {df[c].skew():.1f}", "transformer" if abs(df[c].skew()) > 1 else "mettre à l'échelle")
                 for c in num])
    fiche = pd.DataFrame(regles, columns=["variable", "constat", "operation"])
    return fiche.sort_values("operation", key=lambda o: o.map(ORDRE_OPERATIONS.index), kind="stable", ignore_index=True)


def mesures_datees(df, col_heures, debut="2026-01-01", fenetre=24):
    """Amont : EDA-02. Mesures datées : index temporel depuis un compteur d'heures, moyennes par jour, lissage centré.
    >>> t = pd.DataFrame({"h": range(48), "x": range(48)}); d, jour, lisse = mesures_datees(t, "h")
    >>> d.index[1].hour, jour["x"]["mean"].tolist(), float(lisse["x"].iloc[24])
    (1, [11.5, 35.5], 23.5)
    """
    d = df.set_index(pd.Timestamp(debut) + pd.to_timedelta(df[col_heures], unit="h")).drop(columns=col_heures)
    d.index.name = "horodatage"
    jour = d.resample("D").agg(["mean", "std", "min", "max"])
    lisse = d.rolling(window=fenetre, center=True).mean()
    return d, jour, lisse


# Découpage : X et y, stratification, validation

def preprocessing_Xy(df_in, cible=None):
    """Amont : EDA-13, DEC-01. Sépare X et y : la dernière colonne est la cible, sauf si on la nomme.
    >>> X, y = preprocessing_Xy(JOUET, "etat"); X.shape, y.name, preprocessing_Xy(JOUET)[1].name
    ((6, 4), 'etat', 'etat')
    """
    df = df_in.copy()
    cible = df.columns[-1] if cible is None else cible
    return df.drop(columns=cible), df[cible]


def decouper_stratifie(X, y, test_size=0.2, random_state=42):
    """Amont : DEC-02. Découpage qui garde les proportions de classes ; renvoie X_train, X_test, y_train, y_test.
    >>> y = pd.Series(["a"] * 8 + ["b"] * 2); r = decouper_stratifie(pd.DataFrame({"x": range(10)}), y, 0.5)
    >>> r[3].value_counts().tolist()
    [4, 1]
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def sensibilite_split(modele, X, y, graines=range(5), test_size=0.25):
    """Amont : DEC-03. Score de validation selon la graine du découpage ; à jouer sur l'apprentissage seul.
    >>> X = pd.DataFrame({"x": range(40)}); s = sensibilite_split(KNeighborsClassifier(3), X, pd.Series([0, 1] * 20))
    >>> len(s), bool(0 <= s.min() <= s.max() <= 1)
    (5, True)
    """
    scores = []
    for graine in graines:
        Xa, Xv, ya, yv = train_test_split(X, y, test_size=test_size, random_state=graine)
        scores.append(clone(modele).fit(Xa, ya).score(Xv, yv))
    return np.array(scores)


def separateurs_validation(modele, X, y, separateurs, scoring="accuracy", groupes=None):
    """Amont : DEC-04. Score moyen et écart de la validation croisée pour chaque séparateur (KFold, StratifiedKFold...).
    >>> X = pd.DataFrame({"x": range(40)}); r = separateurs_validation(KNeighborsClassifier(3), X, pd.Series([0, 1] * 20),
    ...                                                                 {"kfold": KFold(4)}); list(r), len(r["kfold"])
    (['kfold'], 2)
    """
    resultats = {}
    for nom, cv in separateurs.items():
        s = cross_val_score(modele, X, y, cv=cv, scoring=scoring, groups=groupes)
        resultats[nom] = (s.mean(), s.std())
    return resultats


def evaluation_lc(model, X_train, y_train, scoring="r2", ax=None):
    """Amont : DEC-05. Courbe d'apprentissage : tailles, score moyen en apprentissage, score moyen en validation.
    >>> X = pd.DataFrame({"x": np.arange(60.0)}); tailles, appr, valid = evaluation_lc(LinearRegression(), X, 2 * X["x"])
    >>> len(tailles), float(valid[-1]) > 0.99
    (5, True)
    """
    tailles, appr, valid = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.2, 1.0, 5), cv=5,
                                          scoring=scoring)
    if ax is not None:
        ax.plot(tailles, appr.mean(axis=1), color="0.6", label="apprentissage")
        ax.plot(tailles, valid.mean(axis=1), color="black", label="validation")
        ax.set_xlabel("lignes d'apprentissage"); ax.legend(frameon=False)
    return tailles, appr.mean(axis=1), valid.mean(axis=1)


# ENISE : dans l'ordre du sigle, comme le livre Méca-IA (encoder, normaliser, imputer, sélectionner, extraire)
# Un projet les enchaîne dans un autre ordre (ORDRE_OPERATIONS : imputer d'abord). Chaque réglage est appris sur
# l'apprentissage (fit), puis appliqué aux deux parties (transform) : le test ne règle rien.

def encoder_cible(y_train, y_test=None):
    """Amont : DEC-06. Encode la cible en entiers (LabelEncoder appris sur l'apprentissage) ; renvoie (encodeur, ya, yt).
    >>> enc, ya, yt = encoder_cible(pd.Series(["b", "a", "b"]), pd.Series(["a"]))
    >>> enc.classes_.tolist(), ya.tolist(), yt.tolist()
    (['a', 'b'], [1, 0, 1], [0])
    """
    transformer_Encoder = LabelEncoder().fit(y_train.unique())
    y_test_code = None if y_test is None else transformer_Encoder.transform(y_test)
    return transformer_Encoder, transformer_Encoder.transform(y_train), y_test_code


def encoder_variables(X_train, X_test, colonnes, methode="onehot"):
    """Amont : E-1. Encode les variables qualitatives (matrice creuse en one-hot) ; renvoie (encodeur, A, T).
    >>> enc, A, T = encoder_variables(JOUET, JOUET, ["materiau"]); A.shape, A.nnz
    ((6, 3), 6)
    """
    if methode == "onehot":
        transformer_Encoder = OneHotEncoder(handle_unknown="ignore")
    else:
        transformer_Encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    transformer_Encoder.fit(X_train[colonnes])
    return transformer_Encoder, transformer_Encoder.transform(X_train[colonnes]), transformer_Encoder.transform(X_test[colonnes])


def encoder_manuel(X_train, X_test, colonnes):
    """Amont : E-2. Codes pandas (cat.codes) après remplissage par le mode de l'apprentissage ; -1 si modalité inconnue.
    >>> a, t = encoder_manuel(JOUET, JOUET.iloc[:1].assign(materiau="titane"), ["materiau"])
    >>> a["materiau"].tolist(), t["materiau"].tolist()
    ([0, 1, 0, 0, 1, 0], [-1])
    """
    A, T = X_train.copy(), X_test.copy()
    for col in colonnes:
        mode = X_train[col].mode()[0]
        categories = X_train[col].fillna(mode).astype("category").cat.categories
        codes = {modalite: i for i, modalite in enumerate(categories)}
        A[col] = A[col].fillna(mode).map(codes).astype(int)
        T[col] = T[col].fillna(mode).map(codes).fillna(-1).astype(int)
    return A, T


def pourquoi_echelles(x, y):
    """Amont : E-3. Pente de la droite ajustée selon les unités : la pente change, le coefficient r reste le même.
    >>> p = pourquoi_echelles(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([2.0, 4.0, 6.0, 8.0])); [round(v[0], 6) for v in p.values()]
    [2e-06, 2000000.0, 2.0, 1.0]
    """
    def z(s): return (s - s.mean()) / s.std(ddof=0)
    cas = {"x fois 1000, y divisé par 1000": (x * 1000, y / 1000), "x divisé par 1000, y fois 1000": (x / 1000, y * 1000),
           "unités de départ": (x, y), "centré réduit": (z(x), z(y))}
    return {nom: (float(stats.linregress(a, b).slope), float(stats.linregress(a, b).rvalue)) for nom, (a, b) in cas.items()}


def mise_echelle_minmax(X_train, X_test):
    """Amont : N-1. MinMaxScaler appris sur l'apprentissage, vérifié contre le calcul à la main ; renvoie (scaler, A, T).
    >>> a = pd.DataFrame({"x": [0.0, 5.0, 10.0]}); s, A, T = mise_echelle_minmax(a, pd.DataFrame({"x": [15.0]}))
    >>> A["x"].tolist(), T["x"].tolist()
    ([0.0, 0.5, 1.0], [1.5])
    """
    a_la_main = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    transformer_Scaler = MinMaxScaler().set_output(transform="pandas").fit(X_train)
    A, T = transformer_Scaler.transform(X_train), transformer_Scaler.transform(X_test)
    assert np.allclose(A.fillna(0), a_la_main.fillna(0)), "le calcul à la main doit égaler MinMaxScaler"
    return transformer_Scaler, A, T


def mise_echelle_standard(X_train, X_test):
    """Amont : N-2. StandardScaler appris sur l'apprentissage, vérifié contre numpy ; renvoie (scaler, A, T).
    >>> a = pd.DataFrame({"x": [1.0, 2.0, 3.0]}); s, A, T = mise_echelle_standard(a, pd.DataFrame({"x": [5.0]}))
    >>> round(float(A["x"].std(ddof=0)), 6), round(float(T["x"].iloc[0]), 4)
    (1.0, 3.6742)
    """
    x = X_train.to_numpy(float)
    a_la_main = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    transformer_Scaler = StandardScaler().set_output(transform="pandas").fit(X_train)
    A, T = transformer_Scaler.transform(X_train), transformer_Scaler.transform(X_test)
    assert np.allclose(A.to_numpy(), a_la_main, equal_nan=True), "le calcul numpy doit égaler StandardScaler"
    return transformer_Scaler, A, T


def choisir_echelle(X_train, part_aberrants=0.01):
    """Amont : N-3, EDA-11. Par colonne : robuste (médiane, quartiles) si trop d'aberrants de Tukey, sinon standard.
    >>> choix, tr = choisir_echelle(JOUET[["epaisseur_mm", "charge_kN"]], 0.1); choix
    {'epaisseur_mm': 'robuste', 'charge_kN': 'standard'}
    """
    hors = {c: float(aberrants_tukey(X_train[c])[2].mean()) for c in X_train}
    choix = {c: "robuste" if part > part_aberrants else "standard" for c, part in hors.items()}
    groupes = [(nom, ECHELLES[nom](), [c for c in choix if choix[c] == nom]) for nom in ("standard", "robuste")]
    transformer_Scaler = ColumnTransformer([g for g in groupes if g[2]], verbose_feature_names_out=False)
    return choix, transformer_Scaler.set_output(transform="pandas")


def imputer_statistique(X_train, X_test, strategie="mean", **options):
    """Amont : N-4. Remplace les manquants par une statistique de l'apprentissage ; renvoie (imputer, A, T).
    >>> A = pd.DataFrame({"x": [1.0, np.nan, 3.0]}); imp, Xa, Xt = imputer_statistique(A, A.iloc[:1].assign(x=np.nan))
    >>> Xa["x"].tolist(), Xt["x"].tolist()
    ([1.0, 2.0, 3.0], [2.0])
    """
    transformer_Imputer = SimpleImputer(missing_values=np.nan, strategy=strategie, **options)
    transformer_Imputer.set_output(transform="pandas").fit(X_train)
    return transformer_Imputer, transformer_Imputer.transform(X_train), transformer_Imputer.transform(X_test)


def imputer_voisins(X_train, X_test, k=2):
    """Amont : I-1. KNNImputer pondéré par la distance ; colonnes déjà à la même échelle de préférence.
    >>> A = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [1.0, 2.0, 3.0, 4.0]}); imp, Xa, Xt = imputer_voisins(A, A, 2)
    >>> round(float(Xa["x"].iloc[2]), 2)
    3.0
    """
    transformer_Imputer = KNNImputer(missing_values=np.nan, n_neighbors=k, weights="distance")
    transformer_Imputer.set_output(transform="pandas").fit(X_train)
    return transformer_Imputer, transformer_Imputer.transform(X_train), transformer_Imputer.transform(X_test)


def indicateur_manquants(X_train, X_test, features="missing-only"):
    """Amont : I-2. Colonnes 0 et 1 qui gardent la trace des manquants, colonnes choisies sur l'apprentissage.
    >>> A = pd.DataFrame({"x": [1.0, np.nan], "y": [1.0, 2.0]}); ind, Ia, It = indicateur_manquants(A, A)
    >>> Ia.columns.tolist(), Ia["manquant_x"].tolist()
    (['manquant_x'], [0, 1])
    """
    transformer_Imputer = MissingIndicator(features=features).fit(X_train)
    noms = [f"manquant_{c}" for c in X_train.columns[transformer_Imputer.features_]]
    def trace(X): return pd.DataFrame(transformer_Imputer.transform(X).astype(int), index=X.index, columns=noms)
    return transformer_Imputer, trace(X_train), trace(X_test)


def filtrer_variance(X_train, X_test, seuil=0.01):
    """Amont : I-3. Retire les colonnes de variance sous le seuil ; colonnes à la même échelle (min-max) avant.
    >>> a = pd.DataFrame({"u": [0.0, 1.0, 0.0, 1.0], "v": [0.5, 0.5, 0.5, 0.5]}); tr, A, T = filtrer_variance(a, a)
    >>> A.columns.tolist()
    ['u']
    """
    transformer_xSelecter = VarianceThreshold(threshold=seuil).set_output(transform="pandas").fit(X_train)
    return transformer_xSelecter, transformer_xSelecter.transform(X_train), transformer_xSelecter.transform(X_test)


def selection_score(X_train, y_train, X_test, k=4, score=f_regression):
    """Amont : S-1. SelectKBest : garde les k colonnes les mieux notées par un test univarié, noté sur l'apprentissage.
    >>> a = pd.DataFrame({"u": [1.0, 2.0, 3.0, 4.0], "v": [1.0, 0.0, 1.0, 0.0]}); tr, A, T = selection_score(a, 2 * a["u"], a, 1)
    >>> A.columns.tolist()
    ['u']
    """
    transformer_xSelecter = SelectKBest(score_func=score, k=k).set_output(transform="pandas")
    transformer_xSelecter.fit(X_train, y_train)
    return transformer_xSelecter, transformer_xSelecter.transform(X_train), transformer_xSelecter.transform(X_test)


def selection_modele(X_train, y_train, X_test, modele=None, max_features=4, seuil="1.25*mean"):
    """Amont : S-2. SelectFromModel : garde les colonnes dont le poids dans le modèle dépasse le seuil.
    >>> a = pd.DataFrame({"u": [1.0, 2.0, 3.0, 4.0], "v": [1.0, 0.0, 1.0, 0.0]}); tr, A, T = selection_modele(a, 2 * a["u"], a, max_features=1)
    >>> A.columns.tolist()
    ['u']
    """
    transformer_xSelecter = SelectFromModel(LinearRegression() if modele is None else modele, threshold=seuil,
                                            max_features=max_features).set_output(transform="pandas")
    transformer_xSelecter.fit(X_train, y_train)
    return transformer_xSelecter, transformer_xSelecter.transform(X_train), transformer_xSelecter.transform(X_test)


def elimination_recursive(X_train, y_train, X_test, modele=None, n=4, cv=None):
    """Amont : S-3. RFE retient n colonnes ; avec cv, RFECV choisit le nombre de colonnes par validation croisée.
    >>> a = pd.DataFrame({"u": [1.0, 2.0, 3.0, 4.0], "v": [1.0, 0.0, 1.0, 0.0]}); tr, A, T = elimination_recursive(a, 2 * a["u"], a, n=1)
    >>> A.columns.tolist()
    ['u']
    """
    modele = LinearRegression() if modele is None else modele
    transformer_xSelecter = RFE(modele, n_features_to_select=n, step=1) if cv is None \
        else RFECV(modele, step=1, min_features_to_select=n, cv=cv)
    transformer_xSelecter.set_output(transform="pandas").fit(X_train, y_train)
    return transformer_xSelecter, transformer_xSelecter.transform(X_train), transformer_xSelecter.transform(X_test)


def controle_selection(X_train, y_train, classement, modele=None, cv=5, ax=None):
    """Amont : S-4. Score de validation croisée selon le nombre de colonnes gardées, rangées du meilleur au pire.
    >>> a = pd.DataFrame({"u": np.arange(30.0), "v": np.arange(30.0) % 3}); s = controle_selection(a, 2 * a["u"], ["u", "v"])
    >>> len(s), bool(s[0] > 0.99)
    (2, True)
    """
    modele = LinearRegression() if modele is None else modele
    scores = [cross_val_score(modele, X_train[classement[:k]], y_train, cv=cv).mean() for k in range(1, len(classement) + 1)]
    if ax is not None:
        ax.plot(range(1, len(scores) + 1), scores, marker="o", color="black")
        ax.set_xlabel("nombre de variables gardées"); ax.set_ylabel("score de validation")
    return scores


def preprocessing_grandeurs(df_in, exclure=()):
    """Amont : S-5. Nettoyage du suivi de production (nom de l'auteur) : colonnes écartées, énergie par pièce.
    >>> t = pd.DataFrame({"consommation_electrique_en_kWh_par_heure": [12.0], "productivite_moyenne_en_pieces_par_heure": [60.0]})
    >>> round(float(preprocessing_grandeurs(t)["energie_par_piece_kWh"].iloc[0]), 2)
    0.2
    """
    df = df_in.drop(columns=list(exclure)).copy()
    df["energie_par_piece_kWh"] = (df["consommation_electrique_en_kWh_par_heure"]
                                   / df["productivite_moyenne_en_pieces_par_heure"])
    return df


def termes_polynomiaux(X_train, X_test, degree=2):
    """Amont : X-5. PolynomialFeatures sans biais : puissances et produits croisés, colonnes nommées.
    >>> a = pd.DataFrame({"u": [1.0, 2.0], "v": [3.0, 4.0]}); tr, A, T = termes_polynomiaux(a, a); A.columns.tolist()
    ['u', 'v', 'u^2', 'u v', 'v^2']
    """
    transformer_xEngineer = PolynomialFeatures(degree=degree, include_bias=False)
    transformer_xEngineer.set_output(transform="pandas").fit(X_train)
    return transformer_xEngineer, transformer_xEngineer.transform(X_train), transformer_xEngineer.transform(X_test)


def transformer_distribution(X_train, X_test):
    """Amont : X-1. PowerTransformer de Yeo et Johnson : rend la distribution plus symétrique, centrée réduite.
    >>> a = pd.DataFrame({"x": [1.0, 1.1, 1.2, 1.3, 9.0]}); tr, A, T = transformer_distribution(a, a)
    >>> bool(abs(A["x"].skew()) < abs(a["x"].skew()))
    True
    """
    transformer_xEngineer = PowerTransformer(method="yeo-johnson", standardize=True)
    transformer_xEngineer.set_output(transform="pandas").fit(X_train)
    return transformer_xEngineer, transformer_xEngineer.transform(X_train), transformer_xEngineer.transform(X_test)


def decouper_en_classes(X_train, X_test, n_bins=4, strategie="kmeans", seuil=None):
    """Amont : X-2. Classes ordonnées (KBinsDiscretizer) ou deux classes autour d'un seuil (Binarizer).
    >>> a = pd.DataFrame({"x": [1.0, 2.0, 8.0, 9.0]}); tr, A, T = decouper_en_classes(a, a, seuil=5.0); A["x"].tolist()
    [0.0, 0.0, 1.0, 1.0]
    """
    if seuil is None:
        transformer_xEngineer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategie)
    else:
        transformer_xEngineer = Binarizer(threshold=seuil)
    transformer_xEngineer.set_output(transform="pandas").fit(X_train)
    return transformer_xEngineer, transformer_xEngineer.transform(X_train), transformer_xEngineer.transform(X_test)


def transformer_fonction(X_train, X_test, f, f_inverse=None):
    """Amont : X-3. Fonction choisie par l'ingénieur (FunctionTransformer), avec sa réciproque pour revenir aux unités.
    >>> a = pd.DataFrame({"x": [0.0, 4.0, 16.0]}); tr, A, T = transformer_fonction(a, a, np.sqrt, np.square)
    >>> A["x"].tolist(), tr.inverse_transform(A)["x"].tolist()
    ([0.0, 2.0, 4.0], [0.0, 4.0, 16.0])
    """
    transformer_xEngineer = FunctionTransformer(f, inverse_func=f_inverse, check_inverse=False, feature_names_out="one-to-one")
    transformer_xEngineer.set_output(transform="pandas").fit(X_train)
    return transformer_xEngineer, transformer_xEngineer.transform(X_train), transformer_xEngineer.transform(X_test)


def workflow_quatre_temps():
    """Amont : X-4. Les quatre temps de l'auteur : référence, standardisation sans pipeline, pipeline, amélioration.
    >>> [temps for temps, geste in workflow_quatre_temps()]
    ['1 Référence', '2 Standardisation sans pipeline', '3 Pipeline', "4 Boucle d'amélioration"]
    """
    return [("1 Référence", "colonnes complètes, découpage, modèle simple : le score à battre"),
            ("2 Standardisation sans pipeline", "échelle apprise sur tout l'apprentissage avant la validation croisée : les plis la partagent"),
            ("3 Pipeline", "l'échelle est apprise dans chaque pli : score sans fuite"),
            ("4 Boucle d'amélioration", "branches numérique et catégorielle, grille d'hyperparamètres, retour à l'EDA")]


def pipeline_simple(modele, X_train, y_train, grille=None, cv=5):
    """Amont : A-1. Pipeline (mise à l'échelle puis modèle) et, avec une grille, recherche d'hyperparamètres.
    >>> X = pd.DataFrame({"x": range(40)}); p = pipeline_simple(KNeighborsClassifier(3), X, pd.Series([0, 1] * 20))
    >>> list(p.named_steps)
    ['scaler', 'classifier']
    """
    pipe = Pipeline([("scaler", StandardScaler()), ("classifier", modele)])
    if grille is None:
        return pipe.fit(X_train, y_train)
    return GridSearchCV(pipe, grille, cv=cv).fit(X_train, y_train)


def preparer(num_cols, cat_cols, k="all", xeng="passthrough", score=f_regression):
    """Amont : A-2. Chaîne complète de préparation : branche numérique, branche catégorielle, k meilleures colonnes.
    >>> X, y = JOUET[["epaisseur_mm", "materiau"]], JOUET["charge_kN"]
    >>> preparer(["epaisseur_mm"], ["materiau"]).fit(X, y).transform(X).shape
    (6, 3)
    """
    num_cols_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                                    ("xeng", xeng)])
    cat_cols_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(transformers=[("num_cols_preproc", num_cols_pipe, num_cols),
                                                   ("cat_cols_preproc", cat_cols_pipe, cat_cols)])
    k_best = SelectKBest(score, k=k)
    return Pipeline(steps=[("preprocessor", preprocessor), ("k_best", k_best)])


def decrire_pipeline(preparation):
    """Amont : A-3. Flowchart en texte : une ligne par branche, les étapes dans l'ordre, puis la sélection.
    >>> p = preparer(["epaisseur_mm"], ["materiau"])
    >>> decrire_pipeline(p)[0]
    'num_cols_preproc : SimpleImputer puis StandardScaler sur 1 colonnes'
    """
    branches = preparation.named_steps["preprocessor"].transformers
    lignes = [f"{nom} : {' puis '.join(type(e).__name__ for _, e in branche.steps if e != 'passthrough')} "
              f"sur {len(cols)} colonnes" for nom, branche, cols in branches]
    return lignes + ["sélection des k meilleures colonnes (SelectKBest)"]


# contrôle du module

def verifier_briques():
    """Contrôle les briques : quinze lignes au plus, brique amont citée avant elles, exemple présent."""
    anomalies, vus = [], []
    for ident, nom, section, amont in ORDRE_EMPLOI:
        source = inspect.getsource(globals()[nom]).rstrip().splitlines()
        doc = inspect.getdoc(globals()[nom]) or ""
        cite = re.match(r"Amont : ([^.]*)\.", doc)
        attendu = ", ".join(amont) if amont else "aucune, départ de la chaîne"
        if len(source) > 15:
            anomalies.append(f"{ident} {nom} : {len(source)} lignes")
        if not cite or cite.group(1) != attendu:
            anomalies.append(f"{ident} {nom} : brique amont non citée comme attendu ({attendu})")
        if ">>>" not in doc:
            anomalies.append(f"{ident} {nom} : aucun exemple testé")
        if any(a not in vus for a in amont):
            anomalies.append(f"{ident} {nom} : brique amont placée après elle")
        vus.append(ident)
    if len(set(vus)) != 45:
        anomalies.append(f"{len(set(vus))} briques distinctes au lieu de 45")
    return anomalies


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    plt.switch_backend("Agg")
    echecs, essais = doctest.testmod()
    plt.close("all")
    problemes = verifier_briques()
    for p in problemes:
        print("ERREUR :", p)
    print(f"exemples : {essais} essais, {echecs} échecs ; briques : {len(ORDRE_EMPLOI)}, anomalies : {len(problemes)}")
    sys.exit(1 if echecs or problemes else 0)
