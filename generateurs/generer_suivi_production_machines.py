# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""Jeu entièrement simulé : MecaIA-Suivi_Production_Machines.xlsx.

Ce script construit vingt mille relevés de machines, quatorze colonnes,
sans lire aucune donnée existante. Tous les nombres viennent de tirages
pseudo-aléatoires et des relations écrites ici. Il n'utilise ni table
publique, ni coordonnées, ni ordre de lignes provenant d'un autre jeu.

Principe. Un parc surtout composé de machines mécaniques simples travaille
en lots, le plus souvent à une consigne de charge de 50,1 %. L'ancienneté,
un aléa d'usure et quelques incidents composent une usure latente. Celle-ci
augmente vibrations, température et consommation, et diminue la santé.
La complexité augmente le nombre de capteurs et la consommation. La charge
augmente la température et la consommation. Un facteur
de rendement commun lie efficacité et productivité. Les états de santé
sont les sorties du même indice arrondi : Panne_imminente jusqu'à 32,
Degradation_severe jusqu'à 42, Degradation_legere jusqu'à 64, puis RAS.
L'indice est donc une FUITE pour la classification de l'état. Pour prédire
l'indice, il faut écarter l'état. La classe de productivité est elle aussi
un recodage, aux seuils 70 et 100 pièces par heure.

Rôle pédagogique. Lecture, nettoyage et regroupements (chapitres 7 et 9),
distributions, densités, paires, régression, violons et facettes (17), ENISE
(18), classification et régression (19). Exactement trois capteurs ont des
vides : trois tirages de Bernoulli indépendants de probabilité 0,10, posés
après les calculs, indépendants de toutes les valeurs (MCAR par construction).
La suppression des lignes incomplètes perd environ un quart du parc.
Les queues de complexité, de consommation et de capteurs rendent visibles
les limites des règles d'aberrants et du min-max. L'ancienneté est plafonnée
à 40 ans. Le taux d'utilisation a un écart interquartile nul : une consigne
commune et quelques écarts simulent ce piège de la règle de Tukey.
Deux catégories de texte sont presque constantes. Des quotas minuscules
garantissent néanmoins leurs trois modalités, même dans les essais de
sensibilité : les types rares équipent les machines les plus complexes,
les modes rares ont les charges les plus élevées. Aucun ordre externe
n'intervient dans ces classements internes.

Lecture du jeu. Les relations sont plausibles, sans être calées sur des
essais industriels. Coefficients, quotas et bornes sont des hypothèses
pédagogiques, pas des mesures ni des recommandations de conduite. Une ligne
décrit une machine fictive, pas une observation datée d'un équipement réel.
Les valeurs extrêmes représentent ici des configurations rares ou des
incidents ; un test statistique ne suffit pas pour décider de les supprimer.
Les classes de santé visent les ordres de grandeur 57 %, 30 %, 6 %, 7 %,
sans imposer leurs effectifs. La graine 42 est retenue d'emblée ; les graines
0 à 39 servent uniquement à contrôler la stabilité, jamais à choisir le jeu.

Reproductibilité. Uniformes PCG64 de graine 42, bornées à [1e-12, 1-1e-12],
puis quantiles SciPy (normale et Poisson), comme le générateur de soudures.
Les arrondis sont explicites. Les propriétés XML et les dates ZIP sont
fixes, sans nom de personne. Les octets sont reproductibles sur une même
installation ; entre installations, XML et zlib peuvent changer les octets.
La vérification compare aussi chaque cellule, son type et les vides.
Les versions utilisées sont consignées dans les preuves accompagnant le jeu.

Usage :
    python generer_suivi_production_machines.py
    python generer_suivi_production_machines.py ecrire <dossier>
    python generer_suivi_production_machines.py verifier [<dossier>]
La dernière commande régénère sans écrire, contrôle les propriétés, fait
un aller-retour CSV en mémoire et compare le classeur s'il existe.
Le simulateur generer ne lit aucun fichier ; seule la commande verifier
peut lire son propre classeur pour le confronter à la simulation.
"""
import argparse
import datetime
import hashlib
import io
import numbers
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.packaging.core import DocumentProperties
from openpyxl.xml.functions import tostring
from scipy.special import ndtri
from scipy.stats import poisson

GRAINE = 42
N_LIGNES = 20000
NOM_FICHIER = 'MecaIA-Suivi_Production_Machines.xlsx'
FEUILLE = 'Sheet1'
COLONNES = (
    'anciennete_machine_en_annees', 'niveau_complexite_machine',
    'nombre_capteurs_installes', 'taux_utilisation_machine',
    'indice_efficacite_energetique', 'temperature_fonctionnement_en_degC',
    'niveau_vibrations_en_mm_par_s', 'consommation_electrique_en_kWh_par_heure',
    'productivite_moyenne_en_pieces_par_heure', 'indice_sante_machine_0_100',
    'etat_sante_machine', 'type_de_machine', 'mode_de_fonctionnement',
    'classe_niveau_productivite')
CAPTEURS = COLONNES[5:8]
ETATS = ('RAS', 'Degradation_legere', 'Degradation_severe', 'Panne_imminente')
TYPES = ('Mecanique_simple', 'CNC_standard', 'Robotisee_avancee')
MODES = ('Batch', 'Semi_continu', 'Continu')
CLASSES = ('Faible', 'Moyenne', 'Elevee')
DECIMALES = (1, 1, 0, 3, 3, 1, 2, 1, 1, 1)
BORNES = ((1, 40), (1, 10), (5, 30), (.5, 1), (.8, 1.5),
          (65, 118), (.1, 6), (11, 40), (20, 280), (0, 100))
DATE_FIXE = datetime.datetime(2026, 10, 1, 0, 0, 0)
DATE_ZIP = (2026, 10, 1, 0, 0, 0)


def entier_valide(valeur, minimum, nom):
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(valeur, numbers.Integral) or valeur < minimum:
        raise ValueError('%s doit être un entier au moins égal à %d' % (nom, minimum))
    return int(valeur)


class Tirages:
    """Uniformes PCG64 transformées par les fonctions quantiles de SciPy."""

    def __init__(self, graine):
        self.flux = np.random.Generator(np.random.PCG64(entier_valide(graine, 0, 'graine')))

    def uniforme(self, n):
        return np.clip(self.flux.random(n), 1e-12, 1 - 1e-12)

    def normale(self, n):
        return ndtri(self.uniforme(n))

    def poisson(self, moyenne, n):
        return poisson.ppf(self.uniforme(n), moyenne)


def etats_depuis_indice(indice):
    """Recodage des valeurs publiées, sans ambiguïté aux seuils."""
    return np.array(ETATS, dtype=object)[np.select(
        [indice <= 32, indice <= 42, indice <= 64], [3, 2, 1], default=0)]


def generer(n=N_LIGNES, graine=GRAINE):
    """Tableau simulé dépendant seulement de n et de la graine, sans lecture."""
    n = entier_valide(n, 100, 'n')
    t = Tirages(graine)
    age = np.round(np.clip(22 + 11 * t.normale(n), 1, 40), 1)
    # Une petite queue lognormale représente des architectures complexes.
    complexite = np.round(np.clip(1 + .3 * np.exp(.9*t.normale(n)), 1, 10), 1)
    capteurs = np.clip(5 + t.poisson(.6, n) + np.floor(2*(complexite-1)), 5, 30).astype(int)
    # Quotas rares, proportionnels à la taille, attribués par complexité.
    nb_cnc, nb_robot = max(1, round(.0004*n)), max(1, round(.0001*n))
    rang = np.argsort(complexite + .01*t.uniforme(n), kind='stable')
    type_i = np.zeros(n, dtype=int)
    type_i[rang[-nb_cnc-nb_robot:-nb_robot]] = 1
    type_i[rang[-nb_robot:]] = 2
    capteurs[type_i == 1] = np.maximum(capteurs[type_i == 1], 12)
    capteurs[type_i == 2] = 30
    # Consigne commune, petites dérives, rares marches à charge élevée.
    charge = np.where(t.uniforme(n) < .90, .501, .501 + .002*t.normale(n))
    charge = np.round(np.clip(charge, .5, .52), 3)
    nb_semi, nb_continu = max(1, round(.0001*n)), max(1, round(.00005*n))
    rang_charge = np.argsort(t.uniforme(n), kind='stable')
    mode_i = np.zeros(n, dtype=int)
    mode_i[rang_charge[:nb_semi]] = 1
    mode_i[rang_charge[nb_semi:nb_semi+nb_continu]] = 2
    charge[mode_i == 1] = np.round(.65 + .15*t.uniforme(nb_semi), 3)
    charge[mode_i == 2] = 1.0
    rendement = t.normale(n)
    efficacite = np.round(np.clip(.8 + .15*np.exp(.5*rendement), .8, 1.5), 3)
    incident = t.uniforme(n) < .055
    usure = .7*(age-22)/11 + .7*t.normale(n) + 2.0*incident
    temperature = np.round(np.clip(83 + 4.6*usure + 8*(charge-.501) + 2.6*t.normale(n), 65, 118), 1)
    vibrations = np.round(np.clip(2.1 + .7*usure + .30*t.normale(n), .1, 6), 2)
    consommation = np.round(np.clip(11.6 + .85*complexite + .25*usure
                                   + 2*(charge-.501) + .12*t.normale(n), 11, 40), 1)
    productivite = np.round(np.clip(20 + 50*np.exp(.55*rendement + .2*t.normale(n))
                                   - .18*(age-22) - 3*usure, 20, 280), 1)
    sante = np.round(np.clip(69 - 22*usure + 20*(efficacite-.96) + 3*t.normale(n), 0, 100), 1)
    donnees = [age, complexite, capteurs, charge, efficacite, temperature,
               vibrations, consommation, productivite, sante,
               etats_depuis_indice(sante), np.array(TYPES, dtype=object)[type_i],
               np.array(MODES, dtype=object)[mode_i],
               np.array(CLASSES, dtype=object)[np.where(productivite <= 70, 0,
                                                        np.where(productivite <= 100, 1, 2))]]
    df = pd.DataFrame(dict(zip(COLONNES, donnees)))
    for c in CAPTEURS:
        df.loc[t.uniforme(n) < .10, c] = np.nan
    return df


def classeur_en_octets(df):
    """Écriture reprise du générateur de soudures, propriétés et ZIP fixes."""
    wb = Workbook()
    ws = wb.active
    ws.title = FEUILLE
    ws.append(list(df.columns))
    for ligne in df.itertuples(index=False, name=None):
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else
                   (v.item() if hasattr(v, 'item') else v) for v in ligne])
    for k, nom in enumerate(df.columns, start=1):
        ws.column_dimensions[ws.cell(row=1, column=k).column_letter].width = max(12, min(30, len(nom)+5))
    tampon = io.BytesIO()
    wb.save(tampon)
    tampon.seek(0)
    proprietes = DocumentProperties(creator='Meca-IA', lastModifiedBy='Meca-IA',
                                    title='MecaIA-Suivi_Production_Machines',
                                    created=DATE_FIXE, modified=DATE_FIXE)
    sortie = io.BytesIO()
    with zipfile.ZipFile(tampon) as entree, zipfile.ZipFile(sortie, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as arc:
        for info in entree.infolist():
            octets = entree.read(info.filename)
            if info.filename == 'docProps/core.xml':
                octets = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + tostring(proprietes.to_tree())
            zi = zipfile.ZipInfo(info.filename, date_time=DATE_ZIP)
            zi.compress_type, zi.create_system, zi.external_attr = zipfile.ZIP_DEFLATED, 0, 0
            arc.writestr(zi, octets, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return sortie.getvalue()


def sha256(octets):
    return hashlib.sha256(octets).hexdigest()


def colonnes_differentes(df, relu):
    """Comparaison exacte des valeurs, types et vides, cellule par cellule."""
    if list(df.columns) != list(relu.columns) or len(df) != len(relu):
        return ['forme du tableau']
    def genre(s):
        return 'entier' if s.dtype.kind == 'i' else ('décimal' if s.dtype.kind == 'f' else 'texte')
    diff = []
    for c in df:
        a, b = df[c], relu[c]
        if genre(a) != genre(b):
            diff.append(c)
        elif genre(a) in ('entier', 'décimal'):
            if not np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True):
                diff.append(c)
        elif a.astype(object).where(a.notna(), None).tolist() != b.astype(object).where(b.notna(), None).tolist():
            diff.append(c)
    return diff


def proprietes(df):
    """Contrat de structure et de signal pédagogique, bornes larges de stabilité."""
    res = []
    def noter(libelle, ok, mesure):
        res.append((libelle, bool(ok), str(mesure)))
    noter('vingt mille lignes', len(df) == N_LIGNES, len(df))
    noter('14 noms dans le même ordre', tuple(df.columns) == COLONNES, df.shape[1])
    if tuple(df.columns) != COLONNES or not len(df):
        return res
    noter('10 numériques dont un entier, 4 textes',
          all(df[c].dtype.kind == ('i' if i == 2 else 'f') for i, c in enumerate(COLONNES[:10]))
          and all(df[c].dropna().map(lambda v: isinstance(v, str)).all() for c in COLONNES[10:]),
          ', '.join(str(v) for v in df.dtypes))
    for c, valeurs in zip(COLONNES[10:], (ETATS, TYPES, MODES, CLASSES)):
        noter('modalités '+c, set(df[c]) == set(valeurs), sorted(df[c].unique()))
    for c, bornes, dec in zip(COLONNES[:10], BORNES, DECIMALES):
        x = df[c].dropna()
        noter('plage et arrondi '+c,
              len(x) > 0 and np.isfinite(x).all() and x.between(*bornes).all()
              and np.array_equal(x.to_numpy(), x.round(dec).to_numpy()),
              '[%g ; %g], %d décimale(s)' % (x.min(), x.max(), dec))
    for c in CAPTEURS:
        taux = df[c].isna().mean()
        noter('vides proches de 10 % : '+c, .09 <= taux <= .11, '%.3f %%' % (100*taux))
    noter('aucun vide ailleurs', not df.drop(columns=list(CAPTEURS)).isna().any().any(), '11 colonnes')
    complet = df.notna().all(axis=1).mean()
    noter('71 à 75 % de lignes complètes', .71 <= complet <= .75, '%.3f %%' % (100*complet))
    masques = df[list(CAPTEURS)].isna().astype(int)
    maxi = np.max(np.abs(masques.corr().to_numpy()-np.eye(3)))
    noter('masques peu corrélés entre capteurs', maxi < .04, 'max |r| = %.4f' % maxi)
    numeriques = df[list(COLONNES[:10])]
    # Pour son propre capteur, un masque vaut toujours zéro là où la valeur
    # est connue : sa corrélation n'est pas définie, on écarte cette paire.
    maxi_valeurs = max(numeriques.drop(columns=c).corrwith(masques[c]).abs().max() for c in CAPTEURS)
    noter('vides peu liés aux valeurs', maxi_valeurs < .04, 'max |r| = %.4f' % maxi_valeurs)
    parts = df.etat_sante_machine.value_counts(normalize=True)
    for e, (bas, haut) in zip(ETATS, ((.53,.62),(.26,.34),(.035,.085),(.045,.10))):
        part = parts.get(e, 0)
        noter('proportion '+e, bas <= part <= haut, '%.3f %%' % (100*part))
    noter('indice : fuite exacte vers les quatre états',
          np.array_equal(etats_depuis_indice(df.iloc[:,9]), df.etat_sante_machine), 'seuils 32, 42, 64')
    prod = df.iloc[:,8]
    attendu = np.array(CLASSES)[np.where(prod <= 70, 0, np.where(prod <= 100, 1, 2))]
    noter('classe de productivité : recodage exact', np.array_equal(attendu, df.iloc[:,13]), 'seuils 70, 100')
    r = numeriques.corr().iloc[:,9]
    noter('vibrations, première corrélation avec la santé',
          r.drop(COLONNES[9]).abs().idxmax() == CAPTEURS[1] and r[CAPTEURS[1]] < -.75, 'r = %.4f' % r[CAPTEURS[1]])
    noter('température et ancienneté liées à la santé', r[CAPTEURS[0]] < -.65 and r.iloc[0] < -.5,
          'r température %.4f, ancienneté %.4f' % (r[CAPTEURS[0]], r.iloc[0]))
    noter('efficacité et productivité liées', numeriques.iloc[:,4].corr(prod) > .75,
          'r = %.4f' % numeriques.iloc[:,4].corr(prod))
    moyennes = df.groupby('etat_sante_machine')[CAPTEURS[1]].mean().reindex(ETATS)
    noter('vibrations croissantes selon la dégradation', moyennes.is_monotonic_increasing
          and moyennes.iloc[-1] > 1.8*moyennes.iloc[0], moyennes.round(3).to_dict())
    noter('ancienneté plafonnée à 40 ans', (df.iloc[:,0] == 40).mean() > .02, df.iloc[:,0].max())
    for c in (COLONNES[1], COLONNES[7], COLONNES[8]):
        noter('queue à droite '+c, df[c].skew() > 1, 'asymétrie %.3f' % df[c].skew())
    charge = df.iloc[:,3]
    noter('charge : IQR nul et valeurs extrêmes', charge.quantile(.75) == charge.quantile(.25)
          and charge.max() == 1 and charge.skew() > 20, 'asymétrie %.3f' % charge.skew())
    for c in COLONNES[11:13]:
        dominante = df[c].value_counts(normalize=True).max()
        noter('texte presque constant '+c, dominante >= .999, '%.4f %%' % (100*dominante))
    noter('capteurs rares en nombre élevé', df.iloc[:,2].max() == 30 and (df.iloc[:,2] > 10).sum() > 10,
          '%d machines au-dessus de 10' % (df.iloc[:,2] > 10).sum())
    noter('aucun doublon exact', not df.duplicated().any(), int(df.duplicated().sum()))
    noter('graine de diffusion fixée à 42', GRAINE == 42, GRAINE)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description='Suivi de production entièrement simulé, graine 42.')
    p.add_argument('action', nargs='?', choices=('ecrire', 'verifier'), default='ecrire')
    p.add_argument('dossier', nargs='?', type=Path, default=Path('.'))
    args = p.parse_args(argv)
    if not args.dossier.is_dir():
        p.error('le dossier doit exister : %s' % args.dossier)
    df = generer()
    controles = proprietes(df)
    octets = classeur_en_octets(df)
    chemin = args.dossier / NOM_FICHIER
    echecs = sum(not ok for _, ok, _ in controles)
    if args.action == 'ecrire':
        if echecs:
            for libelle, ok, mesure in controles:
                if not ok:
                    print('KO', libelle, mesure)
            return 1
        chemin.write_bytes(octets)
        print('%s : %d lignes, %d colonnes, %d octets, sha256 %s' %
              (chemin, len(df), df.shape[1], len(octets), sha256(octets)))
        return 0
    def controle(libelle, ok, mesure):
        nonlocal echecs
        print('%-4s %s : %s' % ('OK' if ok else 'KO', libelle, mesure))
        echecs += not ok
    echecs = 0
    for libelle, ok, mesure in controles:
        controle(libelle, ok, mesure)
    controle('deux générations donnent les mêmes octets', classeur_en_octets(generer()) == octets, sha256(octets))
    csv = io.StringIO(df.to_csv(index=False))
    diff = colonnes_differentes(df, pd.read_csv(csv, float_precision='round_trip'))
    controle('CSV en mémoire : toutes les cellules et types', not diff, diff or 'identiques')
    if chemin.is_file():
        wb = load_workbook(chemin, read_only=True)
        controle('feuille unique Sheet1', wb.sheetnames == [FEUILLE], wb.sheetnames)
        wb.close()
        diff = colonnes_differentes(df, pd.read_excel(chemin))
        controle('classeur : toutes les cellules, types et vides', not diff, diff or 'identiques')
        print('INFO classeur : octets identiques :', chemin.read_bytes() == octets)
    print('%d propriété(s) en échec' % echecs)
    return 1 if echecs else 0


if __name__ == '__main__':
    sys.exit(main())
