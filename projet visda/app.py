"""Uber Fares NYC - les etapes de datavis.py servies par Flask.

Le code d'analyse vit dans datavis.py ; ce fichier l'importe, execute les etapes dans l'ordre
et les affiche dans la page : tableaux, figures, carte folium et boxplots a choisir.

Lancer :  pip install -r requirements.txt  puis  python app.py
Ouvrir :  http://127.0.0.1:5000        (uber.csv doit etre a cote de ce fichier)
"""
import io
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, Response, abort, jsonify, render_template

import datavis

app = Flask(__name__)

STEPS = []   # ordre de la page : etapes 'tables', 'boxplot', 'figure', 'map'


def fmt(v):
    """Nombres lisibles : entiers sans decimales, sinon 4 decimales sans zeros inutiles."""
    if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer()):
        return f"{int(v):,}".replace(",", " ")
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".").replace(",", " ")
    return v


def to_html(table):
    return table.rename_axis(None).to_html(classes="tbl", border=0, float_format=fmt, na_rep="-")


def add_tables(title, desc, tables):
    STEPS.append({"kind": "tables", "title": title, "desc": desc,
                  "tables": [{"caption": c, "html": to_html(t)} for c, t in tables]})


def add_figure(fig_id, title, desc):
    STEPS.append({"kind": "figure", "id": fig_id, "title": title, "desc": desc})


# =========================================================== etapes (fonctions de datavis.py)
raw = datavis.load_data()          # donnees brutes : sert aussi aux boxplots

add_tables(
    "1. Chargement et inspection",
    "Lecture de uber.csv, suppression des colonnes 'Unnamed: 0' et 'key' et des lignes vides, puis "
    "conservation des 20 000 premieres courses. Apercu, structure, statistiques descriptives, "
    "valeurs manquantes et doublons.",
    datavis.inspect_tables(raw),
)

BOX_DATA = datavis.boxplot_data(raw)
STEPS.append({
    "kind": "boxplot", "title": "2. Boxplots des valeurs aberrantes (donnees brutes)",
    "desc": "Choisissez les variables a afficher. Chaque boxplot suit la regle de l'IQR : les points "
            "hors de 1,5 fois l'ecart interquartile sont des valeurs aberrantes (coordonnees "
            "impossibles, tarifs extremes, courses a 0 passager).",
    "columns": datavis.BOX_COLUMNS,
    "table": to_html(datavis.boxplot_table(raw)),
})

df = datavis.clean(raw)
add_tables(
    "3. Nettoyage des donnees",
    "On garde les tarifs strictement positifs, 1 a 6 passagers, des coordonnees valides "
    "(longitude entre -180 et 180, latitude entre -90 et 90) et differentes de 0. "
    "Statistiques descriptives apres nettoyage.",
    [("Statistiques apres nettoyage", df.describe())],
)

df = datavis.add_datetime_features(df)
add_tables(
    "4. Variables de date et d'heure",
    "La date de prise en charge est convertie en vraie date, puis on en extrait l'annee, l'heure, "
    "le nom du mois et le nom du jour.",
    [("Apercu", df.head())],
)

df = datavis.sort_by_date(df)
add_tables(
    "5. Tri chronologique",
    "Les courses sont triees de la plus ancienne a la plus recente, puis la colonne de date "
    "d'origine est supprimee.",
    [("Apercu", df.head())],
)

df = datavis.add_distance(df)
add_tables(
    "6. Distance de la course",
    "La distance a vol d'oiseau (formule de haversine, rayon terrestre 6 371 km) entre la prise "
    "en charge et le depot est ajoutee ; on garde les trajets de plus de 0 et jusqu'a 200 km.",
    [("Apercu", df.head())],
)

# ----------------------------------------------------------- style des figures (theme sombre)
BG, INK, INK2, GRID, AMBER = "#0f1218", "#f4efe4", "#bdb8aa", "#262b36", "#f2b233"
sns.set_theme(style='whitegrid', rc={
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "grid.color": GRID, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold"})

add_figure(
    "fig1", "7. Figure 1 : vue d'ensemble (2 x 2)",
    "Quatre graphiques : distribution des tarifs (histogramme + courbe de densite), relation "
    "distance / tarif (nuage de points), tarif moyen par heure de la journee, nombre de courses "
    "par jour de la semaine.",
)

nyc_map, n_pickups, n_dropoffs = datavis.build_map(df)
STEPS.append({
    "kind": "map", "title": "8. Carte interactive des prises en charge et des depots",
    "desc": "Seules les courses dans la zone de New York (latitude 40,5 a 40,9 ; longitude -74,25 a "
            "-73,7) sont gardees. Deux calques : prises en charge (rouge) et depots (bleu), avec un "
            "menu de calques. La carte est aussi enregistree dans un fichier HTML.",
    "counts": (n_pickups, n_dropoffs),
})

add_tables(
    "9. Courses par mois et montant total par jour",
    "Nombre de courses par mois (de janvier a decembre), puis somme des tarifs par jour de la semaine.",
    datavis.monthly_and_daily_totals(df),
)

add_figure(
    "fig2", "10. Figure 2 : montants totaux par jour et par mois",
    "Deux diagrammes en barres cote a cote : somme des tarifs par jour de la semaine, puis par mois.",
)
add_figure(
    "fig3", "11. Figure 3 : tarif moyen par nombre de passagers",
    "Barres du tarif moyen pour 1 a 6 passagers.",
)

# nom -> (fonction qui dessine, donnees, dpi)
FIGS = {
    "fig1": (datavis.fig_overview, df, 110),
    "fig2": (datavis.fig_totals, df, 140),
    "fig3": (datavis.fig_passengers, df, 140),
}


@lru_cache(maxsize=None)
def render_fig(name):
    fn, data, dpi = FIGS[name]
    fig = fn(data, color=AMBER)
    for ax in fig.axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


# =========================================================== routes
@app.route("/")
def index():
    return render_template("index.html", steps=STEPS, n=len(df))


@app.route("/plot/<name>.png")
def plot(name):
    if name not in FIGS:
        abort(404)
    return Response(render_fig(name), mimetype="image/png")


@app.route("/api/boxplots")
def api_boxplots():
    """Statistiques de tous les boxplots (le navigateur dessine celui qu'on choisit)."""
    return jsonify(BOX_DATA)


@app.route("/map")
def folium_map():
    return Response(nyc_map.get_root().render(), mimetype="text/html")


if __name__ == "__main__":
    app.run(debug=True)
