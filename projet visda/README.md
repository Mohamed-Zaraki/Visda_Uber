# Uber Fares NYC - Flask

`datavis.py` contient toute l'analyse (fonctions) ; `app.py` l'importe et affiche les etapes dans
`templates/index.html` : tableaux, figures, carte folium et boxplots a choisir.

1. `pip install -r requirements.txt`
2. Placer `uber.csv` a cote de `app.py`
3. `python app.py` puis http://127.0.0.1:5000

`python datavis.py` rejoue le script (tableaux en texte + plt.show()).

Routes : `/` page, `/plot/<nom>.png` graphiques (`fig1`, `fig2`, `fig3`), `/api/boxplots` donnees des
boxplots (JSON, dessines dans le navigateur), `/map` carte folium.
