"""Uber Fares NYC - analyse et visualisations.

Ce module ne fait rien a l'import : chaque etape est une fonction.
  - `python datavis.py` rejoue tout le script (prints + plt.show()).
  - `app.py` importe ces fonctions et les affiche dans la page Flask.
"""
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import FastMarkerCluster

CSV_PATH = Path(__file__).with_name('uber.csv')
MAP_PATH = Path(__file__).with_name('uber_pickup_dropoff_map.html')

DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
             'Friday', 'Saturday', 'Sunday']
MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
               'August', 'September', 'October', 'November', 'December']

# colonnes etudiees dans les boxplots
BOX_COLUMNS = ['fare_amount', 'pickup_longitude', 'pickup_latitude',
               'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

NYC_MIN_LAT, NYC_MAX_LAT = 40.5, 40.9
NYC_MIN_LON, NYC_MAX_LON = -74.25, -73.7


# ----------------------------------------------------------- donnees
def load_data(path=CSV_PATH):
    """Lit uber.csv, retire 'Unnamed: 0' et 'key', supprime les vides, garde 20 000 lignes."""
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0', 'key'], errors='ignore')
    df = df.dropna()
    return df[:20000]


def info_table(df):
    """Equivalent tabulaire de df.info() : une ligne par colonne."""
    return pd.DataFrame({
        'Colonne': df.columns,
        'Valeurs non nulles': df.notna().sum().values,
        'Type': df.dtypes.astype(str).values,
    }).set_index('Colonne')


def inspect_tables(df):
    """Tableaux de l'inspection : apercu, structure, statistiques, manquants, doublons."""
    return [
        ('Apercu (5 premieres courses)', df.head()),
        ('Structure du tableau', info_table(df)),
        ('Statistiques descriptives', df.describe()),
        ('Valeurs manquantes par colonne', df.isnull().sum().to_frame('Manquantes')),
        ('Lignes en double', pd.DataFrame({'Doublons': [int(df.duplicated().sum())]}, index=['Total'])),
    ]


def clean(df):
    df = df[df['fare_amount'] > 0]
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['pickup_longitude'].between(-180, 180)) & (df['pickup_latitude'].between(-90, 90))]
    df = df[(df['dropoff_longitude'].between(-180, 180)) & (df['dropoff_latitude'].between(-90, 90))]
    df = df[(df['pickup_longitude'] != 0) & (df['pickup_latitude'] != 0)]
    df = df[(df['dropoff_longitude'] != 0) & (df['dropoff_latitude'] != 0)]
    return df


def add_datetime_features(df):
    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['pickup_year'] = df['pickup_datetime'].dt.year
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_month'] = df['pickup_datetime'].dt.month_name()
    df['pickup_day'] = df['pickup_datetime'].dt.day_name()
    return df


def sort_by_date(df):
    df = df.sort_values(by='pickup_datetime')
    return df.drop(columns=['pickup_datetime'])


def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371  # rayon de la Terre en kilometres
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def add_distance(df):
    df = df.copy()
    df['distance_km'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    return df[(df['distance_km'] > 0) & (df['distance_km'] <= 200)]


# ----------------------------------------------------------- boxplots (valeurs aberrantes)
def boxplot_stats(series):
    """Statistiques d'un boxplot (regle de l'IQR) : quartiles, moustaches, valeurs aberrantes."""
    q1, med, q3 = series.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = series[(series >= low) & (series <= high)]
    outliers = series[(series < low) | (series > high)]
    return {
        'q1': float(q1), 'median': float(med), 'q3': float(q3),
        'whisker_low': float(inside.min()), 'whisker_high': float(inside.max()),
        'outliers': sorted(outliers.round(6).tolist()),
        'n_outliers': int(len(outliers)),
    }


def boxplot_data(df, columns=BOX_COLUMNS):
    """Donnees de tous les boxplots (colonne -> statistiques), servies en JSON par Flask."""
    return {col: boxplot_stats(df[col]) for col in columns}


def boxplot_table(df, columns=BOX_COLUMNS):
    rows = {}
    for col, st in boxplot_data(df, columns).items():
        rows[col] = {'Q1': st['q1'], 'Mediane': st['median'], 'Q3': st['q3'],
                     'Moustache basse': st['whisker_low'], 'Moustache haute': st['whisker_high'],
                     'Outliers (IQR)': st['n_outliers']}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------- figures
def fig_overview(df, color=None):
    fig = plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    sns.histplot(df['fare_amount'], bins=30, kde=True, **({'color': color} if color else {}))
    plt.title('Distribution of Fare Amount')
    plt.xlabel('Fare Amount')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    sns.scatterplot(x='distance_km', y='fare_amount', data=df, alpha=0.5,
                    **({'color': color, 's': 16, 'linewidth': 0} if color else {}))
    plt.title('Relationship between Distance and Fare Amount')
    plt.xlabel('Distance (km)')
    plt.ylabel('Fare Amount')

    plt.subplot(2, 2, 3)
    df.groupby('pickup_hour')['fare_amount'].mean().plot(
        kind='bar', color=color or 'orange', width=0.55)
    plt.title('Average Fare Amount by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Fare Amount')

    daily_rides = df['pickup_day'].value_counts().reindex(DAY_ORDER, fill_value=0)

    ax = plt.subplot(2, 2, 4)
    sns.barplot(x=daily_rides.index, y=daily_rides.values, color=color or 'steelblue', ax=ax)
    ax.set(title='Ride count by day', xlabel='Day', ylabel='Ride count')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(pad=2.0, h_pad=4.0, w_pad=4.0)
    return fig


def build_map(df, save_to=MAP_PATH):
    """Carte folium (calques Pickups / Drop-offs). Retourne (carte, nb_pickups, nb_dropoffs)."""
    pickup_locations = df.loc[
        df['pickup_latitude'].between(NYC_MIN_LAT, NYC_MAX_LAT)
        & df['pickup_longitude'].between(NYC_MIN_LON, NYC_MAX_LON),
        ['pickup_latitude', 'pickup_longitude'],
    ].values.tolist()

    dropoff_locations = df.loc[
        df['dropoff_latitude'].between(NYC_MIN_LAT, NYC_MAX_LAT)
        & df['dropoff_longitude'].between(NYC_MIN_LON, NYC_MAX_LON),
        ['dropoff_latitude', 'dropoff_longitude'],
    ].values.tolist()

    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

    FastMarkerCluster(
        pickup_locations,
        callback="""function (row) {
        var icon = L.divIcon({
            className: 'pickup-marker',
            html: '<div style="background:#e63946;width:9px;height:9px;border:1px solid white;border-radius:50%;"></div>',
            iconSize: [11, 11],
            iconAnchor: [5, 5]
        });
        return L.marker(new L.LatLng(row[0], row[1]), {icon: icon});
    }""",
        name='Pickups',
    ).add_to(nyc_map)

    FastMarkerCluster(
        dropoff_locations,
        callback="""function (row) {
        var icon = L.divIcon({
            className: 'dropoff-marker',
            html: '<div style="background:#1d70b8;width:9px;height:9px;border:1px solid white;border-radius:50%;"></div>',
            iconSize: [11, 11],
            iconAnchor: [5, 5]
        });
        return L.marker(new L.LatLng(row[0], row[1]), {icon: icon});
    }""",
        name='Drop-offs',
    ).add_to(nyc_map)

    folium.LayerControl().add_to(nyc_map)
    nyc_map.save(str(save_to))
    return nyc_map, len(pickup_locations), len(dropoff_locations)


def monthly_and_daily_totals(df):
    """Courses par mois et montant total par jour (tableaux)."""
    monthly_rides = df['pickup_month'].value_counts().reindex(index=MONTH_ORDER)
    total_amount_per_day = df.groupby('pickup_day')['fare_amount'].sum().reindex(DAY_ORDER)
    return (
        ('Courses par mois', monthly_rides.to_frame('Courses')),
        ('Montant total des tarifs par jour', total_amount_per_day.to_frame('Montant total')),
    )


def fig_totals(df, color=None):
    total_amount_per_day = df.groupby('pickup_day')['fare_amount'].sum().reindex(DAY_ORDER)
    fig = plt.figure(figsize=(12, 5.5))

    plt.subplot(1, 2, 1)
    ax = sns.barplot(x=total_amount_per_day.index, y=total_amount_per_day.values, color=color or 'green')
    ax.set(title='Total Fare Amount by Day', xlabel='Day', ylabel='Total Fare Amount')
    ax.tick_params(axis='x', rotation=45)
    plt.title('Total Fare Amount by Day')

    total_amount_per_month = df.groupby('pickup_month')['fare_amount'].sum().reindex(MONTH_ORDER)
    plt.subplot(1, 2, 2)
    ax = sns.barplot(x=total_amount_per_month.index, y=total_amount_per_month.values, color=color or 'purple')
    ax.set(title='Total Fare Amount by Month', xlabel='Month', ylabel='Total Fare Amount')
    ax.tick_params(axis='x', rotation=45)
    plt.title('Total Fare Amount by Month')
    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout(pad=2.0, w_pad=5.0)
    return fig


def fig_passengers(df, color=None):
    mean_fares = df.groupby('passenger_count')['fare_amount'].mean()
    fig = plt.figure(figsize=(10, 5))
    plt.bar(mean_fares.index, mean_fares.values, color=color or 'teal', width=0.5)
    plt.title('Average Fare Amount by Passenger Count')
    plt.xlabel('Passenger Count')
    plt.ylabel('Average Fare Amount')
    plt.tight_layout(pad=2.0)
    return fig


# ----------------------------------------------------------- execution en script
def show(tables):
    for caption, table in tables:
        print(f'\n== {caption} ==')
        print(table.to_string())


def main():
    raw = load_data()
    show(inspect_tables(raw))
    show([('Boxplots (donnees brutes)', boxplot_table(raw))])

    df = clean(raw)
    show([('Apres nettoyage', df.describe())])
    df = add_datetime_features(df)
    df = sort_by_date(df)
    df = add_distance(df)
    show([('Apercu final', df.head())])

    sns.set_theme(style='whitegrid')
    fig_overview(df)
    _, n_pick, n_drop = build_map(df)
    print(f'Carte enregistree : {MAP_PATH.name} ({n_pick} prises en charge, {n_drop} depots)')
    show(monthly_and_daily_totals(df))
    fig_totals(df)
    fig_passengers(df)
    plt.show()


if __name__ == '__main__':
    main()
