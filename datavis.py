import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import FastMarkerCluster


df = pd.read_csv('uber.csv')
df = df.drop(columns=['Unnamed: 0', 'key'], errors='ignore')
df = df.dropna()
df.head()
df = df[:20000]

print(df.head())
df.info()
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())


##cleaning the data
df = df[df['fare_amount'] > 0]

df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]

df = df[(df['pickup_longitude'].between(-180, 180)) & (df['pickup_latitude'].between(-90, 90))]
df = df[(df['dropoff_longitude'].between(-180, 180)) & (df['dropoff_latitude'].between(-90, 90))]

df = df[(df['pickup_longitude'] != 0) & (df['pickup_latitude'] != 0)]
df = df[(df['dropoff_longitude'] != 0) & (df['dropoff_latitude'] != 0)]

print(df.describe())

## extracting date and time features
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df['pickup_year'] = df['pickup_datetime'].dt.year
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_month'] = df['pickup_datetime'].dt.month_name()
df['pickup_day'] = df['pickup_datetime'].dt.day_name()

print(df.head())

df.sort_values(by='pickup_datetime', inplace=True)
df.drop(columns=['pickup_datetime'], inplace=True)
print(df.head())

def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371  # نصف قطر الأرض بالكيلومتر
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c

df['distance_km'] = haversine_distance(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

df = df[(df['distance_km'] > 0) & (df['distance_km'] <= 200)]

print(df.head())

sns.set_theme(style='whitegrid')
plt.figure(figsize=(15, 12))


plt.subplot(2, 2, 1)
sns.histplot(df['fare_amount'], bins=30, kde=True)
plt.title('Distribution of Fare Amount')
plt.xlabel('Fare Amount')
plt.ylabel('Frequency')


plt.subplot(2, 2, 2)
sns.scatterplot(x='distance_km', y='fare_amount', data=df, alpha=0.5)
plt.title('Relationship between Distance and Fare Amount')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount')


plt.subplot(2, 2, 3)
df.groupby('pickup_hour')['fare_amount'].mean().plot(kind='bar' , color="orange")
plt.title('Average Fare Amount by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Fare Amount')

monthly_rides = df['pickup_month'].value_counts()
daily_rides = df['pickup_day'].value_counts()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
             'Friday', 'Saturday', 'Sunday']

daily_rides = df['pickup_day'].value_counts().reindex(day_order, fill_value=0)

ax = plt.subplot(2, 2, 4)
sns.barplot(x=daily_rides.index, y=daily_rides.values, color='steelblue', ax=ax)
ax.set(title='Ride count by day', xlabel='Day', ylabel='Ride count')
ax.tick_params(axis='x', rotation=45)


plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=1.5)
plt.show()

nyc_min_lat, nyc_max_lat = 40.5, 40.9
nyc_min_lon, nyc_max_lon = -74.25, -73.7

pickup_locations = df.loc[
    df['pickup_latitude'].between(nyc_min_lat, nyc_max_lat)
    & df['pickup_longitude'].between(nyc_min_lon, nyc_max_lon),
    ['pickup_latitude', 'pickup_longitude'],
].values.tolist()

dropoff_locations = df.loc[
    df['dropoff_latitude'].between(nyc_min_lat, nyc_max_lat)
    & df['dropoff_longitude'].between(nyc_min_lon, nyc_max_lon),
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
nyc_map.save('uber_pickup_dropoff_map.html')
print('Saved interactive map to uber_pickup_dropoff_map.html')




month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']

monthly_rides = monthly_rides.reindex(index=month_order)

print(monthly_rides)

plt.figure(figsize=(10, 8))

total_amount_per_day = df.groupby('pickup_day')['fare_amount'].sum().reindex(day_order)

print(total_amount_per_day)

plt.figure(figsize=(10, 5))

##i want to subplots
plt.subplot(1, 2, 1)
ax = sns.barplot(x=total_amount_per_day.index, y=total_amount_per_day.values, color='green')
ax.set(title='Total Fare Amount by Day', xlabel='Day', ylabel='Total Fare Amount')
ax.tick_params(axis='x', rotation=45)   
plt.title('Total Fare Amount by Day')

total_amount_per_month = df.groupby('pickup_month')['fare_amount'].sum().reindex(month_order)
plt.subplot(1, 2, 2)
ax = sns.barplot(x=total_amount_per_month.index, y=total_amount_per_month.values, color='purple')
ax.set(title='Total Fare Amount by Month', xlabel='Month', ylabel='Total Fare Amount')
ax.tick_params(axis='x', rotation=45)
plt.title('Total Fare Amount by Month')
plt.show()

mean_fares_by_pessenger_count = df.groupby('passenger_count')['fare_amount'].mean()
plt.figure(figsize=(10, 5))
plt.bar(mean_fares_by_pessenger_count.index, mean_fares_by_pessenger_count.values, color='teal')
plt.title('Average Fare Amount by Passenger Count')
plt.xlabel('Passenger Count')
plt.ylabel('Average Fare Amount')   
plt.show()
