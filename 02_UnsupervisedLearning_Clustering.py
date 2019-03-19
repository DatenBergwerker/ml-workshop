import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import euclidean_distances


def read_df(path: str, value_name: str):
    """
    Reads in and processes a world bank dataset to have a common form.
    """
    df = pd.read_csv(path)
    df = df.drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
    df = df.melt(id_vars='Country Name', var_name='year', value_name=value_name)
    df['year'] = pd.to_numeric(df['year'], downcast='integer')
    df = df.rename(columns={'Country Name': 'country_name'})
    return df


def plot_cluster_results(df: pd.DataFrame, x: str, y: str, ax: plt.Axes,
                         cluster_vector, cluster_centroids=None,
                         plot_centroids: bool = True):
    """
    This function takes a dataframe df and plots the specified x and y axis.
    It also shows the color-coded cluster association.
    """
    cluster_labels = set(cluster_vector)
    df['cluster'] = cluster_vector
    col_pal = sns.color_palette(palette='husl', n_colors=len(cluster_labels))
    col_map = {cluster_label: col_pal[i] for i, cluster_label in enumerate(cluster_labels)}
    ax.scatter(x=df[x], y=df[y], c=df['cluster'].apply(lambda cluster: col_map[cluster]), alpha=0.7)

    if plot_centroids:
        cluster_centroids = pd.DataFrame(cluster_centroids, columns=df.columns[2:5])
        cluster_centroids['cluster'] = range(len(cluster_labels))
        ax.scatter(x=cluster_centroids[x], y=cluster_centroids[y],
                   c=cluster_centroids['cluster'].apply(lambda cluster: col_map[cluster]),
                   marker='+', s=500)
    return ax


def determine_eps_range(scaled_data: np.array, k: int, ax: plt.Axes):
    """
    This function takes a scaled distance matrix, sorts each row and returns
    a plot of sorted distance values to the k_th neighbor for each data point.
    The idea is to choose an eps cutoff before which all points have roughly the same
    distance, or in other words, the graph does plateau.
    """
    euc_dist = euclidean_distances(X=scaled_data, Y=scaled_data)
    kth_values = np.sort(euc_dist, axis=1)[:, k - 1]
    kth_values_sorted = np.sort(kth_values)
    ax.plot(kth_values_sorted)
    ax.set_ylabel(f'Distance to {k}th neighbor')
    return ax


# allgemeine optionen
sns.set_style('whitegrid')

# data import
datasets = [('data/world-bank-data/country_population.csv', 'population'),
            ('data/world-bank-data/fertility_rate.csv', 'fertility_rate'),
            ('data/world-bank-data/life_expectancy.csv', 'life_exp')]

dataset_list = [read_df(path=ds[0], value_name=ds[1]) for ds in datasets]

total_world_bank_data = (dataset_list[0]
                         .merge(right=dataset_list[1], on=['country_name', 'year'])
                         .merge(right=dataset_list[2], on=['country_name', 'year']))

# Ausschneiden eines Jahres Fensters zur verringerung des Zeiteffekts
wb_data_nomiss = total_world_bank_data.loc[(~total_world_bank_data.isnull().any(axis=1)) &
                                           (total_world_bank_data['year'] == 2016)]

wb_data_nomiss.loc[:, 'population'] = wb_data_nomiss.loc[:, 'population'] / 1e6
wb_num_data = wb_data_nomiss.drop(['country_name', 'year'], axis=1)

# preprocessing
scaler = StandardScaler()
wb_data_scaled = scaler.fit_transform(X=wb_num_data)

# Verschiedene Clustertechniken ausprobieren
k_range = range(5, 11)
kmean_results = {'k': k_range, 'cluster_assignment': [],
                 'cluster_sse': [], 'cluster_centroids': []}

# KMeans
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmean_results['cluster_assignment'].append(kmeans.fit_predict(X=wb_data_scaled))
    kmean_results['cluster_sse'].append(kmeans.inertia_)
    kmean_results['cluster_centroids'].append(
        scaler.inverse_transform(kmeans.cluster_centers_))

# Plotten der Daten (in 2D) mit cluster labeln als Farben

fig, ax = plt.subplots(nrows=2, ncols=3)
i = 0
for row in ax:
    for col in row:
        plot_cluster_results(df=wb_data_nomiss, x='population', y='fertility_rate',
                             cluster_vector=kmean_results['cluster_assignment'][i],
                             cluster_centroids=kmean_results['cluster_centroids'][i],
                             ax=col)
        col.title(f'K-Means Clusters with k = {k_range[i]}')
        i += 1
plt.show()

# Plot inter cluster errors
fig = plt.figure()
plt.bar(range(5, 11), kmean_results['cluster_sse'])
plt.xticks(range(5, 11))
plt.title('K-Means Cluster SSE for varying k')
plt.xlabel('k')
plt.ylabel('Intra cluster error SSE')
plt.show()
plt.clf()

# Plot

dbscan = DBSCAN(eps=0.5)
dbscan_results = {'k': k_range,
                  'cluster_assignment': dbscan.fit_predict(X=wb_data_scaled)}

fig, ax = plt.subplots(nrows=1, ncols=1)
plot_cluster_results(df=wb_data_nomiss, x='population', y='fertility_rate',
                     cluster_vector=dbscan_results['cluster_assignment'], ax=ax,
                     plot_centroids=False)
plt.show()


fix, ax = plt.subplots(nrows=1, ncols=1)
determine_eps_range(scaled_data=wb_data_scaled, k=4, ax=ax)
plt.show()