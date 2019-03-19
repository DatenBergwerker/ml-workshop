import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# drop doppelte index spalte
dc_properties = pd.read_csv('data/dc_properties.csv').iloc[:, 1:]


# Preprocessing


# Durchführen der PCA auf den numerischen Features

# Plotten der PCA feature importance

# Weiterverarbeiten, Clustering mit PCA Representation
