import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# drop unnecessary column
dc_properties = pd.read_csv('data/dc_properties.csv').iloc[:, 1:]

# apply correct preprocessing for numeric / categorical cols

# apply PCA dimension reduction

# cluster and plot the data


