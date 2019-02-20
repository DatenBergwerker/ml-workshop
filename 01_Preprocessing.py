import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler, \
    MinMaxScaler, Normalizer, OneHotEncoder


total_data = pd.read_csv('data/agriculture_yields.csv')

total_data.plot(x='Field Area', y='Yield', kind='scatter')
plt.show()
total_data['Field Area'].describe()

scaler = StandardScaler()
total_data_scaled = scaler.fit_transform(total_data[['Field Area', 'Yield']])
total_data_scaled = pd.DataFrame(total_data_scaled, columns=['Field Area', 'Yield'])
total_data_scaled.plot(x='Field Area', y='Yield', kind='scatter')
plt.show()
total_data_scaled['Field Area'].describe()

r_scaler = RobustScaler()
total_data_rscaled = r_scaler.fit_transform(total_data[['Field Area', 'Yield']])
total_data_rscaled = pd.DataFrame(total_data_rscaled, columns=['Field Area', 'Yield'])
total_data_rscaled.plot(x='Field Area', y='Yield', kind='scatter')
plt.show()
total_data_rscaled['Field Area'].describe()

minmax_scaler = MinMaxScaler()
total_data_mmscaled = minmax_scaler.fit_transform(total_data[['Field Area', 'Yield']])
total_data_mmscaled = pd.DataFrame(total_data_mmscaled, columns=['Field Area', 'Yield'])
total_data_mmscaled.plot(x='Field Area', y='Yield', kind='scatter')
plt.show()
total_data_mmscaled['Field Area'].describe()

norm = Normalizer()
total_data_norm = minmax_scaler.fit_transform(total_data[['Field Area', 'Yield']])
total_data_norm = pd.DataFrame(total_data_norm, columns=['Field Area', 'Yield'])
total_data_norm.plot(x='Field Area', y='Yield', kind='scatter')
plt.show()
total_data_norm['Field Area'].describe()

categoricals = total_data.columns[1:-1]
total_data[categoricals].shape
ohc = OneHotEncoder(sparse=False)
total_data_ohc = ohc.fit_transform(total_data[categoricals])
total_data_ohc = pd.DataFrame(total_data_ohc, columns=ohc.get_feature_names())
total_data_ohc['Field Area'] = total_data['Field Area']
total_data_ohc.shape