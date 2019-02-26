import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


def read_df(path: str, value_name: str):
    """
    Reads in and processes a world bank dataset to have a common form.
    """
    df = pd.read_csv(path)
    df = df.drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
    df = df.melt(id_vars='Country Name', var_name='year', value_name=value_name)
    df = df.rename(columns={'Country Name': 'country_name'})
    return df


datasets = [('data/world-bank-data/country_population.csv', 'population'),
            ('data/world-bank-data/fertility_rate.csv', 'fertility_rate'),
            ('data/world-bank-data/life_expectancy.csv', 'life_exp')]

dataset_list = [read_df(path=ds[0], value_name=ds[1]) for ds in datasets]

total_world_bank_data = (dataset_list[0]
                         .merge(right=dataset_list[1], on=['country_name', 'year'])
                         .merge(right=dataset_list[2], on=['country_name', 'year']))

# preprocess the data

# pick a slice of 10 years

# cluster it using different techniques, try a pca to check the direction of the variance

