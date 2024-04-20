import pandas as pd

dataset_path = '../data/Titanic Dataset.csv'

titanic_df = pd.read_csv(dataset_path)

print(titanic_df.head())

print(titanic_df.describe())
