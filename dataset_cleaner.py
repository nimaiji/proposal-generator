import pandas as pd
import numpy as np
# df = pd.read_csv('./companies-2023-q4-sm.csv')

# df = df[df['country_code'] == 'GB'].dropna()

# df.to_csv('cleaned_dataset.csv', index=False)

df = pd.read_csv('./LM/cleaned_dataset.csv')

# print(df.columns)
# print(df.info())
# print(df['city'].value_counts())

print(np.array(list(df.itertuples()))[0][8])