"""
Created on Sun Feb 03 12:45am 2019
@author: De Jong Yeong (T00185309)
"""

# Import Statement
import pandas as pd

# Load CSV file
filename = '../datasets/amazon_unlocked_mobile_datasets.csv'
data = pd.read_csv(filename, na_values=['NA'], low_memory=False)

# Data Understanding and Analyzing
shape = data.shape
types = data.dtypes
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Null Values: {data.isnull().sum().sum()}')
print(f'\nType for each attribute:\n{types}')