"""
Created on Sun Feb 03 12:45am 2019
@author: De Jong Yeong (T00185309)
"""

# Import Statement
import pandas as pd

# Load CSV file with specific column only
filename = '../datasets/amazon_unlocked_mobile_datasets.csv'
fields = ['Product Name', 'Brand Name', 'Reviews']
data = pd.read_csv(filename, low_memory=False, usecols=fields)

# Data Understanding and Analyzing
shape = data.shape
types = data.dtypes
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Dimensions: {data.ndim}')
print(f'No. of Null Values: {data.isnull().sum().sum()}')

# Remove Null Values
data = data.dropna(axis=0, how='any')
shape = data.shape
print(f'\nAfter Cleaning:')
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Null Values: {data.isnull().sum().sum()}')

# Remove Trailing Spaces
data.columns = data.columns.str.strip()