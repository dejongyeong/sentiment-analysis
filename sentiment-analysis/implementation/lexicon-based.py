"""
Created on Sun Feb 03 12:45am 2019
@author: De Jong Yeong (T00185309)
"""

# Import Statement
import pandas as pd

# Load CSV file
data = pd.read_csv('../datasets/amazon_unlocked_mobile_datasets.csv', header=None, na_values=['NA'], low_memory=False)