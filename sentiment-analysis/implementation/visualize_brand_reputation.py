"""
Created on Tue April 16 2:16pm 2019
@author: De Jong Yeong

Visualize a summary of positive, negative, and neutral review for only the 'Apple and BlackBerry Brand'.

Output file: -
"""

# import library
import pandas as pd
from matplotlib import pyplot as plt

# load data
filename = '../datasets/amazon_unlocked_mobile_datasets_with_sentiment.csv'
names = ['product.name', 'brand.name', 'review.text', 'review.process', 'review.tokened', 'score', 'sentiment']
fields = ['brand.name', 'sentiment']
data = pd.read_csv(filename, names=names, usecols=fields)
data.rename(columns={'brand.name': 'brand'}, inplace=True)  # rename column

print()

# replace brand name to same phrase for each brand
# e.g. apple -> Apple
data = data.replace({'brand': ['Apple Computer', 'apple']}, 'Apple')

# BlackBerry and Research in Motion
bb = 'BlackBerry Storm 9530 Smartphone Unlocked GSM Wireless Handheld Device w/Camera Bluetooth 3.25" Touchscreen LCD'
data = data.replace({'brand': [bb, 'BLACKBERRY', 'Black Berry', 'Blackberry', 'blackberry', 'Research In Motion',
                               'Research in Motion', 'BlackBerry (RIM)']}, 'BlackBerry')

# analyse only Apple and BlackBerry
df = data[data['brand'].isin(['Apple', 'BlackBerry'])]

# unique brand
brands = df.brand.unique()

# count sentiment label
result = df.groupby(['brand', 'sentiment']).size()
result = result.unstack(level=1)  # convert to dataframe

# drop null values
result = result.dropna()

# plot graph
fig = plt.figure()
ax = result.plot.bar()
plt.title('Total Sentiment Labels of Apple and BlackBerry Mobile Reviews\n')
plt.ylabel('Total')
plt.xlabel('Brands')
plt.xticks(rotation=-360)  # rotate x labels
plt.savefig('../results/brand_sentiment_labels.png', format='png', transparent=False)  # save result
plt.show()

# end visualization


"""
Appendix
"""
# Amazon
# data = data.replace({'brand': ['Amazon.com, LLC *** KEEP PORules ACTIVE ***']}, 'Amazon')

# Asus
# data = data.replace({'brand': ['Asus', 'ASUS Computers', 'asus']}, 'ASUS')

# Yezz
# data = data.replace({'brand': ['Yezz', 'Yezz Wireless Ltd.']}, 'Yezz')

# rename brand name
# before = ['Cedar Tree Technologies', 'VKworld', 'ARGOM TECH', 'iDROID USA', 'amar']
# after = ['Cedar Tree', 'VKWORLD', 'Agrom Tech', 'iDroid USA', 'Amar']
# data = data.replace(before, after)
