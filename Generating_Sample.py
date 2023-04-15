import pandas as pd
import numpy as np
from scipy import stats
import math

# read csv data
data = pd.read_csv('marketing_AB.csv')

# check for duplicate and null data
# print(f'Number of user id duplicate is {data.duplicated(["user id"]).sum()}')
# print(f' Number of null data: {data.isna().sum()}')
# no duplicate or null data found

# calculate the conversion rate of control group
filt = (data['test group'].str.contains('psa')) & (data['converted'] == True)
num_conv_psa = len(data[filt])
num_psa = data['test group'].value_counts()['psa']
conv_rate_psa = num_conv_psa / num_psa
# calculate standard deviation based on the control group conversion rate
std_dev = np.sqrt(conv_rate_psa * (1 - conv_rate_psa))

# calculate sample size
alpha = 0.05  # significance level, industry standard value
power = 0.8  # industry standard value
delta = 0.01  # given value
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power)
sample_size = math.ceil(2 * (std_dev ** 2) * ((z_beta + z_alpha) ** 2) / (delta ** 2))

# generate sample with the size as calculated from the initial dataset
df_psa = data[data['test group'] == 'psa'].sample(n=sample_size)
df_ad = data[data['test group'] == 'ad'].sample(n=sample_size)
