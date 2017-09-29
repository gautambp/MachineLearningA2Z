# -*- coding: utf-8 -*-

# read dataset
# each row in the dataset is user action for each ad.. (ads are columns)
# if cell value is 1, then it indicates that the user is likely to click on the ad if it's presented
# the cell value of 1 or 0 can be used as reward.. if algo selects an ad for a user and if
# the user is likely to click on the ad (as per the csv file), then the algo receives reward of 1
import pandas as pd
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
round_count, ad_count = dataset.shape

# Thompson Sampling Implementation
import random
# variable to store ad selection based on max upper bound in each round
ad_selection_in_round = []
# variables to store ad count with reward 0 & 1 (for each ad)
no_rewards_1 = [0] * ad_count
no_rewards_0 = [0] * ad_count
total_reward = 0
# iterate through each round and each ad and compute selections and sum for each ad
for n in range(0, round_count):
    selected_ad = 0
    max_random = 0
    for i in range(0, ad_count):
        random_beta = random.betavariate(no_rewards_1[i]+1, no_rewards_0[i]+1)
        # select the ad with max upper bound in this round
        if random_beta > max_random:
            max_random = random_beta
            selected_ad = i
    ad_selection_in_round.append(selected_ad)
    reward = dataset.values[n, selected_ad]
    if reward == 1:
        no_rewards_1[selected_ad] += 1
    else:
        no_rewards_0[selected_ad] += 1
    total_reward += reward

# visualize the results
import matplotlib.pyplot as plt
plt.hist(ad_selection_in_round)    
