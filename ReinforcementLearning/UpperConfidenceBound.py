# -*- coding: utf-8 -*-

# read dataset
# each row in the dataset is user action for each ad.. (ads are columns)
# if cell value is 1, then it indicates that the user is likely to click on the ad if it's presented
# the cell value of 1 or 0 can be used as reward.. if algo selects an ad for a user and if
# the user is likely to click on the ad (as per the csv file), then the algo receives reward of 1
import pandas as pd
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
round_count, ad_count = dataset.shape

# UCB Implementation
import math
# variable to store ad selection based on max upper bound in each round
ad_selection_in_round = []
# selection counts and rewards sum for each ad
no_selections = [0] * ad_count
rewards_sum = [0] * ad_count
total_reward = 0
# iterate through each round and each ad and compute selections and sum for each ad
for n in range(0, round_count):
    selected_ad = 0
    max_upper_bound = 0
    for i in range(0, ad_count):
        # calculate upper bound using reward and delta for each ad
        # since we're dividing by no of selections for an ad, we need to handle
        # the case when no of selection is 0 for an ad (initial state)
        if no_selections[i] > 0:
            avg_reward = rewards_sum[i] / no_selections[i]
            ad_delta = math.sqrt(3/2 * math.log(n+1) / no_selections[i])
            upper_bound = avg_reward + ad_delta
        else:
            # hard code upper bound to very high no for initial state
            upper_bound = 1e400
        # select the ad with max upper bound in this round
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            selected_ad = i
    ad_selection_in_round.append(selected_ad)
    no_selections[selected_ad] += 1
    reward = dataset.values[n, selected_ad]
    rewards_sum[selected_ad] += reward
    total_reward += reward

# visualize the results
import matplotlib.pyplot as plt
plt.hist(ad_selection_in_round)    

    