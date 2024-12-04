import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# need to load the run information across the windy cases
# need to load the wind profiles themselves
# extract some useful features of the wind profiles (mean wind? high altitude wind?, variance?)
# do some logistical regression?
    # what's safe and what's not safe to fly in? 
# do some linear regression?
    # number of successful runs for different wind conditions

def load_wind_profiles(num_profiles=20):
    wind_profiles = []
    # Loop through all CSV files and plot each profile
    for i in range(1, num_profiles + 1):
        # Read the CSV file
        filename = f'wind_data/wind_profile_{i}.csv'
        table = np.genfromtxt(filename, delimiter=',', skip_header=1)
        wind_profiles.append(table)
    return np.array(wind_profiles)
winds = load_wind_profiles()
mean_wind = np.mean(winds[:, :, 1], axis=1)
high_alt_values = winds[:, -1, 1]
landing_alt_values = winds[:, 0, 1]
range_in_wind = np.max(abs(winds[:, :, 1]), axis=1) - np.min(abs(winds[:, :, 1]), axis=1)
# how does this compare to a run that was trained in a windy condition?
success_no_wind_policy = np.load('./wind_evaluations_with_nowind_policy/success_with_different_winds.npz')['arr_0']
fig = plt.figure()
plt.plot(np.arange(len(success_no_wind_policy))+1, success_no_wind_policy)
plt.grid()
plt.show()
fig = plt.figure()
plt.scatter(mean_wind, success_no_wind_policy)
plt.grid()
plt.xlabel('Mean wind force (N)')
plt.show()
fig = plt.figure()
plt.scatter(high_alt_values, success_no_wind_policy)
plt.grid()
plt.xlabel('High altitude mean force (N)')
plt.show()
fig = plt.figure()
plt.scatter(landing_alt_values, success_no_wind_policy)
plt.grid()
plt.xlabel('Low altitude mean force (N)')
plt.show()
fig = plt.figure()
plt.scatter(range_in_wind, success_no_wind_policy)
plt.grid()
plt.xlabel('Range in wind (N)')
plt.show()