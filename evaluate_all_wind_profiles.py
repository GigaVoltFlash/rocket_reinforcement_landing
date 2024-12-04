from evaluate_dqn import evaluate_dqn
import os
import numpy as np
import re

wind_folder = './wind_data'
wind_files = [file for file in os.listdir(wind_folder) if file.endswith('.csv')]

################# RUNS EVALUATED WITH NO-WIND TRAINED POLICY #######################
wind_data_output = './wind_evaluations_with_nowind_policy'
if not os.path.exists(wind_data_output):
    os.makedirs(wind_data_output)

no_wind_profiles = len(wind_files)
successful_run_counts = np.zeros(no_wind_profiles)
for file in wind_files:
    wind_path = os.path.join(wind_folder, file)
    wind_number = re.findall(r'\d+', wind_path)[0] # Regex to get the wind number
    num_succesful_runs = evaluate_dqn(wind_path=wind_path, savefile_suffix=wind_number, save_folder=wind_data_output)
    successful_run_counts[int(wind_number)-1] = num_succesful_runs
    
np.savez(os.path.join(wind_data_output, 'success_with_different_winds.npz'), successful_run_counts)

################# RUNS EVALUATED WITH WIND TRAINED POLICY #######################
wind_data_output = './wind_evaluations_with_windy_policy'
if not os.path.exists(wind_data_output):
    os.makedirs(wind_data_output)

no_wind_profiles = len(wind_files)
successful_run_counts = np.zeros(no_wind_profiles)
for file in wind_files:
    wind_path = os.path.join(wind_folder, file)
    wind_number = re.findall(r'\d+', wind_path)[0] # Regex to get the wind number
    num_succesful_runs = evaluate_dqn(weights_path='placeholder', wind_path=wind_path, savefile_suffix=wind_number, save_folder=wind_data_output)
    successful_run_counts[int(wind_number)-1] = num_succesful_runs
    
np.savez(os.path.join(wind_data_output, 'success_with_different_winds.npz'), successful_run_counts)