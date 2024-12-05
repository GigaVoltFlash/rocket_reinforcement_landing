import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from scipy.signal import lfilter
from scipy.ndimage import gaussian_filter1d
'''
The goal of these wind profiles is to create a random wind disturbance as a function of altitude y, which then maybe exert a 2D
force on the rocket in one direction or the other. Then we see how the trained policy is able to handle it.
Then, can we train it based on the wind? That would be cool.
'''
# Function to generate smooth wind profiles
def generate_smooth_profile(num_profiles, 
                            altitudes=np.linspace(0, 500, 100), 
                            high_alt_force=np.linspace(-4.0, 4.0, 100), 
                            middle_alt_force = np.linspace(-4.0, 4.0, 100), 
                            low_alt_force = np.linspace(-1.0, 1.0, 100)):
    # Generate wind force values
    wind_forces = np.zeros((num_profiles, len(altitudes)))

    for j in range(num_profiles):

        high_alt_force_choice = random.choice(high_alt_force)
        low_alt_force_choice = random.choice(low_alt_force)
        middle_alt_force_choice = random.choice(middle_alt_force)
        change_point = random.randint(20, 80)  # Altitude index where the direction changes

        for i, altitude in enumerate(altitudes):
            if i < change_point:
                wind_forces[j, i] = (altitude - altitudes[0]) * (middle_alt_force_choice - low_alt_force_choice)/(altitudes[change_point] - altitudes[0]) + low_alt_force_choice + random.uniform(0, 0.25)
            else:
                wind_forces[j, i] = (altitude - altitudes[change_point]) * (high_alt_force_choice - middle_alt_force_choice)/(altitudes[-1] - altitudes[change_point]) + middle_alt_force_choice + random.uniform(0, 0.25)

        wind_forces[j, :] = gaussian_filter1d(wind_forces[j, :], 2.5)
    
    return wind_forces

def generate_wind_profile(num_profiles=10, altitude_range=(0, 500), force_range=(-5, 5)):
    # Define the altitudes
    altitudes = np.linspace(altitude_range[0], altitude_range[1], 100)
    high_alt_force = np.linspace(force_range[0], force_range[1], 100)
    middle_alt_force = np.linspace(force_range[0], force_range[1], 100)
    low_alt_force = np.linspace(-1.0, 1.0, 100)
    
    # Generate and save profiles
    for i in range(num_profiles):
        wind_forces = generate_smooth_profile(num_profiles, high_alt_force=high_alt_force, middle_alt_force =middle_alt_force, low_alt_force = low_alt_force)
        df = pd.DataFrame({'Altitude': altitudes, 'Horizontal Force': wind_forces[i, :]})
        filename = f'wind_data/wind_profile_{i+1}.csv'
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")

def plot_wind_profiles_from_csv(csv_folder=".", num_profiles=10):
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Loop through all CSV files and plot each profile
    for i in range(1, num_profiles + 1):
        # Read the CSV file
        filename = os.path.join(csv_folder, f'wind_data/wind_profile_{i}.csv')
        df = pd.read_csv(filename)
        
        # Plot the wind profile using the Altitude and Horizontal Force columns
        plt.plot(df['Horizontal Force'], df['Altitude'], label=f'Profile {i}')
    
    # Adding labels and grid
    plt.ylabel('Altitude (m)')
    plt.xlabel('Horizontal Force on the Rocket (N)')
    plt.title('Wind Profiles')
    plt.grid(True)
    # plt.legend()

    # Save the plot as a PDF
    plt.savefig('wind_data/wind_profiles.pdf')
    plt.close()

if __name__ == '__main__':
    # Generate 10 wind profiles and save them
    generate_wind_profile(num_profiles=50)

    # Example usage: Assuming CSV files are in the current directory
    plot_wind_profiles_from_csv(num_profiles=50)