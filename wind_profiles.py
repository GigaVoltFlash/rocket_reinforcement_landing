import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

'''
The goal of these wind profiles is to create a random wind disturbance as a function of altitude y, which then maybe exert a 2D
force on the rocket in one direction or the other. Then we see how the trained policy is able to handle it.
Then, can we train it based on the wind? That would be cool.
'''




def generate_wind_profile(num_profiles=10, altitude_range=(0, 500), force_range=(-4, 4)):
    # Define the altitudes
    altitudes = np.linspace(altitude_range[0], altitude_range[1], 100)
    
    # Function to generate smooth wind profiles
    def generate_smooth_profile():
        # Randomly decide if the wind turns or not
        direction_change = random.choice([True, False])
        
        # Generate wind force values
        wind_forces = []
        
        if direction_change:
            # Wind starts in one direction, then turns at some point
            change_point = random.randint(20, 80)  # Altitude index where the direction changes
            initial_direction = random.choice([1, -1])  # Random initial direction: 1 or -1
            
            # Generate a profile that changes direction
            for i, altitude in enumerate(altitudes):
                if i < change_point:
                    wind_forces.append(initial_direction * random.uniform(0, 4))  # Increasing wind in one direction
                else:
                    wind_forces.append(-initial_direction * random.uniform(0, 4))  # Wind changes direction
        else:
            # Wind stays in one direction (constant)
            direction = random.choice([1, -1])  # Random constant direction
            for i, altitude in enumerate(altitudes):
                wind_forces.append(direction * random.uniform(0, 4))  # Constant wind force in one direction
        
        return wind_forces
    
    # Generate and save profiles
    for i in range(1, num_profiles + 1):
        wind_forces = generate_smooth_profile()
        # Create a DataFrame to save
        df = pd.DataFrame({'Altitude': altitudes, 'Horizontal Force': wind_forces})
        # Save to CSV
        filename = f'wind_data/wind_profile_{i}.csv'
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
        plt.plot(df['Altitude'], df['Horizontal Force'], label=f'Profile {i}')
    
    # Adding labels and grid
    plt.xlabel('Altitude (m)')
    plt.ylabel('Horizontal Force (N)')
    plt.title('Wind Profiles')
    plt.grid(True)
    plt.legend()

    # Save the plot as a PDF
    plt.savefig('wind_data/wind_profiles.pdf')
    plt.close()

    
# Generate 10 wind profiles and save them
generate_wind_profile()

# Example usage: Assuming CSV files are in the current directory
plot_wind_profiles_from_csv()