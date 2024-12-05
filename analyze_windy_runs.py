import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm  # for color maps
from wind_profiles import generate_smooth_profile

# Load wind files from wind database
def load_wind_profiles(num_profiles=50):
    wind_profiles = []
    # Loop through all CSV files and plot each profile
    for i in range(1, num_profiles + 1):
        # Read the CSV file
        filename = f'wind_data/wind_profile_{i}.csv'
        table = np.genfromtxt(filename, delimiter=',', skip_header=1)
        wind_profiles.append(table)
    return np.array(wind_profiles)

# Class for linear regression definitions
class WindLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = WindLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, x, y):
        super(WindLinearRegression, self).__init__()
        self.x = x
        self.y = y

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = np.shape(x)[0]
        y_predict = np.zeros((m, 1))
        import pdb
        pdb.set_trace()
        theta = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y
        y_predict = x @ theta
        
        return theta, y_predict

winds = load_wind_profiles()

# Potential metrics
mean_wind = np.mean(winds[:, :, 1], axis=1)
high_alt_values = winds[:, -1, 1]
landing_alt_values = winds[:, 0, 1]
range_in_wind = np.max(abs(winds[:, :, 1]), axis=1) - np.min(abs(winds[:, :, 1]), axis=1)
# Successes over this no wind policy
success_no_wind_policy = np.load('./wind_evaluations_with_nowind_policy/success_with_different_winds.npz')['arr_0']

# Generate 500 new arrays using arbitrary linear combinations
n_eval_winds = 500
evaluation_winds = generate_smooth_profile(500)

fig = plt.figure()
for i in range(n_eval_winds):
    plt.plot(winds[0, :, 0], evaluation_winds[i, :])

plt.grid()
plt.show()

regression = WindLinearRegression(winds[:, :, 1], success_no_wind_policy)
theta_vals, y_predict = regression.predict(evaluation_winds)
theta_vals = theta_vals/np.linalg.norm(theta_vals)
print(theta_vals)
print(y_predict)

fig = plt.figure()
plt.plot(winds[0, :, 0], theta_vals)
plt.grid()
plt.xlabel('Altitude of wind value (m)')
plt.ylabel('Normalized importance to success percentage')
plt.show()

#### PLOTTING ###
fig = plt.figure()
plt.plot(np.arange(len(success_no_wind_policy))+1, success_no_wind_policy)
plt.grid()
# plt.show()
fig = plt.figure()
plt.scatter(mean_wind, success_no_wind_policy)
plt.grid()
plt.xlabel('Mean wind force (N)')
# plt.show()
fig = plt.figure()
plt.scatter(high_alt_values, success_no_wind_policy)
plt.grid()
plt.xlabel('High altitude mean force (N)')
# plt.show()
fig = plt.figure()
plt.scatter(landing_alt_values, success_no_wind_policy)
plt.grid()
plt.xlabel('Low altitude mean force (N)')
# plt.show()
fig = plt.figure()
plt.scatter(range_in_wind, success_no_wind_policy)
plt.grid()
plt.xlabel('Range in wind (N)')
# plt.show()

fig,ax = plt.subplots()
cmap = cm.viridis  # You can use other colormaps, like cm.plasma, cm.inferno, etc.
norm = plt.Normalize(vmin=0.0, vmax=100.0)  # Normalize z values for color mapping
for i in range(winds.shape[0]):
    ax.plot(winds[i, :, 1], winds[i, :, 0], color=cmap(norm(success_no_wind_policy[i])))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed to add the color bar; no data passed here
plt.colorbar(sm, ax=ax, label='Success probability')
ax.grid()
ax.set_xlabel('Wind lateral force (N)')
ax.set_ylabel('Altitude (m)')
fig.savefig('imgs/success_of_different_winds.pdf')

