import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm  # for color maps
from wind_profiles import generate_smooth_profile
from sklearn import tree

# Load wind files from wind database
def load_wind_profiles(num_profiles=200, test=False):
    wind_profiles = []
    # Loop through all CSV files and plot each profile
    for i in range(1, num_profiles + 1):
        # Read the CSV file
        if test:
            filename = f'test_wind_data/wind_profile_{i}.csv'
        else:
            filename = f'wind_data/wind_profile_{i}.csv'
        table = np.genfromtxt(filename, delimiter=',', skip_header=1)
        wind_profiles.append(table)
    return np.array(wind_profiles)

def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.005

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print(np.linalg.norm(prev_theta - theta))
        if np.linalg.norm(prev_theta - theta) < 1e-4:
            print('Converged in %d iterations' % i)
            break
        if i % 1e7 == 0:
            print('Still hasn\'t converged, providing best case theta')
            break
    return theta

# For regression fitting
def add_intercept(x):
    """
    Add intercept to matrix x.    
    Args:
        x: 2D NumPy array.    
    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x    
    return new_x

# Perform Principal Component Analysis to reduce the dimensionality of the wind data
def pca(x, k=2):
    # x is expected to be of dimensions m x n (m is number of datapoints, n is number of features)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_norm = (x - x_mean)/x_std
    x_cov = np.cov(x_norm.T)
    U, S, Vt = np.linalg.svd(x_cov)
    u_vectors = Vt[:k]
    return u_vectors

# Class for linear regression definitions
class WindLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = WindLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, x, y, u):
        super(WindLinearRegression, self).__init__()
        self.u = u
        xhat = x @ u.T # Reduced dimensions
        xhat_mod = add_intercept(xhat)
        self.theta = np.linalg.inv(xhat_mod.T @ xhat_mod) @ xhat_mod.T @ y

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        xhat = x @ self.u.T
        xhat_mod = add_intercept(xhat)
        y_predict = xhat_mod @ self.theta
        return y_predict

# Load train wind profiles
train_winds = load_wind_profiles()
test_winds = load_wind_profiles(num_profiles=10, test=True)

# Potential metrics
mean_wind = np.mean(train_winds[:, :, 1], axis=1)
high_alt_values = train_winds[:, -1, 1]
landing_alt_values = train_winds[:, 0, 1]
mid_alt_values = train_winds[:, 20, 1]
range_in_wind = np.max(abs(train_winds[:, :, 1]), axis=1) - np.min(abs(train_winds[:, :, 1]), axis=1)
# Successes over this no wind policy
success_label_train = np.load('./wind_evaluations_with_nowind_policy/success_with_different_winds.npz')['arr_0']
success_label_test = np.load('./wind_evaluations_with_nowind_policy_test/success_with_different_winds.npz')['arr_0']

success_label_train_logic = np.where(success_label_train > 90, 1.0, 0.0)
success_label_test_logic = np.where(success_label_test > 90, 1.0, 0.0)


# Successes over windy policy
windy_success_label_train = np.load('./wind_evaluations_with_windy_policy/success_with_different_winds.npz')['arr_0']
windy_success_label_test = np.load('./wind_evaluations_with_windy_policy_test/success_with_different_winds.npz')['arr_0']

windy_success_label_train_logic = np.where(windy_success_label_train > 90, 1.0, 0.0)
windy_success_label_test_logic = np.where(windy_success_label_test > 90, 1.0, 0.0)

# LINEAR REGRESSION
# u = pca(train_winds[:, :, 1], k=5)
# u = np.eye(100)
# # Initialize and fit regression on train data
# regression = WindLinearRegression(train_winds[:, :, 1], success_label_train, u)
# theta_vals = regression.theta

# # Check error on train data
# y_predict = regression.predict(train_winds[:, :, 1])
# print(theta_vals)
# print(y_predict)
# print(success_label_train)
# print(np.sqrt(np.mean((success_label_train - np.clip(y_predict, 0.0, 100.0))**2)))

# # Evaluate fit on test data
# y_predict = regression.predict(test_winds[:, :, 1])
# print(y_predict)
# print(success_label_test)
# print(np.sqrt(np.mean((success_label_test - np.clip(y_predict, 0.0, 100.0))**2)))

# # Expand the theta from the linear regression to the full information space
# theta_vals_expanded = np.zeros(train_winds.shape[1] + 1, dtype=theta_vals.dtype)
# theta_vals_expanded[0] = theta_vals[0]
# theta_vals_expanded[1:] = (u.T @ theta_vals[1:].reshape(-1, 1)).squeeze()

# fig = plt.figure()
# plt.plot(train_winds[0, :, 0], theta_vals_expanded[1:])
# plt.grid()
# plt.xlabel('Altitude of wind value (m)')
# plt.ylabel('Normalized importance to success percentage')
# plt.show()

# LOGISTIC REGRESSION
# Initialize and fit regression on train data
# theta = logistic_regression(train_winds[:40, :, 1], success_label_train_logic[:40])
# y_predict = 1. / (1 + np.exp(-(train_winds[:40, :, 1]).dot(theta)))
# print(y_predict)
# print(success_label_train_logic[:40])
# print(np.sqrt(np.mean((success_label_train_logic[:40] - np.clip(y_predict, 0.0, 1.0))**2)))

# y_predict = 1. / (1 + np.exp(-(test_winds[:, :, 1]).dot(theta)))
# print(y_predict)
# print(success_label_test_logic)
# print(np.sqrt(np.mean((success_label_test_logic - np.clip(y_predict, 0.0, 1.0))**2)))
# print(theta)
# fig = plt.figure()
# plt.plot(train_winds[0, :, 0], theta)
# plt.grid()
# plt.xlabel('Altitude of wind value (m)')
# plt.ylabel('Normalized importance to success percentage')
# plt.show()

# DECISION TREE SKLEARN (WORKS)
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(train_winds[:, :, 1], success_label_train_logic)
# y_predict = clf.predict(train_winds[:, :, 1])
# print(np.sqrt(np.mean((success_label_train_logic- np.clip(y_predict, 0.0, 1.0))**2)))

# y_predict = clf.predict(test_winds[:, :, 1])
# print(np.sqrt(np.mean((success_label_test_logic- np.clip(y_predict, 0.0, 1.0))**2)))

# tree.plot_tree(clf)

# #### PLOTTING ###
fig = plt.figure()
plt.plot(np.arange(len(success_label_train))+1, success_label_train)
plt.grid()
# plt.show()
fig = plt.figure()
plt.scatter(mean_wind, success_label_train)
plt.grid()
plt.xlabel('Mean wind force (N)')
# plt.show()
fig = plt.figure()
plt.scatter(high_alt_values, success_label_train)
plt.grid()
plt.xlabel('High altitude mean force (N)')
# plt.show()
fig = plt.figure()
plt.scatter(mid_alt_values, success_label_train)
plt.grid()
plt.xlabel('Low altitude mean force (N)')
plt.show()
fig = plt.figure()
plt.scatter(range_in_wind, success_label_train)
plt.grid()
plt.xlabel('Range in wind (N)')
# plt.show()

# No wind Policy
fig,ax = plt.subplots()
cmap = cm.viridis  # You can use other colormaps, like cm.plasma, cm.inferno, etc.
norm = plt.Normalize(vmin=0.0, vmax=100.0)  # Normalize z values for color mapping
for i in range(train_winds.shape[0]):
    ax.plot(train_winds[i, :, 1], train_winds[i, :, 0], color=cmap(norm(success_label_train[i])))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed to add the color bar; no data passed here
plt.colorbar(sm, ax=ax, label='Success probability')
ax.grid()
ax.set_xlabel('Wind lateral force (N)')
ax.set_ylabel('Altitude (m)')
fig.savefig('imgs/success_of_different_winds.pdf')

# Windy Policy
fig,ax = plt.subplots()
cmap = cm.viridis  # You can use other colormaps, like cm.plasma, cm.inferno, etc.
norm = plt.Normalize(vmin=0.0, vmax=100.0)  # Normalize z values for color mapping
for i in range(train_winds.shape[0]):
    ax.plot(train_winds[i, :, 1], train_winds[i, :, 0], color=cmap(norm(windy_success_label_train[i])))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed to add the color bar; no data passed here
plt.colorbar(sm, ax=ax, label='Success probability')
ax.grid()
ax.set_xlabel('Wind lateral force (N)')
ax.set_ylabel('Altitude (m)')
fig.savefig('imgs/success_of_different_winds_windy.pdf')

