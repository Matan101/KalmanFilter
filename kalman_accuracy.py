import numpy as np
from matplotlib import pyplot as plt

from kalman_filter import kalman_filter
from noisy_path import generate_noisy_path


# Calculate accuracy (RMSE)
def calculate_kalman_accuracy(simulations_num, noise_mean, noise_standard_deviation, initial_state, motion_variance,
                              measurement_variance):
    deviation_values = []

    for i in range(simulations_num):
        x_true, y_true, noisy_path = generate_noisy_path(noise_mean, noise_standard_deviation)
        filtered_path = kalman_filter(noisy_path, initial_state, motion_variance, measurement_variance)
        deviation_values.append((filtered_path - y_true) ** 2)

    mean_rmse_values = np.sqrt(np.mean(np.array(deviation_values), axis=0))

    return mean_rmse_values


# Print mean RMSE over time
def print_kalman_accuracy(mean_rmse_values):
    plt.plot(mean_rmse_values)
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('RMSE over Time')
    plt.show()
