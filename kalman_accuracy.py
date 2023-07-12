import numpy as np
from matplotlib import pyplot as plt

from kalman_filter import kalman_filter
from noisy_path import generate_noisy_path


# Calculate accuracy (RMSE)
def calculate_kalman_accuracy(simulations_num, noise_mean, noise_standard_deviation, initial_state, motion_variance,
                              measurement_variance):
    rmse_values = []
    for _ in range(simulations_num):
        x_true, y_true, noisy_path = generate_noisy_path(noise_mean, noise_standard_deviation)
        filtered_path = kalman_filter(noisy_path, initial_state, motion_variance, measurement_variance)
        rmse = np.sqrt(np.mean((filtered_path - y_true) ** 2))
        rmse_values.append(rmse)

    return rmse_values


# Print RMSE over simulations
def print_kalman_accuracy(print_accuracy, rmse_values):
    plt.plot(range(print_accuracy), rmse_values)
    plt.xlabel('Simulation')
    plt.ylabel('RMSE')
    plt.title('RMSE over Simulations')
    plt.show()
