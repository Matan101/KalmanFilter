import numpy as np
from matplotlib import pyplot as plt

from kalman_filter import kalman_filter_uniform_velocity, kalman_filter_uniform_acceleration
from noisy_path import generate_noisy_uniform_velocity_path, generate_noisy_uniform_acceleration_path


# Compute Kalman Filter accuracy (RMSE) for # Kalman Filter Algorithm for uniform velocity
def compute_kalman_accuracy_uniform_velocity(simulations_num, x_initial, y_initial, x_velocity, y_velocity, time,
                                             noise_mean, noise_standard_deviation, motion_variance,
                                             measurement_variance):
    rmse_values = []

    for i in range(simulations_num):
        # Generate original path and noisy path
        x_path, y_path, x_noisy, y_noisy = generate_noisy_uniform_velocity_path(x_initial, y_initial, x_velocity,
                                                                                y_velocity, time,
                                                                                noise_mean, noise_standard_deviation)

        # Apply Kalman Filter
        x_filtered, y_filtered = kalman_filter_uniform_velocity(x_noisy, y_noisy, x_initial, y_initial, x_velocity,
                                                                y_velocity,
                                                                motion_variance, measurement_variance)

        # Compute RMSE separately for each time unit
        rmse_values.append(np.sqrt((x_filtered - x_path) ** 2 + (y_filtered - y_path) ** 2))

    # Compute mean RMSE separately for each time unit
    mean_rmse_values = np.mean(np.array(rmse_values), axis=0)

    return mean_rmse_values


# Compute Kalman Filter accuracy (RMSE) for # Kalman Filter Algorithm for uniform acceleration
def compute_kalman_accuracy_uniform_acceleration(simulations_num, x_initial, y_initial, x_velocity, y_velocity,
                                                 x_acceleration, y_acceleration, time, noise_mean,
                                                 noise_standard_deviation, motion_variance, measurement_variance):
    rmse_values = []

    for i in range(simulations_num):
        # Generate original path and noisy path
        x_path, y_path, x_noisy, y_noisy = generate_noisy_uniform_acceleration_path(x_initial, y_initial, x_velocity,
                                                                                    y_velocity, x_acceleration,
                                                                                    y_acceleration, time, noise_mean,
                                                                                    noise_standard_deviation)

        # Apply Kalman Filter
        x_filtered, y_filtered = kalman_filter_uniform_acceleration(x_noisy, y_noisy, x_initial, y_initial, x_velocity,
                                                                    y_velocity, x_acceleration, y_acceleration,
                                                                    motion_variance, measurement_variance)

        # Compute RMSE separately for each time unit
        rmse_values.append(np.sqrt((x_filtered - x_path) ** 2 + (y_filtered - y_path) ** 2))

    # Compute mean RMSE separately for each time unit
    mean_rmse_values = np.mean(np.array(rmse_values), axis=0)

    return mean_rmse_values


# Print mean RMSE over time
def print_kalman_accuracy(mean_rmse_values):
    plt.plot(mean_rmse_values)
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('RMSE over Time')
    plt.show()
