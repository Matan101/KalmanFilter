from kalman_accuracy import print_kalman_accuracy, \
    compute_kalman_accuracy_uniform_velocity, compute_kalman_accuracy_uniform_acceleration
from kalman_filter import print_kalman_graph, kalman_filter_uniform_acceleration, kalman_filter_uniform_velocity
from noisy_path import generate_noisy_uniform_acceleration_path, generate_noisy_uniform_velocity_path

# Define path parameters
x_initial = 0
y_initial = 0
x_velocity = 2
y_velocity = 2
x_acceleration = 5
y_acceleration = 5
time = 100
noise_mean = 0
noise_standard_deviation = 1

# Define Kalman Filter parameters
motion_variance = 0.5
measurement_variance = 5

# -------------------------------------------------- Uniform Velocity --------------------------------------------------
# Generate original path and noisy path
x_path, y_path, x_noisy, y_noisy = generate_noisy_uniform_velocity_path(x_initial, y_initial, x_velocity, y_velocity,
                                                                        time, noise_mean,
                                                                        noise_standard_deviation)

# Apply Kalman Filter
x_filtered, y_filtered = kalman_filter_uniform_velocity(x_noisy, y_noisy, x_initial, y_initial, x_velocity, y_velocity,
                                                        motion_variance,
                                                        measurement_variance)
print_kalman_graph(x_path, y_path, x_noisy, y_noisy, x_filtered, y_filtered)

# Compute Kalman Filter accuracy (RMSE)
simulations_num = 100
mean_rmse_values = compute_kalman_accuracy_uniform_velocity(simulations_num, x_initial, y_initial, x_velocity,
                                                            y_velocity, time, noise_mean, noise_standard_deviation,
                                                            motion_variance, measurement_variance)
print_kalman_accuracy(mean_rmse_values)

# ------------------------------------------------ Uniform Acceleration ------------------------------------------------
# Generate original path and noisy path
x_path, y_path, x_noisy, y_noisy = generate_noisy_uniform_acceleration_path(x_initial, y_initial, x_velocity,
                                                                            y_velocity, x_acceleration, y_acceleration,
                                                                            time, noise_mean, noise_standard_deviation)
# Apply Kalman Filter
x_filtered, y_filtered = kalman_filter_uniform_acceleration(x_noisy, y_noisy, x_initial, y_initial, x_velocity,
                                                            y_velocity, x_acceleration, y_acceleration, motion_variance,
                                                            measurement_variance)
print_kalman_graph(x_path, y_path, x_noisy, y_noisy, x_filtered, y_filtered)

# Compute Kalman Filter accuracy (RMSE)
simulations_num = 100
mean_rmse_values = compute_kalman_accuracy_uniform_acceleration(simulations_num, x_initial, y_initial, x_velocity,
                                                                y_velocity, x_acceleration, y_acceleration, time,
                                                                noise_mean, noise_standard_deviation, motion_variance,
                                                                measurement_variance)
print_kalman_accuracy(mean_rmse_values)
