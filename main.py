from kalman_accuracy import compute_kalman_accuracy, print_kalman_accuracy
from kalman_filter import print_kalman_graph, kalman_filter
from noisy_path import generate_noisy_path, generate_noisy_acceleration_path

# Define path parameters
x_initial = 0
y_initial = 0
x_velocity = 1
y_velocity = 1
acceleration = 2
time = 100
noise_mean = 0
noise_standard_deviation = 1

# Define Kalman Filter parameters
motion_variance = 0.5
measurement_variance = 5

# ------------------- Uniform Velocity -------------------
# Generate original path and noisy path
x_path, y_path, x_noisy, y_noisy = generate_noisy_path(x_initial, y_initial, x_velocity, y_velocity, time, noise_mean,
                                                       noise_standard_deviation)

# Apply Kalman Filter
x_filtered, y_filtered = kalman_filter(x_noisy, y_noisy, x_initial, y_initial, x_velocity, y_velocity, motion_variance,
                                       measurement_variance)
print_kalman_graph(x_path, y_path, x_noisy, y_noisy, x_filtered, y_filtered)

# Compute Kalman Filter accuracy (RMSE)
simulations_num = 100
mean_rmse_values = compute_kalman_accuracy(simulations_num, x_initial, y_initial, x_velocity, y_velocity, time,
                                           noise_mean, noise_standard_deviation, motion_variance,
                                           measurement_variance)
print_kalman_accuracy(mean_rmse_values)

# ------------------- Uniform Acceleration -------------------
# Generate original path and noisy path
x_path, y_path, x_noisy, y_noisy = generate_noisy_acceleration_path(x_initial, y_initial, x_velocity, y_velocity,
                                                                    acceleration, time, noise_mean,
                                                                    noise_standard_deviation)
# print_kalman_graph(x_path, y_path, x_noisy, y_noisy)
