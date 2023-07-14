from kalman_accuracy import calculate_kalman_accuracy, print_kalman_accuracy
from kalman_filter import kalman_filter, print_kalman_graph
from noisy_path import generate_noisy_path

# Define path parameters
noise_mean = 0
noise_standard_deviation = 0.2

# Generate original constant speed path with noise
x_true, y_true, noisy_path = generate_noisy_path(noise_mean, noise_standard_deviation)

# Define Kalman Filter parameters
initial_state = [0, 1]
motion_variance = 0.1
measurement_variance = 0.2

# Apply Kalman Filter
filtered_path = kalman_filter(noisy_path, initial_state, motion_variance, measurement_variance)

# Print the original path next to the noisy measurements and the filtered path
print_kalman_graph(x_true, y_true, noisy_path, filtered_path)

# Calculate accuracy (RMSE)
simulations_num = 100
mean_rmse_values = calculate_kalman_accuracy(simulations_num, noise_mean, noise_standard_deviation, initial_state,
                                             motion_variance, measurement_variance)
print_kalman_accuracy(mean_rmse_values)
