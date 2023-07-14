import numpy as np


# Generate original path and noisy path
def generate_noisy_path(x_initial, y_initial, x_velocity, y_velocity, time, noise_mean, noise_standard_deviation):
    x_path = np.linspace(x_initial, x_initial + x_velocity * time, time)
    y_path = np.linspace(y_initial, y_initial + y_velocity * time, time)

    # Add noise to the original path
    x_noisy = x_path + np.random.normal(noise_mean, noise_standard_deviation, len(x_path))
    y_noisy = y_path + np.random.normal(noise_mean, noise_standard_deviation, len(y_path))

    return x_path, y_path, x_noisy, y_noisy
