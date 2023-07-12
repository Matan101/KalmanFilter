import numpy as np


def add_noise(path, mean, standard_deviation):
    noise = np.random.normal(mean, standard_deviation, path.shape)
    return path + noise


# Generate noisy path
def generate_noisy_path(noise_mean, noise_standard_deviation):
    x = np.linspace(0, 10, 100)  # y=x

    # Add noise to the original path
    noisy_path = add_noise(x, noise_mean, noise_standard_deviation)

    return x, x, noisy_path
