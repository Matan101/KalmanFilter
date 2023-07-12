import matplotlib.pyplot as plt
import numpy as np


def add_noise(path, mean, standard_deviation):
    noise = np.random.normal(mean, standard_deviation, path.shape)
    return path + noise


# Generate the original constant speed path
x = np.linspace(0, 10, 100)
y = x

# Add noise to the original path
mean = 0
standard_deviation = 0.2
noisy_path = add_noise(y, mean, standard_deviation)

# Plot the original path and the noisy measurements
plt.plot(x, y, label='Original Path')
plt.scatter(x, noisy_path, c='hotpink', label='Noisy Measurements')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
