import numpy as np
from matplotlib import pyplot as plt


# Kalman Filter Algorithm
def kalman_filter(x_noisy, y_noisy, x_initial, y_initial, x_velocity, y_velocity, motion_variance,
                  measurement_variance):
    A = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # Transition matrix - according to velocity formula
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])  # Measurement matrix
    Q = motion_variance * np.eye(4)  # Motion noise matrix
    R = measurement_variance * np.eye(2)  # Measurement noise matrix

    x_filtered = []
    y_filtered = []

    # Initial state
    state = np.array([x_initial, y_initial, x_velocity, y_velocity])

    for i in range(len(x_noisy)):
        # Predict
        state = A.dot(state)

        # Update
        S = H.dot(A).dot(Q).dot(A.T).dot(H.T) + R
        K = A.dot(Q).dot(A.T).dot(H.T).dot(np.linalg.inv(S))
        z = np.array([x_noisy[i], y_noisy[i]])
        state = state + K.dot(z - H.dot(state))

        # Save next filtered to filtered path
        x_filtered.append(state[0])
        y_filtered.append(state[1])

    return x_filtered, y_filtered


# Print the original path next to the noisy measurements and the filtered path
def print_kalman_graph(x_path, y_path, x_noisy, y_noisy, x_filtered, y_filtered):
    plt.plot(x_path, y_path, c='green', label='Original Path')
    plt.scatter(x_noisy, y_noisy, c='hotpink', label='Noisy Measurements')
    plt.plot(x_filtered, y_filtered, c='blue', label='Filtered Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
