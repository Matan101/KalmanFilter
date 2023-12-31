import numpy as np
from matplotlib import pyplot as plt


# Kalman Filter Loop
def kalmanFilterLoop(state, P, Q, A, H, R, x_noisy, y_noisy):
    x_filtered = []
    y_filtered = []

    # Kalman filter loop
    for i in range(len(x_noisy)):
        # Predict
        state = A @ state
        P = A @ P @ A.T + Q

        # Save next filtered to filtered path
        x_filtered.append(state[0])
        y_filtered.append(state[1])

        # Update
        z = np.array([x_noisy[i], y_noisy[i]])
        residual_mean = z - H @ state
        residual_cov = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(residual_cov)
        state = state + K @ residual_mean
        P = P - K @ H @ P

    return np.array(x_filtered), np.array(y_filtered)


# Kalman Filter Algorithm for uniform velocity
def kalman_filter_uniform_velocity(x_noisy, y_noisy, x_initial, y_initial, x_velocity, y_velocity, motion_variance,
                                   measurement_variance):
    # Covariance matrix
    P = [[measurement_variance, 0, 0, 0],
         [0, measurement_variance, 0, 0],
         [0, 0, motion_variance, 0],
         [0, 0, 0, motion_variance]]

    # Transition matrix - based on velocity formula
    A = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Motion noise matrix
    Q = np.eye(4) * motion_variance

    # Measurement matrix
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    # Measurement noise matrix
    R = np.eye(2) * measurement_variance

    # Initialize state
    state = np.array([x_initial, y_initial, x_velocity, y_velocity])

    # Kalman filter loop
    return kalmanFilterLoop(state, P, Q, A, H, R, x_noisy, y_noisy)


# Kalman Filter Algorithm for uniform acceleration
def kalman_filter_uniform_acceleration(x_noisy, y_noisy, x_initial, y_initial, x_velocity, y_velocity, x_acceleration,
                                       y_acceleration, motion_variance, measurement_variance):
    # Covariance matrix
    P = [[measurement_variance, 0, 0, 0, 0, 0],
         [0, measurement_variance, 0, 0, 0, 0],
         [0, 0, measurement_variance, 0, 0, 0],
         [0, 0, 0, measurement_variance, 0, 0],
         [0, 0, 0, 0, motion_variance, 0],
         [0, 0, 0, 0, 0, motion_variance]]

    # Transition matrix - based on acceleration formula
    A = np.array([[1, 0, 1, 0, 0.5, 0],
                  [0, 1, 0, 1, 0, 0.5],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    # Motion noise matrix
    Q = np.eye(6) * motion_variance

    # Measurement matrix
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])

    # Measurement noise matrix
    R = np.eye(2) * measurement_variance

    # Initialize state
    state = np.array([x_initial, y_initial, x_velocity, y_velocity, x_acceleration, y_acceleration])

    # Kalman filter loop
    return kalmanFilterLoop(state, P, Q, A, H, R, x_noisy, y_noisy)


# Print the original path next to the noisy measurements and the filtered path
def print_kalman_graph(x_path, y_path, x_noisy, y_noisy, x_filtered, y_filtered):
    plt.plot(x_path, y_path, c='green', label='Original Path')
    plt.scatter(x_noisy, y_noisy, c='hotpink', label='Noisy Measurements')
    plt.plot(x_filtered, y_filtered, c='blue', label='Filtered Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
