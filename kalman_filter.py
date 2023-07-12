from matplotlib import pyplot as plt


# Kalman Filter Algorithm
def kalman_filter(measurements, initial_state, motion_variance, measurement_variance):
    x = initial_state[0]
    P = initial_state[1]

    filtered_predictions = []

    for z in measurements:
        x_pred = x
        P_pred = P + motion_variance
        K = P_pred / (P_pred + measurement_variance)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred

        filtered_predictions.append(x)

    return filtered_predictions


# Print the original path next to the noisy measurements and the filtered path
def print_kalman_graph(x_true, y_true, noisy_path, filtered_path):
    plt.plot(x_true, y_true, c='green', label='Original Path')
    plt.scatter(x_true, noisy_path, c='hotpink', label='Noisy Measurements')
    plt.plot(x_true, filtered_path, c='blue', label='Filtered Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
