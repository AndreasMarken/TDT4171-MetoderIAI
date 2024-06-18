import numpy as np

def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / vector.sum()

def print_probability(message: np.ndarray) -> None:
    for i in range(1, len(message)):
        print(f"Day {i}: True={message[i][0]}. False={message[i][1]}", end="\n")

if __name__ == "__main__":
    observation_model = np.array([
        [[0.9, 0.0], [0.0, 0.2]], 
        [[0.1, 0.0], [0.0, 0.8]]
    ])
    transition_model = np.array([[.7, .3], [.3, .7]])
    # evidence = [True, True] # First bullet point
    evidence = [True, True, False, True, True] # Second bullet point
    forward_message = [np.array([[.0], [.0]]) for _ in range(len(evidence) + 1)]
    forward_message[0] = np.array([[.5], [.5]])
    for i in range(1, len(evidence) + 1):
        sensor_model = observation_model[0] if evidence[i-1] else observation_model[1]
        forward_message[i] = normalize(
            sensor_model @ transition_model.T @ forward_message[i-1]
        ) # The equation from 14.12
    print_probability(forward_message)