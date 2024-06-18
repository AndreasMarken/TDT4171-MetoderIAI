import numpy as np

class NeuralNetwork():
    def __init__(self, alpha=0.1, epochs=1000) -> None:
        self.W1 = np.random.rand(2, 2)
        self.b1 = np.random.rand(2, 1)
        self.W2 = np.random.rand(1, 2)
        self.b2 = np.random.rand(1, 1)
        self.alpha = alpha
        self.epochs = epochs

    def __forward_prop(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Z1 = self.W1.dot(x.T) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.linear_activation(Z2)
        return Z1, A1, Z2, A2
    
    def __back_prop(self, Z1, A1, Z2, A2, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  
        m = y.size
        dZ2 = self.loss_derivative(y, A2)
        dW2 = dZ2.dot(A1.T) / m
        db2 = np.sum(dZ2) / m
        dZ1 = self.W2.T.dot(dZ2) * self.sigmoid_derivative(Z1)
        dW1 = dZ1.dot(x) / m
        db1 = np.sum(dZ1) / m 
        return dW1, db1, dW2, db2
    
    def __update_params(self, dW1, db1, dW2, db2) -> None:
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2

    def __gradient_descent(self, x: np.ndarray, y: np.ndarray) -> None:
        for i in range(self.epochs):
            Z1, A1, Z2, A2 = self.__forward_prop(x)
            dW1, db1, dW2, db2 = self.__back_prop(Z1, A1, Z2, A2, x, y)
            self.__update_params(dW1, db1, dW2, db2)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def linear_activation(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def loss(self, y, A2) -> np.ndarray:
        return (y - A2) ** 2 / 2
    
    def mse(self, y, A2) -> float:
        return np.sum(self.loss(y, A2)) / A2.size
    
    def loss_derivative(self, y, A2) -> np.ndarray:
        return A2 - y
    
    def predict(self, x: np.ndarray) -> float:
        return self.__forward_prop(x)
    
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.__gradient_descent(x, y)

def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    NN = NeuralNetwork()
    NN.train(X_train, y_train)
    print(f"MSE of training data: {NN.mse(y_train, NN.predict(X_train)[-1])}")
    print(f"MSE of test data: {NN.mse(y_test, NN.predict(X_test)[-1])}")