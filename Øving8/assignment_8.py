import pickle
from typing import Dict, List, Any, Union
import numpy as np
# Keras
from tensorflow import keras
from keras.utils import pad_sequences

epochs = 2

def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)

    return data


def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """
    maxlen = data["max_length"]//16
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])

    return data


def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward") -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    vocab_size = data["vocab_size"]
    max_length = data["max_length"]

    model = keras.Sequential()
    # model.add(keras.layers.Embedding(input_dim=max_length, output_dim=48))
    model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=48))

    if model_type == "feedforward":
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation="relu"))
        model.add(keras.layers.Dense(units=10, activation="relu"))
        model.add(keras.layers.Dense(units=1, activation="sigmoid"))
    elif model_type == "recurrent":
        model.add(keras.layers.LSTM(units=48))
        model.add(keras.layers.Dense(units=1, activation="relu"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, verbose=1)
    _, accuracy = model.evaluate(x_test, y_test, verbose=1)
    return accuracy


def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    print('Model: Feedforward NN.\n'
          f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training recurrent neural network...")
    rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    print('Model: Recurrent NN.\n'
          f'Test accuracy: {rnn_test_accuracy:.3f}')



if __name__ == '__main__':
    main()

