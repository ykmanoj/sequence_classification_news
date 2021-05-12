from datetime import time, datetime

import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense, Dropout, GRU, SimpleRNN
from tensorflow.python.keras.models import Sequential
import tensorflow as tf

from ML_Pipeline.Constants import epochs, batch_size, lstm_size, gru_size


#Building Sequential network with
#   Embeding Layer
#   LSTM
#   Dense
#   Output Layer
def build_network_lstm(embedding_layer):

    print(" Building Sequential network ")
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(lstm_size))#, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def build_network_GRU(embedding_layer):

    print(" Building Sequential network ")
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(gru_size))#, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def build_network_RNN(embedding_layer):

    print(" Building Sequential network ")
    model = Sequential()
    model.add(embedding_layer)
    model.add(SimpleRNN(100))#, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, X_train, y_train, X_test, y_test,epochs=epochs, batch_size=batch_size):
    # Compile Model with loss function,
    # optimizer and metricecs as minimum parameter
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Train model with Train and test set data
    # Number of epochs, batch size as minimum parameter
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        #validation_split=0.2)
                        validation_data=(X_test, y_test))
    return model, history

