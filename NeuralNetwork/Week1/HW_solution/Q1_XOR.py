

import numpy as np
from keras.layers import Input,Dense, Activation
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects


def XOR_train_fully_keras():
    #Since Q1 (XOR) only asks for drawing, in the code, I used Keras/TF codes.
    x = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
    y = np.array([0,1,1,0])

    input = Input(shape=(2,))
    hidden1 = Dense(10, activation='relu')(input)
    hidden2 = Dense(10, activation='relu')(hidden1)
    out = Dense(1, activation='linear')(hidden2)
    model = Model(inputs=[input], outputs=[out])
    model.compile(optimizer='adam', loss='mse', loss_weights=[1])

    model.fit(x=[x], y=[y], batch_size=4, epochs=800)

    y_p = model.predict(x)
    print(y_p)
    y_p = np.round(y_p, 2)
    print(y_p)

def XOR_check_drawing():
    # first columns are for the bias
    x = np.array([[1, 0,0],
                  [1, 0,1],
                  [1, 1,0],
                  [1, 1,1]])
    # label for xor
    y = np.array([0,1,1,0])

    # define weight vectors
    w1 = np.array([-1, 1, -1])
    w2 = np.array([-1, -1, 1])
    w3 = np.array([-1, 1, 1])

    # select an input
    input = x[0,:]

    # calculate layer1
    percept1_logit = np.dot(input, w1)
    percept1_out = 1 if percept1_logit >= 0 else 0

    percept2_logit = np.dot(input, w2)
    percept2_out = 1 if percept2_logit >= 0 else 0

    layer1_out = np.array([1,percept1_out, percept2_out])

    # calculate layer2
    percept3_logit = np.dot(layer1_out, w3)
    percept3_out = 1 if percept3_logit >= 0 else 0
    out = percept3_out

    # print output of the custom neural net
    print(out)

if __name__ =='__main__':
    XOR_check_drawing()
