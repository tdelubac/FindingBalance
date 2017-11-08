from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop

def MLP(input_shape,label_size):
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(label_size))

    rmsprop = RMSprop(lr=0.00025)
    model.compile(loss='mse',optimizer=rmsprop)
    return model

def Mnih15(input_shape,label_size):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), input_shape=input_shape, strides=(4, 4), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(label_size))

    rmsprop = RMSprop(lr=0.00025)
    model.compile(loss='mse',optimizer=rmsprop)
    return model
