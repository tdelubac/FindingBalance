from keras.models import Sequential
from keras.layers.core import Dense, Activation
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