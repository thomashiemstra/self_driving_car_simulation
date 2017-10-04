import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from utils import INPUT_SHAPE, batch_generator

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.models import load_model

def load_data():
    data_df = pd.read_csv('driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size = 0.2)

    return X_train, X_valid, y_train, y_valid

def build_model():

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, X_train, X_valid, y_train, y_valid, 
                data_dir = "IMG", batch_size = 1000, nb_epoch = 20, samples_per_epoch = 100000, learning_rate = 1.0e-4):
    #Saves the best model so far.
    checkpoint = ModelCheckpoint('model2-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    
    # keras 2 has other ideas about certain things
    steps_per_epoch = samples_per_epoch/batch_size
    v_steps = int(np.floor(len(X_valid)/batch_size))
    
    model.fit_generator(batch_generator(data_dir, X_train, y_train, batch_size, True), 
                        steps_per_epoch, nb_epoch, max_queue_size=1,
                        validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
                        validation_steps=v_steps, callbacks=[checkpoint])


data  = load_data()

model = build_model()

#continue from previously trained model
#model = load_model("model-010.h5")

train_model(model, *data)

