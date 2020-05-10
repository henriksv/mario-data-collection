
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten,  Conv2D, MaxPooling2D
import cv2
import os
import csv
from synchronize_bvp import synch_bvp
import numpy as np
import sys
import glob
import random

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import numpy as np
from tensorflow.keras.utils import Sequence

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization

class DataLoader(Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(84,84), n_channels=1, 
                    n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))#, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load_(ID)
            img = load_img(ID, color_mode='grayscale')
            array = img_to_array(img)
            # Normalization
            array /= 255
            X[i,] = array

            # Store class
            y[i] = self.labels[ID]

        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


def load_bvp(session):
    bvp = synch_bvp(session, normalize=True, halve=True)
    return bvp


def make_model(input_shape):
    # for input shape 1, 84, 84

    #input_shape = (84, 84, 1)
    #input_shape = (1, 84, 84)
    model = Sequential()

    model.add(Conv2D(32, (36, 36), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (36, 36), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())

    model.add(Conv2D(64, (18, 18), padding='same', activation='relu'))
    model.add(Conv2D(64, (18, 18), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())


    model.add(Conv2D(64, (9, 9), padding='same', activation='relu'))
    model.add(Conv2D(64, (9, 9), padding='same', activation='relu'))
    model.add(Conv2D(64, (9, 9), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    #model.add(BatchNormalization())

    model.add(Flatten()) #Assumed
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(1, activation="sigmoid"))
    model.add(Dense(1, activation="linear"))

    print(model.summary)

    model.compile(
        #loss='binary_crossentropy',
        loss='mean_squared_error',
        optimizer="adam",
        metrics=['accuracy']
    )

    return model

def make_test_model():
    input_shape = (84, 84, 1)

    model = Sequential()

    model.add(Conv2D(32, (36, 36), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten()) #Assumed
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
    )

    return model

def train_on_bvp(bvp, input_shape, model_id):

    params = {
        #'dim': (84,84,1),
        'dim': input_shape,
        'batch_size': 32,
        'n_classes': 1,
        'n_channels': 1,
        'shuffle': True
    }

    file_count = len(bvp)
    x_path = './DATA/game_frames/6/'

    #frame_IDs = list(glob.glob((x_path + "*")))
    frame_IDs = []

     # BVP values
    labels = {}
    for i in range(file_count):
        frame_path = x_path + 'state' + str(i+1) + '.png'
        frame_IDs.append(frame_path)
        labels[frame_path] = bvp[i]

    random.shuffle(frame_IDs)

    partition = {
        'train' : [],
        'validation' : []
    }

    length = len(frame_IDs)
    split = int((length / 4) * 3)

    partition['train'] = frame_IDs[0:split]
    partition['validation'] = frame_IDs[split:length]



    print(len(partition['train']))
    print(len(partition['validation']))


    # Generators
    print('Making generators')
    training_generator = DataLoader(partition['train'], labels, **params)
    validation_generator = DataLoader(partition['validation'], labels, **params)

    print('Making model')
    model = make_model(input_shape)

    print('Fitting model')
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=1,
                        )

    model_string = 'models/'+model_id+'.h5'    
    model.save(model_string)



if __name__ == "__main__":

    input_shape = (84,84,1)
    model_identifier = 'first_test'
    participant = 6

    bvp = load_bvp(participant)
    train_on_bvp(bvp, input_shape, model_identifier)

"""
    # BVP values
    labels = {}


    # Datasets
    partition = # IDs
    labels = # Labels

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    # Design model
    model = Sequential()
    [...] # Architecture
    model.compile()

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6)


    #def train():

    #load_frames()
    #bvp = load_bvp(6)
    #frames, n_frames = load_frames(len(bvp))

    #frames, n_frames = load_frames(62597)
"""