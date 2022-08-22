from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
 
 
def load_train(path):
    train_datagen = ImageDataGenerator(
    rescale=1./255,horizontal_flip=True, vertical_flip=True,
                                        width_shift_range=0.2, height_shift_range=0.2) # ,rotation_range=90
 
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345)
 
    return train_datagen_flow
 
 
def create_model(input_shape):
    model = Sequential()
    optimizer = Adam(lr=0.001)
 
 
    model.add(Conv2D(6, (5, 5), padding='same', activation='relu',
                 input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(26, (5, 5), padding='same', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(36, (5, 5), padding='same', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(46, (5, 5), padding='same', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=46, activation='relu'))
    model.add(Dense(units=24, activation='relu'))
    model.add(Dense(units=12, activation='softmax'))
 
 
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model
 
def train_model(model, train_data, test_data, batch_size=16, epochs=10,
               steps_per_epoch=None, validation_steps=None):
 
    model.fit(train_data,
          validation_data=test_data,
             epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
 
    return model