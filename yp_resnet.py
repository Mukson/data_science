import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50
import pandas as pd

def load_train(path):
      df = pd.read_csv(path + 'labels.csv')
      train_datagen = ImageDataGenerator(rescale=1./255,
                                         horizontal_flip=True,
                                         width_shift_range=0.2, height_shift_range=0.2,
                                         validation_split=0.25
                                         )
      train_datagen_flow = train_datagen.flow_from_dataframe(
              dataframe = df,
              directory = path + 'final_files',
              x_col = 'file_name',
              y_col = 'real_age',
              class_mode = 'raw',
              target_size = (150,150),
              batch_size = 32,
              subset='training',
              seed=12345
              )
      return train_datagen_flow
 
def load_test(path):
      df = pd.read_csv(path + 'labels.csv')
      test_datagen = ImageDataGenerator(rescale=1./255,
                                        validation_split=0.25
                                        )
      test_datagen_flow = test_datagen.flow_from_dataframe(
              dataframe = df,
              directory = path + 'final_files',
              x_col = 'file_name',
              y_col = 'real_age',
              class_mode = 'raw',
              target_size = (150,150),
              batch_size = 32,
              subset='validation',
              seed=12345
              )
      return test_datagen_flow
 

def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape,
                        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False)
   
    model = Sequential()
    optimizer = Adam(lr=0.001)
    
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32, activation='relu'))
    #model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['mae']) #'sparse_categorical_crossentropy'
    return model


def train_model(model, train_data, test_data, batch_size=32, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    model.fit(train_data,
              validation_data=test_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model