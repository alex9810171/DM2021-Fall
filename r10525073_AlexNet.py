from service.analyticService.core.analyticCore.classificationBase import classification
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from service.analyticService.core.analyticCore.utils import XYdataGenerator,XdataGenerator
from math import ceil
from keras.optimizers import Adam
class r10525073_AlexNet(classification):
    def trainAlgo(self):
        self.model=Sequential()
        # First layer
        self.model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(224, 224, 3), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # Second layer
        self.model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # Three layer
        self.model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # Fully connected layer
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation=self.param['activation']))
        self.model.add(Dropout(self.param['dropout']))
        self.model.add(Dense(4096, activation=self.param['activation']))
        self.model.add(Dropout(self.param['dropout']))
        # Classfication layer
        self.model.add(Dense(2, activation='sigmoid'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])
        self.model.fit_generator(
            XYdataGenerator(self.inputData['X'],self.outputData['Y'],224,224,self.param['batch_size']),
            steps_per_epoch=int(ceil((len(self.inputData['X'])/self.param['batch_size']))),
            epochs=self.param['epochs']
        )
    def predictAlgo(self):
        r=self.model.predict_generator(
            XdataGenerator(self.inputData['X'],224,224,self.param['batch_size']),
            steps=int(ceil((len(self.inputData['X'])/self.param['batch_size'])))
        )
        self.result['Y']=r