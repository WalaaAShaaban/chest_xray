import tensorflow as tf
from tensorflow.keras import layers, utils, models


class CNN():

    X_train = None
    X_test = None
    y_train = None
    y_test = None
    model = None


    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.buildModel()
        self.compileModel()
        self.fitModel()


        

    def buildModel(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(64, padding='same', strides=1, activation='relu',   
                            kernel_size=(3, 3), input_size=(self.X_train[0].shape)))
        self.model.add(layers.MaxPool2D((2, 2)))
        self.model.add(layers.Conv2D(64, padding='same', strides=1, activation='relu',   
                            kernel_size=(3, 3)))
        self.model.add(layers.MaxPool2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(2, activation='softmax'))
        self.model.summary()

    def compileModel(self):
        self.model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy']
              )
        
    def fitModel(self):
        self.model.fitModel(self.X_train, self.y_train, epochs=10, validation_data=(self.X_test, self.y_test))
