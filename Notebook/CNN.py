import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class CNN():

    X_train = None
    X_test = None
    y_train = None
    y_test = None
    y_pred = None
    model = None


    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.buildModel()
        self.compileModel()
        self.fitModel()
        self.predictModel()
        self.reportModel()
        self.visualization_metrics()
        self.saveModel()


        

    def buildModel(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(64, padding='same', strides=1, activation='relu',   
                    kernel_size=(3, 3), input_shape=(self.X_train[0].shape), ))
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
        self.model.fit(self.X_train, self.y_train, epochs=7, validation_data=(self.X_test, self.y_test))

    def saveModel(self):
        self.model.save('./models/cnn_model.h5')

    def predictModel(self):
        self.y_pred=self.model.predict(X_test) 
        self.y_pred=np.argmax(self.y_pred, axis=1)

    def reportModel(self):
        print(classification_report(self.y_test, self.y_pred))

    def visualization_metrics(self):
        metrics = confusion_matrix(self.y_test, self.y_pred)
        ConfusionMatrixDisplay(metrics, self.model.classes_).plot()