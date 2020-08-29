from tensorflow.keras.models import Sequential
from tensorflow.keras.kayers import BatchNormalization
from tensorflow.keras.kayers import SeperableConv2D
from tensorflow.keras.kayers import MaxPooling2D
from tensorflow.keras.kayers import Activation
from tensorflow.keras.kayers import Flatten
from tensorflow.keras.kayers import Dropout
from tensorflow.keras.kayers import Dense

class FireDetectionNet:
    @staticmethod
    def build(width, height, depth, classes):
        """
        Initialize the model along with the input shape to be 'channels last'and the channel dimension itself
        """
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # CONV -> RELU -> POOL
        model.add(SeperableConv2D(16, (7, 7), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # CONV -> RELU -> POOL
        model.add(SeperableConv2D(32, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #CONV -> RELU -> POOL
        model.add(SeperableConv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeperableConv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # First set of FC -> RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Second set of FC => RELU layers
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax for last activation layer
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # return the model
        return model
