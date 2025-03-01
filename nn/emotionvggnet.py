from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # Block #1: first CONV => ELU => CONV => ELU => POOL
        # layer set
        model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", input_shape=inputShape))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), kernel_initializer="he_normal", padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Block #2: second CONV => ELU => CONV => ELU => POOL
        # layer set
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Block #3: third CONV => ELU => CONV => ELU => POOL
        # layer set
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Block  4: set Full-Connected => Relu layers
        # layer set
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Block  5: second set Full-Connected => Relu layers
        # layer set
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Block 7: Softmax classifier
        model.add(Dense(classes, kernel_initializer="he_normal"))
        model.add(Activation("softmax"))

        return model
