from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from keras.utils.vis_utils import plot_model as plot

def LaneNNModel(inputShape = (160, 320, 3), batchSize = 64, epochs = 10, poolSize = (2,2)):
    
    model = Sequential()

    #Encoder Network

    #Conv Section 1
    model.add(BatchNormalization(input_shape=inputShape))
    model.add(Conv2D(8, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv1'))
    model.add(Conv2D(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv2'))
    model.add(MaxPooling2D(pool_size=poolSize))

    #Conv Section 2
    model.add(Conv2D(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv3'))
    model.add(Dropout(0.2))

    #Conv Section 3
    model.add(Conv2D(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv4'))
    model.add(Dropout(0.2))

    #Conv Section 4
    model.add(Conv2D(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv5'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=poolSize))

    #Conv Section 5
    model.add(Conv2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv6'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=poolSize))


    #Decoder Network

    #Deconv Section 1
    model.add(UpSampling2D(size=poolSize))
    model.add(Conv2DTranspose(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv1'))
    model.add(Dropout(0.2))

    #Deconv Section 2
    model.add(Conv2DTranspose(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv2'))
    model.add(Dropout(0.2))
    model.add(UpSampling2D(size=poolSize))

    #Deconv Section 3
    model.add(Conv2DTranspose(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv3'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv4'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv5'))
    model.add(Dropout(0.2))
    model.add(UpSampling2D(size=poolSize))

    #Deconv Section 4
    model.add(Conv2DTranspose(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv6'))
    model.add(Conv2DTranspose(1, (3,3), padding='valid', strides=(1,1), activation='sigmoid', name='Final'))

    return model

if __name__ == "__main__":
    model = LaneNNModel()
    model.summary()
    plot(model, to_file='LaneNN_OLD.png', show_shapes=True)


    
    
