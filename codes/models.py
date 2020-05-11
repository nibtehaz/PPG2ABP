"""
    Models used in experiments
"""

from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def UNet(length, n_channel=1):
    """
        Standard U-Net
    
    Arguments:
        length {int} -- length of the input signal
    
    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})
    
    Returns:
        keras.model -- created model
    """
    
    x = 32

    inputs = Input((length, n_channel))
    conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv1D(1, 1)(conv9)

    model = Model(inputs=[inputs], outputs=[conv10]) 

    return model


def UNetWide64(length, n_channel=1):
    """
       Wider U-Net with kernels multiples of 64
    
    Arguments:
        length {int} -- length of the input signal
    
    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})
    
    Returns:
        keras.model -- created model
    """
    
    x = 64

    inputs = Input((length, n_channel))
    conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv1D(1, 1)(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    

    return model


def UNetDS64(length, n_channel=1):
    """
        Deeply supervised U-Net with kernels multiples of 64
    
    Arguments:
        length {int} -- length of the input signal
    
    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})
    
    Returns:
        keras.model -- created model
    """
    
    x = 64

    inputs = Input((length, n_channel))
    conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    level4 = Conv1D(1, 1, name="level4")(conv5)

    up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    level3 = Conv1D(1, 1, name="level3")(conv6)

    up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    level2 = Conv1D(1, 1, name="level2")(conv7)

    up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    level1 = Conv1D(1, 1, name="level1")(conv8)

    up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    out = Conv1D(1, 1, name="out")(conv9)

    model = Model(inputs=[inputs], outputs=[out, level1, level2, level3, level4])
    
    

    return model


def UNetWide40(length, n_channel=1):
    """
       Wider U-Net with kernels multiples of 40
    
    Arguments:
        length {int} -- length of the input signal
    
    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})
    
    Returns:
        keras.model -- created model
    """
    
    x = 40

    inputs = Input((length, n_channel))
    conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv1D(1, 1)(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    

    return model


def UNetWide48(length, n_channel=1):
    """
       Wider U-Net with kernels multiples of 48
    
    Arguments:
        length {int} -- length of the input signal
    
    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})
    
    Returns:
        keras.model -- created model
    """
    
    x = 48

    inputs = Input((length, n_channel))
    conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv1D(1, 1)(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    

    return model


def UNetLite(length, n_channel=1):
    """
       Shallower U-Net
    
    Arguments:
        length {int} -- length of the input signal
    
    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})
    
    Returns:
        keras.model -- created model
    """

    inputs = Input((length, n_channel))
    conv1 = Conv1D(32,3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(32,3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(64,3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(64,3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(128,3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(128,3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)    

    conv5 = Conv1D(256, 3, activation='relu', padding='same')(pool3)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(256, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    
    up7 = concatenate([UpSampling1D(size=2)(conv5), conv3], axis=2)
    conv7 = Conv1D(128, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(128, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
    conv8 = Conv1D(64, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(64, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
    conv9 = Conv1D(32, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(32, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv1D(1, 1)(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    

    return model


def MultiResUNet1D(length, n_channel=1):
    """
       1D MultiResUNet
    
    Arguments:
        length {int} -- length of the input signal
    
    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})
    
    Returns:
        keras.model -- created model
    """

    def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
        
        kernel = 3

        x = Conv1D(filters, kernel,  padding=padding)(x)
        x = BatchNormalization()(x)

        if(activation == None):
            return x

        x = Activation(activation, name=name)(x)
        return x


    def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
 
        x = UpSampling1D(size=2)(x)        
        x = BatchNormalization()(x)
        
        return x


    def MultiResBlock(U, inp, alpha = 2.5):
        '''
        MultiRes Block
        
        Arguments:
            U {int} -- Number of filters in a corrsponding UNet stage
            inp {keras layer} -- input layer 
        
        Returns:
            [keras layer] -- [output layer]
        '''

        W = alpha * U

        shortcut = inp

        shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                            int(W*0.5), 1, 1, activation=None, padding='same')

        conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                            activation='relu', padding='same')

        conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                            activation='relu', padding='same')

        conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                            activation='relu', padding='same')

        out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
        out = BatchNormalization()(out)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

        return out


    def ResPath(filters, length, inp):
        '''
        ResPath
        
        Arguments:
            filters {int} -- [description]
            length {int} -- length of ResPath
            inp {keras layer} -- input layer 
        
        Returns:
            [keras layer] -- [output layer]
        '''


        shortcut = inp
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                            activation=None, padding='same')

        out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

        for i in range(length-1):

            shortcut = out
            shortcut = conv2d_bn(shortcut, filters, 1, 1,
                                activation=None, padding='same')

            out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization()(out)

        return out





    inputs = Input((length, n_channel))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling1D(pool_size=2)(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling1D(pool_size=2)(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling1D(pool_size=2)(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling1D(pool_size=2)(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([UpSampling1D(size=2)(mresblock5), mresblock4], axis=-1)
    mresblock6 = MultiResBlock(32*8, up6)

    up7 = concatenate([UpSampling1D(size=2)(mresblock6), mresblock3], axis=-1)
    mresblock7 = MultiResBlock(32*4, up7)

    up8 = concatenate([UpSampling1D(size=2)(mresblock7), mresblock2], axis=-1)
    mresblock8 = MultiResBlock(32*2, up8)

    up9 = concatenate([UpSampling1D(size=2)(mresblock8), mresblock1], axis=-1)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = Conv1D(1, 1)(mresblock9)
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def MultiResUNetDS(length, n_channel=1):
    """
       1D Deeply Supervised MultiResUNet
    
    Arguments:
        length {int} -- length of the input signal
    
    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})
    
    Returns:
        keras.model -- created model
    """

    def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
        
        kernel = 3

        x = Conv1D(filters, kernel,  padding=padding)(x)
        x = BatchNormalization()(x)

        if(activation == None):
            return x

        x = Activation(activation, name=name)(x)
        return x


    def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
 
        x = UpSampling1D(size=2)(x)        
        x = BatchNormalization()(x)
        
        return x


    def MultiResBlock(U, inp, alpha = 2.5):
        '''
        MultiRes Block
        
        Arguments:
            U {int} -- Number of filters in a corrsponding UNet stage
            inp {keras layer} -- input layer 
        
        Returns:
            [keras layer] -- [output layer]
        '''

        W = alpha * U

        shortcut = inp

        shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                            int(W*0.5), 1, 1, activation=None, padding='same')

        conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                            activation='relu', padding='same')

        conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                            activation='relu', padding='same')

        conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                            activation='relu', padding='same')

        out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
        out = BatchNormalization()(out)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

        return out


    def ResPath(filters, length, inp):
        '''
        ResPath
        
        Arguments:
            filters {int} -- [description]
            length {int} -- length of ResPath
            inp {keras layer} -- input layer 
        
        Returns:
            [keras layer] -- [output layer]
        '''


        shortcut = inp
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                            activation=None, padding='same')

        out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

        for i in range(length-1):

            shortcut = out
            shortcut = conv2d_bn(shortcut, filters, 1, 1,
                                activation=None, padding='same')

            out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization()(out)

        return out


    inputs = Input((length, n_channel))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling1D(pool_size=2)(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling1D(pool_size=2)(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling1D(pool_size=2)(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling1D(pool_size=2)(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    level4 = Conv1D(1, 1, name="level4")(mresblock5)

    up6 = concatenate([UpSampling1D(size=2)(mresblock5), mresblock4], axis=-1)
    mresblock6 = MultiResBlock(32*8, up6)

    level3 = Conv1D(1, 1, name="level3")(mresblock6)

    up7 = concatenate([UpSampling1D(size=2)(mresblock6), mresblock3], axis=-1)
    mresblock7 = MultiResBlock(32*4, up7)

    level2 = Conv1D(1, 1, name="level2")(mresblock7)

    up8 = concatenate([UpSampling1D(size=2)(mresblock7), mresblock2], axis=-1)
    mresblock8 = MultiResBlock(32*2, up8)
    
    level1 = Conv1D(1, 1, name="level1")(mresblock8)

    up9 = concatenate([UpSampling1D(size=2)(mresblock8), mresblock1], axis=-1)
    mresblock9 = MultiResBlock(32, up9)

    out = Conv1D(1, 1, name="out")(mresblock9)
    
    model = Model(inputs=[inputs], outputs=[out,level1,level2,level3,level4])

    return model
    