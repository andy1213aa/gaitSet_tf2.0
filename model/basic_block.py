from multiprocessing import pool
import tensorflow as tf
from tensorflow.keras import layers


class BasicConv2D(layers.Layer):
    def __init__(self, filter_number, kernel_size, strides, **kwargs):
        super(BasicConv2D, self).__init__()

        self.conv2D = layers.Conv2D(filter_number, kernel_size, strides, bias= False, **kwargs)
        self.leaky_ReLU = layers.LeakyReLU()

    def call(self, input):
        x = self.conv2D(input)
        x = self.leaky_ReLU(x)
        return x


class SetBlock(layers.Layer):
    def __init__(self, foward_block, pooling=False):
        super(SetBlock, self).__init__()

        self.foward_block = foward_block
        self.pooling = pooling

        if pooling:
            self.pool2D = layers.MaxPooling2D()

    def call(self, input):
        '''
        n: number of ppl
        s: frame of each gait
        h: height
        w: width
        c: channel
        '''

        n, s, h, w, c = input.shape
        x = input.reshape((-1, h, w, c))
        x = self.foward_block(x)
        if self.pooling:
            x = self.pool2D(x)
        _, h, w, c = x.shape()
        return x.reshape(n, s, h, w, c)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


