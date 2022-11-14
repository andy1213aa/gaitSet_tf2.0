
from typing import Set
import tensorflow as tf
from .basic_block import BasicConv2D, SetBlock, HPM
from tensorflow.keras import layers
# from tensorflow.keras.utils import plot_model


class GaitSet(tf.keras.Model):
    def __init__(self, hidden_dim=256):
        super(GaitSet, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_frame = None

        # Set Network
        _set_channels = [64, 128, 256]

        self.set_layer1 = SetBlock(BasicConv2D(
            _set_channels[0], 5, (1, 1), padding='same'))

        self.set_layer2 = SetBlock(BasicConv2D(
            _set_channels[0], 3, (1, 1), padding='same'), pooling=True)

        self.set_layer3 = SetBlock(BasicConv2D(
            _set_channels[1], 3, (1, 1), padding='same'))

        self.set_layer4 = SetBlock(BasicConv2D(
            _set_channels[1], 3, (1, 1), padding='same'), pooling=True)

        self.set_layer5 = SetBlock(BasicConv2D(
            _set_channels[2], 3, (1, 1), padding='same'))

        self.set_layer6 = SetBlock(BasicConv2D(
            _set_channels[2], 3, (1, 1), padding='same'))

        # MGP
        _gl_channels = [128, 256]

        self.gl_layer1 = BasicConv2D(
            _gl_channels[0], 3, (1, 1), padding='same')

        self.gl_layer2 = BasicConv2D(
            _gl_channels[0], 3, (1, 1), padding='same')

        self.gl_layer3 = BasicConv2D(
            _gl_channels[1], 3, (1, 1), padding='same')

        self.gl_layer4 = BasicConv2D(
            _gl_channels[1], 3, (1, 1), padding='same')

        self.gl_pooling = layers.MaxPool2D()

        # HPM
        self.gl_hpm = HPM(_set_channels[-1], hidden_dim)
        self.x_hpm = HPM(_set_channels[-1], hidden_dim)

        self.bin_num = [1, 2, 4, 8, 16]

    def frame_max(self, x):
        return tf.reduce_max(x, axis=1)
    #     n, h, w, c = x.shape
    #     k = 8 # number of ppl
    #     x = tf.reshape(x, (k, -1, h, w, c))
    #     x =
        # return tf.reduce_max(x, axis=)

    def call(self, input):

        x = self.set_layer1(input)
        x = self.set_layer2(x)
        gl = self.gl_layer1(self.frame_max(x))
        # gl = self.gl_layer1(x)
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)

        x = self.set_layer3(x)
        x = self.set_layer4(x)
        gl = self.gl_layer3(gl + self.frame_max(x))
        # gl = self.gl_layer3(gl)
        gl = self.gl_layer4(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)

        gl = gl + x

        gl_f = self.gl_hpm(gl)
        x_f = self.x_hpm(x)

        return tf.concat([gl_f, x_f], axis=1)

    def model(self, inputsize: tuple):
        input = tf.keras.Input(
            shape=inputsize, name='input_layer'
        )
        model = tf.keras.models.Model(
            inputs=input, outputs=self.call(input)
        )
        # plot_model(model, to_file='model.png', show_shapes=True)
        return model
