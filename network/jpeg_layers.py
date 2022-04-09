import tensorflow as tf
import tensorflow.keras.layers as tfklayers
from .jpeg_utils import *

class RGBToYCbCr(tfklayers.Layer):
    def __init__(self, diff_round, name='rgb_to_ycbcr'):
        super(RGBToYCbCr, self).__init__(name=name)
        self.diff_round = diff_round

    def call(self, inputs):
        return rgb_to_ycbcr(inputs, self.diff_round)


class YCbCrToRGB(tfklayers.Layer):
    def __init__(self, diff_round, name='ycbcr_to_rgb'):
        super(YCbCrToRGB, self).__init__(name=name)
        self.diff_round = diff_round

    def call(self, inputs):
        return ycbcr_to_rgb(inputs, self.diff_round)


class Split(tfklayers.Layer):
    def __init__(self, name='split'):
        super(Split, self).__init__(name=name)
  
    def call(self, inputs):
        return batch_split_v1(inputs)


class Tile(tfklayers.Layer):
    def __init__(self, name='tile'):
        super(Tile, self).__init__(name=name)
  
    def call(self, inputs):
        return batch_tile_v1(inputs)


class DCT2D(tfklayers.Layer):
    def __init__(self, name='dct2d'):
        super(DCT2D, self).__init__(name=name)

    def call(self, inputs):
        return dct2d_v2(inputs)


class IDCT2D(tfklayers.Layer):
    def __init__(self, name='idct2d'):
        super(IDCT2D, self).__init__(name=name)

    def call(self, inputs):
        return idct2d_v2(inputs)


class Transform(tfklayers.Layer):
    def __init__(self, trainable=True, initialize='identity', name='transform'):
        assert initialize in ['identity', 'dct', 'he']
        super(Transform, self).__init__(name=name)
        self.trainable = trainable
        self.initialize = initialize

    def build(self, input_shape):
        if self.initialize == 'identity':
            self.transform_matrix = tf.Variable(tf.linalg.diag(tf.ones((8,))),
                                                trainable=self.trainable, dtype=tf.float32)
        elif self.initialize == 'dct':
            self.transform_matrix = tf.Variable(_DCT_MATRIX_8x8,
                                                trainable=self.trainable, dtype=tf.float32)
        elif self.initialize == 'he':
            he_initializer = tf.tf.keras.initializers.HeUniform()
            self.transform_matrix = tf.Variable(he_initializer(shape=(8, 8)))

    def call(self, inputs, inverse=False):
        matrix = self.transform_matrix
        inv_matrix = tf.linalg.inv(matrix)
        if not inverse:
            return tf.matmul(tf.matmul(matrix, inputs), inv_matrix)
        else:
            return tf.matmul(tf.matmul(inv_matrix, inputs), matrix)