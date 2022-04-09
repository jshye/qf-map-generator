import tensorflow as tf
import tensorflow.keras.layers as tfklayers
from .layers import *


class QFMapPredictor(tf.keras.Model):
    """
    Input:
        8*8 Quantized DCT blocks of 224*224*3 images [28,28,8,8,3]
    Output:
        QF maps for luma and chroma channels
        upsampled from [14, 14, 2] to [224, 224, 2] by block size = 16
    """
    def __init__(self, name='qf_map_predictor'):
        super(QFMapPredictor, self).__init__(name=name)

        self.reshape = tfklayers.Reshape((28,28,192))

        self.conv2d_1 = tfklayers.Conv2D(filters=48,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2d_2 = tfklayers.Conv2D(filters=96,
                                         kernel_size=3,
                                         strides=2,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2d_3 = tfklayers.Conv2D(filters=192,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2d_4 = tfklayers.Conv2D(filters=384,    
                                         kernel_size=3,
                                         strides=2,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2d_t1 = tfklayers.Conv2DTranspose(filters=48,
                                                   kernel_size=3,
                                                   strides=2,
                                                   padding='same',
                                                   activation='relu',
                                                   kernel_initializer='he_uniform')

        self.conv2d_5 = tfklayers.Conv2D(filters=96,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2d_6 = tfklayers.Conv2D(filters=192,
                                         kernel_size=3,
                                         strides=2,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2d_7 = tfklayers.Conv2D(filters=384,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2d_t2 = tfklayers.Conv2DTranspose(filters=192,
                                                   kernel_size=3,
                                                   strides=2,
                                                   padding='same',
                                                   activation='relu',
                                                   kernel_initializer='he_uniform')

        self.conv2d_8 = tfklayers.Conv2D(filters=96,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2d_9 = tfklayers.Conv2D(filters=192,
                                         kernel_size=3,
                                         strides=2,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2d_t3 = tfklayers.Conv2DTranspose(filters=192,
                                                   kernel_size=3,
                                                   strides=2,
                                                   padding='same',
                                                   activation='relu',
                                                   kernel_initializer='he_uniform')
        
        self.conv2d_10 = tfklayers.Conv2D(filters=2,
                                          kernel_size=3,
                                          strides=1,
                                          padding='same',
                                          activation='sigmoid')
                                            
        self.upsample = tfklayers.UpSampling2D(size=(16,16), interpolation='nearest')

    def call(self, inputs):
        dec_qfmap = self.reshape(inputs)

        dec_qfmap = self.conv2d_1(dec_qfmap) 
        dec_qfmap = self.conv2d_2(dec_qfmap) 
        dec_qfmap = self.conv2d_3(dec_qfmap) 
        dec_qfmap = self.conv2d_4(dec_qfmap) 
        dec_qfmap = self.conv2d_t1(dec_qfmap)

        dec_qfmap = self.conv2d_5(dec_qfmap) 
        dec_qfmap = self.conv2d_6(dec_qfmap) 
        dec_qfmap = self.conv2d_7(dec_qfmap) 
        dec_qfmap = self.conv2d_t2(dec_qfmap)

        dec_qfmap = self.conv2d_8(dec_qfmap) 
        dec_qfmap = self.conv2d_9(dec_qfmap) 
        dec_qfmap = self.conv2d_t3(dec_qfmap)

        dec_qfmap = self.conv2d_10(dec_qfmap)
        dec_qfmap = self.upsample(dec_qfmap) 
        return dec_qfmap

    def summary(self, print_fn=None):
        x = tf.keras.Input(shape=(28,28,8,8,3))
        _qf_map_predictor = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return _qf_map_predictor.summary(print_fn=print_fn)