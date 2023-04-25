import tensorflow as tf
from utils import *
from .jpeg_layers import *
from .generator import *


class JPEGEncoder(tf.keras.Model):
    def __init__(self,
                 qf_bias=0,
                 clip_qf_min=2,
                 clip_qf_max=50,
                 qf_map_generator=None,
                 name='jpeg_encoder'):
        super(JPEGEncoder, self).__init__(name=name)
        self.qf_bias = qf_bias
        self.clip_qf_min = clip_qf_min
        self.clip_qf_max = clip_qf_max

        self.diff_round = hard_round
        self.rgb_to_ycbcr = RGBToYCbCr(self.diff_round)
        self.split = Split()
        self.tile = Tile()
        self.dct2d = DCT2D()

        if qf_map_generator is None:
            self.qf_map_generator = QFMapGenerator()
        else:
            self.qf_map_generator = qf_map_generator

        self.rank_to_qf = RankToQF(diff_round=self.diff_round,
                                   qf_bias=self.qf_bias,
                                   clip_qf_min=self.clip_qf_min,
                                   clip_qf_max=self.clip_qf_max,
                                   )
        self.qf_to_scale = QFToScale(self.diff_round)

        self.y_qtb = tf.constant([16,11,10,16,24,40,51,61,
                                  12,12,14,19,26,58,60,55,
                                  14,13,16,24,40,57,69,56,
                                  14,17,22,29,51,87,80,62,
                                  18,22,37,56,68,109,103,77,
                                  24,36,55,64,81,104,113,92,
                                  49,64,78,87,103,121,120,101,
                                  72,92,95,98,112,100,103,99],
                                 shape=(8,8), dtype=tf.float32)
        
        self.c_qtb = tf.constant([17,18,24,47,99,99,99,99,
                                  18,21,26,66,99,99,99,99,
                                  24,26,56,99,99,99,99,99,
                                  47,66,99,99,99,99,99,99,
                                  99,99,99,99,99,99,99,99,
                                  99,99,99,99,99,99,99,99,
                                  99,99,99,99,99,99,99,99,
                                  99,99,99,99,99,99,99,99],
                                 shape=(8,8), dtype=tf.float32)
        
    def call(self, rgb):
        # 1. Color space conversion: RGB to YCbCr
        ycbcr = self.rgb_to_ycbcr(rgb) - 128
        
        shape = tf.shape(rgb)
        h, w = shape[1], shape[2]

        blocks = self.split(ycbcr)

        y_blocks  = blocks[..., 0]
        cb_blocks = blocks[..., 1]
        cr_blocks = blocks[..., 2]

        # 2. Discrete Cosine Transform (DCT)
        y_dct_coefficient  = self.dct2d(y_blocks)
        cb_dct_coefficient = self.dct2d(cb_blocks)
        cr_dct_coefficient = self.dct2d(cr_blocks)

        # 3. Quantization
        qf_ref = tf.stack([y_dct_coefficient, 
                           cb_dct_coefficient, 
                           cr_dct_coefficient], axis=-1)
        
        enc_qf = self.qf_map_generator(qf_ref)
        enc_qf = self.rank_to_qf(enc_qf)

        y_qf_min = tf.reduce_min(enc_qf[...,0], axis=[1,2], keepdims=True)
        y_qf_max = tf.reduce_max(enc_qf[...,0], axis=[1,2], keepdims=True)
        c_qf_min = tf.reduce_min(enc_qf[...,1], axis=[1,2], keepdims=True)
        c_qf_max = tf.reduce_max(enc_qf[...,1], axis=[1,2], keepdims=True)

        enc_scale = self.qf_to_scale(enc_qf) 

        enc_scale_y = tf.expand_dims(enc_scale[..., 0], axis=-1)
        enc_scale_y = self.split(enc_scale_y)
        enc_scale_y = tf.transpose(enc_scale_y[...,0], perm=[1, 2, 0, 3, 4])
        
        enc_scale_c = tf.expand_dims(enc_scale[..., 1], axis=-1)
        enc_scale_c = self.split(enc_scale_c)
        enc_scale_c = tf.transpose(enc_scale_c[...,0], perm=[1, 2, 0, 3, 4])
        
        y_qdct_coefficient = (tf.transpose(y_dct_coefficient, perm=[1, 2, 0, 3, 4]) / 
                              (enc_scale_y * self.y_qtb))
        cb_qdct_coefficient = (tf.transpose(cb_dct_coefficient, perm=[1, 2, 0, 3, 4]) / 
                               (enc_scale_c * self.c_qtb))
        cr_qdct_coefficient = (tf.transpose(cr_dct_coefficient, perm=[1, 2, 0, 3, 4]) / 
                               (enc_scale_c * self.c_qtb))

        y_qdct_coefficient = tf.transpose(self.diff_round(y_qdct_coefficient),
                                          perm=[2, 0, 1, 3, 4])
        cb_qdct_coefficient = tf.transpose(self.diff_round(cb_qdct_coefficient),
                                           perm=[2, 0, 1, 3, 4])
        cr_qdct_coefficient = tf.transpose(self.diff_round(cr_qdct_coefficient),
                                           perm=[2, 0, 1, 3, 4])

        ycbcr_qdct_blocks = tf.stack([y_qdct_coefficient, 
                                      cb_qdct_coefficient, 
                                      cr_qdct_coefficient], axis=1)
        qdct = tf.identity(ycbcr_qdct_blocks)

        shape = tf.shape(ycbcr_qdct_blocks)

        batch_size = shape[0]
        num_channels = shape[1]
        num_blocks = shape[2] * shape[3]
        num_coefficients = shape[4] * shape[5]

        ycbcr_qdct_blocks = tf.reshape(ycbcr_qdct_blocks,
                                       [batch_size, num_channels, num_blocks, num_coefficients])
        ycbcr_qdct_blocks = tf_delta_encode(ycbcr_qdct_blocks)
        ycbcr_qdct_blocks = tf_raster_scan(ycbcr_qdct_blocks)

        blocks = tf.reshape(ycbcr_qdct_blocks,
                            [batch_size, num_channels * num_blocks, num_coefficients])
        blocks = tf.math.log(tf.abs(blocks) + 1) / tf.math.log(2.)
        blocks = tf.reshape(blocks, (-1, 64))
    
        output = {'qdct':qdct, 'qdct_blocks': blocks, 
                  'qf_map': enc_qf, 'enc_scale': enc_scale, 
                  'qf_range': [y_qf_min, y_qf_max, c_qf_min, c_qf_max]}

        return output