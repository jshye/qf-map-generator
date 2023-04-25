import tensorflow as tf
from utils import *
from .jpeg_layers import *
from .predictor import *


class JPEGDecoder(tf.keras.Model):
    def __init__(self, qf_map_predictor=None, name='jpeg_decoder'):
        super(JPEGDecoder, self).__init__(name=name)
        self.diff_round = hard_round
        self.rank_to_qf = RankToQFNorm(diff_round=self.diff_round)
        self.qf_to_scale = QFToScale(self.diff_round)
        self.idct2d = IDCT2D()
        self.ycbcr_to_rgb = YCbCrToRGB(self.diff_round)
        self.split = Split()
        self.tile = Tile()

        if qf_map_predictor is None:
            self.qf_map_predictor = QFMapPredictor()
        else:
            self.qf_map_predictor = qf_map_predictor


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
        
    def call(self, inputs):
        qdct = inputs['qdct']
        ymin, ymax, cmin, cmax = inputs['qf_range']

        shape = tf.shape(qdct)
        h, w = shape[2] * shape[4], shape[3] * shape[5]

        # De-quantization
        dec_qf_ref = tf.transpose(qdct, perm=[0, 2, 3, 4, 5, 1])

        dec_qf = self.qf_map_predictor(dec_qf_ref) 
        dec_qf=  self.rank_to_qf([dec_qf, ymin, ymax, cmin, cmax])
        dec_scale = self.qf_to_scale(dec_qf)

        dec_scale_y = tf.expand_dims(dec_scale[..., 0], axis=-1)
        dec_scale_y = self.split(dec_scale_y)        
        dec_scale_y = dec_scale_y[...,0]
        
        dec_scale_c = tf.expand_dims(dec_scale[..., 1], axis=-1)
        dec_scale_c = self.split(dec_scale_c)
        dec_scale_c = dec_scale_c[...,0]

        y_qdct_coef, cb_qdct_coef, cr_qdct_coef = tf.unstack(qdct, axis=1)
        y_dct_coef = y_qdct_coef * (dec_scale_y * self.y_qtb)
        cb_dct_coef = cb_qdct_coef * (dec_scale_c * self.c_qtb)
        cr_dct_coef = cr_qdct_coef * (dec_scale_c * self.c_qtb)

        # Inverse DCT
        y_blocks  = self.idct2d(y_dct_coef)
        cb_blocks = self.idct2d(cb_dct_coef)
        cr_blocks = self.idct2d(cr_dct_coef)

        ycbcr_blocks = tf.stack([y_blocks, cb_blocks, cr_blocks], axis=-1)
        ycbcr = self.tile(ycbcr_blocks) + 128
        ycbcr = ycbcr[:, :h, :w, :]

        # Color space conversion: YCbCr to RGB
        rgb = self.ycbcr_to_rgb(ycbcr)
        
        output = {'rgb': rgb, 'qf_map': dec_qf, 'dec_scale': dec_scale}

        return output
