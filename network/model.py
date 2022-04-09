import tensorflow as tf
from .utils import *
from .jpeg_layers import *
from .layers import RankToQF, RankToQFDec, QFToScale
from .generator import QFMapGenerator
from .predictor import QFMapPredictor


class JPEGEndToEnd(tf.keras.Model):
    def __init__(self,
                 bpp_estimator,           
                 classifier,          
                 qf_bias=0,
                 clip_qf_min=2,
                 clip_qf_max=50,
                 rate_loss_coef=4,
                 name='jpeg_end_to_end'):
        super(JPEGEndToEnd, self).__init__(name=name)
        self.bpp_estimator = bpp_estimator
        self.classifier = classifier

        self.diff_round = hard_round
        self.rgb_to_ycbcr = RGBToYCbCr(self.diff_round)
        self.ycbcr_to_rgb = YCbCrToRGB(self.diff_round)
        self.split = Split()
        self.tile = Tile()
        self.dct2d = DCT2D()
        self.idct2d = IDCT2D()

        self.qf_bias = qf_bias
        self.clip_qf_min = clip_qf_min
        self.clip_qf_max = clip_qf_max

        self.qf_map_generator = QFMapGenerator()
        self.qf_map_predictor = QFMapPredictor()

        self.rank_to_qf = RankToQF(diff_round=self.diff_round,
                                   qf_bias=self.qf_bias,
                                   clip_qf_min=self.clip_qf_min,
                                   clip_qf_max=self.clip_qf_max)
        self.rank_to_qf_dec = RankToQFDec(diff_round=self.diff_round)
        
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

        self.rate_loss_coef = rate_loss_coef

    def call(self, rgb):
        # RGB-to-YCbCr and DCT
        ycbcr = self.rgb_to_ycbcr(rgb) - 128
        
        shape = tf.shape(rgb)
        h = shape[1]
        w = shape[2]

        blocks = self.split(ycbcr)

        y_blocks  = blocks[..., 0]
        cb_blocks = blocks[..., 1]
        cr_blocks = blocks[..., 2]

        y_dct_coefficient  = self.dct2d(y_blocks)
        cb_dct_coefficient = self.dct2d(cb_blocks)
        cr_dct_coefficient = self.dct2d(cr_blocks)

        # Quantization
        qf_ref = tf.stack([y_dct_coefficient, cb_dct_coefficient, cr_dct_coefficient], axis=-1)
        
        enc_qf = self.qf_map_generator(qf_ref)
        enc_qf = self.rank_to_qf(enc_qf)
        enc_scale = self.qf_to_scale(enc_qf) 

        enc_scale_y = tf.expand_dims(enc_scale[..., 0], axis=-1)
        enc_scale_y = self.split(enc_scale_y)
        enc_scale_y = tf.transpose(enc_scale_y[...,0], perm=[1, 2, 0, 3, 4])
        
        enc_scale_c = tf.expand_dims(enc_scale[..., 1], axis=-1)
        enc_scale_c = self.split(enc_scale_c)
        enc_scale_c = tf.transpose(enc_scale_c[...,0], perm=[1, 2, 0, 3, 4])
        
        y_qdct_coefficient = tf.transpose(y_dct_coefficient,
                                          perm=[1, 2, 0, 3, 4]) / (enc_scale_y * self.y_qtb)
        cb_qdct_coefficient = tf.transpose(cb_dct_coefficient,
                                           perm=[1, 2, 0, 3, 4]) / (enc_scale_c * self.c_qtb)
        cr_qdct_coefficient = tf.transpose(cr_dct_coefficient,
                                           perm=[1, 2, 0, 3, 4]) / (enc_scale_c * self.c_qtb)

        y_qdct_coefficient = tf.transpose(self.diff_round(y_qdct_coefficient),
                                          perm=[2, 0, 1, 3, 4])
        cb_qdct_coefficient = tf.transpose(self.diff_round(cb_qdct_coefficient),
                                           perm=[2, 0, 1, 3, 4])
        cr_qdct_coefficient = tf.transpose(self.diff_round(cr_qdct_coefficient),
                                           perm=[2, 0, 1, 3, 4])

        # BPP Estimation
        ycbcr_qdct_blocks = tf.stack([y_qdct_coefficient, cb_qdct_coefficient, cr_qdct_coefficient], axis=1)
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

        pred_code_len = self.bpp_estimator(blocks, training=False)
        pred_code_len = tf.reshape(pred_code_len, (batch_size, -1))
        pred_code_len = tf.reduce_sum(pred_code_len, axis=-1)
        pred_code_len = tf.reduce_mean(pred_code_len, keepdims=True)

        self.add_metric(pred_code_len / (224.*224.), name='bpp_pred')

        # De-quantization
        dec_qf_ref = tf.transpose(qdct, perm=[0, 2, 3, 4, 5, 1])
        dec_qf = self.qf_map_predictor(dec_qf_ref) 
        dec_qf = self.rank_to_qf_dec([dec_qf, enc_qf])
        dec_scale = self.qf_to_scale(dec_qf)

        dec_scale_y = tf.expand_dims(dec_scale[..., 0], axis=-1)
        dec_scale_y = self.split(dec_scale_y)        
        dec_scale_y = dec_scale_y[...,0]
        
        dec_scale_c = tf.expand_dims(dec_scale[..., 1], axis=-1)
        dec_scale_c = self.split(dec_scale_c)
        dec_scale_c = dec_scale_c[...,0]

        y_dct_coefficient = y_qdct_coefficient * (dec_scale_y * self.y_qtb)
        cb_dct_coefficient = cb_qdct_coefficient * (dec_scale_c * self.c_qtb)
        cr_dct_coefficient = cr_qdct_coefficient * (dec_scale_c * self.c_qtb)

        # Inverse DCT and YCbCr-to-RGB
        y_blocks  = self.idct2d(y_dct_coefficient)
        cb_blocks = self.idct2d(cb_dct_coefficient)
        cr_blocks = self.idct2d(cr_dct_coefficient)

        ycbcr_blocks = tf.stack([y_blocks, cb_blocks, cr_blocks], axis=-1)
        ycbcr = self.tile(ycbcr_blocks) + 128

        ycbcr = ycbcr[:, :h, :w, :]

        rgb = self.ycbcr_to_rgb(ycbcr)

        # Inference on reconstructed images
        rgb_input = self.classifier.preprocess(rgb)
        logits = self.classifier(rgb_input, training=False)

        qf_mse = tf.reduce_mean(tf.math.squared_difference(enc_qf, dec_qf))
        self.add_metric(qf_mse, name='qf_mse')

        return logits, pred_code_len, enc_scale, dec_scale, rgb
        

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred, pred_code_len, enc_scale, dec_scale, rgb = self(x, training=True)
            task_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            rate_loss = pred_code_len / (224.*224.)
            loss = task_loss + rate_loss * self.rate_loss_coef
                    
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}