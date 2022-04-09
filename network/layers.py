import tensorflow as tf
import tensorflow.keras.layers as tfklayers


class ReshapeDCT(tfklayers.Layer):
    def __init__(self, extract_dct_type='all', name='reshape_dct'):
        super(ReshapeDCT, self).__init__(name=name)
        assert extract_dct_type in ['all', 'dc', 'ac_9', '2x2']
        if extract_dct_type == 'dc':
            self.reshape = tfklayers.Reshape((28,28,3))
        elif extract_dct_type == 'ac_9':
            self.reshape = tfklayers.Reshape((28,28,27))
        elif extract_dct_type == '2x2':
            self.reshape = tfklayers.Reshape((28,28,12))
        else:
            self.reshape = tfklayers.Reshape((28,28,192))

    def call(self, inputs):
        y = self.reshape(inputs)
        return y


class RankToQF(tfklayers.Layer):
    def __init__(self, diff_round, qf_bias, clip_qf_min, clip_qf_max, name='rank_to_qf'):
        super(RankToQF, self).__init__(name=name)
        self.diff_round = diff_round
        self.qf_bias = qf_bias
        self.clip_qf_min = clip_qf_min
        self.clip_qf_max = clip_qf_max

    def call(self, inputs):
        qf = inputs + self.qf_bias
        qf = self.diff_round(qf)
        qf = tf.clip_by_value(qf, self.clip_qf_min, self.clip_qf_max)
        return qf


class RankToQFDec(tfklayers.Layer):
    def __init__(self, diff_round, name='rank_to_qf_dec'):
        super(RankToQFDec, self).__init__(name=name)
        self.diff_round = diff_round

    def call(self, inputs):
        dec_qf, enc_qf = inputs[0], inputs[1]

        y_dec_qf_min = tf.reduce_min(enc_qf[...,0], axis=[1,2], keepdims=True)
        y_dec_qf_max = tf.reduce_max(enc_qf[...,0], axis=[1,2], keepdims=True)
        c_dec_qf_min = tf.reduce_min(enc_qf[...,1], axis=[1,2], keepdims=True)
        c_dec_qf_max = tf.reduce_max(enc_qf[...,1], axis=[1,2], keepdims=True)
        
        y_dec_qf = y_dec_qf_min + (y_dec_qf_max - y_dec_qf_min) * dec_qf[...,0]
        c_dec_qf = c_dec_qf_min + (c_dec_qf_max - c_dec_qf_min) * dec_qf[...,1]
        
        dec_qf = tf.stack([y_dec_qf, c_dec_qf], axis=-1)
        dec_qf = self.diff_round(dec_qf)
        return dec_qf


class QFToScale(tfklayers.Layer):
    def __init__(self, diff_round, name='qf_to_scale'):
        super(QFToScale, self).__init__(name=name)
        self.diff_round = diff_round

    def call(self, inputs):
        scale = self.diff_round(5000/inputs)
        scale = scale/100.
        scale = tf.clip_by_value(scale, 1., 50.)
        return scale