import tensorflow as tf
from .encoder import JPEGEncoder
from .decoder import JPEGDecoder


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

        self.encoder = JPEGEncoder(qf_bias=qf_bias,
                                   clip_qf_min=clip_qf_min,
                                   clip_qf_max=clip_qf_max)
        self.decoder = JPEGDecoder()

        self.qf_bias = qf_bias
        self.clip_qf_min = clip_qf_min
        self.clip_qf_max = clip_qf_max
        self.rate_loss_coef = rate_loss_coef

    def call(self, rgb):
        enc_outputs = self.encoder(rgb)

        pred_code_len = self.bpp_estimator(enc_outputs['qdct_blocks'], training=False)
        self.add_metric(pred_code_len / (224.*224.), name='bpp_pred')

        dec_inputs = {'qdct': enc_outputs['qdct'], 'qf_range': enc_outputs['qf_range']} 
        dec_outputs = self.decoder(dec_inputs)

        classifier_inputs = self.classifier.preprocess(dec_outputs['rgb'])
        logits = self.classifier(classifier_inputs, training=False)

        qf_mse = tf.reduce_mean(tf.math.squared_difference(enc_outputs['qf_map'],
                                                           dec_outputs['qf_map']))
        self.add_metric(qf_mse, name='qf_mse')

        return logits, pred_code_len, enc_outputs['enc_scale'], dec_outputs['dec_scale'], dec_outputs['rgb']

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred, pred_code_len, enc_scale, dec_scale, rgb = self(x, training=True)

            task_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            rate_loss = pred_code_len / (224.*224.)
            loss = task_loss + self.rate_loss_coef * rate_loss

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.compiled_metrics.update_state(y, y_pred)

            return {m.name: m.result() for m in self.metrics}
