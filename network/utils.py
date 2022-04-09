import tensorflow as tf
import numpy as np


def hard_round(x):
    """Differitiable Round Operation"""
    x_ = tf.round(x)
    x_ = x_ - tf.stop_gradient(x) + x
    return x_


def tf_delta_encode(coefs):
    ac = coefs[..., 1:]
    dc = coefs[..., 0:1]
    dc = tf.concat([dc[..., 0:1, :], dc[..., 1:, :] - dc[..., :-1, :]], axis=-2)
    return tf.concat([dc, ac], axis=-1)


FROM_ZIGZAG_INDEX = np.array([
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
])

TO_ZIGZAG_INDEX = np.argsort(FROM_ZIGZAG_INDEX)

def tf_raster_scan(coefs):
    return tf.gather(coefs, TO_ZIGZAG_INDEX, axis=-1, batch_dims=0)