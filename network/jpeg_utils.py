import tensorflow as tf
import numpy as np
import random
import math
from PIL import Image
from io import BytesIO


def rgb_to_ycbcr(input, diff_round):
    dtype = input.dtype
    kernel = tf.constant([[ 0.299,  0.587,  0.114],
                          [-0.299, -0.587,  0.886],
                          [ 0.701, -0.587, -0.114]], dtype)
    nomi = tf.constant([1., 1.772, 1.402], dtype)
    offset = tf.constant([0., 128., 128.], dtype)

    output = tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))
    output = diff_round(output)
    output = output / nomi + offset
    return tf.clip_by_value(output, 0.0, 255.0)


def ycbcr_to_rgb(input, diff_round):
    dtype = input.dtype
    offset = tf.constant([0.0, -128.0, -128.0], dtype)
    kernel = tf.constant([[ 1.0,                0.0,              1.402],
                          [ 1.0, -0.114*1.772/0.587, -0.299*1.402/0.587],
                          [ 1.0,              1.772,                0.0]], dtype)
    
    output = input + offset
    output = tf.tensordot(output, tf.transpose(kernel), axes=((-1,), (0,)))
    return tf.clip_by_value(diff_round(output), 0.0, 255.0)


def batch_split_v1(input):
    """Split batched images into 8x8 blocks"""
    shape = tf.shape(input)
    output = tf.reshape(input, [shape[0], shape[1]//8, 8, shape[2]//8, 8, shape[3]])
    output = tf.transpose(output, perm=[0, 1, 3, 2, 4, 5])
    return output


def batch_tile_v1(input):
    """Tile blocks into batched images"""
    shape = tf.shape(input)
    output = tf.transpose(input, perm=[0, 1, 3, 2, 4, 5])
    output = tf.reshape(output, [shape[0], shape[1]*8, shape[2]*8, 3])
    return output


def get_2d_dct_matrix(n, transpose=False):
  """Generate 2-D Discrete cosine transform matrix
    Args:
      n: a size of height and width of the matrix.
      transpose: a boolean to select a type of the transformation matrix.
        (inverse or forward)
  """
  matrix = np.sqrt(2/n) * np.array([
      [1/np.sqrt(2)] + [np.cos(np.pi*(2*i+1)*j / (2*n)) for j in range(1, n)] 
      for i in range(0, n)]).astype('float32')
  if not transpose:
      matrix = matrix.transpose()
  return matrix


_DCT_MATRIX_8x8 = get_2d_dct_matrix(8)
_DCT_MATRIX_8x8_T = get_2d_dct_matrix(8, True)


def dct2d_v2(mcu_block):
    return tf.matmul(tf.matmul(_DCT_MATRIX_8x8, mcu_block), _DCT_MATRIX_8x8_T)


def idct2d_v2(mcu_block):
    return tf.matmul(tf.matmul(_DCT_MATRIX_8x8_T, mcu_block), _DCT_MATRIX_8x8)


def qf_to_scale(qf):
    qf = int(qf)
    if qf < 50 and qf >= 1:
        scale = math.floor(5000 / qf)
    elif qf < 100 and qf >= 50:
        scale = 200 - 2 * qf
    else:
        scale = 10  # QF95
    return scale / 100.


def scale_to_qf(scale):
    if scale < 1:
        qf = tf.round((200 - scale * 100) / 2)
    else:
        qf = tf.round(5000 / (scale * 100))
    return int(qf)