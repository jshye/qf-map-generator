import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .jpeg_utils import *


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


def viz_results(rgb, enc_hmap, dec_hmap, hmap_type='qf', save_dir='./results', 
                figname='result.png', display=False, verbose=False):
    """ Visualize in a figure
        - decoded RGB image
        - luma & chroma QF maps (or scale maps) used when encoding
        - luma & chroma QF maps (or scale maps) used when decoding
        
    Args:
        rgb: decoded RGB image [224,224,3]
        enc_hmap: QF maps (or scale maps) used in the encoder [224,224,2]
        dec_hmap: QF maps (or scale maps) used in the deocder [224,224,2]
        hmap_type:
            'qf': recieve QF maps and display values in QF
            'scale': receive scale maps and display values in scale
            'scale_to_qf': recieve scale maps and disply values in QF
        disaply: 
            True: plt.show() and return None 
            False: return fig
    """
    assert hmap_type in ['qf', 'scale', 'scale_to_qf']

    enc_hmap_y = enc_hmap[...,0]
    enc_hmap_c = enc_hmap[...,1]
    
    enc_hmap_y = tf.expand_dims(enc_hmap_y, axis=-1)
    enc_hmap_c = tf.expand_dims(enc_hmap_c, axis=-1)

    y_min, y_max = tf.reduce_min(enc_hmap_y), tf.reduce_max(enc_hmap_y)
    c_min, c_max = tf.reduce_min(enc_hmap_c), tf.reduce_max(enc_hmap_c)
    
    dec_hmap_y = dec_hmap[...,0]
    dec_hmap_c = dec_hmap[...,1]

    dec_hmap_y = tf.expand_dims(dec_hmap_y, axis=-1)
    dec_hmap_c = tf.expand_dims(dec_hmap_c, axis=-1)

    dec_y_min, dec_y_max = tf.reduce_min(dec_hmap_y), tf.reduce_max(dec_hmap_y)
    dec_c_min, dec_c_max = tf.reduce_min(dec_hmap_c), tf.reduce_max(dec_hmap_c)
    
    hmap_min = min([y_min, c_min, dec_y_min, dec_c_min])
    hmap_max = max([y_max, c_max, dec_y_max, dec_c_max])


    fig = plt.figure(figsize=(20,4))
    grid_spec = gridspec.GridSpec(1, 5, width_ratios=[8,10,10,10,10])

    # decoded output image
    plt.subplot(grid_spec[0])
    plt.imshow(rgb/255.)
    plt.title('Decoded Image')
    plt.axis('off')

    # encoder luma heatmap
    cbax_y = plt.subplot(grid_spec[1])
    boundaries_y, _ = tf.unique(tf.reshape(enc_hmap_y, [-1]))
    y_plt = plt.imshow(enc_hmap_y[:,:,0], vmin=hmap_min, vmax=hmap_max, cmap='viridis_r')
    
    # titles in QF for all hmap_type
    if hmap_type=='qf':
        plt.title(f'Luma Gen QF {y_min:.0f}-{y_max:.0f}')
    else:
        plt.title(f'Luma Gen QF {scale_to_qf(y_max)}-{scale_to_qf(y_min)}')

    plt.axis('off')

    cbar_y = plt.colorbar(y_plt, ax=cbax_y, shrink=0.8, ticks=boundaries_y)
    if hmap_type=='scale_to_qf':
        cbar_y.ax.set_yticklabels(list(map(scale_to_qf, boundaries_y)))

    # encoder chroma heatmap
    cbax_c = plt.subplot(grid_spec[2])
    boundaries_c, _ = tf.unique(tf.reshape(enc_hmap_c, [-1]))
    c_plt = plt.imshow(enc_hmap_c[:,:,0], vmin=hmap_min, vmax=hmap_max, cmap='viridis_r')

    if hmap_type=='qf':
        plt.title(f'Chroma Gen QF {c_min:.0f}-{c_max:.0f}')
    else:
        plt.title(f'Chroma Gen QF {scale_to_qf(c_max)}-{scale_to_qf(c_min)}')

    plt.axis('off')

    cbar_c = plt.colorbar(c_plt, ax=cbax_c, shrink=0.8, ticks=boundaries_c)
    if hmap_type=='scale_to_qf':
        cbar_c.ax.set_yticklabels(list(map(scale_to_qf, boundaries_c)))

    # decoder luma heatmap
    dec_cbax_y = plt.subplot(grid_spec[3])
    dec_boundaries_y, _ = tf.unique(tf.reshape(dec_hmap_y, [-1]))
    dec_y_plt = plt.imshow(dec_hmap_y[:,:,0], vmin=hmap_min, vmax=hmap_max, cmap='viridis_r')
    
    if hmap_type=='qf':
        plt.title(f'Luma Pred QF {dec_y_min:.0f}-{dec_y_max:.0f}')
    else:
        plt.title(f'Luma Pred QF {scale_to_qf(dec_y_max)}-{scale_to_qf(dec_y_min)}')

    plt.axis('off')

    dec_cbar_y = plt.colorbar(dec_y_plt, ax=dec_cbax_y, shrink=0.8, ticks=dec_boundaries_y)
    if hmap_type=='scale_to_qf':
        dec_cbar_y.ax.set_yticklabels(list(map(scale_to_qf, dec_boundaries_y)))

    # decoder chroma heatmap
    dec_cbax_c = plt.subplot(grid_spec[4])
    dec_boundaries_c, _ = tf.unique(tf.reshape(dec_hmap_c, [-1]))
    dec_c_plt = plt.imshow(dec_hmap_c[:,:,0], vmin=hmap_min, vmax=hmap_max, cmap='viridis_r')

    if hmap_type=='qf':
        plt.title(f'Chroma Pred QF {dec_c_min:.0f}-{dec_c_max:.0f}')
    else:
        plt.title(f'Chroma Pred QF {scale_to_qf(dec_c_max)}-{scale_to_qf(dec_c_min)}')

    plt.axis('off')

    dec_cbar_c = plt.colorbar(dec_c_plt, ax=dec_cbax_c, shrink=0.8, ticks=dec_boundaries_c)
    if hmap_type=='scale_to_qf':
        dec_cbar_c.ax.set_yticklabels(list(map(scale_to_qf, dec_boundaries_c)))


    plt.tight_layout()

    if display:
        plt.show()

    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, figname)
    plt.savefig(fig_path)
    plt.close()

    if verbose:
        print(f'Figure saved at {fig_path}')
    
    return fig
