from argparse import ArgumentParser
from glob import glob
from functools import partial
import tensorflow as tf
from utils import *
from models import *


def read_image(img_path):
    img_file = tf.io.read_file(img_path)
    return img_file


def preprocess_img(img_file, size):
    shape = tf.image.extract_jpeg_shape(img_file)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        0.875 * tf.cast(tf.minimum(image_height, image_width), tf.float32),
        tf.int32
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, 
                            padded_center_crop_size])
    
    x = tf.image.decode_and_crop_jpeg(img_file,
                                      crop_window, 
                                      channels=3,
                                      dct_method='INTEGER_ACCURATE')
    x = tf.image.resize([x], [size, size], method='bicubic')[0]
    x = tf.clip_by_value(x, 0., 255.)
    x = tf.cast(x, 'float32')

    return x


def main(args):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    preprocess_fn = partial(preprocess_img, size=args.img_size)

    image_paths = glob(os.path.join(args.datadir, '*.JPEG'))
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(read_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(preprocess_fn)
    dataset = dataset.batch(args.batch_size)

    with tf.device('/CPU:0'):
        qf_map_generator = tf.keras.models.load_model(os.path.join('./artifacts', args.generator))
        for layer in qf_map_generator.layers:
            layer.trainable = False

        qf_map_predictor = tf.keras.models.load_model(os.path.join('./artifacts', args.predictor))
        for layer in qf_map_predictor.layers:
            layer.trainable = False

        jpeg_encoder = JPEGEncoder(qf_map_generator=qf_map_generator)
        jpeg_decoder = JPEGDecoder(qf_map_predictor=qf_map_predictor)


    for bidx, images in enumerate(dataset):
        enc_outputs = jpeg_encoder(images)
        dec_inputs = {'qdct': enc_outputs['qdct'], 'qf_range': enc_outputs['qf_range']}
        dec_outputs = jpeg_decoder(dec_inputs)

        for i, (img, enc_qf, dec_qf) in enumerate(zip(dec_outputs['rgb'], 
                                                    enc_outputs['qf_map'], 
                                                    dec_outputs['qf_map'])):
            fig = viz_results(img, enc_qf, dec_qf, display=False, verbose=True,
                              figname=f'result{args.batch_size*bidx+i:04d}.png')


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--datadir', type=str, default='./samples', help='Input image data directory.')
    parser.add_argument('--img_size', type=int, default=224, help='Image size.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')

    parser.add_argument('--generator', type=str, default='generator-48593829', help='Trained QF Map Generator artifact.')
    parser.add_argument('--predictor', type=str, default='predictor-48593829', help='Trained QF Map Predictor artifact.')

    args = parser.parse_args()

    main(args)
