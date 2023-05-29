import os
import argparse
import functools
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from models import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default='run-01', help='Experiment title.')
    parser.add_argument('--data_dir', type=str, default='./tensorflow_datasets', help='Data directory.')
    parser.add_argument('--logdir', type=str, default='./results', help='Directory to save experiment logs.')
    parser.add_argument('--save_ckpt', action='store_true', help='Save checkpoints per epoch.')
    parser.add_argument('--viz', action='store_true', help='Visualize sample results per epoch')
    parser.add_argument('--n_samples', type=int, default=4, help='The number of samples to visualize.')

    parser.add_argument('--input_size', type=int, default=224, help='Input image size.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs.')
    parser.add_argument('--steps', type=int, default=1000, help='The number of training steps.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    
    parser.add_argument('--rate_loss_coef', type=int, default=4, help='Weight coefficient for the rate loss term.')
    parser.add_argument('--qf_bias', type=int, default=0, help='Bias added to QF map values.')
    parser.add_argument('--clip_qf_min', type=int, default=2, help='Lower limit of QF map values')
    parser.add_argument('--clip_qf_max', type=int, default=2, help='Upper limit of QF map values')
    parser.add_argument('--bpp_estimator', type=str, default='./artifacts/bpp_estimator',
                        help='Path to the pre-trained BPP estimator.')

    args = parser.parse_args()

    return args


class VizSamples(tf.keras.callbacks.Callback):
    def __init__(self, sample_data, args):
        super(VizSamples, self).__init__()
        self.sample_data = sample_data
        self.args = args

    def on_epoch_end(self, epoch, logs=None):
        for x, y in self.sample_data:
            _, _, enc_scale, dec_scale, rgb = self.model.predict_on_batch(x)

        for i in range(self.args.n_samples):
            viz_results(rgb[i], enc_scale[i], dec_scale[i], hmap_type='scale_to_qf',
                        save_dir=f'{self.args.logdir}/{self.args.run_name}/samples/epoch{epoch:04d}',
                        figname=f'sample-{i}.png',
                        display=False)
            plt.close()


def main(args):
    train_dataset = tfds.load('imagenet2012',
                            split='train',
                            decoders={'image': tfds.decode.SkipDecoding()},
                            as_supervised=True,
                            data_dir=args.data_dir)

    valid_dataset = tfds.load('imagenet2012',
                            split='validation',
                            decoders={'image': tfds.decode.SkipDecoding()},
                            as_supervised=True,
                            data_dir=args.data_dir)

    func_train = functools.partial(preprocess_train, size=args.input_size)
    func_valid = functools.partial(preprocess_valid, size=args.input_size)

    train_data = (train_dataset.map(func_train, -1)
                               .shuffle(5000)
                               .repeat(-1)
                               .batch(args.batch_size, drop_remainder=True)
                               .prefetch(-1))

    valid_data = (valid_dataset.map(func_valid, -1)
                               .batch(1000)
                               .prefetch(-1))

    with tf.device('/device:GPU:0'):
        bpp_estimator = tf.keras.models.load_model(args.bpp_estimator)
        classifier = tf.keras.applications.ResNet50(weights='imagenet')
        classifier.compile(metrics=['accuracy'])

    classifier.preprocess = tf.keras.applications.resnet.preprocess_input

    for layer in bpp_estimator.layers:
        layer.trainable = False

    for layer in classifier.layers:
        layer.trainable = False

    with tf.device('/device:GPU:0'):
        model = JPEGEndToEnd(bpp_estimator=bpp_estimator,
                            classifier=classifier,
                            qf_bias=args.qf_bias,
                            clip_qf_min=args.clip_qf_min,
                            clip_qf_max=args.clip_qf_max)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    os.makedirs(f'{args.logdir}/{args.run_name}', exist_ok=True)
    csv_logger = tf.keras.callbacks.CSVLogger(
        f'{args.logdir}/{args.run_name}/log.csv', append=True
    )

    callbacks = [csv_logger]

    if args.viz:
        sample_data = (valid_dataset.map(func_valid, -1)
                                    .batch(args.n_samples, drop_remainder=True)
                                    .take(1))

        callbacks.append(VizSamples(sample_data, args))

    if args.save_ckpt:
        ckpt_dir = f'{args.logdir}/{args.run_name}/checkpoints/'

        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_dir + '{epoch:04d}.ckpt',
            save_weights_only=True,
            save_best_only=False,
            verbose=1,
            save_freq='epoch'
        )

        callbacks.append(ckpt_cb)
        print(f'Checkpoints will be saved in {ckpt_dir}')

    else:
        print('** Checkpoints will not be saved **')

    model.fit(train_data,
              steps_per_epoch=args.steps,
              epochs=args.epochs,
              callbacks=callbacks,
              verbose=1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
