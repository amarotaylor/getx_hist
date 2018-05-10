import tensorflow as tf
import sys
import os
import time
import glob
import argparse
import yaml

from tf_model_functions import *

parser = argparse.ArgumentParser(description='Train CNN for classification of variant selection.')
parser.add_argument('--training_data', type=str, help='Directory containing training data in tfrecords format.')
parser.add_argument('--checkpoint_dir', type=str, help='Directory where model checkpoints will be saved.')
parser.add_argument('--epochs', '-e', default=1, type=int, help='Training epochs.')
parser.add_argument('--dropout_frequency', '-d', default=0.50, type=float,
                    help='Set frequency for dropout normalization during training.')
parser.add_argument('--batch_size', '-b', default=200, type=int, help='Batch size for training.')
parser.add_argument('--input_threads', '-p', default=2, type=int,
                    help='Number of input threads to use (must be 2 or more).')
parser.add_argument('--num_gpus', '-g', default=2, type=int, help='specify number of GPUs to use.')
parser.add_argument('--summary_dir', '-s', default='/tmp/tensorflow_summaries', help='Data containing summary info')
parser.add_argument('--checkpoint_frequency', '-c', default=5000, type=int,
                    help="Number of steps between checkpoint saves.")
parser.add_argument('--collect_run_metadata', '-m', action='store_true',
                    help="Flag to store run metadata during training. Warning: buggy API component.")
parser.add_argument('--nn_architecture', '-nn_a', type=str, help="YAML file containing nn_architecture specs.")
parser.add_argument('--learning_rate',default=0.01, type=float)
parser.add_argument('--epsilon',default = 1e-8, type = float)
parser.add_argument('--variational', default = False, help='if variational true use KL divergence + reconstruction loss')
parser.add_argument('--variational_hidden_units', default=100, help='number of variational parameters')
parser.add_argument('--image', default = None, help='image to generate encodings')

args = parser.parse_args()


def evaluate_image(args):
    # Define nn_arch
    nn_path = args.nn_architecture
    nn_arch = yaml.safe_load(open(nn_path, 'r'))
    print nn_arch

    # set run-parameters
    num_epochs = args.epochs
    save_file = os.path.join(args.checkpoint_dir, 'my_model')
    set_dtype = tf.float32
    adam_lr = args.learning_rate
    adam_ep = args.epsilon
    # Make checkpoint dir if doesn't exist
    try:
        os.makedirs(args.checkpoint_dir)
        RESTORE = False
    except OSError:
        if not os.path.isdir(args.checkpoint_dir):
            raise
        else:
            if glob.glob(save_file + "*"):
                RESTORE = True
            else:
                RESTORE = False
    # Make summary dir if doesn't exist
    try:
        os.makedirs(args.summary_dir)
    except OSError:
        if not os.path.isdir(args.summary_dir):
            raise
    #
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_pipes = inputs(args.image,
                             batch_size=args.batch_size,
                             num_epochs=num_epochs,
                             num_threads=args.input_threads,
                             shuffle=False,
                             dtype=set_dtype)
        images,flat_target = input_pipes
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images['image'],flat_target], capacity=2 * args.batch_size * args.num_gpus)
        global_step = tf.get_variable(name='global_step', shape=[],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        with tf.variable_scope("autoencoder", reuse = tf.AUTO_REUSE):

            for index in xrange(args.num_gpus):
                with tf.device('/gpu:{}'.format(index)):
                    with tf.name_scope('tower_{}'.format(index)) as scope:
                        image_batch, target_batch = batch_queue.dequeue()

                        if args.variational == True:
                            sd, mn, decoded,_ = inference(image_batch, nn_arch,batch_size=args.batch_size,
                                                    dtype=set_dtype, training=True, latent_params = args.variational_hidden_units)
                            loss_op = tf.reduce_mean(
                                img_loss(y_hat=decoded, targets_flat=target_batch) + kl_loss(sd=sd, mn=mn))
                        else:
                            _,_,decoded,encoded = inference(image_batch, nn_arch, batch_size=args.batch_size,
                                                dtype=set_dtype, training=True)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        vars_to_restore = tf.contrib.slim.get_variables_to_restore()
        saver = tf.train.Saver(vars_to_restore, keep_checkpoint_every_n_hours=2, max_to_keep=8)
        encoded_arrays =[]
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if RESTORE == True:
                sys.stderr.write("Restoring Checkpoint\n")
                restore_checkpoint(sess, saver, args.checkpoint_dir)
            try:
                incr = 0
                step = 0
                _, _, decoded, encoded = sess.run([inference,global_step]  # ,
                                                                # feed_dict={keep_prob: args.dropout_frequency}
                                                                )
                if len(encoded_arrays)> 1:
                    encoded_arrays = np.concatenate([np.asarray(encoded),encoded_arrays])
                else:
                    encoded_arrays = np.asarray(encoded)


            finally:
                np.save('{}.npy'.format(args.image), encoded_arrays)
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    evaluate_image(args)