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
parser.add_argument('--epochs', '-e', default=20, type=int, help='Training epochs.')
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
args = parser.parse_args()


def run_training(args):
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
        input_pipes = inputs(os.path.join(args.training_data, '*tfrecords'),
                             batch_size=args.batch_size,
                             num_epochs=num_epochs,
                             num_threads=args.input_threads,
                             shuffle=True,
                             dtype=set_dtype)
        images,flat_target = input_pipes
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images['image'],flat_target], capacity=2 * args.batch_size * args.num_gpus)
        global_step = tf.get_variable(name='global_step', shape=[],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=adam_lr, epsilon=adam_ep)
        tower_grads = []
        with tf.variable_scope("autoencoder", reuse = tf.AUTO_REUSE):

            for index in xrange(args.num_gpus):
                with tf.device('/gpu:{}'.format(index)):
                    with tf.name_scope('tower_{}'.format(index)) as scope:
                        image_batch, target_batch = batch_queue.dequeue()

                        if args.variational == True:
                            sd, mn, decoded = inference(image_batch, nn_arch,batch_size=args.batch_size,
                                                    dtype=set_dtype, training=True)
                            loss_op = tf.reduce_mean(
                                img_loss(y_hat=decoded, targets_flat=target_batch) + kl_loss(sd=sd, mn=mn))
                        else:
                            _,_,decoded = inference(image_batch, nn_arch, batch_size=args.batch_size,
                                                dtype=set_dtype, training=True)
                            loss_op = img_loss(y_hat = decoded, targets_flat=target_batch)
                        for i in xrange(0,9):
                            tf.summary.image("reconstruction_{}".format(i), tf.reshape(decoded[i,:],[-1,100,100,3]))
                        for i in xrange(0, 9):
                            tf.summary.image("source_{}".format(i),
                                             tf.reshape(target_batch[i, :], [-1, 100, 100, 3]))
                        variable_summaries(loss_op)
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = optimizer.compute_gradients(loss_op)
                        tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        summary_op = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        vars_to_restore = tf.contrib.slim.get_variables_to_restore()
        saver = tf.train.Saver(vars_to_restore, keep_checkpoint_every_n_hours=2, max_to_keep=8)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if RESTORE == True:
                sys.stderr.write("Restoring Checkpoint\n")
            try:
                incr = 0
                step = 0
                start_time = time.time()
                sys.stderr.write("Start time: {}\n".format(start_time))
                while not coord.should_stop():
                    incr += 1
                    if ((step % 500) == 498) and (args.collect_run_metadata):
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, loss_value, summary, step = sess.run([train_op, loss_op, summary_op, global_step],
                                                                # feed_dict={keep_prob: args.dropout_frequency},
                                                                options=run_options,
                                                                run_metadata=run_metadata)
                        duration = time.time() - start_time
                        summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                        summary_writer.add_summary(summary, step)
                        sys.stderr.write("Step {}: loss = {} ({} sec)\n".format((step + 1),
                                                                                loss_value,
                                                                                duration))
                        start_time = time.time()
                    elif (step % 100) == 98:
                        _, loss_value, summary, step = sess.run([train_op, loss_op, summary_op, global_step]  # ,
                                                                # feed_dict={keep_prob: args.dropout_frequency}
                                                                )

                        duration = time.time() - start_time
                        summary_writer.add_summary(summary, step)
                        sys.stderr.write("Step {}: loss = {} ({} sec)\n".format((step + 1),
                                                                                loss_value,
                                                                                duration))
                        start_time = time.time()
                    else:
                        _, loss_value, summary, step = sess.run([train_op, loss_op, summary_op, global_step]  # ,
                                                                # feed_dict={keep_prob: args.dropout_frequency}
                                                                )
                        duration = time.time() - start_time
                    if (step % args.checkpoint_frequency) == (args.checkpoint_frequency - 1):
                        saver.save(sess, save_file, global_step=global_step)
            except tf.errors.OutOfRangeError:
                sys.stderr.write('Done training for {} epochs, {} steps.\n'.format(num_epochs,
                                                                                   step + 1))
            finally:
                coord.request_stop()
            coord.join(threads)
            saver.save(sess, save_file, global_step=global_step)
            summary_writer.add_summary(summary, step)


if __name__ == '__main__':
    run_training(args)