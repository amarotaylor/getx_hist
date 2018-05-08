import tensorflow as tf
from keras import backend as K
import numpy as np
import re
import sys
import glob
import math
import yaml


# example yaml file :
## YAML structure
"""
layer0:
    type: 'conv'
    width: 12
    output: 64
    pool: 4
layer1:
    type: 'incept'
    downsample: 16
    output: 16
    width1: 4
    width2: 12
    pool: 4
layer2:
    type: 'fc'
    output: 1280
layer3:
    type: 'dropout'
layer4:
    type: 'fc'
    output: 2
"""
def erprint(in_text):
    sys.stderr.write(in_text+'\n')


def read_and_decode(filename_queue, dtype = tf.float32):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    image = decode(serialized_example, dtype=dtype)

    return {'image': image}


def decode(serialized_example, height=100, width=100, dtype = tf.float32):
    features = tf.parse_single_example(
        serialized_example, features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'file_name': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)

    image_shape = tf.stack([height, width, 3])

    image = tf.reshape(image, image_shape)

    processed_example = tf.image.resize_image_with_crop_or_pad(image=image,
                                                               target_height=height,
                                                               target_width=width)
    return tf.cast(processed_example, dtype= dtype)



def inputs(file_regex, batch_size, num_epochs, num_threads=2, shuffle=True, dtype=tf.float32):
    if not num_epochs:
        num_epochs = None
    filenames = glob.glob(file_regex)
    with tf.name_scope('input'):
        min_after_dequeue = 200
        capacity = min_after_dequeue + ( num_threads + 3 ) * batch_size
        if shuffle:
            filename_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs,shuffle=True)
            example_list = [ read_and_decode(filename_queue, dtype=dtype)
                             for _ in range(num_threads) ]
            images = tf.train.shuffle_batch_join(
                example_list,
                batch_size=batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue)
        else:
            filename_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs,shuffle=False)
            example_list = read_and_decode(filename_queue, dtype=dtype)
            images = tf.train.batch(
                example_list,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity,
                allow_smaller_final_batch=True)
        flat_targets = tf.reshape(images['image'],[-1,100*100*3])
        return images, flat_targets


def restore_checkpoint(sess,saver,save_dir):
    erprint("Model checkpoints found at {}".format(save_dir))
    tf.train.get_checkpoint_state(save_dir)
    erprint("Recovered checkpoint state.")
    most_recent_save = tf.train.latest_checkpoint(save_dir)
    erprint("Loading checkpoint: {}".format(most_recent_save))
    saver.restore(sess, most_recent_save)
    erprint("Model restored")



def weight_variable(shape, name, stddev=0.1, dtype=tf.float32):
    with tf.device('/cpu:0'):
        final = tf.get_variable(name=name,shape=shape,dtype=dtype,
            initializer=tf.truncated_normal_initializer(mean=0,stddev=stddev))
    return final

def bias_variable(shape, name, dtype=tf.float32):
    with tf.device('/cpu:0'):
        final = tf.get_variable(name=name,shape=shape,dtype=dtype,
            initializer=tf.constant_initializer(0.1))
    return final


def variable_summaries(var):
    tensor_name = re.sub('{}_[0-9]*/'.format('tower'), '', var.op.name)
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(tensor_name + '/mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(tensor_name + '/stddev',stddev)
        tf.summary.scalar(tensor_name + '/max', tf.reduce_max(var))
        tf.summary.scalar(tensor_name + '/min', tf.reduce_min(var))
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(var))
        tf.summary.histogram(tensor_name + '/activations', var)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(grad, 0)
        var = grad_and_vars[0][1]
        grad_and_var = (grad, var)
        average_grads.append(grad_and_var)
    return average_grads

def training(loss, learning_rate):
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, global_step

# difference of decoded image from target image
# y_hat and targets_flat should be batch_size x (length*width*channels)
def img_loss(y_hat, targets_flat,scale_factor = 100*100*3):
    with tf.name_scope('image_loss'):
        return tf.divide(tf.reduce_sum(tf.squared_difference(y_hat, targets_flat), 1), scale_factor)

# kl divergence from standard normal
def kl_loss(sd,mn):
    with tf.name_scope('kl_loss'):
        return -0.5 * K.sum(1 + sd - K.square(mn) - K.exp(sd), axis=-1)

class layer_maker:
    def __init__(self,in_tensor,in_chn,in_width,dtype=tf.float32,training=True,dformat='channels_last', hidden = 1000):
        self.Ops = {}
        self.in_tensor = in_tensor
        self.in_chn = in_chn
        self.in_width = in_width
        self.training = training
        self.dtype = dtype
        self.dformat = dformat
        self.hidden = hidden

    def conv2d(self,x, f, k, name, stride=1, padding='same', format='channels_last',activation = tf.nn.leaky_relu):
        ''' wrapper for tf.layers.conv2d'''
        layer = tf.layers.conv2d(x, filters=f, kernel_size=k, strides=stride, padding=padding,
                                 name=name, data_format=format,
                                 activation=activation)
        return layer

    def deconv2d(self,x, f, k, name, stride=1, padding='same', format='channels_last',activation = tf.nn.leaky_relu):
        ''' wrapper for tf.layers.conv2d_transpose'''
        layer = tf.layers.conv2d_transpose(x, filters=f, kernel_size=k, padding='same', strides=stride,
                                           data_format=format
                                           , name=name,activation=activation)
        return layer

    def max_pool_2d(self,x, p, name, stride=1, format='channels_last', padding='valid'):
        '''wrapper for tf.layers.max_pooling2d'''
        layer = tf.layers.max_pooling2d(x, pool_size=p,
                                        strides=stride,
                                        padding=padding, data_format=format)
        return layer

    def fully_connected(self,x, u, name, activation = tf.nn.leaky_relu):
        ''' wrapper for tf.layers.dense'''
        layer = tf.layers.dense(x, units = u,activation = activation,
                                 name=name)
        return layer

    def sampling(self,z_mean, z_log_var):
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.hidden), mean=0.,
                                  stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def make_layer(self,l_info,l_index,last=False):
        ''' uses YAML file to generate layers '''
        layer_name = "layer_{}_{}".format(l_info['type'],l_index)
        layer_id = "layer_{}".format(l_index)
        data_format = self.dformat

        if l_info['type'] == 'variational':
            with tf.variable_scope(layer_id, reuse=tf.AUTO_REUSE):
                self.mn = tf.layers.dense(self.in_tensor, units=self.hidden, activation=tf.nn.leaky_relu, name='encode_mean')
                self.sd = 0.5 * tf.layers.dense(self.in_tensor, units=self.hidden, activation=tf.nn.leaky_relu, name='encode_sd')
                layer = self.sampling(self.mn, self.sd)
                out_chn = self.hidden
                out_width = 1
                variable_summaries(self.mn)
                variable_summaries(self.sd)

        assert l_info['type'] in ['conv','incept','deconv','fc','dropout','flatten','maxpool', 'variational']

        if l_info['type'] == 'conv':
            try:
                l_info['stride']
            except KeyError:
                l_info['stride'] = 1
            with tf.variable_scope(layer_id,reuse=tf.AUTO_REUSE):
                layer = self.conv2d(x = self.in_tensor, f= l_info['filters'], k = l_info['kernel_size'],
                                name = layer_id, stride=l_info['stride'])
            out_chn = l_info['filters']
            out_width = self.in_width

        elif l_info['type'] == 'deconv':
            try:
                l_info['stride']
            except KeyError:
                l_info['stride'] = 1
            with tf.variable_scope(layer_id,reuse=tf.AUTO_REUSE):
                layer = self.deconv2d(self.in_tensor,f = l_info['filters'],
                                    k = l_info['kernel_size'],name=layer_id, stride=l_info['stride'],padding='same')
            out_chn = l_info['filters']
            out_width = int(math.ceil( self.in_width / float(l_info['stride'])))
        elif l_info['type'] == 'fc':
            with tf.variable_scope(layer_id, reuse=tf.AUTO_REUSE):
                layer = self.fully_connected(self.in_tensor, l_info['units'], name = layer_id,
                                         activation = tf.nn.leaky_relu)
            out_chn = l_info['units']
            out_width = 1
        elif l_info['type'] == 'flatten':

            with tf.variable_scope(layer_id, reuse=tf.AUTO_REUSE):

                out_chn = self.in_width * self.in_chn
                out_width = 1

                layer = tf.reshape(self.in_tensor, shape=[-1, out_chn], name=layer_name)

        elif l_info['type'] == 'maxpool':
            try:
                l_info['stride']
            except KeyError:
                l_info['stride'] = 1

            with tf.name_scope(layer_id,reuse=tf.AUTO_REUSE):
                layer = self.max_pool_2d(self.in_tensor,p = l_info['pool'], name=layer_name,
                                    stride=l_info['stride'], format='channels_last', padding='same')
                out_width = int(math.ceil( (self.in_width - float(l_info['pool']) / float(l_info['stride'])))) + 1
                out_chn = self.in_chn
        self.in_tensor = layer
        self.in_chn = out_chn
        self.in_width = out_width
        self.Ops[l_index] = layer


def inference(images, nn_architecture, dtype=tf.float32, training=True):
    #print images.shape
    in_tensor = images
    in_chn = 3
    in_width = 100
    #print in_tensor.shape
    builder = layer_maker(in_tensor, in_chn, in_width, dtype=dtype, training=training)
    with tf.name_scope("autoencoder"):
        for layer_index in range(len(nn_architecture.keys())):
            l_info = nn_architecture['layer{}'.format(layer_index)]
            builder.make_layer(l_info,layer_index)
        return builder.sd,builder.mn,builder.in_tensor