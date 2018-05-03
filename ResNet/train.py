"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf

import resnet_v2 as resnet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from functools import partial
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('src', 'amazon', 'Source domain')
tf.app.flags.DEFINE_string('tar', 'webcam', 'Target domain')
tf.app.flags.DEFINE_integer('epoches', 40, 'Number of training epoches')
tf.app.flags.DEFINE_float('lr0', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('bs', 32, 'Batch size')
tf.app.flags.DEFINE_integer('num_class', 31, 'number of classes')
tf.app.flags.DEFINE_float('weight_decay', 0.00001, 'weight decay')
tf.app.flags.DEFINE_bool('group_or_not', True, 'use_group_attention')
tf.app.flags.DEFINE_float('gp_weight', 1.0, 'group_attention')
tf.app.flags.DEFINE_float('mmd_weight', 1.0, 'group_attention')
tf.app.flags.DEFINE_float('gap_weight', 0.8, 'channel_attention')
tf.app.flags.DEFINE_float('fc_weight', 0.8, 'fc_attention')

def compute_pairwise_distances(x, y):
  
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
  
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')
  
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))
  
        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


# Path to the textfiles for the trainings and validation set
train_file = './'+FLAGS.src+'.txt'
val_file = './'+FLAGS.tar+'.txt'

# Learning params
learning_rate = FLAGS.lr0
num_epochs = FLAGS.epoches
batch_size = FLAGS.bs

# Network params
dropout_rate = 0.5
num_classes = FLAGS.num_class

display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./tensorboard"
checkpoint_path = "./checkpoints"
checkpointpath = "./ckpt/resnet_v2_50.ckpt"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='training',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=True)

    # create an reinitializable iterator given the dataset structure
    iterator_s = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    iterator_t = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)

    images_batch_s, labels_batch_s = iterator_s.get_next()
    images_batch_t, labels_batch_t = iterator_t.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator_s.make_initializer(tr_data.data)
validation_init_op = iterator_t.make_initializer(val_data.data)



# Initialize model
# Op for calculating the loss
#with tf.name_scope("cross_ent"):
#def LOSS(Gap5, CONV5, score, num_classes, batch_size, y):
images_batch = tf.concat([images_batch_s, images_batch_t], axis=0)
y = tf.concat([labels_batch_s, labels_batch_t], axis=0)
with slim.arg_scope(resnet.resnet_arg_scope()):
    Gap5, CONV5,net,_ = resnet.resnet_v2_50(images_batch)#,is_training=False)
net = tf.nn.dropout(net, 0.5)
net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                    normalizer_fn=None, scope='logits')
score = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    #d = 13*13# num_classes #tf.shape(Bs)[1]
    #elt = CONV5*Ex5
    #Conv5_abs = tf.abs(elt)
#Conv5_abs = tf.abs(CONV5)
#Att = tf.reduce_mean(Conv5_abs, axis = 3, keep_dims=True)
#ATT_fl = tf.reshape(Att, (batch_size*2, -1))
#source_samples = tf.slice(ATT_fl, [0, 0], [batch_size, -1])
#target_samples = tf.slice(ATT_fl, [batch_size, 0], [batch_size, -1])
#Bs = tf.slice(ATT_fl, [0, 0], [batch_size, d])
#Bt = tf.slice(ATT_fl, [batch_size, 0], [batch_size, d])
#source_samples = tf.slice(score, [0, 0], [batch_size, d])
#target_samples = tf.slice(score, [batch_size, 0], [batch_size, d])
sigmas = [
    1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 7e-2, 1e-1, 2e-1, 4e-1, 8e-1, 1, 5, 10, 15, 20, 25, 30, 35, 50, 80, 100, 300, 800,
    1e3, 1e4, 1e5, 1e6
]
#sigmas = [
#    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
#    1e3, 1e4, 1e5, 1e6
#]
gaussian_kernel = partial(
    gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

if(FLAGS.group_or_not):
    Conv5_abs_0 = tf.abs(tf.slice(CONV5,[0,0,0,0   ],[-1,-1,-1,512]))
    Conv5_abs_1 = tf.abs(tf.slice(CONV5,[0,0,0,512 ],[-1,-1,-1,512]))
    Conv5_abs_2 = tf.abs(tf.slice(CONV5,[0,0,0,1024],[-1,-1,-1,512]))
    Conv5_abs_3 = tf.abs(tf.slice(CONV5,[0,0,0,1536],[-1,-1,-1,512]))
    Conv5_abs_4 = tf.abs(tf.slice(CONV5,[0,0,0,0   ],[-1,-1,-1,1024]))
    Conv5_abs_5 = tf.abs(tf.slice(CONV5,[0,0,0,1024],[-1,-1,-1,1024]))
    Att0 = tf.reduce_mean(Conv5_abs_0, axis = 3)
    ATT_fl0 = tf.reshape(Att0, (batch_size*2, -1))
    source_samples_0 = tf.slice(ATT_fl0, [0, 0], [batch_size, -1])
    target_samples_0 = tf.slice(ATT_fl0, [batch_size, 0], [batch_size, -1])
    Att1 = tf.reduce_mean(Conv5_abs_1, axis = 3)
    ATT_fl1 = tf.reshape(Att1, (batch_size*2, -1))
    source_samples_1 = tf.slice(ATT_fl1, [0, 0], [batch_size, -1])
    target_samples_1 = tf.slice(ATT_fl1, [batch_size, 0], [batch_size, -1])
    Att2 = tf.reduce_mean(Conv5_abs_2, axis = 3)
    ATT_fl2 = tf.reshape(Att2, (batch_size*2, -1))
    source_samples_2 = tf.slice(ATT_fl2, [0, 0], [batch_size, -1])
    target_samples_2 = tf.slice(ATT_fl2, [batch_size, 0], [batch_size, -1])
    Att3 = tf.reduce_mean(Conv5_abs_3, axis = 3)
    ATT_fl3 = tf.reshape(Att3, (batch_size*2, -1))
    source_samples_3 = tf.slice(ATT_fl3, [0, 0], [batch_size, -1])
    target_samples_3 = tf.slice(ATT_fl3, [batch_size, 0], [batch_size, -1])
    Att4 = tf.reduce_mean(Conv5_abs_4, axis = 3)
    ATT_fl4 = tf.reshape(Att4, (batch_size*2, -1))
    source_samples_4 = tf.slice(ATT_fl4, [0, 0], [batch_size, -1])
    target_samples_4 = tf.slice(ATT_fl4, [batch_size, 0], [batch_size, -1])
    Att5 = tf.reduce_mean(Conv5_abs_5, axis = 3)
    ATT_fl5 = tf.reshape(Att5, (batch_size*2, -1))
    source_samples_5 = tf.slice(ATT_fl5, [0, 0], [batch_size, -1])
    target_samples_5 = tf.slice(ATT_fl5, [batch_size, 0], [batch_size, -1])
    source_samples = tf.concat([source_samples_0, source_samples_1, source_samples_2, source_samples_3, source_samples_4, source_samples_5],1)
    target_samples = tf.concat([target_samples_0, target_samples_1, target_samples_2, target_samples_3, target_samples_4, target_samples_5],1)
else:
    Conv5_abs = tf.abs(CONV5)
    Att = tf.reduce_mean(Conv5_abs, axis = 3)
    ATT_fl = tf.reshape(Att, (batch_size*2, -1))
    source_samples = tf.slice(ATT_fl, [0, 0], [batch_size, -1])
    target_samples = tf.slice(ATT_fl, [batch_size, 0], [batch_size, -1])
	#
Gap_fl = tf.reshape(Gap5, (batch_size*2, -1))
source_samples_gap = tf.slice(Gap_fl, [0, 0], [batch_size, -1])
target_samples_gap = tf.slice(Gap_fl, [batch_size, 0], [batch_size, -1])
loss_value = maximum_mean_discrepancy(
    source_samples, target_samples, kernel=gaussian_kernel)
if(FLAGS.group_or_not):
    dal_loss = tf.maximum(1e-4, loss_value) * FLAGS.gp_weight 
else:
    dal_loss = tf.maximum(1e-4, loss_value) * FLAGS.mmd_weight 

loss_value_gap = maximum_mean_discrepancy(
    source_samples_gap, target_samples_gap, kernel=gaussian_kernel)
dal_loss_gap = tf.maximum(1e-4, loss_value_gap) * FLAGS.gap_weight

SCORE = tf.reshape(score, [batch_size*2, -1])
source_samples_fc = tf.slice(SCORE, [0, 0], [batch_size, -1])
target_samples_fc = tf.slice(SCORE, [batch_size, 0], [batch_size, -1])
loss_value_fc = maximum_mean_discrepancy(
    source_samples_fc, target_samples_fc, kernel=gaussian_kernel)
dal_loss_fc = tf.maximum(1e-4, loss_value_fc) * FLAGS.fc_weight

cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.slice(SCORE, [0, 0], [batch_size, num_classes]), labels=tf.slice(y, [0, 0], [batch_size, num_classes])))

VARS = tf.trainable_variables()
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in VARS ]) * FLAGS.weight_decay

loss = dal_loss + cls_loss + lossL2 + dal_loss_gap + dal_loss_fc
# Tran op

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(tf.slice(SCORE, [0, 0], [batch_size, num_classes]), 1), tf.argmax(tf.slice(y, [0, 0], [batch_size, num_classes]), 1))
    accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

global_step = tf.Variable(0, trainable=False, name='global_step')
boundaries = [int(train_batches_per_epoch * num_epochs * 0.25), int(train_batches_per_epoch * num_epochs * 0.5), int(train_batches_per_epoch * num_epochs * 0.75)]
values = [learning_rate, 0.1 * learning_rate, 0.01 * learning_rate, 0.001 * learning_rate]
learning_rate_ = tf.train.piecewise_constant(global_step, boundaries, values)

trainable = tf.global_variables()
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op1 = tf.train.MomentumOptimizer(learning_rate_ * 10, 0.9).minimize(loss, var_list=trainable[-3:])
    train_op2 = tf.train.MomentumOptimizer(learning_rate_, 0.9).minimize(loss, var_list=trainable[216:-3])
    train_op = tf.group(train_op1, train_op2)


# Start Tensorflow session
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=trainable[:-3], max_to_keep=1000)
    saver.restore(sess, checkpointpath)
    saver2 = tf.train.Saver()

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)
        sess.run(validation_init_op)

        for step in range(train_batches_per_epoch):

            if ((step + 1) % val_batches_per_epoch) == 0:
                sess.run(validation_init_op)

            _,tloss, accuracy_ = sess.run([train_op, loss, accuracy])
            print("{} Training Accuracy = {:.4f} cls_loss = {:.4f}".format(datetime.now(), accuracy_/batch_size, tloss))


        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'aw_model_epoch' + str(epoch + 1)+'.ckpt')
        if(epoch>25):
            save_path = saver2.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))

