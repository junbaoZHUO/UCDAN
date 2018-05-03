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
tf.app.flags.DEFINE_string('tar', 'webcam', 'Target domain')
tf.app.flags.DEFINE_integer('num_class', 31, 'number of classes')
tf.app.flags.DEFINE_integer('bs', 1, 'Batch size')

val_file = './'+FLAGS.tar+'.txt'

batch_size = FLAGS.bs

# Network params
num_classes = FLAGS.num_class

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    test_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator_test = Iterator.from_structure(test_data.data.output_types,
                                       test_data.data.output_shapes)

    images_batch_test, labels_batch_test = iterator_test.get_next()

# Ops for initializing the two different iterators
test_init_op = iterator_test.make_initializer(test_data.data)

images_batch = tf.concat([images_batch_test, images_batch_test], axis=0)
y = tf.concat([labels_batch_test, labels_batch_test], axis=0)
with slim.arg_scope(resnet.resnet_arg_scope()):
    Gap5, CONV5,net,_ = resnet.resnet_v2_50(images_batch, is_training=False)
net = tf.nn.dropout(net, 1.0)
net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                    normalizer_fn=None, scope='logits')
score = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
SCORE = tf.reshape(score, [2*batch_size, -1])

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(tf.slice(SCORE, [0, 0], [batch_size, num_classes]), 1), tf.argmax(tf.slice(y, [0, 0], [batch_size, num_classes]), 1))
    accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

val_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    saver = tf.train.Saver()#tf.global_variables())#tf.trainable_variables())
    saver.restore(sess, './checkpoints/aw_model_epoch40.ckpt')

    sess.run(test_init_op)
    test_acc = 0.
    test_count = 0
    for it in range(val_batches_per_epoch + 0):
        accuracy_ = sess.run(accuracy)
        test_acc += accuracy_
        test_count += batch_size
    test_acc /= test_count
    print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
