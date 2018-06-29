import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from functools import partial

#FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('src', 'amazon', 'Source domain')
tf.app.flags.DEFINE_string('tar', 'webcam', 'Target domain')
tf.app.flags.DEFINE_integer('epoches', 90, 'Number of training epoches')
tf.app.flags.DEFINE_float('lr0', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('bs', 128, 'Batch size')
tf.app.flags.DEFINE_integer('num_class', 31, 'number of classes')
tf.app.flags.DEFINE_float('weight_decay', 0.00001, 'weight decay')
tf.app.flags.DEFINE_bool('group_or_not', True, 'use_group_attention')
tf.app.flags.DEFINE_float('gp_weight', 6.0, 'group_attention')
tf.app.flags.DEFINE_float('mmd_weight', 6.0, 'group_attention')
tf.app.flags.DEFINE_float('gap_weight', 3.0, 'channel_attention')
tf.app.flags.DEFINE_float('fc_weight', 1.3, 'fc_attention')

FLAGS = tf.app.flags.FLAGS

#if __name__ == "__main__":
#    tf.app.run()
#tf.app.run(main=None)

def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
      ValueError: if the inputs do no matched the specified dimensions.
    """
  
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
  
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')
  
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))
  
        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

"""
Configuration Part.
"""
def main(_):    
    print(FLAGS)
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
    train_layers = ['fc8']
    train_layers1 = ['fc8']
    train_layers2 = ['dropout7', 'fc7', 'dropout6', 'fc6', 'conv5', 'conv4']#, 'conv3', 'norm2', 'conv2', 'norm1', 'conv1', 'pool1', 'pool2']
    
    # How often we want to write the tf.summary data to disk
    display_step = 20
    
    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = "./tensorboard"
    checkpoint_path = "./checkpoints"
    
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
        test_data = ImageDataGenerator(val_file,
                                      mode='inference',
                                      batch_size=batch_size,
                                      num_classes=num_classes,
                                      shuffle=False)
    
        # create an reinitializable iterator given the dataset structure
        iterator_s = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        iterator_t = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        iterator_test = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
    
        next_batch_s = iterator_s.get_next()
        next_batch_t = iterator_t.get_next()
        next_batch_test = iterator_test.get_next()
    
    # Ops for initializing the two different iterators
    training_init_op = iterator_s.make_initializer(tr_data.data)
    validation_init_op = iterator_t.make_initializer(val_data.data)
    test_init_op = iterator_test.make_initializer(test_data.data)
    
    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size * 2, 227, 227, 3])
    y = tf.placeholder(tf.float32, [batch_size * 2, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    
    # Initialize model
    model = AlexNet(x, keep_prob, num_classes, train_layers)
    
    # Link variable to model output
    CONV5 = model.CONV5
    CONV4 = model.CONV4
    score = model.fc8
    Gap5 = model.GAP5
    
    # List of trainable variables of the layers we want to train
    var_list1 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers1]
    var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers2]
    
    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
    val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
    
    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        d = 13*13# num_classes #tf.shape(Bs)[1]
        if(FLAGS.group_or_not):
    	    print('USE GROUP!')
            Conv5_abs = tf.abs(CONV5)
            Att = tf.reduce_mean(Conv5_abs, axis = 3)
            ATT_fl = tf.reshape(Att, (batch_size*2, -1))
            source_samples_gl = tf.slice(ATT_fl, [0, 0], [batch_size, d])
            target_samples_gl = tf.slice(ATT_fl, [batch_size, 0], [batch_size, d])
    
            Conv5_abs_0 = tf.abs(tf.slice(CONV5,[0,0,0,0  ],[-1,-1,-1,128]))
            Conv5_abs_1 = tf.abs(tf.slice(CONV5,[0,0,0,128],[-1,-1,-1,128]))
            Conv5_abs_2 = tf.abs(tf.slice(CONV5,[0,0,0,128],[-1,-1,-1,64]))
            Conv5_abs_3 = tf.abs(tf.slice(CONV5,[0,0,0,192],[-1,-1,-1,64]))
            Conv5_abs_4 = tf.abs(tf.slice(CONV5,[0,0,0,0  ],[-1,-1,-1,64]))
            Conv5_abs_5 = tf.abs(tf.slice(CONV5,[0,0,0,64 ],[-1,-1,-1,64]))
    
            Att0 = tf.reduce_mean(Conv5_abs_0, axis = 3)
            ATT_fl0 = tf.reshape(Att0, (batch_size*2, -1))
            source_samples_0 = tf.slice(ATT_fl0, [0, 0], [batch_size, d])
            target_samples_0 = tf.slice(ATT_fl0, [batch_size, 0], [batch_size, d])
            Att1 = tf.reduce_mean(Conv5_abs_1, axis = 3)
            ATT_fl1 = tf.reshape(Att1, (batch_size*2, -1))
            source_samples_1 = tf.slice(ATT_fl1, [0, 0], [batch_size, d])
            target_samples_1 = tf.slice(ATT_fl1, [batch_size, 0], [batch_size, d])
            Att2 = tf.reduce_mean(Conv5_abs_2, axis = 3)
            ATT_fl2 = tf.reshape(Att2, (batch_size*2, -1))
            source_samples_2 = tf.slice(ATT_fl2, [0, 0], [batch_size, d])
            target_samples_2 = tf.slice(ATT_fl2, [batch_size, 0], [batch_size, d])
            Att3 = tf.reduce_mean(Conv5_abs_3, axis = 3)
            ATT_fl3 = tf.reshape(Att3, (batch_size*2, -1))
            source_samples_3 = tf.slice(ATT_fl3, [0, 0], [batch_size, d])
            target_samples_3 = tf.slice(ATT_fl3, [batch_size, 0], [batch_size, d])
            Att4 = tf.reduce_mean(Conv5_abs_4, axis = 3)
            ATT_fl4 = tf.reshape(Att4, (batch_size*2, -1))
            source_samples_4 = tf.slice(ATT_fl4, [0, 0], [batch_size, d])
            target_samples_4 = tf.slice(ATT_fl4, [batch_size, 0], [batch_size, d])
            Att5 = tf.reduce_mean(Conv5_abs_5, axis = 3)
            ATT_fl5 = tf.reshape(Att5, (batch_size*2, -1))
            source_samples_5 = tf.slice(ATT_fl5, [0, 0], [batch_size, d])
            target_samples_5 = tf.slice(ATT_fl5, [batch_size, 0], [batch_size, d])
    
            source_samples = tf.concat([source_samples_gl, source_samples_0, source_samples_1, source_samples_2, source_samples_3, source_samples_4, source_samples_5],1)
            target_samples = tf.concat([target_samples_gl, target_samples_0, target_samples_1, target_samples_2, target_samples_3, target_samples_4, target_samples_5],1)
            #source_samples = tf.concat([source_samples_0, source_samples_1, source_samples_2, source_samples_3, source_samples_4, source_samples_5],1)
            #target_samples = tf.concat([target_samples_0, target_samples_1, target_samples_2, target_samples_3, target_samples_4, target_samples_5],1)
        else:
    	    print('USE SPATIAL ATTENTION!')
            Conv5_abs = tf.abs(CONV5)
            Att = tf.reduce_mean(Conv5_abs, axis = 3)
            ATT_fl = tf.reshape(Att, (batch_size*2, -1))
            source_samples = tf.slice(ATT_fl, [0, 0], [batch_size, d])
            target_samples = tf.slice(ATT_fl, [batch_size, 0], [batch_size, d])
    
        sigmas = [
            1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 7e-2, 1e-1, 2e-1, 4e-1, 8e-1, 1, 5, 10, 15, 20, 25, 30, 35, 50, 80, 100, 300, 800,
            1e3, 1e4, 1e5, 1e6
        ]
    
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
      
        loss_value = maximum_mean_discrepancy(
            source_samples, target_samples, kernel=gaussian_kernel)
        cov_loss = tf.maximum(1e-4, loss_value) * FLAGS.gp_weight 
    
        Gap_fl = tf.reshape(Gap5, (batch_size*2, -1))
        source_samples_gap = tf.slice(Gap_fl, [0, 0], [batch_size, 256])
        target_samples_gap = tf.slice(Gap_fl, [batch_size, 0], [batch_size, 256])
        loss_value_gap = maximum_mean_discrepancy(
            source_samples_gap, target_samples_gap, kernel=gaussian_kernel)
        cov_loss_gap = tf.maximum(1e-4, loss_value_gap) * FLAGS.gap_weight
    
        source_samples_fc8 = tf.slice(score, [0, 0], [batch_size, num_classes])
        target_samples_fc8 = tf.slice(score, [batch_size, 0], [batch_size, num_classes])
        loss_value_fc8 = maximum_mean_discrepancy(
            source_samples_fc8, target_samples_fc8, kernel=gaussian_kernel)
        cov_loss_fc8 = tf.maximum(1e-4, loss_value_fc8) * FLAGS.fc_weight
    
        cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.slice(score, [0, 0], [batch_size, num_classes]), labels=tf.slice(y, [0, 0], [batch_size, num_classes])))
    
        VARS = tf.trainable_variables()
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in VARS ]) * FLAGS.weight_decay
    
        loss = cov_loss + cls_loss + lossL2 + cov_loss_gap + cov_loss_fc8
    # Tran op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
    
        # Create optimizer and apply gradient descent to the trainable variables
        global_step = tf.Variable(0, trainable=False)#, name='global_setp')
        increment_op = tf.assign_add(global_step,tf.constant(1))
        boundaries = [int(train_batches_per_epoch * num_epochs * 0.25), int(train_batches_per_epoch * num_epochs * 0.5), int(train_batches_per_epoch * num_epochs * 0.75)]
        values = [learning_rate, 0.1 * learning_rate, 0.01 * learning_rate]
    
        learning_rate_ = tf.train.piecewise_constant(global_step, boundaries, values)
    
        train_op1 = tf.train.MomentumOptimizer(learning_rate_ * 10, 0.9).minimize(loss, var_list=var_list1)
        train_op2 = tf.train.MomentumOptimizer(learning_rate_, 0.9).minimize(loss, var_list=var_list2)
        train_op = tf.group(train_op1, train_op2)
    
    tf.summary.scalar('cross_entropy', cls_loss)
    tf.summary.scalar('total', loss)
    
    
    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(tf.slice(score, [0, 0], [batch_size, num_classes]), 1), tf.argmax(tf.slice(y, [0, 0], [batch_size, num_classes]), 1))
        accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    
    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)
    
    # Merge all summaries together
    merged_summary = tf.summary.merge_all()
    
    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)
    
    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()
    
    
    # Start Tensorflow session
    with tf.Session() as sess:
    
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
    
        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)
    
        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)
    
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
                sess.run(increment_op)
                img_batch_s, label_batch_s = sess.run(next_batch_s)
    
                if ((step + 1) % val_batches_per_epoch) == 0:
                    sess.run(validation_init_op)
                img_batch_t, label_batch_t = sess.run(next_batch_t)
    
                # And run the training op
                cov_loss_, cls_loss_, accuracy_, _ = sess.run([cov_loss, cls_loss, accuracy, train_op], feed_dict={x: np.concatenate((img_batch_s, img_batch_t), axis=0),
                                              y: np.concatenate((label_batch_s, label_batch_t), axis = 0),
                                              keep_prob: dropout_rate})
                print("{} Training Accuracy = {:.4f} cls_loss = {:.4f} MMD_loss = {:.4f}".format(datetime.now(), accuracy_/batch_size, cls_loss_, cov_loss_))
    
                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: np.concatenate((img_batch_s, img_batch_t), axis = 0),
                                                            y: np.concatenate((label_batch_s, label_batch_t), axis = 0),
                                                            keep_prob: 1.})
    
                    writer.add_summary(s, epoch*train_batches_per_epoch + step)
    
            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.now()))
            sess.run(test_init_op)
            test_acc = 0.
            test_count = 0
            for it in range(val_batches_per_epoch + 1):
                img_batch, label_batch = sess.run(next_batch_test)
                if(it == val_batches_per_epoch):
                    res_size = img_batch.shape[0]
                    img_batch = np.concatenate((img_batch, np.zeros((batch_size - img_batch.shape[0], 227, 227, 3))) ,axis = 0)
                    label_batch = np.concatenate((label_batch, np.zeros((batch_size - label_batch.shape[0], num_classes))) ,axis = 0)
                    correct_pred_ = sess.run(correct_pred, feed_dict={x: np.concatenate((img_batch, img_batch), axis = 0),
                                                        y: np.concatenate((label_batch,label_batch), axis = 0),
                                                        keep_prob: 1.})
                    acc = np.sum(correct_pred_.astype(float)[0:res_size])
                    batch_size_ = res_size
                else:
                    acc = sess.run(accuracy, feed_dict={x: np.concatenate((img_batch, img_batch), axis = 0),
                                                        y: np.concatenate((label_batch,label_batch), axis = 0),
                                                    keep_prob: 1.})
                    batch_size_ = batch_size
                test_acc += acc
                test_count += batch_size_
            test_acc /= test_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
            print("{} Saving checkpoint of model...".format(datetime.now()))
    
            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch' + str(epoch + 1)+'.ckpt')
            #save_path = saver.save(sess, checkpoint_name)
    
            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))
if __name__ == "__main__":
    tf.app.run()    
