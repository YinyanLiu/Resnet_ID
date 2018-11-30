#!/usr/bin/env python
# _*_coding:utf-8 _*_

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l2_regularizer
from model.network.nnet import NNet

def tensor_summary(tensor):
    """Add histogram and scalar summary of the tensor"""
    tensor_name = tensor.op.name
    tf.summary.histogram(tensor_name+"/activations", tensor)
    tf.summary.scalar(tensor_name+"/sparsity", tf.nn.zero_fraction(tensor))

class ResnetConfig(object):
    """
    The default hyper parameters config
    """
    # The batch normalization variance epsilon
    bn_var_epsilon = 0.001
    # weight decay for regularization in fully connected layer
    fc_weight_decay = 0.0002
    # weight decay for regularization in convolutional layer
    conv_weight_decay = 0.0002


class ResNet(NNet):

    def build_resnet(self, is_training, inputs):
        # resnet50  1 + (3+4+6+3)*3 + 1 =50

        layers = []
        if self.net_name == 'resnet50':
            blocks = self.resnet50()
        elif self.net_name == 'resnet101':
            blocks = self.resnet101()
        elif self.net_name == 'resnet152':
            blocks = self.resnet152()
        elif self.net_name == 'resnet200':
            blocks = self.resnet200()
        else:
            raise Exception(('Error: net_name shoulbe be resnet50, resnet101, resnet152, or resnet200, but the input is %d' ) % self.net_name)

        if is_training is True:
            reuse = tf.AUTO_REUSE
        else:
            reuse = tf.AUTO_REUSE

        # the first block, root block with convolution and pooling layer
        with tf.variable_scope('conv0', reuse=reuse):
            net1 = self.root_block(inputs=inputs)
            tensor_summary(net1)
            layers.append(net1)

        outputs = net1
        for block in range(4):
            with tf.variable_scope('block'+str(block+1), reuse=reuse): # there are four block
                block_para = blocks[block]
                base_depth = block_para['base_depth']
                for i in range(block_para['num_unit']): # in each block there are different residual unit
                    # except the last block, the stride of last unit for each block is 2
                    is_first_unit=True if (block==0) and (i==0) else False
                    stride = 2 if (i ==block_para['num_unit']-1) and (block != 3) else 1
                    with tf.variable_scope('unit' + str(i+1), reuse=reuse):
                        outputs = self.residual_unit(inputs=outputs, is_first_unit=is_first_unit, base_depth=base_depth, stride=stride)
                        layers.append(outputs)
        # full connection layer
        with tf.variable_scope("full_connection", reuse=reuse):

            fc_bn = tf.layers.batch_normalization(inputs=outputs, axis=0, renorm_momentum=0.95)
            fc_relu = tf.nn.relu(fc_bn)
            global_pool = tf.reduce_mean(fc_relu, axis=[1, 2])
            fc_n_input = global_pool.get_shape().as_list()[-1]
            weights = tf.get_variable('fc_weight', [fc_n_input, self.class_nums],
                                      initializer=tf.initializers.variance_scaling(scale=1.0),
                                      regularizer=l2_regularizer(scale=ResnetConfig.fc_weight_decay))
            bias = tf.get_variable('fc_bias', shape=[self.class_nums], initializer=tf.zeros_initializer,
                                   regularizer=l2_regularizer(ResnetConfig.fc_weight_decay))
            net_out = tf.add(tf.matmul(global_pool, weights), bias)
            net_out = tf.identity(input=net_out)
            layers.append(outputs)
        return net_out


    def root_block(self, inputs):

        x_shape = np.shape(inputs)
        inputs = tf.reshape(tensor=inputs, shape=[x_shape[0], -1, x_shape[2], 1])

        fiter0 = tf.get_variable('conv0', shape=[7, 7, 1, 1], initializer=xavier_initializer(),
                                 regularizer=l2_regularizer(ResnetConfig.conv_weight_decay))
        net = tf.nn.conv2d(input=inputs, filter=fiter0, strides=[1, 2, 2, 1], padding='SAME')
        # since the input dimension of our data is [50, 40], so the stride of pooling has been changed to 1 rather than 2
        net = tf.nn.max_pool(value=net, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")

        return net



    def residual_unit(self, inputs, is_first_unit, base_depth, stride):
        """"
        inputs:
        out_channels:
        stride:
        return:
        """
        # 1*1 64,  3*3 64, 1*1 256
        # 1*1 128, 3*3 128 1*1 512
        # 1*1 256, 3*3 256 1*1 1024
        # 1*1 512  3*3 512 1*1 2048
        in_channels = inputs.get_shape().as_list()[-1]

        # because of the different dimension between input and residual output,
        # the input dimension should be dealt with convolution and batch_normalization layers
        filter_shortcut = tf.get_variable('conv_shortcut',
                                          shape=[stride, stride, in_channels, base_depth*4],
                                          initializer=xavier_initializer(),
                                          regularizer=l2_regularizer(ResnetConfig.conv_weight_decay))
        x_shortcut = tf.nn.conv2d(input=inputs, filter=filter_shortcut, strides=[1, stride, stride, 1], padding='SAME')
        x_shortcut = tf.layers.batch_normalization(inputs=x_shortcut, axis=0)


        # conv + BN + Relu 1*1 64 1
        filter1 = tf.get_variable('conv1', shape=[1, 1, in_channels, base_depth], initializer=xavier_initializer(),
                                  regularizer=l2_regularizer(ResnetConfig.conv_weight_decay))

        res_conv1 = tf.nn.conv2d(input=inputs, filter=filter1, strides=[1, 1, 1, 1], padding='SAME')
        res_bn1 = tf.layers.batch_normalization(inputs=res_conv1, axis=0, renorm_momentum=0.95)
        res_relu1 = tf.nn.relu(res_bn1)

        # conv + BN + Relu 3*3 64 1
        filter2 = tf.get_variable('conv2', shape=[3, 3, base_depth, base_depth], initializer=xavier_initializer(),
                                  regularizer=l2_regularizer(ResnetConfig.conv_weight_decay))

        res_conv2 = tf.nn.conv2d(input=res_relu1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
        res_bn2 = tf.layers.batch_normalization(inputs=res_conv2, axis=0, renorm_momentum=0.95)
        res_relu2 = tf.nn.relu(res_bn2)

        # conv + BN 1*1 256 2
        filter3 = tf.get_variable('conv3', shape=[1, 1, base_depth, base_depth*4], initializer=xavier_initializer(),
                                  regularizer=l2_regularizer(ResnetConfig.conv_weight_decay))
        res_conv3 = tf.nn.conv2d(input=res_relu2, filter=filter3, strides=[1, stride, stride, 1], padding='SAME')
        res_relu3 = tf.nn.relu(res_conv3)
        out = x_shortcut + res_relu3
        out = tf.layers.batch_normalization(inputs=out, axis=0, renorm_momentum=0.95)

        return out

    def resnet50(self):

        block1 = dict(name='block1', base_depth=64, num_unit=3, stride=2)
        block2 = dict(name='block2', base_depth=128, num_unit=4, stride=2)
        block3 = dict(name='block3', base_depth=256, num_unit=6, stride=2)
        block4 = dict(name='block4', base_depth=512, num_unit=3, stride=1)
        blocks = [block1, block2, block3, block4]
        return blocks

    def resnet101(self):
        block1 = dict(name='block1', base_depth=64, num_unit=3, stride=2)
        block2 = dict(name='block2', base_depth=128, num_unit=4, stride=2)
        block3 = dict(name='block3', base_depth=256, num_unit=23, stride=2)
        block4 = dict(name='block4', base_depth=512, num_unit=3, stride=1)
        blocks = [block1, block2, block3, block4]

        return blocks


    def resnet152(self):
        block1 = dict(name='block1', base_depth=64, num_unit=3, stride=2)
        block2 = dict(name='block2', base_depth=128, num_unit=8, stride=2)
        block3 = dict(name='block3', base_depth=256, num_unit=36, stride=2)
        block4 = dict(name='block4', base_depth=512, num_unit=3, stride=1)
        blocks = [block1, block2, block3, block4]

        return blocks

    def resnet200(self):
        block1 = dict(name='block1', base_depth=64, num_unit=3, stride=2)
        block2 = dict(name='block2', base_depth=128, num_unit=24, stride=2)
        block3 = dict(name='block3', base_depth=256, num_unit=36, stride=2)
        block4 = dict(name='block4', base_depth=512, num_unit=3, stride=1)
        blocks = [block1, block2, block3, block4]
        return blocks