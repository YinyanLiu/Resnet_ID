#!/usr/bin/env python
# _*_coding:utf-8 _*_

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l2_regularizer

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


class ResNet(ResnetConfig):
    def __init__(self):
        """
        :param input_tensor: 4-D input tensor
        :param is_training: bool, create new variables if True, otherwise, reuse the variables
        """
        self.base_depth = [64, 128, 256, 512]
        self.strides = [2, 2, 2, 1]
        self.config = ResnetConfig()

    def nnet(self, is_training, nnet_nums, inputs, class_nums):
        # resnet50  1 + (3+4+6+3)*3 + 1 =50
        layers = []
        if nnet_nums == 50:
            unit_nums = [3, 4, 6, 3]
        elif nnet_nums == 101:
            unit_nums = [3, 4, 23, 3]
        elif nnet_nums == 152:
            unit_nums = [3, 8, 36, 3]
        elif nnet_nums == 200:
            unit_nums = [3, 24, 36, 3]
        else:
            raise Exception(('Error: nnet_nums shoulbe be 50, 101, 152, or 200, but the input is %d' ) % nnet_nums)

        if self.is_training is True:
            reuse = tf.AUTO_REUSE
        else:
            reuse = False

        # the first block, root block with convolution and pooling layer
        with tf.variable_scope('conv0', reuse=reuse):
            net0 = self.root_block(inputs=inputs)
            tensor_summary(net0)
            layers.append(net0)

        outputs = net0
        for block in range(4):
            with tf.variable_scope('block'+str(block+1), reuse=reuse):
                for i in range(unit_nums[block]):
                    if i==0:
                        first_block = True
                    else:
                        first_block = False

                    outputs = self.residual_block(inputs=outputs, first_block=first_block)
                    layers.append(outputs)
        # full connection layer
        with tf.variable_scope("full connection", reuse=reuse):

            fc_bn = tf.layers.batch_normalization(inputs=outputs, axis=0, renorm_momentum=0.95)
            fc_relu = tf.nn.relu(fc_bn)
            global_pool = tf.reduce_mean(fc_relu, axis=[1, 2])
            fc_n_input = global_pool.get_shape().as_list()[-1]
            weights = tf.get_variable('fc_weight', [fc_n_input, class_nums],
                                      initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                                      regularizer=l2_regularizer(scale=self.config.fc_weight_decay))
            bias = tf.get_variable('fc_bias', shape=[class_nums], initializer=tf.zeros_initializer,
                                   regularizer=l2_regularizer(self.config.fc_weight_decay))
            net_out = tf.add(tf.matmul(global_pool, weights), bias)
            net_out = tf.identity(input=net_out)
            layers.append(outputs)
        return net_out


    def root_block(self, inputs):

        fiter0 = tf.get_variable('conv0', shape=[7, 7, 1, 1], initializer=xavier_initializer(),
                                 regularizer=l2_regularizer(self.config.conv_weight_decay))
        net = tf.nn.conv2d(input=inputs, filter=fiter0, strides=[1, 2, 2, 1], padding='SAME')
        # since the input dimension of our data is [50, 40], so the stride of pooling has been changed to 1 rather than 2
        net = tf.nn.max_pool(input=net, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")

        return net



    def residual_block(self, inputs, first_block=False):
        # 1*1 64,  3*3 64, 1*1 256
        # 1*1 128, 3*3 128 1*1 512
        # 1*1 256, 3*3 256 1*1 1024
        # 1*1 512  3*3 512 1*1 2048
        x_short = inputs
        in_channels = inputs.get_shape().as_list()[-1]

        if first_block: # block2
            out_channels = 64
        else:
            out_channels = in_channels / 2

        final_channels = out_channels * 4

        # conv + BN + Relu 1*1 64 1
        filter1 = tf.get_variable('conv1', shape=[1, 1, in_channels, out_channels], initializer=xavier_initializer(),
                                  regularizer=l2_regularizer(self.config.conv_weight_decay))

        res_conv1 = tf.nn.conv2d(inputs=inputs, filter=filter1, strides=[1, 1, 1, 1], padding='SAME')
        res_bn1 = tf.layers.batch_normalization(inputs=res_conv1, axis=0, renorm_momentum=0.95)
        res_relu1 = tf.nn.relu(res_bn1)

        # conv + BN + Relu 3*3 64 1
        filter2 = tf.get_variable('conv2', shape=[3, 3, out_channels, out_channels], initializer=xavier_initializer(),
                                  regularizer=l2_regularizer(self.config.conv_weight_decay))

        res_conv2 = tf.nn.conv2d(inputs=res_relu1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
        res_bn2 = tf.layers.batch_normalization(inputs=res_conv2, axis=0, renorm_momentum=0.95)
        res_relu2 = tf.nn.relu(res_bn2)

        # conv + BN 1*1 256 2
        filter3 = tf.get_variable('conv3', shape=[1, 1, out_channels, final_channels], initializer=xavier_initializer(),
                                  regularizer=l2_regularizer(self.conv_weight_decay))
        res_conv3 = tf.nn.conv2d(input=res_relu2, filter=filter3, strides=[1, 2, 2, 1], padding='SAME')
        res_relu3 = tf.nn.relu(res_conv3)
        out = x_short + res_relu3
        out = tf.layers.batch_normalization(inputs=out, axis=0, renorm_momentum=0.95)

        return x_short + res_relu3




