#!/usr/bin/env python
# _*_coding:utf-8 _*_

"""
# Env       : python3.6
# @Author   : Liuyinyan
# @Contact  : yinyan.liu@rokid.com
# @Site     : 
# @File     : nnet.py
# @Time     : 28/11/2018, 9:04 PM
"""
import tensorflow as tf

class NNet(object):
    def __init__(self, gpu_nums, net_name, class_nums, batch_size, fea_dim):
        self.gpu_nums = gpu_nums
        self.net_name = net_name
        self.class_nums = class_nums + 1
        self.batch_size = batch_size
        self.fea_dim = fea_dim

    def build_resnet(self, is_training, inputs):
        raise NotImplementedError("Abstract method")

    def build_trn_graph(self, inputs, labels,learning_rate):
        print('Building Train Graph ...')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # build dnn
        tower_grads = []
        self.loss_sum = tf.convert_to_tensor(0.0, dtype=tf.float32)
        for i in range(self.gpu_nums):
            x_input = tf.reshape(tensor=inputs[i], shape=[self.batch_size//self.gpu_nums, -1, self.fea_dim])
            y_label = tf.one_hot(indices=labels[i], depth=self.class_nums)

            with tf.device('/gpu:%d' % i):
                logit = self.build_resnet(is_training=True, inputs=x_input)
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=logit)
                loss_mean = tf.reduce_mean(input_tensor=loss, axis=0)
                self.loss_sum += loss_mean

                grad = optimizer.compute_gradients(loss_mean)
                tower_grads.append(grad)
        with tf.device('/gpu:0'):
            self.loss_sum = self.loss_sum / self.gpu_nums
            print('merge gradients ...')
            grads_and_vars = self.average_gradients(tower_grads=tower_grads)

            print('minimize loss ...')
            self.trn_opt = optimizer.apply_gradients(grads_and_vars)
        return self.loss_sum, self.trn_opt

    def run_trn_graph(self, sess):
        trn_loss = 0
        print('Do Training ... ')
        trn_step_count1 = 10000000//self.batch_size

        for trn_step in range(trn_step_count1):
            _, loss_sum = sess.run([self.trn_opt, self.loss_sum])

            trn_loss += loss_sum

            if (trn_step + 1) % 10 == 0:
                print(' %d /  %d: loss: %0.5f' %(trn_step+1, trn_step_count1, trn_loss / (trn_step+1)))

        return trn_loss / trn_step_count1

    def build_test_graph(self, inputs, labels):
         print('Building Test Graph ...')
         self.test_loss_sum = tf.convert_to_tensor(0.0, dtype=tf.float32)

         for i in range(self.gpu_nums):
             x_input = tf.reshape(tensor=inputs[i], shape=[self.batch_size // self.gpu_nums, -1, self.fea_dim])
             y_label = tf.one_hot(indices=labels[i], depth=self.class_nums)
             with tf.device('/gpu:%d' % i):
                 logits = self.build_resnet(is_training=False, inputs=x_input)
                 loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=logits)
                 loss_mean = tf.reduce_mean(loss, axis=0)
                 self.test_loss_sum +=loss_mean

         with tf.device('/gpu:0'):
             self.test_loss_sum  = self.test_loss_sum / self.gpu_nums

         return self.test_loss_sum

    def run_test_graph(self, sess):
        test_loss = 0

        cv_step_count = 10000//self.batch_size
        for cv_step in range(cv_step_count):
            test_loss_sum = sess.run(self.test_loss_sum)
            test_loss += test_loss_sum

            if (cv_step +1) % 100 == 0:
                print(' %d /  %d: loss: %0.5f' % (cv_step + 1, cv_step_count, test_loss / (cv_step + 1)))
        return test_loss / cv_step_count


    def average_gradients(self, tower_grads):
        avg_grads = []
        # list all the gradient obtained from different GPU
        # grad_and_vars represents gradient of w1, b1, w2, b2 of different gpu respectively
        for grad_and_vars in zip(*tower_grads):  # w1, b1, w2, b2
            # calculate average gradients
            # print('grad_and_vars: ', grad_and_vars)
            grads = []
            for g, _ in grad_and_vars:  # different gpu
                expanded_g = tf.expand_dims(g, 0)  # expand one dimension (5, 10) to (1, 5, 10)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)  # for 4 gpu, 4 (1, 5, 10) will be (4, 5, 10),concat the first dimension
            grad = tf.reduce_mean(grad, 0)  # calculate average by the first dimension
            # print('grad: ', grad)

            v = grad_and_vars[0][1]  # get w1 and then b1, and then w2, then b2, why?
            # print('v',v)
            grad_and_var = (grad, v)
            # print('grad_and_var: ', grad_and_var)
            # corresponding variables and gradients
            avg_grads.append(grad_and_var)
        return avg_grads








