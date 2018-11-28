#!/usr/bin/env python
# _*_coding:utf-8 _*_

"""
# Env       : python3.6
# @Author   : Liuyinyan
# @Contact  : yinyan.liu@rokid.com
# @Site     : 
# @File     : RunTrain.py
# @Time     : 26/11/2018, 1:19 PM
"""
import os
import time
import argparse
import tensorflow as tf
from prodata.batchdata import GetData
from model.nnet.Resnet import ResNet



def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=100, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000, help='epoch to train the network')
    parser.add_argument('--batch_size', default=64, help='batch size to train network')
    parser.add_argument('--lr_steps', default=[40000, 60000, 80000], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    parser.add_argument('--eval_datasets', default=['lfw', 'cfp_fp'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--inputs_size', default=[50, 40], help='the image size')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--tfrecords_trn_lst', default='./tfdata/train.lst', type=str,
                        help='path list to the output of tfrecords file path for traindataset')
    parser.add_argument('--tfrecords_test_lst', default='./tfdata/test.lst', type=str,
                        help='path list to the output of tfrecords file path for testdataset')
    parser.add_argument('--feacfg_trn_path', default='./tfdata/tf/train/fea.cfg', type=str,
                        help='path to the feature cofiguration of traindataset')
    parser.add_argument('--feacfg_test_path', default='./tfdata/tf/test/fea.cfg', type=str,
                        help='path to the feature cofiguration of testdataset')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=100000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=5000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to show information')
    parser.add_argument('--num_gpus', default=2, help='the num of gpus')
    parser.add_argument('--tower_name', default='tower', help='tower name')
    args = parser.parse_args()
    return args

def run_all():
    class_nums = 1000
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7, 8, 9, 10, 11, 12, 13, 14, 15"
    # 1. defin global parameters
    args = get_parser()
    # inputs: [batch, time, frequency, channel=1]
    inputs = tf.placeholder(name='speech_inputs', shape=[None, *args.inputs_size, 1], dtype=tf.float32)
    labels = tf.placeholder(name='speech_labels', shape=[None,], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    # 2. prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    _fetch = GetData(buffer_size=args.buffer_size, gpu_nums=args.num_gpus, epoch=args.epoch, batch_size=args.batch_size)
    X_trn_s, Y_trn_s = _fetch.fetch_data(tf_file_lst=args.tf_trn_lst, cfg_path=args.feacfg_trn_path, cmvn=True)

    # 2.2 test datasets
    X_test_s, Y_test_s = _fetch.fetch_data(tf_file_lst=args.tf_test_lst, cfg_path=args.feacfg_test_path, cmvn=True)

    _nnet = ResNet()
    # build trn network
    net_trn_output = _nnet.nnet(is_training=True, nnet_nums=50, inputs=X_trn_s, class_nums=class_nums)
    loss_trn = tf.nn.softmax_cross_entropy_with_logits(labels=Y_trn_s, logits=net_trn_output)

    # build test network
    net_test_output = _nnet.nnet(is_training=True, nnet_nums=50, inputs=X_trn_s, class_nums=class_nums)
    loss_test = tf.nn.softmax_cross_entropy_with_logits(labels=Y_trn_s, logits=net_test_output)

    # run train and test:










