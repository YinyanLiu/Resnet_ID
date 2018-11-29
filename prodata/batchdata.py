#!/usr/bin/env python
# _*_coding:utf-8 _*_

"""
# Env       : python3.6
# @Author   : Liuyinyan
# @Contact  : yinyan.liu@rokid.com
# @Site     : 
# @File     : batchdata.py
# @Time     : 26/11/2018, 12:57 PM
"""

import tensorflow as tf
from prodata.fea_cfg import FeaCfg
class GetData(object):
    def __init__(self, gpu_nums, buffer_size, epoch, batch_size):
        self.gpu_nums = gpu_nums
        self.buffer_size = buffer_size
        self.epoch = epoch
        self.batch_size = batch_size

    def fetch_data(self, tf_file_lst, cfg_path, cmvn=True):
        self.tf_file_lst = tf_file_lst
        next_element = self.dataset()
        feas = next_element['fea']
        labels = next_element['label']

        if cmvn is True:
            self.feas_cfg = FeaCfg()
            self.feas_cfg.readfrom(cfg_path=cfg_path)
            feas = (feas - self.feas_cfg._fea_mean) / self.feas_cfg._fea_var

        feas_split = tf.split(feas, self.gpu_nums)
        labels_split = tf.split(labels, self.gpu_nums)

        return feas_split, labels_split


    def dataset(self):

        tfrecord_path = self.get_tf_path()
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        new_dataset = dataset.map(self._parse_function)
        shuffle_dataset = new_dataset.shuffle(buffer_size=self.buffer_size)
        repeat_dataset = shuffle_dataset.repeat(self.epoch)
        prefetch_dataset = repeat_dataset.prefetch(2000)
        batch_dataset = prefetch_dataset.padded_batch(self.batch_size, padded_shapes={'fea': [None, None], 'fea_shape': [None], 'label': [None]})
        iterator = batch_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        return next_element


    def _parse_function(self, example_proto):
        dics = {
            'fea': tf.VarLenFeature(dtype=tf.float32),
            'fea_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'label': tf.VarLenFeature(dtype=tf.float32)}

        parsed_example = tf.parse_single_example(example_proto, dics)
        parsed_example['fea'] = tf.sparse_tensor_to_dense(parsed_example['fea'])
        parsed_example['label'] = tf.sparse_tensor_to_dense(parsed_example['label'])
        parsed_example['label'] = tf.cast(parsed_example['label'], tf.int32)
        parsed_example['fea'] = tf.reshape(parsed_example['fea'], parsed_example['fea_shape'])
        return parsed_example

    def get_tf_path(self):
        tf_files = []
        with open(self.tf_file_lst) as tflis:
            for line in tflis:
                line = line.strip('\n')
                tf_files.append(line)
        return  tf_files