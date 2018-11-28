#!/usr/bin/env python
# _*_coding:utf-8 _*_

"""
# Env       : python3.6
# @Author   : Liuyinyan
# @Contact  : yinyan.liu@rokid.com
# @Site     : 
# @File     : fea_cfg.py
# @Time     : 13/11/2018, 5:00 PM
"""
import numpy as np
class FeaCfg(object):

    def setcfg(self, cfg_path, fea_dim, utt_num, words_size, fea_num_total, fea_mean, fea_var):
        self.cfg_path = cfg_path
        self._fea_dim = fea_dim
        self._utt_num = utt_num
        self._words_size = words_size
        self._fea_num_total = fea_num_total
        self._fea_mean = np.asarray(fea_mean, dtype=np.float32)
        self._fea_var = np.asarray(fea_var, dtype=np.float32)

    def saveto(self):

        with open(self.cfg_path, "w") as cfg_file:
            cfg_file.write("Fea_Dim %d\n" % self._fea_dim)
            cfg_file.write('Utt_Num %d\n' % self._utt_num)
            cfg_file.write('Words_Size %d\n' % self._words_size)
            cfg_file.write("Fea_Num_Total %d\n" % self._fea_num_total)
            cfg_file.write("Fea_Mean")
            for x in self._fea_mean:
                cfg_file.write(" %0.7f" % x)
            cfg_file.write("\n")

            cfg_file.write("Fea_Var")
            for x in self._fea_var:
                cfg_file.write(" %0.7f" % x)
            cfg_file.write("\n")

    def readfrom(self, cfg_path):
        with open(cfg_path, "r") as cfg_file:
            for line in cfg_file:
                data = line.strip().split(' ')
                if data[0] == 'Fea_Dim':
                    self._fea_dim = int(data[1])
                if data[0] == 'Utt_Num':
                    self._utt_num = int(data[1])
                if data[0] == 'Words_Size':
                    self._words_size = int(data[1])
                if data[0] == 'Fea_Num_Total':
                    self._fea_num_total = int(data[1])
                if data[0] == 'Fea_Mean':
                    self._fea_mean = np.asarray(list(map(float, data[1:])), dtype=np.float32)
                if data[0] == 'Fea_Var':
                    self._fea_var = np.asarray(list(map(float, data[1:])), dtype=np.float32)
        return self._fea_dim, self._utt_num, self._words_size, self._fea_mean, self._fea_var
