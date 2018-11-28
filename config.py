#!/usr/bin/env python
# _*_coding:utf-8 _*_

"""
# Env       : python3.6
# @Author   : Liuyinyan
# @Contact  : yinyan.liu@rokid.com
# @Site     : 
# @File     : config.py
# @Time     : 26/11/2018, 2:06 PM
"""
import os
import configparser

current_dir = os.path.abspath(os.path.dirname(__file__))

class OperationalError(Exception):
    """operation error."""

class Dictionary(dict):
    """ custom dict. """

    def __getattr__(self, key):
        return self.get(key, None)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Config:
    def __init__(self, file_name="conf", cfg=None):
        """

        :param file_name: file name without extension
        :param cfg: configuration file path

        """
        env = {}
        for key, value in os.environ.items():
            if key.startswith('Test_'):
                env[key] = value
        config = configparser.ConfigParser(env)

        if cfg:
            config.read(cfg)
        else:
            config.read(os.path.join(current_dir, 'conf', '%s.ini' % file_name))

        for section in config.sections():
            setattr(self, section, Dictionary())
            for name, raw_value in config.items(section):
                try:
                    if config.get(section, name) in ['0', '1']:
                        raise ValueError
                    value = config.getboolean(section, name)
                except ValueError:
                    try:
                        value = config.getint(section, name)
                    except ValueError:
                        value = config.get(section, name)
                setattr(getattr(self, section), name, value)

    def get(self, section):
        """

        :param section: section to fetch
        :return: option value

        """
        try:
            return getattr(self, section)
        except ArithmeticError as e:
            raise OperationalError('Option %s is not found in'
                                   'configuration, error: %s' %
                                   (section, e))
if __name__ == "__main__":
    conf = Config()

