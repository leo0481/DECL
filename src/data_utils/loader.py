#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import pandas as pd
import scipy.sparse as sp

import json

import os


class Loader:

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_loader'
        self.load_path = flags_obj.load_path
        if not os.path.exists(self.load_path):
            print('Error! Load path ({}) does not exist!'.format(self.load_path))
    
    def load(self, filename, **kwargs):
        raise NotImplementedError


class CsvLoader(Loader):
    
    def load(self, filename, **kwargs):
        filename = os.path.join(self.load_path, filename)
        record = pd.read_csv(filename, **kwargs)
        return record


class CooLoader(Loader):
    
    def load(self, filename, **kwargs):
        filename = os.path.join(self.load_path, filename)
        record = sp.load_npz(filename)
        return record


class JsonLoader(Loader):
    
    def load(self, filename, **kwargs):
        filename = os.path.join(self.load_path, filename)
        with open(filename, 'r') as f:
            record = json.loads(f.read())
        return record


class NpyLoader(Loader):
    
    def load(self, filename, **kwargs):
        filename = os.path.join(self.load_path, filename)
        record = np.load(filename)
        return record
