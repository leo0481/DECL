import datetime
import numbers
import os
import sys

import numpy as np
import torch
from absl import logging
from visdom import Visdom

import config.const as const_util
import data_utils.loader as LOADER
import data_utils.transformer as TRANSFORMER
import recommender
import trainer


class ContextManager:

    def __init__(self, flags_obj):
        self.name = flags_obj.name + '_cm'
        self.exp_name = flags_obj.name
        self.output = flags_obj.output
        self.set_load_path(flags_obj)

    @staticmethod
    def set_load_path(flags_obj):
        is_windows = sys.platform.startswith('win')
        if is_windows:
            flags_obj.load_path = f'D:/pythonProjects/DICE/data/{flags_obj.dataset}/output/'
        else:
            flags_obj.load_path = f'/home/liaojie/projects/dice/data/{flags_obj.dataset}/output/'

    def set_default_ui(self):
        self.set_workspace()
        self.set_logging()

    def set_workspace(self):
        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dir_name = self.exp_name + '_' + date_time
        if not os.path.exists(self.output):
            os.mkdir(self.output)
        self.workspace = os.path.join(self.output, dir_name)
        os.mkdir(self.workspace)

    def set_logging(self):
        logging.get_absl_handler().use_absl_log_file(self.exp_name + '.log', self.workspace)

    @staticmethod
    def logging_flags(flags_obj):
        logging.info('FLAGS:')
        for flag, value in flags_obj.flag_values_dict().items():
            logging.info('{}: {}'.format(flag, value))

    @staticmethod
    def set_trainer(flags_obj, cm, vm, dm):
        if 'IPS' in flags_obj.model:
            return trainer.IPSPairTrainer(flags_obj, cm, vm, dm)
        elif 'CausE' in flags_obj.model:
            return trainer.CausETrainer(flags_obj, cm, vm, dm)
        elif 'DICE' in flags_obj.model:
            return trainer.DICETrainer(flags_obj, cm, vm, dm)
        elif 'DECL' in flags_obj.model:
            return trainer.DECLTrainer(flags_obj, cm, vm, dm)
        elif 'MACR' in flags_obj.model:
            return trainer.MACRTrainer(flags_obj, cm, vm, dm)
        else:
            return trainer.PairTrainer(flags_obj, cm, vm, dm)

    @staticmethod
    def set_recommender(flags_obj, workspace, dm):
        if flags_obj.model == 'MF':
            return recommender.MFRecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'DICE':
            return recommender.DICERecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'IPS':
            return recommender.IPSRecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'CausE':
            return recommender.CausERecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'LGN':
            return recommender.LGNRecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'LGNDICE':
            return recommender.LGNDICERecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'LGNIPS':
            return recommender.LGNIPSRecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'LGNCausE':
            return recommender.LGNCausERecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'DECL':
            return recommender.DECLRecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'MACR':
            return recommender.MACRRecommender(flags_obj, workspace, dm)

    @staticmethod
    def set_device(flags_obj):
        if not flags_obj.use_gpu:
            return torch.device('cpu')
        else:
            return torch.device('cuda:{}'.format(flags_obj.gpu_id))


class VizManager:

    def __init__(self, flags_obj):
        self.name = flags_obj.name + '_vm'
        self.exp_name = flags_obj.name
        self.port = flags_obj.port
        self.viz = Visdom(port=self.port, env=self.exp_name)

    def get_new_text_window(self, title):
        win = self.viz.text(title)
        return win

    def append_text(self, text, win):
        self.viz.text(text, win=win, append=True)

    def show_basic_info(self, flags_obj):
        basic = self.viz.text('Basic Information:')
        self.viz.text('Name: {}'.format(flags_obj.name), win=basic, append=True)
        self.viz.text('Model: {}'.format(flags_obj.model), win=basic, append=True)
        self.viz.text('Dataset: {}'.format(flags_obj.dataset), win=basic, append=True)
        self.viz.text('Embedding Size: {}'.format(flags_obj.embedding_size), win=basic, append=True)
        self.viz.text('Initial lr: {}'.format(flags_obj.lr), win=basic, append=True)
        self.viz.text('Batch Size: {}'.format(flags_obj.batch_size), win=basic, append=True)
        self.viz.text('Weight Decay: {}'.format(flags_obj.weight_decay), win=basic, append=True)
        self.viz.text('Negative Sampling Ratio: {}'.format(flags_obj.neg_sample_rate), win=basic, append=True)

        self.basic = basic

        flags = self.viz.text('FLAGS:')
        for flag, value in flags_obj.flag_values_dict().items():
            self.viz.text('{}: {}'.format(flag, value), win=flags, append=True)

        self.flags = flags

    def show_test_info(self, flags_obj):
        test = self.viz.text('Test Information:')
        self.test = test

    def step_update_line(self, title, value):
        if not isinstance(value, numbers.Number):
            return

        if not hasattr(self, title):
            setattr(self, title, self.viz.line([value], [0], opts=dict(title=title)))
            setattr(self, title + '_step', 1)
        else:
            step = getattr(self, title + '_step')
            self.viz.line([value], [step], win=getattr(self, title), update='append')
            setattr(self, title + '_step', step + 1)

    def step_update_multi_lines_singleGraph(self, title, record):
        if not hasattr(self, title):
            tmp = [[], []]
            for key, value in record.items():
                tmp[1].append(key)
                tmp[0].append(value.item())
            setattr(self, title, self.viz.line([tmp[0]], [0], opts=dict(title=title, legend=tmp[1])))
            setattr(self, title + '_step', 1)
        else:
            step = getattr(self, title + '_step')
            for key, value in record.items():
                try:
                    self.viz.line([value.item()], [step], win=getattr(self, title), update='append', name=key)
                except:
                    self.viz.line([value], [step], win=getattr(self, title), update='append', name=key)
            setattr(self, title + '_step', step + 1)

    def step_update_multi_lines(self, kv_record):
        for title, value in kv_record.items():
            self.step_update_line(title, value)

    def plot_lines(self, y, x, opts):
        title = opts['title']
        if not hasattr(self, title):
            setattr(self, title, self.viz.line(y, x, opts=opts))
        else:
            self.viz.line(y, x, win=getattr(self, title), opts=opts, update='replace')

    def show_result(self, results, topk):
        self.viz.text('-----topk = {}-----'.format(topk), win=self.test, append=True)
        self.viz.text('-----Results-----', win=self.test, append=True)

        for metric, value in results.items():
            self.viz.text('{}: {}'.format(metric, value), win=self.test, append=True)

        self.viz.text('-----------------', win=self.test, append=True)


class DatasetManager:

    def __init__(self, flags_obj):
        self.name = flags_obj.name + '_dm'
        self.make_coo_loader_transformer(flags_obj)
        self.make_npy_loader(flags_obj)
        self.make_csv_loader(flags_obj)

    def make_coo_loader_transformer(self, flags_obj):
        self.coo_loader = LOADER.CooLoader(flags_obj)
        self.coo_transformer = TRANSFORMER.SparseTransformer(flags_obj)

    def make_npy_loader(self, flags_obj):
        self.npy_loader = LOADER.NpyLoader(flags_obj)

    def make_csv_loader(self, flags_obj):
        self.csv_loader = LOADER.CsvLoader(flags_obj)

    def get_dataset_info(self):
        coo_record = self.coo_loader.load(const_util.train_coo_record)

        self.n_user = coo_record.shape[0]
        self.n_item = coo_record.shape[1]

        self.coo_record = coo_record
        print('n_user: ', self.n_user, 'n_item:', self.n_item)

    def get_skew_dataset(self):
        self.skew_coo_record = self.coo_loader.load(const_util.train_skew_coo_record)

    def get_popularity(self):
        self.popularity = self.npy_loader.load(const_util.popularity)
        return self.popularity

    def get_blend_popularity(self):
        self.blend_popularity = self.npy_loader.load(const_util.blend_popularity)
        return self.blend_popularity


class EarlyStopManager:

    def __init__(self, flags_obj):
        self.name = flags_obj.name + '_esm'
        self.min_lr = flags_obj.min_lr
        self.es_patience = flags_obj.es_patience
        self.count = 0
        self.max_metric = 0

    def step(self, lr, metric):
        if lr > self.min_lr:
            if metric > self.max_metric:
                self.max_metric = metric
            return False
        else:
            if metric > self.max_metric:
                self.max_metric = metric
                self.count = 0
                return False
            else:
                self.count = self.count + 1
                if self.count > self.es_patience:
                    return True
                return False


def setSeed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
