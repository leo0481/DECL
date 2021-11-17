import sys

from absl import app
from absl import flags

import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'MF-debug', 'Experiment name.')
flags.DEFINE_enum('model', 'DECL',
                  ['MF', 'DICE', 'IPS', 'CausE', 'LGN', 'LGNDICE', 'LGNIPS', 'LGNCausE', 'DECL'],
                  'Model name.')

flags.DEFINE_integer('num_layers', 2, 'The number of layers for LGN.')
flags.DEFINE_float('dropout', 0.2, 'Dropout ratio for LGN.')
flags.DEFINE_integer('margin', 40, 'Margin for negative sampling.')
flags.DEFINE_integer('pool', 40, 'Pool for negative sampling.')
flags.DEFINE_bool('adaptive', False, 'Adapt hyper-parameters or not.')

flags.DEFINE_float('margin_decay', 0.9, 'Decay of margin and pool.')
flags.DEFINE_float('loss_decay', 0.9, 'Decay of loss.')

flags.DEFINE_enum('weighting_mode', 'nc', ['n', 'c', 'nc', 'x'], 'Mode of IPS technique.')
flags.DEFINE_float('weighting_smoothness', 1.0, 'IPS weighting smoothness.')
flags.DEFINE_enum('dataset', 'ciao', ['ml10m', 'nf', 'lastfm', 'ciao'], 'Dataset.')

flags.DEFINE_integer('embedding_size', 64, 'Embedding size for embedding based models.')
flags.DEFINE_integer('epochs', 1000, 'Max epochs for training.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('min_lr', 0.0001, 'Minimum learning rate.')
flags.DEFINE_float('weight_decay', 5e-8, 'Weight decay.')
flags.DEFINE_integer('batch_size', 128, 'Batch Size.')
flags.DEFINE_enum('dis_loss', 'dcor', ['L1', 'L2', 'dcor'], 'Discrepency loss function.')
flags.DEFINE_float('dis_pen', 0.01, 'Discrepancy penalty.')
flags.DEFINE_float('int_weight', 0.1, 'Weight for interest term.')
flags.DEFINE_float('pop_weight', 0.1, 'Weight for popularity term.')

flags.DEFINE_float('ssl_weight', 0.0001, 'Weight for ssl term.')
flags.DEFINE_float('ssl_ratio', 0.5, 'Probability of ssl data augmentation.')
flags.DEFINE_float('ssl_temp', 0.5, 'Temperature of ssl.')

flags.DEFINE_integer('neg_sample_rate', 4, 'Negative Sampling Ratio.')
flags.DEFINE_bool('shuffle', True, 'Shuffle the training set or not.')
flags.DEFINE_multi_string('metrics', ['recall', 'hit_ratio', 'ndcg', 'mrr'], 'Metrics.')
flags.DEFINE_multi_string('val_metrics', ['recall'], 'Metrics.')
flags.DEFINE_string('watch_metric', 'recall', 'Which metric to decide learning rate reduction.')
flags.DEFINE_integer('patience', 5, 'Patience for reducing learning rate.')
flags.DEFINE_integer('es_patience', 3, 'Patience for early stop.')
flags.DEFINE_integer('num_val_users', 1000000, 'Number of users for validation.')
flags.DEFINE_integer('num_test_users', 1000000, 'Number of users for test.')
flags.DEFINE_enum('test_model', 'best', ['best', 'last'], 'Which model to test.')
flags.DEFINE_multi_integer('topk', [5, 10, 20, 50], 'Topk for testing recommendation performance.')
flags.DEFINE_integer('num_workers', 8, 'Number of processes for training and testing.')
flags.DEFINE_string('load_path', '', 'Load path. Set in ContextManager')
flags.DEFINE_string('workspace', '', 'Path to load ckpt.')
flags.DEFINE_integer('port', 33336, 'Port to show visualization results.')

is_windows = sys.platform.startswith('win')
if is_windows:
    flags.DEFINE_string('output', 'D:/pythonProjects/DECL/output', 'Directory to save model/log/metrics.')
    flags.DEFINE_bool('use_gpu', False, 'Use GPU or not.')
    flags.DEFINE_integer('gpu_id', 1, 'GPU ID.')
    flags.DEFINE_bool('cg_use_gpu', False, 'Use GPU or not for candidate generation.')
    flags.DEFINE_integer('cg_gpu_id', 1, 'GPU ID for candidate generation.')
else:
    flags.DEFINE_string('output', '/home/liaojie/projects/decl/output/', 'Directory to save model/log/metrics.')
    flags.DEFINE_bool('use_gpu', True, 'Use GPU or not.')
    flags.DEFINE_integer('gpu_id', 1, 'GPU ID.')
    flags.DEFINE_bool('cg_use_gpu', True, 'Use GPU or not for candidate generation.')
    flags.DEFINE_integer('cg_gpu_id', 1, 'GPU ID for candidate generation.')


def main(argv):
    utils.setSeed(2021)
    flags_obj = FLAGS
    cm = utils.ContextManager(flags_obj)
    vm = utils.VizManager(flags_obj)
    dm = utils.DatasetManager(flags_obj)
    dm.get_dataset_info()

    cm.set_default_ui()
    cm.logging_flags(flags_obj)
    vm.show_basic_info(flags_obj)
    trainer = utils.ContextManager.set_trainer(flags_obj, cm, vm, dm)
    trainer.train()

    trainer.test()


if __name__ == "__main__":
    app.run(main)
