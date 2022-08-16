import numpy as np


class Judger:

    def __init__(self, flags_obj, dm, topk):
        self.name = flags_obj.name + '_judger'
        self.metrics = flags_obj.metrics
        self.dm = dm
        self.topk = topk
        self.workspace = flags_obj.workspace

    def judge(self, items, test_pos, num_test_pos):
        results = {}
        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            results[metric] = sum(
                [f(items[i], test_pos=test_pos[i], num_test_pos=num_test_pos[i].item()) if num_test_pos[i] > 0 else 0
                 for i in range(len(items))])

        valid_num_users = sum([1 if len(t) > 0 else 0 for t in test_pos])

        return results, valid_num_users


class Metrics:

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_metrics'

    @staticmethod
    def get_metrics(metric):
        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'ndcg': Metrics.ndcg,
            'pre': Metrics.pre,
            'mrr': Metrics.mrr
        }

        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):
        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()
        return hit_count / num_test_pos

    @staticmethod
    def hr(items, **kwargs):
        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def ndcg(items, **kwargs):
        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']

        index = np.arange(len(items))
        k = min(len(items), num_test_pos)
        idcg = (1 / np.log(2 + np.arange(k))).sum()
        dcg = (1 / np.log(2 + index[np.isin(items, test_pos)])).sum()
        return dcg / idcg

    @staticmethod
    def pre(items, **kwargs):
        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()
        return hit_count / len(items)

    @staticmethod
    def mrr(items, **kwargs):
        test_pos = kwargs['test_pos']
        hit = np.isin(items, test_pos)
        hit_index = np.where(hit)[0]
        if hit_index.shape[0]:
            return np.mean([1 / (hit_index[i] + 1) for i in range(hit_index.shape[0])])
        else:
            return 0
