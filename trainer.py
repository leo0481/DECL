import time

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from absl import logging

import utils
from tester import Tester


class Trainer:

    def __init__(self, flags_obj, cm, vm, dm):
        self.name = flags_obj.name + '_trainer'
        self.cm = cm
        self.vm = vm
        self.dm = dm
        self.flags_obj = flags_obj
        self.lr = flags_obj.lr
        self.set_recommender(flags_obj, cm.workspace, dm)
        self.recommender.transfer_model()
        self.tester = Tester(flags_obj, self.recommender)

    def set_recommender(self, flags_obj, workspace, dm):
        self.recommender = utils.ContextManager.set_recommender(flags_obj, workspace, dm)

    def train(self):
        self.set_dataloader()
        self.tester.set_dataloader('val')
        self.tester.set_metrics(self.flags_obj.val_metrics)
        self.set_optimizer()
        self.set_scheduler()
        self.set_esm()
        self.set_leaderboard()

        for epoch in range(self.flags_obj.epochs):
            start_time = time.time()
            self.train_one_epoch(epoch)
            time_cost = time.time() - start_time
            self.vm.step_update_line('train time per epoch cost', time_cost)

            watch_metric_value = self.validate(epoch)

            self.scheduler.step(watch_metric_value)
            self.update_leaderboard(epoch, watch_metric_value)

            stop = self.esm.step(self.lr, watch_metric_value)
            if stop:
                break

            if self.flags_obj.adaptive:
                self.adapt_hyperparameters(epoch)

    def test(self):
        self.tester.set_dataloader('test')
        self.tester.set_metrics(self.flags_obj.metrics)

        if self.flags_obj.test_model == 'best':
            self.recommender.load_ckpt(self.max_epoch)
            logging.info('best epoch: {}'.format(self.max_epoch))

        self.vm.show_test_info(self.flags_obj)

        for topk in self.flags_obj.topk:

            self.tester.max_topk = topk
            results = self.tester.test(self.flags_obj.num_test_users)
            self.vm.show_result(results, topk)

            logging.info('TEST results topk = {}:'.format(topk))
            for metric, value in results.items():
                logging.info('{}: {}'.format(metric, value))

        self.tester.set_topk(self.flags_obj)

    def set_dataloader(self):
        raise NotImplementedError

    def set_optimizer(self):
        self.optimizer = self.recommender.get_optimizer()

    def set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                                    patience=self.flags_obj.patience,
                                                                    min_lr=self.flags_obj.min_lr)

    def set_esm(self):
        self.esm = utils.EarlyStopManager(self.flags_obj)

    def set_leaderboard(self):
        self.max_metric = -1.0
        self.max_epoch = -1
        self.leaderboard = self.vm.get_new_text_window('leaderboard')

    def update_leaderboard(self, epoch, metric):

        if metric > self.max_metric:
            self.max_metric = metric
            self.max_epoch = epoch
            self.recommender.save_ckpt()

            self.vm.append_text('New Record! {} @ epoch {}!'.format(metric, epoch), self.leaderboard)

    def adapt_hyperparameters(self, epoch):
        print("Not Implemented")
        raise NotImplementedError

    def train_one_epoch(self, epoch):
        self.lr = self.train_one_epoch_core(epoch, self.dataloader, self.optimizer, self.lr)

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr):
        start_time = time.time()
        running_loss = 0.0
        total_loss = 0.0
        num_batch = len(dataloader)
        self.distances = np.zeros(num_batch)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        for batch_count, sample in enumerate(dataloader):

            optimizer.zero_grad()

            loss = self.get_loss(sample, batch_count)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()

            if batch_count % 1000 == 0:
                self.vm.step_update_line('loss every 1k step', loss.item())

            if batch_count % (num_batch // 5) == num_batch // 5 - 1:
                logging.info('epoch {}: running loss = {}'.format(epoch, running_loss / (num_batch // 5)))
                running_loss = 0.0

        logging.info('epoch {}: total loss = {}'.format(epoch, total_loss))
        self.vm.step_update_line('epoch loss', total_loss)
        self.vm.step_update_line('distance', self.distances.mean())

        time_cost = time.time() - start_time
        self.vm.step_update_line('train time cost', time_cost)

        return lr

    def get_loss(self, sample, batch_count):
        raise NotImplementedError

    def validate(self, epoch):
        results = self.tester.test(self.flags_obj.num_val_users)
        self.vm.step_update_multi_lines(results)
        logging.info('VALIDATION epoch: {}, results: {}'.format(epoch, results))
        return results[self.flags_obj.watch_metric]


class PairTrainer(Trainer):

    def __init__(self, flags_obj, cm, vm, dm):
        super(PairTrainer, self).__init__(flags_obj, cm, vm, dm)
        self.pair_loss = self.bpr_loss

    def set_dataloader(self):
        self.dataloader = self.recommender.get_pair_dataloader()

    def get_loss(self, sample, batch_count):
        p_score, n_score = self.recommender.pair_inference(sample)

        self.distances[batch_count] = (p_score - n_score).mean().item()

        loss = self.pair_loss(p_score, n_score)

        return loss

    def bpr_loss(self, p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))


class IPSPairTrainer(PairTrainer):

    def set_dataloader(self):
        self.dataloader = self.recommender.get_ips_pair_dataloader()

    def get_loss(self, sample, batch_count):
        p_score, n_score, weight = self.recommender.pair_inference(sample)

        self.distances[batch_count] = (p_score - n_score).mean().item()

        loss = self.pair_loss(p_score, n_score, weight)

        return loss

    def bpr_loss(self, p_score, n_score, weight):
        loss = torch.log(torch.sigmoid(p_score - n_score))
        loss = loss * weight
        loss = -loss.mean()
        return loss


class CausETrainer(Trainer):

    def set_dataloader(self):
        self.dataloader = self.recommender.get_point_dataloader()

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr):
        start_time = time.time()
        running_loss = 0.0
        running_control_loss = 0.0
        running_treatment_loss = 0.0
        running_discrepency_loss = 0.0

        total_loss = 0.0
        total_control_loss = 0.0
        total_treatment_loss = 0.0
        total_discrepency_loss = 0.0

        num_batch = len(dataloader)
        self.control_distances = np.zeros(num_batch)
        self.treatment_distances = np.zeros(num_batch)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        for batch_count, sample in enumerate(dataloader):

            optimizer.zero_grad()

            loss, control_loss, treatment_loss, discrepency_loss = self.get_loss(sample, batch_count)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_control_loss += control_loss.item()
            running_treatment_loss += treatment_loss.item()
            running_discrepency_loss += discrepency_loss.item()

            total_loss += loss.item()
            total_control_loss += control_loss.item()
            total_treatment_loss += treatment_loss.item()
            total_discrepency_loss += discrepency_loss.item()

            if batch_count % 1000 == 0:
                self.vm.step_update_line('loss every 1k step', loss.item())
                self.vm.step_update_line('control loss every 1k step', control_loss.item())
                self.vm.step_update_line('treatment loss every 1k step', treatment_loss.item())
                self.vm.step_update_line('discrepency loss every 1k step', discrepency_loss.item())

            if batch_count % (num_batch // 5) == num_batch // 5 - 1:
                logging.info('epoch {}: running loss = {}'.format(epoch, running_loss / (num_batch // 5)))
                logging.info(
                    'epoch {}: running control loss = {}'.format(epoch, running_control_loss / (num_batch // 5)))
                logging.info(
                    'epoch {}: running treatment loss = {}'.format(epoch, running_treatment_loss / (num_batch // 5)))
                logging.info('epoch {}: running discrepency loss = {}'.format(epoch, running_discrepency_loss / (
                        num_batch // 5)))

                running_loss = 0.0
                running_control_loss = 0.0
                running_treatment_loss = 0.0
                running_discrepency_loss = 0.0

        logging.info('epoch {}: total loss = {}'.format(epoch, total_loss))
        logging.info('epoch {}: total control loss = {}'.format(epoch, total_control_loss))
        logging.info('epoch {}: total treatment loss = {}'.format(epoch, total_treatment_loss))
        logging.info('epoch {}: total discrepency loss = {}'.format(epoch, total_discrepency_loss))
        self.vm.step_update_line('epoch loss', total_loss)
        self.vm.step_update_line('epoch control loss', total_control_loss)
        self.vm.step_update_line('epoch treatment loss', total_treatment_loss)
        self.vm.step_update_line('epoch discrepency loss', total_discrepency_loss)
        self.vm.step_update_line('control distance', self.control_distances.mean())
        self.vm.step_update_line('treatment distance', self.treatment_distances.mean())

        time_cost = time.time() - start_time
        self.vm.step_update_line('train time cost', time_cost)
        return lr

    def get_loss(self, sample, batch_count):
        loss, control_loss, treatment_loss, discrepancy_loss, control_distance, treatment_distance = \
            self.recommender.get_loss(sample)

        self.control_distances[batch_count] = control_distance
        self.treatment_distances[batch_count] = treatment_distance

        return loss, control_loss, treatment_loss, discrepancy_loss


class DICETrainer(Trainer):

    def set_dataloader(self):
        self.dataloader = self.recommender.get_pair_dataloader()

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr):
        num_batch = len(dataloader)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        for batch_count, sample in enumerate(dataloader):
            optimizer.zero_grad()
            loss = self.get_loss(sample, batch_count)
            loss.backward()
            optimizer.step()
        return lr

    def get_loss(self, sample, batch_count):
        loss_dict = self.recommender.get_loss(sample)
        if batch_count % 1000 == 0:
            self.vm.step_update_multi_lines_singleGraph('loss', loss_dict)
        return sum(loss_dict.values())

    def adapt_hyperparameters(self, epoch):
        self.dataloader.dataset.adapt(epoch, self.flags_obj.margin_decay)
        self.recommender.adapt(epoch, self.flags_obj.loss_decay)


class DECLTrainer(Trainer):

    def __init__(self, flags_obj, cm, vm, dm):
        super(DECLTrainer, self).__init__(flags_obj, cm, vm, dm)
        # get directed edges
        graph = self.recommender.graph.remove_self_loop()
        u, v = graph.all_edges()
        length = int(u.shape[0] / 2)
        self.blend_users = np.array(u.split(length)[0].to(torch.device('cpu')))
        self.blend_items = np.array(v.split(length)[0].to(torch.device('cpu')) - self.dm.n_user)

    def set_dataloader(self):
        self.dataloader = self.recommender.get_pair_dataloader()

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr):
        self.get_masked_graphs()
        num_batch = len(dataloader)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        for batch_count, sample in enumerate(dataloader):
            optimizer.zero_grad()
            loss = self.get_loss(sample, batch_count)
            loss.backward()
            optimizer.step()

        return lr

    def get_loss(self, sample, batch_count):
        loss_dict = self.recommender.get_loss(sample)
        ssl_user, ssl_item = self.recommender.get_ssl_loss_graph(sample, self.masked_graphs)
        loss_dict['loss_ssl'] = (ssl_user + ssl_item) * self.flags_obj.ssl_weight
        ssl_dict = {'ssl_user': ssl_user, 'ssl_item': ssl_item}
        if batch_count % 1000 == 0:
            self.vm.step_update_multi_lines_singleGraph('loss', loss_dict)
            self.vm.step_update_multi_lines_singleGraph('ssl_dis', ssl_dict)

        return sum(loss_dict.values())

    def adapt_hyperparameters(self, epoch):
        self.dataloader.dataset.adapt(epoch, self.flags_obj.margin_decay)
        self.recommender.adapt(epoch, self.flags_obj.loss_decay)

    def get_masked_graphs(self):
        self.masked_graphs = []
        self.masked_graphs.append(self.get_masked_graph())

    def get_masked_graph(self):
        n_user, n_item = self.dm.n_user, self.dm.n_item
        n_node = n_user + n_item
        length = len(self.blend_users)
        keep_idx = np.random.choice(np.arange(length),
                                    size=int(length * (1 - self.flags_obj.ssl_ratio)), replace=False)
        user_np = np.array(self.blend_users)[keep_idx]
        item_np = np.array(self.blend_items)[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        # data, (row, col)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.dm.n_user)), shape=(n_node, n_node))

        adj_mat = tmp_adj + tmp_adj.T
        graph = dgl.DGLGraph(adj_mat).to(self.recommender.device)
        graph = graph.add_self_loop()
        return graph


class MACRTrainer(Trainer):

    def set_dataloader(self):
        self.dataloader = self.recommender.get_pair_dataloader()

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr):
        num_batch = len(dataloader)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        for batch_count, sample in enumerate(dataloader):
            optimizer.zero_grad()
            loss = self.get_loss(sample, batch_count)
            loss.backward()
            optimizer.step()
        return lr

    def get_loss(self, sample, batch_count):
        loss_dict = self.recommender.get_loss(sample)
        if batch_count % 1000 == 0:
            self.vm.step_update_multi_lines_singleGraph('loss', loss_dict)
        return sum(loss_dict.values())
