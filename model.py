import math

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):
        super(MF, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.init_params()

    def init_params(self):
        stdv = 1. / math.sqrt(self.users.size()[1])
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)

    def pair_forward(self, user, item_p, item_n):
        user = self.users[user]
        item_p = self.items[item_p]
        item_n = self.items[item_n]

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def point_forward(self, user, item):
        user = self.users[user]
        item = self.items[item]

        score = torch.sum(user * item, 2)

        return score

    def get_item_embeddings(self):
        return self.items.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):
        return self.users.detach().cpu().numpy().astype('float32')


class CausE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):
        super(CausE, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_control = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.items_treatment = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.criterion_factual = nn.BCEWithLogitsLoss()
        self.criterion_counterfactual = nn.MSELoss()

        self.init_params()

    def init_params(self):
        stdv = 1. / math.sqrt(self.users.size()[1])
        self.users.data.uniform_(-stdv, stdv)
        self.items_control.data.uniform_(-stdv, stdv)
        self.items_treatment.data.uniform_(-stdv, stdv)

    def forward(self, user, item, label, mask):
        user_control = self.users[user[~mask]]
        item_control = self.items_control[item[~mask]]
        score_control = torch.sum(user_control * item_control, 2)
        label_control = label[~mask]
        control_loss = self.criterion_factual(score_control, label_control)

        control_distance = (torch.sigmoid(score_control) - label_control).abs().mean().item()

        user_treatment = self.users[user[mask]]
        item_treatment = self.items_treatment[item[mask]]
        score_treatment = torch.sum(user_treatment * item_treatment, 2)
        label_treatment = label[mask]
        treatment_loss = self.criterion_factual(score_treatment, label_treatment)

        treatment_distance = (torch.sigmoid(score_treatment) - label_treatment).abs().mean().item()

        item_all = torch.unique(item)
        item_control_factual = self.items_control[item_all]
        item_control_counterfactual = self.items_treatment[item_all]
        discrepancy_loss = self.criterion_counterfactual(item_control_factual, item_control_counterfactual)

        return control_loss, treatment_loss, discrepancy_loss, control_distance, treatment_distance

    def get_item_control_embeddings(self):
        return self.items_control.detach().cpu().numpy().astype('float32')

    def get_item_treatment_embeddings(self):
        return self.items_treatment.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):
        return self.users.detach().cpu().numpy().astype('float32')


class LGNCausE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout):

        super(LGNCausE, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.embeddings_control = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.embeddings_treatment = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers_control = nn.ModuleList()
        for _ in range(num_layers):
            self.layers_control.append(LGConv(embedding_size, embedding_size, 1))

        self.layers_treatment = nn.ModuleList()
        for _ in range(num_layers):
            self.layers_treatment.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        self.criterion_factual = nn.BCEWithLogitsLoss()
        self.criterion_counterfactual = nn.MSELoss()

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings_control.size()[1])
        self.embeddings_control.data.uniform_(-stdv, stdv)
        self.embeddings_treatment.data.uniform_(-stdv, stdv)

    def forward(self, user, item, label, mask, graph_control, graph_treatment, training=True):

        features_control = [self.embeddings_control]
        h_control = self.embeddings_control
        for layer in self.layers_control:
            h_control = layer(graph_control, h_control)
            h_control = F.dropout(h_control, p=self.dropout, training=training)
            features_control.append(h_control)

        features_control = torch.stack(features_control, dim=2)
        features_control = torch.mean(features_control, dim=2)

        features_treatment = [self.embeddings_treatment]
        h_treatment = self.embeddings_treatment
        for layer in self.layers_treatment:
            h_treatment = layer(graph_treatment, h_treatment)
            h_treatment = F.dropout(h_treatment, p=self.dropout, training=training)
            features_treatment.append(h_treatment)

        features_treatment = torch.stack(features_treatment, dim=2)
        features_treatment = torch.mean(features_treatment, dim=2)

        item = item + self.n_user

        user_control = features_control[user[~mask]]
        item_control = features_control[item[~mask]]
        score_control = torch.sum(user_control * item_control, 2)
        label_control = label[~mask]
        control_loss = self.criterion_factual(score_control, label_control)

        control_distance = (torch.sigmoid(score_control) - label_control).abs().mean().item()

        user_treatment = features_treatment[user[mask]]
        item_treatment = features_treatment[item[mask]]
        score_treatment = torch.sum(user_treatment * item_treatment, 2)
        label_treatment = label[mask]
        treatment_loss = self.criterion_factual(score_treatment, label_treatment)

        treatment_distance = (torch.sigmoid(score_treatment) - label_treatment).abs().mean().item()

        user_control_factual = features_control[user]
        user_control_counterfactual = features_treatment[user]
        item_control_factual = features_control[item]
        item_control_counterfactual = features_treatment[item]
        discrepancy_loss = self.criterion_counterfactual(user_control_factual,
                                                         user_control_counterfactual) + self.criterion_counterfactual(
            item_control_factual, item_control_counterfactual)

        return {"control_loss": control_loss, "treatment_loss": treatment_loss, "discrepancy_loss": discrepancy_loss,
                "control_distance": control_distance, "treatment_distance": treatment_distance}

    def get_control_embeddings(self, graph):

        features = [self.embeddings_control]
        h = self.embeddings_control
        for layer in self.layers_control:
            h = layer(graph, h)
            features.append(h)

        features = torch.stack(features, dim=2)
        features = torch.mean(features, dim=2)

        users = features[:self.n_user]
        items = features[self.n_user:]

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')


class LGConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None):
        super(LGConv, self).__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm

    def forward(self, graph, feat):

        graph = graph.local_var()
        if self._cached_h is not None:
            feat = self._cached_h
        else:
            # compute normalization
            degrees = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degrees, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)
            # compute (D^-(1/2) A D^-(1/2))^k X
            for _ in range(self._k):
                # pytorch broadcast, equal to D^(-1/2) \dot A
                feat = feat * norm
                graph.ndata['h'] = feat
                # aggregate
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm

            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat

        return feat


class LGN(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout):

        super(LGN, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.embeddings = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings.size()[1])
        self.embeddings.data.uniform_(-stdv, stdv)

    def pair_forward(self, user, item_p, item_n, graph, training=True):

        features = [self.embeddings]
        h = self.embeddings
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features.append(h)

        features = torch.stack(features, dim=2)
        features = torch.mean(features, dim=2)

        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        user = features[user]
        item_p = features[item_p]
        item_n = features[item_n]

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def get_embeddings(self, graph):

        features = [self.embeddings]
        h = self.embeddings
        for layer in self.layers:
            h = layer(graph, h)
            features.append(h)

        features = torch.stack(features, dim=2)
        features = torch.mean(features, dim=2)

        users = features[:self.n_user]
        items = features[self.n_user:]

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')


class DICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, dis_loss, dis_pen, int_weight, pop_weight):

        super(DICE, self).__init__()
        self.users_int = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.users_pop = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_int = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.items_pop = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.int_weight = int_weight
        self.pop_weight = pop_weight

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.dis_pen = dis_pen

        self.init_params()

    def adapt(self, epoch, decay):
        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay

    def dcor(self, x, y):
        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(y[:, None] - y, p=2, dim=2)

        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):
        stdv = 1. / math.sqrt(self.users_int.size()[1])
        self.users_int.data.uniform_(-stdv, stdv)
        self.users_pop.data.uniform_(-stdv, stdv)
        self.items_int.data.uniform_(-stdv, stdv)
        self.items_pop.data.uniform_(-stdv, stdv)

    def bpr_loss(self, p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask):
        users_int = self.users_int[user]
        users_pop = self.users_pop[user]
        items_p_int = self.items_int[item_p]
        items_p_pop = self.items_pop[item_p]
        items_n_int = self.items_int[item_n]
        items_n_pop = self.items_pop[item_n]

        p_score_int = torch.sum(users_int * items_p_int, 2)
        n_score_int = torch.sum(users_int * items_n_int, 2)

        p_score_pop = torch.sum(users_pop * items_p_pop, 2)
        n_score_pop = torch.sum(users_pop * items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop,
                                                                                           ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = self.items_int[item_all]
        item_pop = self.items_pop[item_all]
        user_all = torch.unique(user)
        user_int = self.users_int[user_all]
        user_pop = self.users_pop[user_all]
        discrepancy_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int,
                                                                                                       user_pop)

        return {"loss_total": loss_total, "loss_int": self.int_weight * loss_int,
                "loss_pop": self.pop_weight * loss_pop, "discrepancy_loss": - self.dis_pen * discrepancy_loss}

    def get_item_embeddings(self):

        item_embeddings = torch.cat((self.items_int, self.items_pop), 1)
        # item_embeddings = self.items_pop
        return item_embeddings.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        user_embeddings = torch.cat((self.users_int, self.users_pop), 1)
        # user_embeddings = self.users_pop
        return user_embeddings.detach().cpu().numpy().astype('float32')


class LGNDICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout, dis_loss, dis_pen, int_weight,
                 pop_weight):
        super(LGNDICE, self).__init__()
        self.n_user = num_users
        self.n_item = num_items

        self.int_weight = int_weight
        self.pop_weight = pop_weight

        self.embeddings_int = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.embeddings_pop = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.dis_pen = dis_pen

        self.init_params()

    @staticmethod
    def dcor(x, y):

        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(y[:, None] - y, p=2, dim=2)

        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):
        stdv = 1. / math.sqrt(self.embeddings_int.size()[1])
        self.embeddings_int.data.uniform_(-stdv, stdv)
        self.embeddings_pop.data.uniform_(-stdv, stdv)

    def adapt(self, epoch, decay):
        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay

    @staticmethod
    def bpr_loss(p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    @staticmethod
    def mask_bpr_loss(p_score, n_score, mask):
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask, graph, training=True):
        features_int, features_pop = self.dice(graph, training)
        return self.loss(features_int, features_pop, user, item_p, item_n, mask)

    def dice(self, graph, training=True):
        features_int = [self.embeddings_int]
        h = self.embeddings_int
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_int.append(h)

        features_int = torch.stack(features_int, dim=2)
        features_int = torch.mean(features_int, dim=2)

        features_pop = [self.embeddings_pop]
        h = self.embeddings_pop
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_pop.append(h)

        features_pop = torch.stack(features_pop, dim=2)
        features_pop = torch.mean(features_pop, dim=2)
        return features_int, features_pop

    def loss(self, features_int, features_pop, user, item_p, item_n, mask):
        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        users_int = features_int[user]
        users_pop = features_pop[user]
        items_p_int = features_int[item_p]
        items_p_pop = features_pop[item_p]
        items_n_int = features_int[item_n]
        items_n_pop = features_pop[item_n]

        p_score_int = torch.sum(users_int * items_p_int, 2)
        n_score_int = torch.sum(users_int * items_n_int, 2)

        p_score_pop = torch.sum(users_pop * items_p_pop, 2)
        n_score_pop = torch.sum(users_pop * items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop,
                                                                                           ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = features_int[item_all]
        item_pop = features_pop[item_all]
        user_all = torch.unique(user)
        user_int = features_int[user_all]
        user_pop = features_pop[user_all]
        discrepancy_loss = \
            self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

        return {"loss_total": loss_total, "loss_int": self.int_weight * loss_int,
                "loss_pop": self.pop_weight * loss_pop, "discrepancy_loss": - self.dis_pen * discrepancy_loss}

    def get_embeddings(self, graph):
        features_int = [self.embeddings_int]
        h = self.embeddings_int
        for layer in self.layers:
            h = layer(graph, h)
            features_int.append(h)

        features_int = torch.stack(features_int, dim=2)
        features_int = torch.mean(features_int, dim=2)

        users_int = features_int[:self.n_user]
        items_int = features_int[self.n_user:]

        features_pop = [self.embeddings_pop]
        h = self.embeddings_pop
        for layer in self.layers:
            h = layer(graph, h)
            features_pop.append(h)

        features_pop = torch.stack(features_pop, dim=2)
        features_pop = torch.mean(features_pop, dim=2)
        users_pop = features_pop[:self.n_user]
        items_pop = features_pop[self.n_user:]

        items = torch.cat((items_int, items_pop), 1)
        users = torch.cat((users_int, users_pop), 1)

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')


class DECL(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout, dis_loss, dis_pen, int_weight,
                 pop_weight, ssl_weight, ssl_temp):
        super(DECL, self).__init__()
        self.n_user = num_users
        self.n_item = num_items

        self.dis_pen = dis_pen
        self.int_weight = int_weight
        self.pop_weight = pop_weight
        self.ssl_weight = ssl_weight
        self.ssl_temp = ssl_temp
        self.embeddings_int = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.embeddings_pop = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.init_params()

    @staticmethod
    def dcor(x, y):

        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(y[:, None] - y, p=2, dim=2)

        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):
        stdv = 1. / math.sqrt(self.embeddings_int.size()[1])
        self.embeddings_int.data.uniform_(-stdv, stdv)
        self.embeddings_pop.data.uniform_(-stdv, stdv)

    def adapt(self, epoch, decay):
        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay
        self.ssl_weight = self.ssl_weight * decay

    @staticmethod
    def bpr_loss(p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    @staticmethod
    def mask_bpr_loss(p_score, n_score, mask):
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask, graph):
        self.features_int, self.features_pop = self.dice(graph)
        loss_dict = self.loss(user, item_p, item_n, mask)
        return loss_dict

    def dice(self, graph, training=True):
        features_int = [self.embeddings_int]
        h = self.embeddings_int
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_int.append(h)
        features_int = torch.stack(features_int, dim=2)
        features_int = torch.mean(features_int, dim=2)

        features_pop = [self.embeddings_pop]
        h = self.embeddings_pop
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_pop.append(h)
        features_pop = torch.stack(features_pop, dim=2)
        features_pop = torch.mean(features_pop, dim=2)
        return features_int, features_pop

    def loss(self, user, item_p, item_n, mask):
        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        users_int = self.features_int[user]
        users_pop = self.features_pop[user]
        items_p_int = self.features_int[item_p]
        items_p_pop = self.features_pop[item_p]
        items_n_int = self.features_int[item_n]
        items_n_pop = self.features_pop[item_n]

        p_score_int = torch.sum(users_int * items_p_int, 2)
        n_score_int = torch.sum(users_int * items_n_int, 2)

        p_score_pop = torch.sum(users_pop * items_p_pop, 2)
        n_score_pop = torch.sum(users_pop * items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop,
                                                                                           ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = self.features_int[item_all]
        item_pop = self.features_pop[item_all]
        user_all = torch.unique(user)
        user_int = self.features_int[user_all]
        user_pop = self.features_pop[user_all]
        discrepancy_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int,
                                                                                                       user_pop)

        return {'loss_total': loss_total, 'loss_int': self.int_weight * loss_int,
                'loss_pop': self.pop_weight * loss_pop, 'loss_dis': - self.dis_pen * discrepancy_loss}

    def get_embeddings(self, graph):
        features_int = [self.embeddings_int]
        h = self.embeddings_int
        for layer in self.layers:
            h = layer(graph, h)
            features_int.append(h)

        features_int = torch.stack(features_int, dim=2)
        features_int = torch.mean(features_int, dim=2)

        users_int = features_int[:self.n_user]
        items_int = features_int[self.n_user:]

        features_pop = [self.embeddings_pop]
        h = self.embeddings_pop
        for layer in self.layers:
            h = layer(graph, h)
            features_pop.append(h)

        features_pop = torch.stack(features_pop, dim=2)
        features_pop = torch.mean(features_pop, dim=2)
        users_pop = features_pop[:self.n_user]
        items_pop = features_pop[self.n_user:]

        items = torch.cat((items_int, items_pop), 1)
        users = torch.cat((users_int, users_pop), 1)

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')

    def get_ssl_loss_graph(self, users, items, masked_graphs, graph):
        features_aug = self.get_aug_embedding(masked_graphs)
        items = torch.unique(items) + self.n_user
        users = torch.unique(users)

        masked_users_int = features_aug[users]
        masked_items = features_aug[items]
        users_int = self.features_int[users]
        items_int = self.features_int[items]
        users_pop = self.features_pop[users]
        items_pop = self.features_pop[items]

        normalize_masked_users_int = F.normalize(masked_users_int)
        normalize_masked_items_int = F.normalize(masked_items)
        normalize_users_int = F.normalize(users_int)
        normalize_items_int = F.normalize(items_int)
        normalize_users_pop = F.normalize(users_pop)
        normalize_items_pop = F.normalize(items_pop)

        pos_score_user = torch.exp(torch.sum(normalize_masked_users_int * normalize_users_int, 1) / self.ssl_temp)
        neg_score_user = torch.exp(torch.sum(normalize_users_pop * normalize_users_int, 1) / self.ssl_temp)
        ssl_user = max(pos_score_user.norm(2) - neg_score_user.norm(2) + 1e-8, 0)

        pos_score_item = torch.exp(torch.sum(normalize_masked_items_int * normalize_items_int, 1) / self.ssl_temp)
        neg_score_item = torch.exp(torch.sum(normalize_items_pop * normalize_items_int, 1) / self.ssl_temp)
        ssl_item = max(pos_score_item.norm(2) - neg_score_item.norm(2) + 1e-8, 0)
        return ssl_user, ssl_item

    def get_aug_embedding(self, masked_graphs):
        features_aug = [self.embeddings_int]
        h1 = self.embeddings_int
        for i in range(len(self.layers)):
            h1 = self.layers[i](masked_graphs[0], h1)
            features_aug.append(h1)
        features_aug = torch.stack(features_aug, dim=2)
        features_aug = torch.mean(features_aug, dim=2)
        return features_aug


class MACR(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout, batch_size, alpha, beta):
        super(MACR, self).__init__()
        self.n_user = num_users
        self.n_item = num_items

        self.embeddings_user = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.embeddings_item = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.user_branch = nn.Linear(embedding_size, 1)
        self.item_branch = nn.Linear(embedding_size, 1)

        self.dropout = dropout
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.l2_loss = nn.MSELoss()

        self.init_params()

    def init_params(self):
        stdv = 1. / math.sqrt(self.embeddings_user.size()[1])
        self.embeddings_user.data.uniform_(-stdv, stdv)
        self.embeddings_item.data.uniform_(-stdv, stdv)
        self.user_branch.data.uniform_(-stdv, stdv)
        self.item_branch.data.uniform_(-stdv, stdv)

    @staticmethod
    def bce_loss(y, y_hat):
        return - torch.mean(y * torch.log(torch.sigmoid(y_hat)) + (1 - y) * torch.log(1-torch.sigmoid(y_hat)))

    def forward(self, user, item_p, item_n, graph):
        self.users_feat, self.items_feat = self.get_feature(graph)
        loss_dict = self.loss(user, item_p, item_n)
        return None

    def loss(self, user, item_p, item_n):
        user_feat = self.users_feat[user]
        item_p_feat = self.items_feat[item_p]
        item_n_feat = self.items_feat[item_n]
        pos_scores = torch.mean(user_feat * item_p_feat, dim=1)
        neg_scores = torch.mean(user_feat * item_n_feat, dim=1)
        pos_item_scores = self.item_branch(item_p_feat)
        neg_item_scores = self.item_branch(item_n_feat)
        user_scores = self.user_branch(user_feat)
        pos_scores = pos_scores * torch.sigmoid(pos_item_scores) * torch.sigmoid(user_scores)
        neg_scores = neg_scores * torch.sigmoid(neg_item_scores) * torch.sigmoid(user_scores)
        mf_loss_ori = - torch.mean((torch.log(torch.sigmoid(pos_scores) + 1e-10)) +
                                   (torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)))
        mf_loss_item = - torch.mean((torch.log(torch.sigmoid(pos_item_scores) + 1e-10)) +
                                   (torch.log(1 - torch.sigmoid(neg_item_scores) + 1e-10)))
        mf_loss_user = - torch.mean(torch.log(torch.sigmoid(user_scores) + 1e-10) +
                                    torch.log(1 - torch.sigmoid(user_scores) + 1e-10))
        mf_loss = mf_loss_ori + self.alpha * mf_loss_item + self.beta * mf_loss_user
        return {"mf_loss":mf_loss}

    def get_feature(self, graph, training=True):
        users_feat = [self.embeddings_user]
        items_feat = [self.embeddings_item]
        hu, hi = self.embeddings_user, self.embeddings_item
        for layer in self.layers:
            hu_ego = hu
            hu = layer(graph, hi)
            hi = layer(graph, hu_ego)
            hu = F.dropout(hu, p=self.dropout, training=training)
            hi = F.dropout(hi, p=self.dropout, training=training)
            users_feat.append(hu)
            items_feat.append(hi)
        users_feat = torch.stack(users_feat, dim=2)
        users_feat = torch.mean(users_feat, dim=2)
        items_feat = torch.stack(items_feat, dim=2)
        items_feat = torch.mean(items_feat, dim=2)
        return users_feat, items_feat

    def get_embedding(self):
        pass

