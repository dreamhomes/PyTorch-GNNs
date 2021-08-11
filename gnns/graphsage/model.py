import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.autograd import Variable

from gnns.graphsage.aggregators import MeanAggregator
from gnns.graphsage.encoders import Encoder


class SupervisedGraphSage(nn.Module):
    """Simple supervised GraphSAGE model on Cora and Pubmed datasets."""

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("data/cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("data/cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(
        lambda nodes: enc1(nodes).t(),
        enc1.embed_dim,
        128,
        adj_lists,
        agg2,
        base_model=enc1,
        gcn=True,
        cuda=False,
    )
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
    #    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(
            batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)]))
        )
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print("batch: {} loss: {:.2f}".format(batch, loss.item()))

    val_output = graphsage.forward(val)
    print(
        "Validation F1:",
        f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"),
    )
    test_output = graphsage.forward(test)
    print(
        "test F1:",
        f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"),
    )
    print("Average batch time:{:.2f}".format(np.mean(times)))


if __name__ == "__main__":
    run_cora()
