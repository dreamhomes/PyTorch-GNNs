import random

import torch
import torch.nn as nn
from torch.autograd import Variable


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, cuda=False, gcn=False):
        """Initializes the aggregator for a specific graph.

        Parameters
        ----------
        features : [type]
            function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda : bool, optional
            whether to use GPU, by default False
        gcn : bool, optional
            whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """

        Parameters
        ----------
        nodes : [type]
            list of nodes in a batch
        to_neighs : [type]
            list of sets, each set is the set of neighbors for node in batch
        num_sample : int, optional
            number of neighbors to sample. No sampling if None, by default 10

        Returns
        -------
        feats
        """
        # Local pointers to functions (speed hack)
        _set = set
        if num_sample:
            _sample = random.sample
            samp_neighs = [
                _set(
                    _sample(
                        to_neigh,
                        num_sample,
                    )
                )
                if len(to_neigh) >= num_sample
                else to_neigh
                for to_neigh in to_neighs
            ]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats
