# -*- coding: utf-8 -*-

"""
@Date: 2019/1/14

@Author: dreamhome

@Summary: train  semi-supervised setting
"""
import torch
import torch.nn.functional as F

import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from model import GCN
from build_graph import build_karate_club_graph

import warnings
warnings.filterwarnings('ignore')


net = GCN(34, 5, 2)
print(net)
G = build_karate_club_graph()

inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []

for epoch in range(20):
    logits = net(G, inputs)
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)

    # compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    # ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors, with_labels=True, node_size=300, ax=ax)


nx_G = G.to_networkx().to_undirected()
fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
# draw(1)  # draw the prediction of the first epoch

ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
plt.show()

