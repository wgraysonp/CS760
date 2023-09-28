import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import random


class Node:
    def __init__(self, train_idxs, root=False, split=None, leaf=True):
        self.train_idxs = train_idxs
        self.split = split
        self.leaf = leaf
        self.root = root
        self.predict = None
        self.left_child = None
        self.right_child = None


class DecisionTree:

    def __init__(self, X, Y):
        self.X = X
        self.feature_dims = X.shape[1]
        self.Y = Y
        idxs = [True for k in range(X.shape[0])]
        self.root = Node(idxs, root=True, split=None, leaf=True)
        self.graph = nx.Graph()
        label = None
        self.graph.add_node(self.root, label=label)

    def fit(self, print_root_splits=False):
        self.makeSubtree(self.root, print_root_splits=print_root_splits)

    def predict(self, x):
        node = self.root
        while not node.leaf:
            split_val, split_dim = node.split['val'], node.split['dim']
            if x[split_dim] >= split_val:
                node = node.left_child
            else:
                node = node.right_child

        return node.predict

    def makeSubtree(self, node, print_root_splits=False):
        print_splits = node.root if print_root_splits else False
        split_val, split_dim = self.findBestSplit(node, print_splits=print_splits)
        idxs = node.train_idxs
        if split_val is None:
            node.leaf = True
            Y = self.Y[idxs]
            Y_pos = Y[Y == 1]
            Y_neg = Y[Y == 0]
            if len(Y_pos) >= len(Y_neg):
                node.predict = 1
            else:
                node.predict = 0
            self.graph.nodes[node]['label'] = f" y = {node.predict}"

        else:
            node.split = dict(dim=split_dim, val=split_val)
            self.graph.nodes[node]['label'] = r"$x_{} \geq {:.2f}$".format(split_dim + 1, split_val)
            node.leaf = False
            X = self.X
            parent_idx = node.train_idxs
            idx_l = np.logical_and(X[:, split_dim] >= split_val, parent_idx)
            idx_r = np.logical_and(X[:, split_dim] < split_val, parent_idx)
            node.left_child = Node(idx_l)
            node.right_child = Node(idx_r)
            self.graph.add_edge(node.left_child, node, label="True")
            self.graph.add_edge(node.right_child, node, label="False")
            self.makeSubtree(node.left_child)
            self.makeSubtree(node.right_child)

    def findBestSplit(self, node, print_splits=False):
        X = self.X[node.train_idxs]
        Y = self.Y[node.train_idxs]
        max_gain = 0
        split_val = None
        split_dim = None

        for dim in (0, 1):
            for val in set(X[:, dim]):
                if print_splits:
                    print("split val: {}, split dim: {}".format(val, dim))
                gain = gain_ratio(X, Y, dim=dim, val=val, verbose=print_splits)
                if gain > max_gain:
                    max_gain = gain
                    split_val = val
                    split_dim = dim

        return split_val, split_dim

    def print_tree(self, shape='s', size=1000, font_size=10, width=1, vert_gap=0.2, edge_font=8, node_color='b',
                   f_name=None, fig_width=10, fig_height=10):
        fig, ax = plt.subplots()
        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_height)
        pos = hierarchy_pos(self.graph, root=self.root, width=width, vert_gap=vert_gap)
        labels = {node: self.graph.nodes[node]['label'] for node in self.graph.nodes}
        nx.draw(self.graph, pos, node_shape=shape, node_size=size,
                font_size=font_size, ax=ax, labels=labels, with_labels=True, node_color=node_color)
        edge_labels = {edge: self.graph.edges[edge]['label'] for edge in self.graph.edges}
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_color='black',
            font_size=edge_font
        )
        if f_name is not None:
            plt.savefig(f_name)


def split_entropy(X, dim=0, val=0):
    n = X.shape[0]
    idx_l = X[:, dim] >= val
    idx_r = X[:, dim] < val
    X_l = X[idx_l]
    X_r = X[idx_r]
    n_l = X_l.shape[0]
    n_r = X_r.shape[0]

    if n_l == 0 or n_r == 0:
        ent = 0
    else:
        ent = -(n_l/n)*np.log2(n_l/n) - (n_r/n)*np.log2(n_r/n)

    return ent, idx_l, idx_r


def gain_ratio(X, Y, dim=0, val=0, verbose=False):
    n = len(Y)
    n_pos = len(Y[Y == 1])
    n_neg = n - n_pos

    split_ent, idx_l, idx_r = split_entropy(X, dim=dim, val=val)

    if n_pos == n or n_pos == 0:
        ent = 0
    else:
        ent = -(n_pos/n)*np.log2(n_pos/n) - (n_neg/n)*np.log2(n_neg/n)

    cond_int = 0
    for idx in (idx_l, idx_r):
        Yi = Y[idx]
        Xi = X[idx]
        k = Xi.shape[0]

        j = len(Yi[Yi == 1])
        if j == k or j == 0:
            cond_int_on_split = 0
        else:
            cond_int_on_split = -(j/k)*np.log2(j/k) - ((k-j)/k)*np.log2((k-j)/k)

        cond_int += (k/n)*cond_int_on_split

    mutual_info = ent - cond_int
    gain = mutual_info/split_ent if split_ent != 0 else 0

    if verbose:
        if split_ent == 0:
            print("Split entropy zero - mutual info: {:f}".format(mutual_info))
        else:
            print("gain: {:f}".format(gain))

    return gain


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width=width, vert_gap=vert_gap, vert_loc=vert_loc, xcenter=xcenter)


def test():
    X = np.array([[-1, -1], [1, 1], [1, -1], [-1, 1]])
    Y = np.array([[1], [1], [1], [0]])
    tree = DecisionTree(X, Y)
    tree.fit(print_root_splits=True)
    print(tree.root.leaf)
    y = tree.predict([0, 2])
    print(y)


if __name__ == "__main__":
    test()







