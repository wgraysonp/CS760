import numpy as np


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
        else:
            node.split = dict(dim=split_dim, val=split_val)
            node.leaf = False
            X = self.X
            parent_idx = node.train_idxs
            idx_l = np.logical_and(X[:, split_dim] >= split_val, parent_idx)
            idx_r = np.logical_and(X[:, split_dim] < split_val, parent_idx)
            node.left_child = Node(idx_l)
            node.right_child = Node(idx_r)
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







