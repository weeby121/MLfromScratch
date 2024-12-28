import numpy as np
from collections import Counter

def entrophy(y):
    hist= np.bincount(y)
    ps = hist/len(y)
    return -np.sum(p*np.log2(p) for p in ps if p > 0)

class Node:
    def __init__(self,feature = None,threhold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threhold = threhold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None



class DecisionTree:

    def __init__(self,min_samples_split = 2,max_Deapth=10,n_feats=None):
        self.min_samples_split= min_samples_split
        self.max_deapth = max_Deapth
        self.n_feats= n_feats
        self.root = None


    def fit(self,X,y):
        #grow tree
        self.n_feats=X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root=self._grow_tree(X,y)

    def _grow_tree(self,X,y,deapth=0):
        n_samples,n_features = X.shape
        n_labels = len(np.unique(y))

        if (deapth >= self.max_deapth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idx = np.random.choice(n_features,self.n_feats,replace=False)

        #greedy search
        best_feat,best_thresh = self._best_criteria(X,y,feat_idx)
        left_idxs,right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs],deapth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs],deapth+1)
        return Node(best_feat,best_thresh,left,right)
    def _best_criteria(self,X,y,feat_idxs):
        best_gain = -1
        split_idx,split_thresh = None,None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y,X_column,threshold)

                if gain > best_gain:
                    best_gain=gain
                    split_idx= feat_idx
                    split_threh = threshold

    def _information_gain(self,y,X_column,split_treh):
        #parent E
        parent_entrophy = entrophy(y)
        #generate split
        left_idx,right_idx = self._split(X_column,split_treh)
        
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        #weighted avg child E
        n = len(y)
        n_l,n_r = len(left_idx), len(right_idx)
        e_l,e_r = entrophy(left_idx),entrophy(right_idx)
        child_entrophy = (n_l/n)*e_l + (n_r/n)*e_r

        ig = parent_entrophy - child_entrophy
        return ig


    def _split(self,X_colums,split_treh):
        left_idx=np.argwhere(X_colums <= split_treh).flatten()
        right_idx=np.argwhere(X_colums > split_treh).flatten()
        return left_idx,right_idx
        
    def _most_common_label(self,y):
        counter = counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


    def predict(self,X):
        #traverse tree
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def _traverse_tree(self,x,node):
        if Node.is_leaf_node():
            return Node.value
        
        if x[Node.feat_idx] <= Node.threhold:
            return self._traverse_tree(x,Node.left)
        
        return self._traverse_tree(x,Node.right)