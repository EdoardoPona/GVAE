from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp 
import numpy as np  
import tensorflow as tf 
    
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def load_network(path):
    # return sp.csr_matrix(np.load('data/diseasome/disease_network_adj.npy'))
    return sp.csr_matrix(np.load(path))

def load_network_labels(path, one_hot=False):
    data = np.load(path) 
    data = LabelEncoder().fit_transform(data)
    if one_hot:
        return OneHotEncoder(sparse=False).fit_transform(np.array(data).reshape((-1, 1)))
    return data

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    neg_to_pos_ratio = 1 # 100
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    print('DONE: train_edges')
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)


    '''
    Ideally, we would like to sample based on the join distribution, 
    but that doesn't fit to memory (most of the time!)
    The alternative is to sample a row and a column independntly. 

    '''
    rows = np.random.choice(adj.shape[0], size=len(test_edges*neg_to_pos_ratio))
    cols = np.random.choice(adj.shape[0], size=len(test_edges*neg_to_pos_ratio))
    test_edges_false = []
    for row, col in zip(rows, cols):
        if adj[row, col] == 0:
            test_edges_false.append([row, col])
    print('DONE: test_edges_false')

    rows = np.random.choice(adj.shape[0], size=len(val_edges))
    cols = np.random.choice(adj.shape[0], size=len(val_edges))
    val_edges_false = []
    for row, col in zip(rows, cols):
        if adj[row, col] == 0:
            val_edges_false.append([row, col])
    print('DONE: val_edges_false')

    def is_member(x, y):
        return len(set([','.join([str(l) for l in el]) for el in x]) & set([','.join([str(l) for l in el]) for el in y])) == 0

    print(is_member(test_edges_false, edges_all))
    print(is_member(val_edges_false, edges_all))
    print(is_member(val_edges, train_edges))
    print(is_member(test_edges, train_edges))
    print(is_member(val_edges, test_edges))


    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def load_and_build_dataset(experiment_params, network_path, labels_path, epochs=1000):
    # network_path = diseasome_network_path
    # labels_path = diseasome_labels_path

    # network_path = debarment_network_path
    # labels_path = debarment_labels_path 

    '''experiment_params = {
        'learning_rate': 1e-2,
        'epochs': 200,
        'hidden1': 32,
        'hidden2': 16,
        'weight_decay': 0. ,
        'dropout': 0.2 ,
        'model': 'vae',
        'features': 1,      # whether to use features (1) or not (0)
        'auxiliary_pred_dim': None     # will be filled later
    }'''

    adj = load_network(network_path) 
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    adj_normalized = preprocess_graph(adj)


    # target = np.ones((100, 6))
    target = load_network_labels(labels_path, one_hot=True)
    print(target.shape)
    experiment_params['auxiliary_pred_dim'] = target.shape[1]


    if experiment_params['features'] == 0:
        features = sp.identity(adj.shape[0])  # featureless
    else:
        # features = sp.diags(load_regions(WORKING_PATH, YEAR, one_hot=False)[:100])
        # TODO why are the features a diagonal matrix? would a batch of one-hot vectors work? 
        features = sp.diags(load_network_labels(labels_path, one_hot=False))

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = tf.sparse.SparseTensor(*sparse_to_tuple(adj_label))

    # labels = tf.reshape(tf.sparse.to_dense(tf.sparse.reorder(adj_label), validate_indices=False), [-1])
    labels = tf.sparse.to_dense(tf.sparse.reorder(adj_label), validate_indices=False)
    adj_normalized = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(*adj_normalized)))     
    features = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(*features)))

    # TODO the dataset could return also the norm 
    return adj, target, tf.data.Dataset.from_tensor_slices(([tf.cast(adj_normalized, tf.float32)],
                                              [tf.cast(features, tf.float32)], 
                                              [tf.cast(labels, tf.float32)])).repeat(epochs)

NUM_MC_SAMPLES = 50
def mc_kl_divergence(P, Q, seed=None):
    s = P.sample(NUM_MC_SAMPLES, seed)    
    # return tf.reduce_mean(P.prob(s) * (P.log_prob(s) - Q.log_prob(s)), axis=0, name='KL_MSF_MSF')   this gives sometimes better performance
    return tf.reduce_mean(P.log_prob(s) - Q.log_prob(s), axis=0, name='KL_MSF_MSF')

def kl_divergence_upper_bound(P, Q):
    """ upper bound between two gaussian mixtures with the same number of components 
    P and Q are tensorflow_probability distributions
    section 6:
    https://www.researchgate.net/publication/4249249_Approximating_the_Kullback_Leibler_Divergence_Between_Gaussian_Mixture_Models
    """ 
    pi = P.mixture_distribution
    omega = Q.mixture_distribution
    kl = 0
    for a in range(pi.probs.shape[1]):
        kl += pi.prob(a) * (pi.log_prob(a) - omega.log_prob(a) + tfd.kl_divergence(P.components_distribution[:, a], Q.components_distribution[:, a]))
    return kl
