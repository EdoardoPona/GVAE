import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import node
from tensorflow.python.keras.engine.node import Node 
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np
import scipy.sparse as sp

from src.models import GM_VGAE, VGAE
from src import utils 


# loading data 
# network_path = 'data/diseasome/disease_network_adj.npy'
# labels_path = 'data/diseasome/disease_network_types.npy'

network_path = 'data/debarment/2014-2015_adjacency.npz'
labels_path = 'data/debarment/2014-2015_state_id.npy' 

output_path = 'data/saved/diseasome/model/'

data_params = dict(network_path=network_path,
                   labels_path=labels_path,
                   use_features=False,
                   auxiliary_prediction_task=True,
                   epochs=1000)

print('LOAD', utils.load_network(network_path).shape)

res = utils.load_and_build_dataset(data_params)
adj = res['adj']
aux_targets = res['target']
dataset = res['dataset']
val_edges = res['val_edges']
val_edges_false = res['val_edges_false']
test_edges = res['test_edges']
test_edges_false = ['test_edges_false']

adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj_orig.eliminate_zeros()
adj_orig = adj_orig.toarray()
