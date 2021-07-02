from src import utils 

network_path = 'data/diseasome/disease_network_adj.npy'
labels_path = 'data/diseasome/disease_network_types.npy'
output_path = 'data/saved/diseasome/model/'

data_params = dict(network_path=network_path,
                   labels_path=labels_path,
                   use_features=False,
                   auxiliary_prediction_task=True,
                   epochs=1000)


res = utils.load_and_build_dataset(data_params)
adj = res['adj']
aux_targets = res['target']
dataset = res['dataset']
val_edges = res['val_edges']
val_edges_false = res['val_edges_false']
test_edges = res['test_edges']
test_edges_false = ['test_edges_false']