import tensorflow as tf 
import tensorflow_probaility as tfp
from models import GM_VGAE, VGAE
from utils import *


optimizer = tf.keras.optimizers.Adam(lr=1e-3)
reconstruction = []
kl_losses = []
losses = []
classification_losses = []

def train_step(adj_normalized, features, adj_label, norm, pos_weight, class_targets=None, model_type='vgae'):   
    assert model_type in ['vgae', 'gmvgae']
    assert ((model_type=='vgae' and class_targets is None) or (model_type=='gmvgae'))


    with tf.GradientTape() as tape:
        beta = 1
        adj_label = tf.reshape(adj_label, [-1])

        Q, Q_log_std, reconstructed = model(adj_normalized, features)
        reconstruction_loss = norm * tf.math.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(labels=adj_label, logits=reconstructed, pos_weight=pos_weight)
        ) 

        if model_type == 'vgae':
            kl = (0.5 / adj_label.shape[0]) * tf.math.reduce_mean(
                tf.math.reduce_sum(1 + 2 * Q_log_std - tf.math.square(Q.loc) - tf.math.square(Q.scale), axis=1)
            ) 
            # NOTE: why are these two different? they should give the same performance
            kl = tf.reduce_mean(tfd.kl_divergence(Q, model.prior))
            classification_loss = 0
        else:
            kl = tf.reduce_mean(mc_kl_divergence(Q, model.prior))
            # kl = tf.reduce_mean(kl_divergence_upper_bound(Q, model.prior))
            if class_targets is None:
                classification_loss = 0
            else: 
                classification_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=model.cy_logits, labels=class_targets)
                )
        
        vae_loss = reconstruction_loss + beta*kl + classification_loss

    # kls.append(kl.numpy())

    gradients = tape.gradient(vae_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # metrics
    reconstruction.append(reconstruction_loss.numpy())
    kl_losses.append(kl.numpy())
    losses.append(vae_loss.numpy())
    classification_losses.append(classification_loss.numpy())

## TODO actually use all of these values 
experiment_params = {
    'learning_rate': 1e-2,
    'epochs': 200,
    'hidden1': 32,
    'hidden2': 16,
    'weight_decay': 0. ,
    'dropout': 0.2 ,
    'model': 'vae',
    'features': 1,      # whether to use features (1) or not (0)
    'auxiliary_pred_dim': None     # will be filled later
}

network_path = 'PATHTONETWORK'
labels_path = 'PATHTOLABELS'
adj, dataset = load_and_build_dataset(experiment_params, network_path, labels_path)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
# model = VGAE(node_num=adj.shape[0])
node_num = adj.shape[0]
class_num = target.shape[1]
model = GM_VGAE(node_num=adj.shape[0], class_num=target.shape[1])
    

e = 0
for adj_norm, features, label in dataset:
    train_step(adj_norm, features, label, class_targets=target, model_type='gmvgae')
    # train_step(adj_norm, features, label)

    if e % 100 == 0:
        print('total', losses[-1], 'rec', reconstruction[-1], 'classification', classification_losses[-1], 'kl', kl_losses[-1])
    e+=1
