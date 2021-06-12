from src.layers import GraphConvolution, InnerProductDecoder 

from tensorflow.keras import Model
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class VGAE(Model):
  
    def __init__(self, node_num, hidden=32, latent_size=16, dropout=0.2):
        super(VGAE, self).__init__()
        self.shared = GraphConvolution(hidden, dropout)   ## TODO implement sparse graph convolution
        self.gcn_mean = GraphConvolution(latent_size, dropout, activation=lambda x: x)
        self.gcn_log_std = GraphConvolution(latent_size, dropout, activation=lambda x: x)
        self.decoder = InnerProductDecoder(dropout, activation=lambda x: x)

        # self.prior = tfd.Normal(
        #     loc=tf.fill((node_num, latent_size), 0.0),
        #     scale=tf.fill((node_num, latent_size), 1.0)
        # )

        self.prior = tfd.MultivariateNormalDiag(
            loc=tf.fill((node_num, latent_size), 0.0),
            scale_diag=tf.fill((node_num, latent_size), 1.0)
            
        )


    def call(self, adj, features): 
        h = self.shared(adj, features)  
        mean = self.gcn_mean(adj, h)    
        log_std = self.gcn_log_std(adj, h)
        
        # self.Q = tfd.Normal(  # TODO this should actually be a real multivariate normal, not a a batch of normals 
        #     loc=mean, scale=tf.exp(log_std)
        # )

        self.Q = tfd.MultivariateNormalDiag(
            loc=mean, 
            scale_diag=tf.exp(log_std)
        )
        reconstruction = self.decoder(tf.squeeze(self.Q.sample(1)))
        return self.Q, log_std, reconstruction


class GM_VGAE(Model):
  
    def __init__(self, node_num, class_num, hidden=32, latent_size=16, dropout=0.2):
        super(GM_VGAE, self).__init__()
        self.shared = GraphConvolution(hidden, dropout)   ## TODO implement sparse graph convolution

        self.classifier = GraphConvolution(class_num, dropout, activation=lambda x: x)   

        self.mean_layers = [GraphConvolution(latent_size, dropout, activation=lambda x: x) for i in range(class_num)] 
        self.log_std_layers = [GraphConvolution(latent_size, dropout, activation=lambda x: x) for i in range(class_num)]

        self.decoder = InnerProductDecoder(dropout, activation=lambda x: x)

        self.prior =  tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=tf.fill((node_num, class_num), 1/class_num)
            ),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=tf.fill((node_num, class_num, latent_size), 0.0),
                scale_diag=tf.fill((node_num, class_num, latent_size), 1.0)
            )
        )

    def call(self, adj, features): 
        h = self.shared(adj, features)  

        self.cy_logits = self.classifier(adj, h) 
        cy = tf.math.softmax(self.cy_logits)

        mean = tf.stack([l(adj, h) for l in self.mean_layers], axis=1)
        log_std = tf.stack([l(adj, h) for l in self.log_std_layers], axis=1)

        self.Q = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=cy
            ),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=mean,
                scale_diag = tf.exp(log_std)
            )
        )
        reconstruction = self.decoder(tf.squeeze(self.Q.sample(1), axis=0))
        return self.Q, log_std, reconstruction