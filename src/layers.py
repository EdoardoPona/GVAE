import tensorflow as tf 
from tensorflow.keras import Model
import tensorflow.keras.layers as tfkl


class GraphConvolution(Model):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, output_dim, dropout=0.2, activation=tf.nn.relu):
        super(GraphConvolution, self).__init__()
        self.dropout = tfkl.Dropout(dropout)
        self.dense = tfkl.Dense(output_dim, use_bias=False, kernel_initializer='glorot_uniform')
        self.activation = activation

    def call(self, adj, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = tf.matmul(adj, x)
        outputs = self.activation(x)
        return outputs


class InnerProductDecoder(Model):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout=0.2, activation=tf.nn.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = tfkl.Dropout(dropout)
        self.activation = activation

    def call(self, inputs):
        inputs = self.dropout(inputs)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.activation(x)
        return outputs