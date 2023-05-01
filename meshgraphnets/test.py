# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Core learned graph net model."""

import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(snt.AbstractModule):
  def __init__(self,
                output_size,
                latent_size,
                num_heads,
                bias,
                name='MultiHeadAttentionLayer'):
        super(MultiHeadAttentionLayer, self).__init__(name=name)
        
        self.output_size = output_size
        self.num_heads = num_heads
        self.latent_size = latent_size
        self.bias = bias
    
  def propagate_attention(self, k_h, q_h, v_h, proj_e):
      # Compute attention score
      #score = tf.reduce_sum(k_h * q_h, axis=-1)
      score = tf.matmul(k_h, q_h)

      # scaling
      score = tf.divide(score, tf.sqrt(float(self.latent_size)))
      # Use available edge features to modify the scores

      score = tf.matmul(score, proj_e)
      # Copy edge features as e_out to be passed to FFN_e

      e_out = score
      # softmax
      #score = tf.exp(tf.clip_by_value(tf.reduce_sum(score, axis=-1, keepdims=True), -5, 5))
      score = tf.softmax(score)
      # Send weighted values to target nodes
      wV = tf.matmul(v_h, score)
      return e_out, wV, score

  def _make_linear(self, out_dim):
    width = out_dim * self.num_heads
    network = snt.Linear(width, self.bias)
    return network

  def _build(self, h, e):
      Q_h = self._make_linear(self.latent_size)(h)
      K_h = self._make_linear(self.latent_size)(h)
      V_h = self._make_linear(self.latent_size)(h)
      proj_e = self._make_linear(self.latent_size)(e)
      
      # Reshaping into [num_nodes, num_heads, feat_dim] to 
      # get projections for multi-head attention
      q_h = tf.reshape(Q_h, [-1, self.num_heads, self.latent_size])
      k_h = tf.reshape(K_h, [-1, self.num_heads, self.latent_size])
      v_h = tf.reshape(V_h, [-1, self.num_heads, self.latent_size])
      pro_e = tf.reshape(proj_e, [-1, self.num_heads, self.latent_size])

      e_out, wV, score = self.propagate_attention(k_h, q_h, v_h, pro_e)
      
      #h_out = wV / (score + tf.fill(tf.shape(score), 1e-6)) # adding eps to all values here
      h_out = wV
      
      return h_out, e_out


class GraphTransformerLayer(snt.AbstractModule):
  def __init__(self, output_size, latent_size, num_heads=4, dropout=0.0, 
                layer_norm=False, batch_norm=False, 
                residual=True, use_bias=False, is_training=True, name='GraphTransformerLayer'):
    super(GraphTransformerLayer, self).__init__(name=name)
    self.output_size = output_size
    self.latent_size = latent_size
    self.num_heads = num_heads
    self.dropout = dropout
    self.residual = residual
    self.layer_norm = layer_norm     
    self.batch_norm = batch_norm
    self.bias = use_bias
    self.is_training = is_training
    
  def _make_linear(self, out_dim):
    width = out_dim
    network = snt.Linear(width, self.bias)
    return network

  def _make_mlp(self, widths, layer_norm=True):
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _update_edge_features(self, node_features, edge_set):
    sender_features = tf.gather(node_features, edge_set.senders)
    receiver_features = tf.gather(node_features, edge_set.receivers)
    features = [sender_features, receiver_features, edge_set.features]
    with tf.variable_scope(edge_set.name+'_edge_fn'):
      return self._make_linear(self.latent_size)(tf.concat(features, axis=-1))

  def _transform_edge(self, node_features, edge_sets):
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:
      features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                    edge_set.receivers,
                                                    num_nodes))
    return tf.concat(features, axis=-1)

  def _build(self, graph):
    h = graph.node_features
    h1 = h
    e = self._transform_edge(graph.node_features, graph.edge_sets)

    h_att, e_att = MultiHeadAttentionLayer(self.output_size, self.latent_size//self.num_heads, self.num_heads, self.bias)(h, e)
    h = tf.reshape(h_att, [-1, self.latent_size])
    e = tf.reshape(e_att, [-1, self.latent_size])

    new_edge_sets = []
    for edge_set in graph.edge_sets:
      updated_features = self._update_edge_features(h, edge_set)
      new_edge_sets.append(edge_set._replace(features=updated_features))

    h = self._make_linear(self.latent_size)(h)

    if self.residual:
      h += h1
      new_edge_sets = [es._replace(features=es.features + old_es.features)
                     for es, old_es in zip(new_edge_sets, graph.edge_sets)]
    
    if self.layer_norm:
      h = snt.LayerNorm()(h, is_training=self.is_training)
      new_edge_sets = [es._replace(features=snt.LayerNorm()(es.features, is_training=self.is_training))
                     for es in new_edge_sets]

    
    if self.batch_norm:
      h = snt.BatchNorm()(h, is_training=self.is_training)
      new_edge_sets = [es._replace(features=snt.BatchNorm()(es.features, is_training=self.is_training))
                     for es in new_edge_sets]
      
    h2 = h
    e2 = new_edge_sets

    h = self._make_mlp([self.latent_size*2, self.latent_size])(h)
    new_edge_sets = [es._replace(features=self._make_mlp([self.latent_size*2, self.latent_size])(es.features))
                     for es in new_edge_sets]

    if self.residual:
      h += h2
      new_edge_sets = [es._replace(features=es.features + old_es.features)
                     for es, old_es in zip(new_edge_sets, e2)]
    
    if self.layer_norm:
      h = snt.LayerNorm()(h, is_training=self.is_training)
      new_edge_sets = [es._replace(features=snt.LayerNorm()(es.features, is_training=self.is_training))
                     for es in new_edge_sets]

    
    if self.batch_norm:
      h = snt.BatchNorm()(h, is_training=self.is_training)
      new_edge_sets = [es._replace(features=snt.BatchNorm()(es.features, is_training=self.is_training))
                     for es in new_edge_sets]

    return MultiGraph(h, new_edge_sets)
  
"GraphTransformer"
class EncodeProcessDecode(snt.AbstractModule):

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               use_bias=False,
               is_training=False,
               name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self.latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps
    self.bias = use_bias
    self.is_training = is_training

  def _make_linear(self, out_dim):
    width = out_dim
    network = snt.Linear(width, self.bias)
    return network

  def _make_mlp(self, widths, layer_norm=True):
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _encoder(self, graph):
    """Encodes node and edge features into latent features."""
    with tf.variable_scope('encoder'):
        #h = snt.Embed(vocab_size=9, embed_dim=self._latent_size)(graph.node_features)  
        #h = self._make_linear(self._latent_size)(graph.node_features)
        h = self._make_mlp([self.latent_size, self.latent_size])(graph.node_features)
        new_edges_sets = []
        for edge_set in graph.edge_sets:
            #latent = self._make_linear(self._latent_size)(edge_set.features)
            latent = self._make_mlp([self.latent_size, self.latent_size])(edge_set.features)
            new_edges_sets.append(edge_set._replace(features=latent))
        latentgraph = MultiGraph(h, new_edges_sets)
        for conv in range(self._num_layers):
            latentgraph = GraphTransformerLayer(self._output_size, self.latent_size, use_bias=self.bias, is_training=self.is_training)(latentgraph)
        return latentgraph

  def _decoder(self, graph):
    """Decodes node features from graph."""
    with tf.variable_scope('decoder'):
      decoder = self._make_mlp([self.latent_size, self.latent_size, self._output_size], layer_norm=False)
      return decoder(graph.node_features)

  def _build(self, graph):
    latent_graph = self._encoder(graph)
    return self._decoder(latent_graph)