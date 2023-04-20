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
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets', 'lap_pos'])


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
      score = k_h * q_h
      # scaling
      score /= tf.sqrt(self.latent_size)
      # Use available edge features to modify the scores
      score *= proj_e
      # Copy edge features as e_out to be passed to FFN_e
      e_out = score
      # softmax
      score = tf.exp(tf.clip_by_value(tf.reduce_sum(score, axis=-1, keepdims=True), -5, 5))
      # Send weighted values to target nodes
      wV = v_h * score
      return e_out, wV, score

  def _make_linear(self, out_dim):
    width = out_dim * self.num_heads
    network = snt.Linear(width, with_bias=self.bias)
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
      
      h_out = wV / (score + tf.fill(tf.shape(score), 1e-6)) # adding eps to all values here
      
      return h_out, e_out


class GraphTransformerLayer(snt.AbstractModule):
  def __init__(self, output_size, latent_size, num_heads=4, dropout=0.0, 
                layer_norm=False, batch_norm=True, 
                residual=True, use_bias=False, name='GraphTransformerLayer'):
    super(GraphTransformerLayer, self).__init__(name=name)
    self.output_size = output_size
    self.latent_size = latent_size
    self.num_heads = num_heads
    self.dropout = dropout
    self.residual = residual
    self.layer_norm = layer_norm     
    self.batch_norm = batch_norm
    self.bias = use_bias

  def _make_linear(self, out_dim):
    width = [out_dim]
    network = snt.Linear(width, with_bias=self.bias)
    return network

  def _make_mlp(self, widths, layer_norm=True):
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _build(self, h, e):
    h1 = h
    e1 = e
    h_att, e_att = MultiHeadAttentionLayer(self.output_size, self.latent_size, self.num_heads, self.bias)(h, e)
    h = tf.reshape(h_att, [-1, self.latent_size])
    e = tf.reshape(e_att, [-1, self.latent_size])

    h = self._make_linear([self.output_size])(h)
    e = self._make_linear([self.output_size])(e)

    if self.residual:
      h += h1
      e += e1
    
    if self.layer_norm:
      h = snt.LayerNorm()(h)
      e = snt.LayerNorm()(e)
    
    if self.batch_norm:
      h = snt.BatchNorm()(h)
      e = snt.BatchNorm()(e)
    
    h2 = h
    e2 = e

    h = self._make_mlp([self.latent_size*2, self.output_size])
    e = self._make_mlp([self.latent_size*2, self.output_size])

    if self.residual:
      h += h2
      e += e2
    
    if self.layer_norm:
      h = snt.LayerNorm()(h)
      e = snt.LayerNorm()(e)
    
    if self.batch_norm:
      h = snt.BatchNorm()(h)
      e = snt.BatchNorm()(e)
     
    return h, e
  
"GraphTransformer"
class EncodeProcessDecode(snt.AbstractModule):

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               use_bias=False,
               name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps
    self.bias = use_bias

  def _make_linear(self, out_dim):
    width = [out_dim]
    network = snt.Linear(width, with_bias=self.bias)
    return network

  def _make_mlp(self, widths, layer_norm=True):
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _build(self, graph):
    #h = snt.Embed(vocab_size=9, embed_dim=self._latent_size)(graph.node_features)
    h = self._make_linear(self._latent_size)(graph.node_features)
    h += self._make_linear(self._latent_size)(graph.lap_pos)
    for edge_set in graph.edge_sets:
        e = self._make_linear(self._latent_size)(edge_set.features)
    for conv in self._num_layers:
       h, e = GraphTransformerLayer(self._output_size, self._latent_size, use_bias=self.bias)(h, e)
    return self._make_mlp([self._latent_size, self._latent_size, self._output_size], layer_norm=False)(h)

  
