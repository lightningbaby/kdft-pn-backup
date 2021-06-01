#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from methods.meta_template import MetaTemplate
from methods import Attention
class ProtoNet(MetaTemplate):
  def __init__(self, model_func, n_way, n_support,n_query, common, proto_attention, tf_path=None, hidden_size=230,num_heads=2, distance='Euclidean'):
    super(ProtoNet, self).__init__(model_func,  n_way, n_support,n_query, tf_path=tf_path)
    self.loss_fn = nn.CrossEntropyLoss()
    self.method = 'ProtoNet'
    self.hidden_size = hidden_size
    weights = np.random.standard_normal((hidden_size, hidden_size * 4))
    bias = np.random.standard_normal((hidden_size * 4,))

    if proto_attention:
      self.protonet_attention = Attention.get_torch_layer_with_weights(hidden_size, num_heads,weights,bias)
    if distance == 'MLP':
      self.linear1 = nn.Linear(2*hidden_size,hidden_size,True)
      self.linear2 = nn.Linear(hidden_size, 1, True)
    if distance == 'pair':
      self.pair_linear1 = nn.Linear(2 * hidden_size, hidden_size, True)
      self.pair_linear2 = nn.Linear(hidden_size, 1, True)
      self.pair_gru = torch.nn.GRU(self.hidden_size*2, self.hidden_size, bidirectional=True)

    self.atten_or_not = proto_attention
    self.distance = distance
    self.common = common

  def reset_modules(self):
    return

  def get_batch_data(self,x):
    if torch.cuda.is_available():
      x = x.cuda()
    z_all = self.forward(x.contiguous())
    z_support = z_all[:-1]  # [6,230]
    z_query = z_all[-1].unsqueeze(0).unsqueeze(0)  # [1,1,230]
    z_support = z_support.view(self.n_way, self.n_support, -1)

    return z_support, z_query

  def set_forward(self,x,is_feature=False,lab=False):# x [8,10,230]
    if lab:
        z_support,z_query = self.get_batch_data(x) # [N,K,D] [1,1,D]
        # z_query = z_query_std.expand(self.n_way,1,self.hidden_size) # [N,1,D]

    else:
      z_support, z_query = self.parse_feature(x,is_feature) # [N,K,D] [N,Q,D]

    if self.atten_or_not:
      z_query = z_query.view(1,-1,self.hidden_size)
      z_support = z_support.view(1,-1,self.hidden_size)
      a = self.protonet_attention(z_query,z_support,z_support) # [N,Q,D] or [N,1,D]weighted query
     # z_query = a/100   #
      z_query = (a/100 + z_query)/2  # [N,Q,D]or [N,1,D]

    z_support   = z_support.contiguous().view(self.n_way, self.n_support, -1 ) #  [N,K,D]
    z_proto     = z_support.float().mean(1)  #[N,D]
    # z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 ).float() # [NQ,D]
    z_query = z_query.contiguous().view(-1,self.hidden_size) # [N*D,D] or [N*1,D]

    if self.distance == 'Euclidean':
      dists = euclidean_dist(z_query, z_proto)  # [NQ,N]
      if self.common:
        dists = dists * self.get_class_weight(z_support, z_proto, self.distance)
      scores = -dists
    elif self.distance == 'MLP':
      scores = self.get_distance_by_MLP(z_proto, z_query)
      if self.common:
        return scores * self.get_class_weight(z_support, z_proto, self.distance)
    else :
      scores = self.get_distance_by_relationnet(z_proto, z_query)
      if self.common:
        return scores * self.get_class_weight(z_support, z_proto, self.distance)

    return scores

  def get_class_weight(self,z_support,z_proto,disatance_type):
    class_common = z_proto  # * common_gain  # (N, D)
    support_differ = abs((z_support - class_common.unsqueeze(1)))  # (N, K, D)
    class_differ = torch.mean(support_differ, 1)  # (N, D)
    class_differ = class_differ.mean(1)  # (N)

    if disatance_type == 'Euclidean':
      class_weight = F.softmax((1 - F.softmax(class_differ / 4, dim=0)), dim=0) * self.n_way
    else:
      class_weight = F.softmax(class_differ / 4, dim=0) * self.n_way
    return class_weight

  def get_distance_by_MLP(self,proto,query):   # concat proto_vector with query_vector
    # proto[N,D],query[N*Q,D]

    sum = torch.cat([query[0], proto[0]], 0).unsqueeze(0) #[1,460] proto[N,230] query [NQ,230]
    for tmp in query:
      for tmp2 in proto:
        sum = torch.cat([sum, torch.cat([tmp, tmp2], 0).unsqueeze(0)], 0)
    sum = sum[1:]# [N*N,D]
    vects = sum.split(self.n_way, 0)
    input = vects[0].unsqueeze(0)
    for vect in vects[1:]:
      input = torch.cat([input, vect.unsqueeze(0)], 0) #[1,1,460]
    x = self.linear1(input) #[NQ,N,D] [1,1,230]
    x = self.linear2(x).squeeze(2)
    return x

  def get_distance_by_relationnet(self,proto,query):
    N, D, Q = proto.size(0), proto.size(-1), query.size(0)
    proto = proto.unsqueeze(0).expand(Q,N,D)
    query = query.unsqueeze(1).expand(Q,N,D)
    # pair = torch.cat([proto,query],-1).view(N*Q,-1)
    pair = torch.cat([proto,query],-1)#[NQ,N,2D]
    x, h = self.pair_gru(pair)
    x = self.pair_linear1(x)
    x = self.pair_linear2(x).squeeze(2)
    return x


  def get_distance(self,x,is_feature = False):
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
    z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
    return euclidean_dist(z_proto, z_proto)[0, :5].cpu().numpy()

  def set_forward_loss(self, x): # x [5,21,3,224,224] [5,10,512]
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query )) #[80]
    if torch.cuda.is_available():
      y_query = y_query.cuda()
    scores = self.set_forward(x) #[80,5]
    loss = self.loss_fn(scores, y_query)
    return scores, loss


def euclidean_dist( x, y):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)






