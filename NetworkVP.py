# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf
import time
from Config import Config

def timing(function):
  if Config.PROFILE_COMM == False:
    return function

  def wrapper(self, *args, **kwargs):
    time1 = time.time()
    ret = function(self, *args, **kwargs)
    time2 = time.time()
    self.total_network_delay += (time2-time1)
    return ret
  return wrapper

class NetworkVP:
    def __init__(self, cluster_spec, job_name, task_index, num_workers, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.total_network_delay = int(0) # in seconds
        
        self.local_graph = tf.Graph()
        with self.local_graph.as_default() as g:
            with tf.device(self.device):
                self._create_local_graph()

                self.sess = tf.Session(
                    graph=self.local_graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

            if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
              vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='local')
              self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=10)
                
        server_config = tf.ConfigProto(log_device_placement = False,
            allow_soft_placement = True, device_filters=["/job:ps",
              "/job:%s/task:%d" % (job_name, task_index)])

        # start a tensor-flow server for a worker task
        self.server = tf.train.Server(cluster_spec, job_name = job_name, task_index = task_index, config = server_config)
        #print("Started server for job_name:%s, task_index:%d, is_chief:%d"% (job_name, task_index, (task_index==0)))
        
        # for the global graph, TF-ops and TF-variables should be on the CPU at
        # the parameter servers
        global_graph_device = 'cpu:0' 

        self.global_graph = tf.Graph()
        with self.global_graph.as_default() as g:
          # between-graph replication
          with tf.device(tf.train.replica_device_setter(worker_device="/job:%s/task:%d/%s" % (job_name, task_index, global_graph_device), cluster=cluster_spec)):
            self._create_global_graph()
            init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(graph=self.global_graph, is_chief = (task_index==0), init_op = init_op)

        # Create a session for running ops on the graph. The init_op is run when we prepare the session 
        self.global_sess = sv.prepare_or_wait_for_session(self.server.target, config = server_config)

        if task_index==0: # only for chief
          print("Asynchronous Distributed TF with %d replicas."%(num_workers))
    
    def _create_global_graph(self):
      """
      global graph vars are distributed over the parameter servers by TF. The
      graph is the same as the local graph, but the TF-ops are not required
      """

      with tf.variable_scope('global'):
        self.global_x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.global_var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        # As implemented in A3C paper
        self.global_n1 = self.conv2d_layer(self.global_x, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        self.global_n2 = self.conv2d_layer(self.global_n1, 4, 32, 'conv12', strides=[1, 2, 2, 1])
        _input = self.global_n2

        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]

        self.global_flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.global_d1 = self.dense_layer(self.global_flat, 256, 'dense1')
        self.global_logits_v = tf.squeeze(self.dense_layer(self.global_d1, 1, 'logits_v', func=None), axis=[1])
        self.global_logits_p = self.dense_layer(self.global_d1, self.num_actions, 'logits_p', func=None)
        
        if Config.USE_LOG_SOFTMAX:
            self.global_softmax_p = tf.nn.softmax(self.global_logits_p)
            self.global_log_softmax_p = tf.nn.log_softmax(self.global_logits_p)
        else:
            self.global_softmax_p = (tf.nn.softmax(self.global_logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
  
        self.global_opt = tf.train.RMSPropOptimizer(
            learning_rate=self.global_var_learning_rate,
            decay=Config.RMSPROP_DECAY,
            momentum=Config.RMSPROP_MOMENTUM,
            epsilon=Config.RMSPROP_EPSILON)
  
        self.global_tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.grads = [tf.placeholder(tf.float32, name='grad'+str(i),
          shape=tv.get_shape().as_list()) for i,tv in enumerate(self.global_tvs)]

        # Apply gradients from local network to global network
        self.apply_global_grads = self.global_opt.apply_gradients(zip(self.grads, self.global_tvs))

    def _create_local_graph(self):
      with tf.variable_scope('local'):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        # As implemented in A3C paper
        self.n1 = self.conv2d_layer(self.x, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        self.n2 = self.conv2d_layer(self.n1, 4, 32, 'conv12', strides=[1, 2, 2, 1])
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])
        _input = self.n2

        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]

        self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.d1 = self.dense_layer(self.flat, 256, 'dense1')

        self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1])
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        self.logits_p = self.dense_layer(self.d1, self.num_actions, 'logits_p', func=None)
        if Config.USE_LOG_SOFTMAX:
            self.softmax_p = tf.nn.softmax(self.logits_p)
            self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
            self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_index, axis=1)

            self.cost_p_1 = self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)
        else:
            self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
            self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)

            self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                        * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                      self.softmax_p, axis=1)
        
        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)
        
        self.cost_all = self.cost_p + self.cost_v
        self.opt = tf.train.RMSPropOptimizer(
            learning_rate=self.var_learning_rate,
            decay=Config.RMSPROP_DECAY,
            momentum=Config.RMSPROP_MOMENTUM,
            epsilon=Config.RMSPROP_EPSILON)

        self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)
        
        local_tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'local')
        
        #Get gradients from local network using local losses
        self.gvs = self.opt.compute_gradients(self.cost_all, local_tvs)
        self.apply_local_grads = self.opt.apply_gradients(self.gvs)

        # gradient accumulator vars
        self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()),
          trainable=False) for tv in local_tvs]

        # reset accum_vars
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]
        self.accum_ops = [self.accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.gvs)]

        self.gvals = [tf.placeholder(tf.float32, name='gval'+str(i),
          shape=tv.get_shape().as_list()) for i,tv in enumerate(local_tvs)]
        
        # Update local variables with values from the global network
        self.local_update = [tv.assign(self.gvals[i]) for i,tv in
            enumerate(local_tvs)]

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_n1", self.n1))
        summaries.append(tf.summary.histogram("activation_n2", self.n2))
        summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x})
        return prediction

    def predict_p(self, x):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x})
        return prediction
    
    def predict_p_and_v(self, x):
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})

    def get_comm_time(self, steps):
      s = steps // Config.SYNC_FREQ 
      return 0 if s == 0 else self.total_network_delay/float(s)
    
    @timing
    def run_timed_op(self, op, feed_dict):
      if feed_dict == None:
        return self.global_sess.run(op)
      else:
        return self.global_sess.run(op, feed_dict=feed_dict)

    def syncGlobal(self):
      accum = self.sess.run(self.accum_vars)
      feed_dict = {self.global_var_learning_rate: self.learning_rate}
      feed_dict.update({k:v for k,v in zip(self.grads, accum)})
      
      # apply gradients to global model
      self.run_timed_op(self.apply_global_grads, feed_dict=feed_dict)

    def syncLocal(self):
      gv = self.run_timed_op(self.global_tvs, feed_dict=None)
      
      # update local model with parameters from global model
      feed_dict={}
      feed_dict.update({k:v for k,v in zip(self.gvals, gv)})
      self.sess.run(self.local_update, feed_dict=feed_dict)

    def train(self, x, y_r, a, trainer_id, steps):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        self.sess.run([self.accum_ops, self.apply_local_grads], feed_dict=feed_dict)
        
        if steps % Config.SYNC_FREQ == 0:
          self.syncGlobal()
          self.syncLocal()
          # reset gradient accumulator
          self.sess.run(self.zero_ops)

    def log(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
