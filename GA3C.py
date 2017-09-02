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

# check python version; warn if not Python3
import sys
import warnings
if sys.version_info < (3,0):
    warnings.warn("Optimized for Python3. Performance may suffer under Python2.", Warning)

from mpi4py import MPI 
import subprocess
import argparse
from Config import Config

def parse():
    global job_name, task_index, parameter_servers, workers

    parser = argparse.ArgumentParser()
    parser.add_argument('-np', '--num_ps', default=1, help="number of parameter server nodes")
    parser.add_argument('-nw', '--num_wrks', default=1, help="number of worker nodes")
    parser.add_argument('-f', '--sync_freq', help="frequency of synchronization")
    parser.add_argument('-b', '--min_batch', help="minimum size of training batch")
    parser.add_argument('-l', '--learning_rate', help="learning rate")
    args = parser.parse_args()

    np = int(args.num_ps)
    nw = int(args.num_wrks)
    assert np
    assert nw                             
    
    if args.sync_freq is not None:
      Config.SYNC_FREQ = int(args.sync_freq) 

    if args.min_batch is not None:
      Config.TRAINING_MIN_BATCH_SIZE = int(args.min_batch)

    if args.learning_rate is not None:
      Config.LEARNING_RATE_START = float(args.learning_rate)
      Config.LEARNING_RATE_END = float(args.learning_rate)  # no annealing

    cmd = "/sbin/ifconfig"
    out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE).communicate()

    ip = str(out).split("inet addr:")[1].split()[0] 
    name = MPI.Get_processor_name()
    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank()     
    num_nodes = int(comm.Get_size())
    #print('Rank ID ', rank, ' has IP ', ip)

    ip = comm.gather(ip)

    if rank != 0:
      ip = None

    ip = comm.bcast(ip, root=0)

    assert(len(ip) == np+nw)

    # cluster config
    parameter_servers = [x+':2222' for x in ip[:np]]
    workers = [x+':2222' for x in ip[np:]]

    if rank == 0:
      print("==Distributed GA3C with {} PS nodes and {} WORKER nodes==".format(np, nw))
      print("==MPI info==")
      print('Communicator size ', num_nodes)
      print('PS:', parameter_servers)
      print('WORKERS:', workers)

      print('\n\n')
      print(vars(Config))  # all Config variables
      print('\n\n')

    if rank < np:
      job_name = 'ps'
      task_index = rank
    else:
      job_name = 'worker'
      task_index = rank-np

    if rank != np:
      # disable printing of stats from all but the first worker
      Config.PRINT_STATS_FREQUENCY = 1E10
      # if saving model, do only from the first worker
      Config.SAVE_MODELS = False

parameter_servers=[]
workers=[]
job_name = None
task_index = None
# parse input command line
parse()

from Server import Server
import tensorflow as tf

# cluster specification
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

def start_training():
  import gym
  gym.undo_logger_setup()

  # Start main program
  Server(cluster, job_name, task_index, len(workers)).main()

if job_name == "ps":
  Config.DEVICE = '/cpu:0'
  
  # prevent the parameter server from hogging memory on the GPU
  server_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
  server = tf.train.Server(cluster, job_name = job_name, task_index =
      task_index, config = server_config) 
  server.join()

elif job_name == "worker":
  start_training()
