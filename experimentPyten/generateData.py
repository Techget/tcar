# based on example on https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/egt/examples/alpharank_example.py
# and pyten/testImgRecovery.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils
import pyspiel
import pyten
from pyten.method import *
import numpy as np
from pyten.tools import tenerror
import scipy.stats as stats
import tensorly as tl
from tensorly.decomposition import tucker, parafac

from numpy.linalg import matrix_rank
from pyten.tools import create  # Import the problem creation function


def main(unused_arg):
  # num_players = 3
  # payoff_tables = []
  # for i in range(num_players):
    # payoff_tables.append(tl.random.random_cp(shape=(11, 11, 11), rank=5, 
    #   full=True, random_state=np.random.RandomState(seed=i*3))) 
    # random_factors = [tl.tensor(np.random.uniform(low=-3, high=3, size=(s,5))) for s in (11,11,11)] # random_sample((s, 5))
    # random_weights = tl.ones(5)
    # tensor = tl.cp_to_tensor((random_weights, random_factors))
    # payoff_tables.append(tensor)
    # siz = [11, 11, 11]  # Size of the Created Synthetic Tensor
    # r = [2, 2, 2]  # Rank of the Created Synthetic Tensor
    # problem = 'basic'  # Define Problem As Basic Tensor Completion Problem
    # miss = 0.2  # Missing Percentage
    # tp = 'Tucker'  # Define Solution Format of the Created Synthetic Tensor As 'CP decomposition'
    # # dims = len(siz)
    # # u = [np.random.random([siz[n], r[n]]) for n in range(dims)]
    # # core = pyten.tenclass.Tensor(np.random.random(r))
    # # sol = pyten.tenclass.Ttensor(core, u)
    # [X1, Omega1, sol1] = create(problem, siz, r, miss, tp)
    # payoff_tables.append((sol1.totensor()).data)

  with open('payoff_open_spiel.npy', 'rb') as f:
    pts = np.load(f)
  payoff_tables = []
  for i in range(len(pts)):
    tensor_to_be_decomposed = tl.tensor(pts[i])
    weight, factors = parafac(tensor_to_be_decomposed, rank=11)
    new_weights = tl.ones(2)
    new_factors = []
    for factor in factors:
      new_factors.append(tl.tensor(factor[:,:2]))
    payoff_tables.append(tl.cp_to_tensor((new_weights, new_factors)))

  print("shape of payoff_table: ", payoff_tables[0].shape)
  temp_sh = payoff_tables[0].shape
  for pt in payoff_tables:
    tpt = tl.tensor(pt)
    rs = []
    for i in range(len(temp_sh)):
      mat = tl.unfold(tpt, i)
      rs.append(matrix_rank(mat))
    print("ranks:", rs)

  with open('payoff_open_spiel_rank2_data.npy', 'wb') as f:
    np.save(f, np.array(payoff_tables))
    

if __name__ == '__main__':
  app.run(main)