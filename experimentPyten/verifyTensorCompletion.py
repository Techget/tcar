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
from tensorly.decomposition import tucker,CP, parafac

from numpy.linalg import matrix_rank

def main(unused_arg):
  low_rank_data_files = ["payoff_open_spiel_rank2_data.npy", "randomly_generated_pyten_low_rank5_data.npy"] # , "randomly_generated_low_rank5_data.npy", 
  r = 2 # human estimation
  R = [2,2,2]

  for file in low_rank_data_files:
    print("data:", file)
    
    with open(file, 'rb') as f:
      payoff_tables = np.load(f)

    for data_keep_rate in np.arange(0.3,0.9,0.2):
      print("data_keep_rate:", data_keep_rate)

      ####### setup tensor completion ##########
      payoff_shape = payoff_tables[0].shape
      omega = (np.random.random(payoff_shape) <= data_keep_rate) * 1

      payoff_missing_tensors = []
      original_payoff_tensors = []
      for p in payoff_tables:
        X1 = pyten.tenclass.Tensor(p)
        X0 = X1.data.copy()
        X0 = pyten.tenclass.Tensor(X0)  # Save the Ground Truth
        # X1.data[omega == 0] = 0
        # ten.data[omega == 0] -= ten.data[omega == 0]
        X1.data[omega==0] -= X1.data[omega==0]
        payoff_missing_tensors.append(X1)
        original_payoff_tensors.append(X0)  

      #### tensor completion ###
      for i in range(len(payoff_missing_tensors)):
        print("payoff_missing_tensors[%d]", i)
        print("cp_als:")
        [_, recoerved_result] = cp_als(payoff_missing_tensors[i], r, omega, maxiter=1000, printitn=500)
        [Err1, ReErr11, ReErr21] = tenerror(recoerved_result, original_payoff_tensors[i], omega)
        print ('The frobenius norm of error of are:', Err1)
        print("\n")

        print("tucker_als:")
        [_, recoerved_result] = tucker_als(payoff_missing_tensors[i], R, omega, max_iter=1000, printitn=500)
        [Err1, ReErr11, ReErr21] = tenerror(recoerved_result, original_payoff_tensors[i], omega)
        print ('The frobenius norm of error of are:', Err1)
        print("\n")

        print("halrtc:")
        recoerved_result = halrtc(payoff_missing_tensors[i], omega)
        [Err1, ReErr11, ReErr21] = tenerror(recoerved_result, original_payoff_tensors[i], omega)
        print ('The frobenius norm of error of are:', Err1)
        print("\n")

        print("TNCP:")
        tncp1 = TNCP(payoff_missing_tensors[i], omega, rank=r, tol=1e-15, max_iter=5000, printitn=0) 
        tncp1.run()
        recoerved_result = tncp1.X
        [Err1, ReErr11, ReErr21] = tenerror(recoerved_result, original_payoff_tensors[i], omega)
        print ('The frobenius norm of error of are:', Err1)
        print("\n")  

      print("####################")
    print("------------")


if __name__ == '__main__':
  app.run(main)