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
from tensorly.decomposition import tucker

from numpy.linalg import matrix_rank

# from pyten.tenclass import Tensor


def get_game_fictitious_play_payoff_data(game, num_players, game_settings):
  """Returns the kuhn poker data for the number of players specified."""
  # game_settings = {key: val for (key, val) in kwargs.items()}
  # logging.info("Using game settings: %s", game_settings)
  # self._game = pyspiel.load_game(game, game_settings)
  # game = pyspiel.load_game('kuhn_poker', {'players': num_players})
  game = pyspiel.load_game(game, game_settings)
  xfp_solver = fictitious_play.XFPSolver(game, save_oracles=True)
  for _ in range(10):
    xfp_solver.iteration()

  # Results are seed-dependent, so show some interesting cases
  # if num_players == 2:
  #   meta_games = xfp_solver.get_empirical_metagame(100, seed=1)
  # elif num_players == 3:
  #   meta_games = xfp_solver.get_empirical_metagame(100, seed=5)
  # elif num_players == 4:
  meta_games = xfp_solver.get_empirical_metagame(100, seed=2)

  # Metagame utility matrices for each player
  payoff_tables = []
  for i in range(num_players):
    payoff_tables.append(meta_games[i])
  return payoff_tables


def main(unused_arg):
  # Construct meta-game payoff tables
  # shape is [# player, # strategies, # strategies ...]
  num_players = 3
  payoff_tables = get_game_fictitious_play_payoff_data('kuhn_poker', num_players, {'players': num_players})
  # with open('test_data.npy', 'wb') as f:
  #   np.save(f, np.array(payoff_tables))    
  # payoff_tables = []
  # for i in range(num_players):
  #   payoff_tables.append(tl.random.random_cp(shape=(11, 11, 11), rank=5, full=True))

  # payoff_tables = [i * 100 for i in payoff_tables]
  print("shape of payoff_table: ", payoff_tables[0].shape)
  # tucker_rank = [2,2,2]
  # for p in payoff_tables:
  #   core, tucker_factors = tucker(p, tucker_rank)
  #   print(core, tucker_factors)
  temp_sh = payoff_tables[0].shape
  for pt in payoff_tables:
    tpt = tl.tensor(pt)
    rs = []
    for i in range(len(temp_sh)):
      mat = tl.unfold(tpt, i)
      print(np.shape(mat))
      rs.append(matrix_rank(mat))
    print("ranks:", rs)

  ####### setup tensor completion ##########
  payoff_shape = payoff_tables[0].shape
  miss = 0.2
  np.random.seed(0)
  omega = (np.random.random(payoff_shape) > miss) * 1

  payoff_missing_tensors = []
  original_payoff_tensors = []
  for p in payoff_tables:
    X1 = pyten.tenclass.Tensor(p)
    X0 = X1.data.copy()
    X0 = pyten.tenclass.Tensor(X0)  # Save the Ground Truth
    X1.data[omega == 0] = 0
    payoff_missing_tensors.append(X1)
    original_payoff_tensors.append(X0)

  ###### tensor completion #########
  completed_payoff_tensors = []
  r = 11 # human estimation
  R = [5,5,5]
  for i in range(len(payoff_missing_tensors)):
    # [_, recoerved_result] = cp_als(payoff_missing_tensors[i], r, omega, maxiter=1000, printitn=500)
    # [_, recoerved_result] = tucker_als(payoff_missing_tensors[i], R, omega, max_iter=500, printitn=100)
    # recoerved_result = falrtc(payoff_missing_tensors[i], omega) # pi error:  0.01557476547851582, tau:  0.8839969947407964
    # recoerved_result = halrtc(payoff_missing_tensors[i], omega)
    tncp1 = TNCP(payoff_missing_tensors[i], omega, rank=r, tol=1e-15, max_iter=5000) # 
    tncp1.run()
    recoerved_result = tncp1.X
    completed_payoff_tensors.append(recoerved_result)
    [Err1, ReErr11, ReErr21] = tenerror(recoerved_result, original_payoff_tensors[i], omega)
    # print("recoerved_result: ", recoerved_result.data)
    print ('\n', 'The Relative Error of the cp_als are:', Err1, ' ', ReErr11)

  ###### alpharank with completed tensors ##########
  completed_payoff_tables = []
  for cpt in completed_payoff_tensors:
    completed_payoff_tables.append(cpt.data)

  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(completed_payoff_tables)
  strat_labels = utils.get_strat_profile_labels(completed_payoff_tables, payoffs_are_hpt_format)
  # Run AlphaRank
  rhos_est, rho_m_est, pi_est, _, _ = alpharank.compute(completed_payoff_tables, alpha=1e2)
  # alpharank.print_results(completed_payoff_tables, payoffs_are_hpt_format, rhos=rhos_est, rho_m=rho_m_est, pi=pi_est)
  utils.print_rankings_table(completed_payoff_tables, pi_est, strat_labels, num_top_strats_to_print=15)

  ##### alpharank with original payoff #################
  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  strat_labels = utils.get_strat_profile_labels(payoff_tables,
                                                payoffs_are_hpt_format)

  # Run AlphaRank
  rhos, rho_m, pi, _, _ = alpharank.compute(payoff_tables, alpha=1e2)

  # Report & plot results
  # alpharank.print_results(
  #     payoff_tables, payoffs_are_hpt_format, rhos=rhos, rho_m=rho_m, pi=pi)
  utils.print_rankings_table(payoff_tables, pi, strat_labels, num_top_strats_to_print=15)
  # m_network_plotter = alpharank_visualizer.NetworkPlot(
  #     payoff_tables, rhos, rho_m, pi, strat_labels, num_top_profiles=8)
  # m_network_plotter.compute_and_draw_network()

  # print metrics to measure alpharank diffs
  print("pi error: ",np.max(np.abs(pi_est - pi)))

  tau, p_value = stats.kendalltau(pi, pi_est)
  print("tau: ", tau)


if __name__ == '__main__':
  app.run(main)