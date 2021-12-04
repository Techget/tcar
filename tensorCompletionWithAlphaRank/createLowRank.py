from tensorly import random
import tensorly as tl
from numpy.linalg import matrix_rank
import numpy as np
import pyspiel
from absl import app
from open_spiel.python.algorithms import fictitious_play
from tensorly.decomposition import CP, parafac


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
  meta_games = xfp_solver.get_empirical_metagame(100, seed=5)

  # Metagame utility matrices for each player
  payoff_tables = []
  for i in range(num_players):
    payoff_tables.append(meta_games[i])
  return payoff_tables

def main(unused_arg):
  # tensor = random.random_cp(shape=(11, 11, 11), rank=5, full=True)
  # print(tensor)

  random_factors = [tl.tensor(np.random.uniform(low=-3, high=3, size=(s,5))) for s in (11,11,11)] # random_sample((s, 5))
  for i in range(len(random_factors)):
    print("random_factors[i].shape:", random_factors[i].shape)
  random_weights = tl.ones(5)
  tensor = tl.cp_to_tensor((random_weights, random_factors))

  temp_sh = tensor.shape    
  tpt = tl.tensor(tensor)
  rs = []
  for i in range(len(temp_sh)):
    mat = tl.unfold(tpt, i)
    print(np.shape(mat))
    rs.append(matrix_rank(mat))
  print("123 ranks:", rs)
  print(random_factors)
  # print(tensor)

  num_players = 3
  payoff_tables = get_game_fictitious_play_payoff_data('kuhn_poker', num_players, {'players': num_players})

  temp_sh = payoff_tables[0].shape
  for pt in payoff_tables:
    tpt = tl.tensor(pt)
    rs = []
    for i in range(len(temp_sh)):
      mat = tl.unfold(tpt, i)
      print(np.shape(mat))
      rs.append(matrix_rank(mat))
    print("payoff_tables ranks:", rs)

  tensor_to_be_decomposed = tl.tensor(payoff_tables[0])

  weight, factors = parafac(tensor_to_be_decomposed, rank=11)
  print(len(weight))
  print("factors[0].shape:",factors[0].shape)
  print("factors[1].shape:",factors[1].shape)
  print("factors[2].shape:",factors[2].shape)
  # print(factors[0])
  # print(factors[1])

  weights = tl.ones(5)
  new_factors = []
  for factor in factors:
    new_factors.append(tl.tensor(factor[:,:5]))
  # new_factors = tl.tensor(new_factors)
  print("new_factors[0].shape:", new_factors[0].shape)
  print("new_factors[1].shape:", new_factors[1].shape)
  print("new_factors[2].shape:", new_factors[2].shape)
  print("new_factors", new_factors)

  cp_reconstruction = tl.cp_to_tensor((weights, new_factors))
  temp_sh = cp_reconstruction.shape
  print("cp_reconstruction shape: ", temp_sh)
  rs=[]
  for i in range(len(temp_sh)):
    mat = tl.unfold(cp_reconstruction, i)
    print(np.shape(mat))
    rs.append(matrix_rank(mat))
  print("ranks:", rs)


if __name__ == '__main__':
  app.run(main)