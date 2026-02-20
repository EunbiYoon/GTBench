# eval.py
# Monte-Carlo evaluation utilities.

import pyspiel
from open_spiel.python import rl_environment

from agents import ActorCritic, sample_action_from_env_timestep
from cfr_opponent import cfr_action_from_state


def eval_mc_vs_fixed(game_name: str, model: ActorCritic, cfr_policy, n_games: int, seed: int, device: str):
    """
    Monte-Carlo evaluation of PPO vs fixed CFR, in both seatings.

    Returns:
      (ev_p0, ev_p1)
      - ev_p0: PPO reward when PPO plays as player 0 vs CFR as player 1
      - ev_p1: PPO reward when PPO plays as player 1 vs CFR as player 0
    """
    env = rl_environment.Environment(game_name)
    env.seed(seed)

    game = pyspiel.load_game(game_name)
    act_dim = int(env.action_spec()["num_actions"])

    # ----- Seating 1: PPO as player 0 -----
    ev_p0 = 0.0
    for _ in range(n_games):
        ts = env.reset()
        st = game.new_initial_state()

        while not ts.last():
            p = int(ts.observations["current_player"])
            if p < 0:
                ts = env.reset()
                st = game.new_initial_state()
                continue

            if p == 0:
                a = sample_action_from_env_timestep(ts, 0, model, act_dim, device)
            else:
                a = cfr_action_from_state(st, 1, cfr_policy)

            ts = env.step([a])
            st.apply_action(a)

        ev_p0 += float(ts.rewards[0])
    ev_p0 /= n_games

    # ----- Seating 2: PPO as player 1 -----
    ev_p1 = 0.0
    for _ in range(n_games):
        ts = env.reset()
        st = game.new_initial_state()

        while not ts.last():
            p = int(ts.observations["current_player"])
            if p < 0:
                ts = env.reset()
                st = game.new_initial_state()
                continue

            if p == 0:
                a = cfr_action_from_state(st, 0, cfr_policy)
            else:
                a = sample_action_from_env_timestep(ts, 1, model, act_dim, device)

            ts = env.step([a])
            st.apply_action(a)

        ev_p1 += float(ts.rewards[1])
    ev_p1 /= n_games

    return float(ev_p0), float(ev_p1)