'''Deep reinforcement learning in gym evironments'''
import numpy as np
import matplotlib.pyplot as plt
import gym
from agent import Agent

GAMMA = 0.99
EPS = 1.0
LR = 5e-4
BATCH_SIZE = 64
EPS_DEC = 1e-3
EPS_MIN = 5e-4
MAX_MEM_SIZE = 1000000
SYNC_EVERY = 100

def main(n_games=500, env_type='LunarLander-v2'):
    '''
    Perform deep reinforcement learning on a
    given gym environment. The hyperparamters can be
    tuned using Optuna.
    '''
    env = gym.make(env_type)
    agent = Agent(gamma=GAMMA, eps=EPS, n_actions=env.action_space.n, lr=LR, batch_size=BATCH_SIZE,
                  eps_dec=EPS_DEC, eps_min=EPS_MIN, input_dims=[env.observation_space.shape[0]],
                  sync_every=SYNC_EVERY, mem_size=MAX_MEM_SIZE)

    scores, epsilons = [], []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        render = i % 50 == 0

        while not done:
            if render:
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        env.close()
        scores.append(score)
        epsilons.append(agent.eps)

        print(f'episode {i}, score {round(score, 2)} average score {np.mean(scores[-100:])}')
        print(f'eps {round(agent.eps, 3)}')

    #agent.save_models()
    x = [i for i in range(n_games)]
    plot_info(x, scores, filename=f'images/{env_type}')

def plot_info(x, scores, filename, window=20):

    '''
    Utility function to visualize the results,
    it creates a graph with the scores of the single
    episodes, as well as their rolling mean
    '''

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    ax.plot(x, scores, 'bo', label='single episode')

    fun = lambda s, w: np.convolve(s, np.ones(w), 'valid')/w
    rolling = fun(scores, window)
    ax.plot(x[window-1:], rolling, label='rolling mean')

    ax.set_ylabel('Score')
    ax.set_xlabel('Episode')
    ax.legend()

    ax.grid()

    fig.savefig(filename)


if __name__ == "__main__":

    main()