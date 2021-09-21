'''Deep reinforcement learning in gym evironments'''
import numpy as np
import gym
from agent import SYNC_EVERY, Agent
from utils import plot_info

GAMMA = 0.99
EPS = 0.99
LR = 0.00007
BATCH_SIZE = 64
EPS_DEC = 0.001
EPS_MIN = 5e-4
MAX_MEM_SIZE = 1000000
SYNC_EVERY = 10000

def main(n_games=5000, env_type='Acrobot-v1'):
    '''
    Perform deep reinforcement learning on a
    given gym environment
    '''
    env = gym.make(env_type)
    agent = Agent(gamma=GAMMA, eps=EPS, n_actions=env.action_space.n, lr=LR, batch_size=BATCH_SIZE,
                  eps_dec=EPS_DEC, eps_min=EPS_MIN, input_dims=[env.observation_space.shape[0]])

    scores, epsilons = [], []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        render = i % 500 == 0

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

    agent.save_models()
    x = [i for i in range(n_games)]
    plot_info(x, scores, epsilons, filename=env_type)

if __name__ == "__main__":

    main()
