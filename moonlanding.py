import gym
import numpy as np
from agent import Agent
from utils import plotLearning

def main(n_games=50):

    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.001)

    scores, eps_history = [], []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        if i % 5 == 0:
            env.render()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(f'episode {i}, score {round(score, 2)} average score {round(avg_score, 2)}')

    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander_2020,png'
    plotLearning(x, scores, eps_history, filename)

if __name__ == "__main__":

    main()