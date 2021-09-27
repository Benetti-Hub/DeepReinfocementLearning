# Deep Reinfocement Learning

The repository shows how to solve a reinforcement learning problem with function approximation (achieved using neural networks). The target network is used to estimate the next set of Q-values for a specific action in a continuous environment. The online network is instead used for the prediction of the instantaneous reward and is updated using the replay buffer each time the agent takes an action.

Install the required packages using

```
pip install -r requirements.txt
```

To speed up the simulation make sure that a CUDA enabled gpu is available

Then run the simulation using

```
python simulation.py
```
