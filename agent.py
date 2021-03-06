import os
import numpy as np
import torch as T

from replaybuffer import ReplayBuffer
from neuralnets import DuelQnet

class Agent():

    def __init__(self, gamma, eps, lr, n_actions, input_dims, mem_size, batch_size,
                 eps_min=0.01, eps_dec=5e-7, sync_every=1000, save_path='models') -> None:

        self.set_rl(eps, gamma, eps_dec, eps_min, n_actions)
        self.set_dl(batch_size, n_actions, lr, input_dims, save_path, sync_every)

        self.state_mem = ReplayBuffer(mem_size, *input_dims)

    def set_rl(self, eps, gamma, eps_dec, eps_min, n_actions) -> None:

        self.gamma = gamma
        self.action_space = [i for i in range(n_actions)]

        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec

    def set_dl(self, batch_size, n_actions,
               lr, input_dims, save_path, sync_every) -> None:

        self.learn_step_counter = 0
        self.batch_size = batch_size

        self.Q_eval = DuelQnet(n_actions, lr, *input_dims)
        self.Q_next = DuelQnet(n_actions, lr, *input_dims)
        self.Q_next.load_state_dict(self.Q_eval.state_dict())

        self.sync_every = sync_every

        self.save_path = save_path
        #self.load_models()

    def store_transition(self, state, action, reward, state_, done) -> None:
        self.state_mem.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation) -> int:
        if np.random.random() > self.eps:
            state  = T.tensor([observation]).to(self.Q_eval.device)
            _, advantage = state = self.Q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self) -> None:
        if self.learn_step_counter % self.sync_every == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def decrement_epsilon(self) -> None:
        self.eps = self.eps - self.eps_dec \
                    if self.eps > self.eps_min else self.eps_min

    def learn(self) -> None:

        if self.state_mem.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        self.replace_target_network()

        state, action, reward, state_, done = \
                self.state_mem.sample_buffer(self.batch_size)

        done = T.tensor(done, dtype=T.int8).to(self.Q_eval.device)
        state = T.tensor(state, dtype=T.float32).to(self.Q_eval.device)
        state_ = T.tensor(state_, dtype=T.float32).to(self.Q_next.device)
        action = T.tensor(action, dtype=T.int64).to(self.Q_eval.device)
        reward = T.tensor(reward, dtype=T.float32).to(self.Q_next.device)


        V_s, A_s = self.Q_eval.forward(state)
        V_s_, A_s_ = self.Q_next.forward(state_)

        q_eval = T.add(V_s, (A_s-A_s.mean(dim=1, keepdim=True))).gather(1,
                       action.unsqueeze(-1)).squeeze(-1)

        q_next = T.add(V_s_, (A_s_-A_s_.mean(dim=1, keepdim=True)))
        q_target = reward + self.gamma*T.max(q_next, dim=1)[0].detach()*(1-done)

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        T.save(self.Q_eval.state_dict(), f'{self.save_path}/Q_eval.ph')
        T.save(self.Q_next.state_dict(), f'{self.save_path}/Q_next.ph')

    def load_models(self):

        q_eval_path = f'{self.save_path}/Q_eval.ph'
        q_next_path = f'{self.save_path}/Q_next.ph'
        if os.path.exists(q_eval_path):
            print("Loading existing networks weights!")
            self.Q_eval.load_state_dict(T.load(q_eval_path))
            self.Q_next.load_state_dict(T.load(q_next_path))

