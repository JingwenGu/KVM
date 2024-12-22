import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class PolicyGradient:
    def __init__(self, env, policy_net, seed=-1, reward_to_go: bool = False):
        """Policy gradient algorithm based on the REINFORCE algorithm.

        Args:
            env (gym.Env): Environment
            policy_net (PolicyNet): Policy network
            seed (int): Seed
            reward_to_go (bool): True if using reward_to_go, False if not
        """
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = policy_net.to(self.device)
        self.reward_to_go = reward_to_go
        if seed != -1:
            self.seed = seed
            self.env.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def select_action(self, state):
        """Select an action based on the policy network

        Args:
            state (np.ndarray): State of the environment

        Returns:
            action (int): Action to be taken
        """
        # TODO: Implement the action selection here based on the policy network output probabilities
        # Hint: You can use torch.distributions.Categorical to sample from the policy network output distribution
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item()

    def compute_loss(self, episode, gamma):
        """Compute the loss function for the REINFORCE algorithm

        Args:
            episode (list): List of tuples (state, action, reward)
            gamma (float): Discount factor

        Returns:
            loss (torch.Tensor): The value of the loss function
        """
        # TODO: Extract states, actions and rewards from the episode
        states = [x[0] for x in episode]
        actions = [x[1] for x in episode]
        rewards = [x[2] for x in episode]
        discounted_rewards = torch.Tensor([gamma**t * r for (t,r) in enumerate(rewards)])

        if not self.reward_to_go:
            # TODO: Part 1: Compute the total discounted reward here
            total_rewards = [torch.sum(discounted_rewards)]*len(episode)

        else:
            # TODO: Part 2: Compute the discounted rewards to go here
            total_rewards = [torch.sum(discounted_rewards[i:]) for i in range(len(episode))]

        # TODO: Implement the loss function for the REINFORCE algorithm here
        probs = []
        loss = torch.tensor(0.0).to(self.device)
        for state, action, reward in zip(states, actions, total_rewards):
            #state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            state = torch.from_numpy(state).float().to(self.device)
            prob = self.policy_net(state)[action]
            loss -= reward*torch.log(prob)
            #probs.append(self.policy_net(state)[action])
        #loss = -(torch.Tensor(total_rewards).requires_grad_(True) @ torch.log(torch.Tensor(probs)))
        #loss = -(torch.Tensor(total_rewards) @ torch.log(torch.Tensor(probs)))

        return loss

    def update_policy(self, episodes, optimizer, gamma):
        """Update the policy network using the batch of episodes

        Args:
            episodes (list): List of episodes
            optimizer (torch.optim): Optimizer
            gamma (float): Discount factor
        """
        # TODO: Compute the loss function for each episode using compute_loss
        # loss = torch.Tensor(0.0, requires_grad=True, device=self.device)
        # for episode in episodes:
        #   loss += self.compute_loss(episode,gamma)
        losses = [self.compute_loss(episode,gamma) for episode in episodes]
        loss = torch.mean(torch.stack(losses))
        # print(losses)
        # print(loss)
        # TODO: Update the policy network using average loss across the batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def run_episode(self,seed=-1):
        """
        Run an episode of the environment and return the episode

        Returns:
            episode (list): List of tuples (state, action, reward)
        """
        if seed != -1:
            self.env.seed(seed)
        state = self.env.reset()
        episode = []
        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def train(self, num_iterations, batch_size, gamma, lr):
        """Train the policy network using the REINFORCE algorithm

        Args:
            num_iterations (int): Number of iterations to train the policy network
            batch_size (int): Number of episodes per batch
            gamma (float): Discount factor
            lr (float): Learning rate
        """
        self.policy_net.train()
        optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # TODO: Implement the training loop for the REINFORCE algorithm here.
        # Update the policy every iteration, and use one batch per iteration.
        rewards = []
        for i in range(num_iterations):
          episodes = []
          print(f'iteration {i}/{num_iterations}')
          for j in range(batch_size):
            episodes.append(self.run_episode())
          self.update_policy(episodes, optimizer, gamma)
          if i % 10 == 0:
            rewards.append(self.evaluate(100))
            print(rewards[-1])
        return rewards

    def evaluate(self, num_episodes = 100):
        """Evaluate the policy network by running multiple episodes.

        Args:
            num_episodes (int): Number of episodes to run

        Returns:
            average_reward (float): Average total reward per episode
        """
        self.policy_net.eval()
        # TODO: Implement an evaluation loop for the REINFORCE algorithm here
        # by running multiple episodes and returning the average total reward
        total_rewards = []
        for i in range(num_episodes):
          episode = self.run_episode()
          total_rewards.append(sum([x[2] for x in episode]))
        return np.mean(total_rewards)