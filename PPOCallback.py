import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from PPOActor import PPOActor
from utils import evaluate_policy
import datetime

class PPOCallback(BaseCallback):
    def __init__(self, save_freq=1000, num_eval_episodes=10, verbose=0, save_path='default', eval_env=None, log=None, reward_lr=False):
        super(PPOCallback, self).__init__(verbose)
        self.rewards = []

        self.save_freq = save_freq
        self.num_eval_episodes = num_eval_episodes
        self.min_reward = -np.inf
        self.actor = None
        self.eval_env = eval_env

        self.save_path = save_path

        self.eval_steps = []
        self.eval_rewards = []
        self.log = log
        self.reward_lr = reward_lr

    def _init_callback(self) -> None:
        pass

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

        self.actor = PPOActor(model=self.model)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        episode_info = self.model.ep_info_buffer
        rewards = [ep_info['r'] for ep_info in episode_info]
        mean_rewards = np.mean(rewards)

        self.rewards.append(mean_rewards)


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.eval_env is None:
            return True

        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps != 0:
            mean_reward = evaluate_policy(self.actor, environment=self.eval_env, num_episodes=self.num_eval_episodes)
            print(f'evaluating {self.num_timesteps=}, {mean_reward=}=======')
            with open(self.log,'a') as file:
                file.writelines([f'{datetime.datetime.now()}>> evaluating {self.num_timesteps=}, {mean_reward=}=======\n'])

            self.eval_steps.append(self.num_timesteps)
            self.eval_rewards.append(mean_reward)
            if mean_reward > self.min_reward:
                self.min_reward = mean_reward
                self.model.save(self.save_path)
                print(f'model saved on eval reward: {self.min_reward}')
            if self.reward_lr:
                new_lr = 10**(min(-5,-10-mean_reward/100))
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f'lr={new_lr}')
                with open(self.log,'a') as file:
                    file.writelines([f'lr={new_lr}\n'])

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print(f'model saved on eval reward: {self.min_reward}')

        plt.plot(self.eval_steps, self.eval_rewards, c='red')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Rewards over Episodes')

        plt.show()
        plt.close()