from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

class PPOActor():
    def __init__(self, ckpt: str=None, environment: VecEnv=None, model=None):
        '''
          Requires environment to be a 1-vectorized environment

          The `ckpt` is a .zip file path that leads to the checkpoint you want
          to use for this particular actor.

          If the `model` variable is provided, then this constructor will store
          that as the internal representing model instead of loading one from the
          checkpoint path

        '''
        assert ckpt is not None or model is not None
        if model is not None:
            self.model = model
            return

        # TODO: MODIFY
        self.model = PPO.load(ckpt, env=environment)
        # End TODO


    def select_action(self, obs):
        '''
          Gives the action prediction of this particular actor
        '''
        # TODO:
        return self.model.predict(obs)[0]
        # END TODO