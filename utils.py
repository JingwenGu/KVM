import torch
from tqdm import trange

def release_kv(past_key_values, i):
    #print(i)
    if i == -1:
        return [(K[:, :, :i, :],V[:, :, :i, :]) for K,V in past_key_values]
    return [(torch.cat((K[:, :, :i, :], K[:, :, i + 1:, :]), dim=2),torch.cat((V[:, :, :i, :], V[:, :, i + 1:, :]), dim=2)) for K,V in past_key_values]

def truncate_kv(past_key_values, i):
    return [(K[:, :, :i, :],V[:, :, :i, :]) for K,V in past_key_values]

# gets the i-th step kv from kv2 and appends it to the end of kv1
def append_kv(kv1,kv2,i):
    kv_i = [(K[:, :, i, :].unsqueeze(2),V[:, :, i, :].unsqueeze(2)) for K,V in kv2]
    return [(torch.cat((K, Ki), dim=2),torch.cat((V,Vi), dim=2)) for (K,V),(Ki,Vi) in zip(kv1,kv_i)]

def kv_to_ndarray(past_key_values):
  return torch.stack([torch.stack(kv,dim=0) for kv in past_key_values],dim=0).detach().numpy()

def ndarray_to_kv(past_key_values):
  return [(torch.from_numpy(K),torch.from_numpy(V)) for K,V in past_key_values]

def release_id(ids,i):
  if i == -1:
    return ids[:,:-1]
  return torch.cat((ids[:,:i],ids[:,i+1:]),dim=1)

def append_id(id1,id2,i):
  id_i = id2[:,i:i+1]
  return torch.cat((id1,id_i),dim=1)

def evaluate_policy(actor, environment, num_episodes=100, progress=True):
    '''
        Returns the mean trajectory reward of rolling out `actor` on `environment

        Parameters
        - actor: PPOActor instance, defined in Part 1
        - environment: classstable_baselines3.common.vec_env.VecEnv instance
        - num_episodes: total number of trajectories to collect and average over
    '''
    total_rew = 0

    iterate = (trange(num_episodes) if progress else range(num_episodes))
    for _ in iterate:
        obs = environment.reset()
        done = False

        while not done:
            action = actor.select_action(obs)

            next_obs, reward, done, info = environment.step(action)
            total_rew += reward

            obs = next_obs

    return (total_rew / num_episodes).item()

def pad_obs_tensor(ids,obs_shape):
    original_shape = ids.shape
    padded_tensor = torch.zeros(obs_shape, dtype=ids.dtype, device=ids.device)
    padded_tensor[..., :original_shape[-1]] = ids
    return padded_tensor

def ids_to_obs(ids,obs_shape):
    return pad_obs_tensor(ids,obs_shape).detach().numpy()