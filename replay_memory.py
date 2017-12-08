from collections import namedtuple
import random
import numpy as np

class replay_memory:
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        self.position=0
        self.Transition=namedtuple('Transition',
                                   ['obs4','act','next_obs4','reward','done'])
    def __len__(self):
        return len(self.memory)
    def add(self,*args):
        '''Add a transition to replay memory
        Parameters
        ----------
        e.g. repay_memory.add(obs4,action,next_obs4,reward,done)
        obs4: {Tensor} of shape torch.Size([4,84,84])
        act: {Tensor} of shape torch.Size([6])
        next_obs4: {Tensor} of shape torch.Size([4,84,84])
        reward: {int}
        done: {bool} the next station is the terminal station or not

        Function
        --------
        the replay_memory will save the latest samples
        '''
        if len(self.memory)<self.capacity:
            self.memory.append(None)
        self.memory[self.position]=self.Transition(*args)
        self.position=(self.position+1)%self.capacity
    def sample(self,batch_size):
        '''Sample a batch from replay memory
        Parameters
        ----------
        batch_size: int
            How many trasitions you want

        Returns
        -------
        obs_batch: {Tensor} of shape torch.Size([BATCH_SIZE,4,84,84])
            batch of observations

        act_batch: {Tensor} of shape torch.Size([BATCH_SIZE,6])
            batch of actions executed w.r.t observations in obs_batch

        nob_batch: {Tensor} of shape torch.Size([BATCH_SIZE,4,84,84])
            batch of next observations w.r.t obs_batch and act_batch

        rew_batch: {ndarray} of shape
            batch of reward received w.r.t obs_batch and act_batch
        '''
        batch = random.sample(self.memory, batch_size)
        batch_zip=self.Transition(*zip(*batch))
        return batch_zip
