import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import copy

# if gpu is to be used

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class dqn_net(nn.Module):
    def __init__(self,ACTION_NUM):
        super(dqn_net,self).__init__()
        self.conv1=nn.Conv2d(in_channels=4,out_channels=16,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2)
        self.fc1=nn.Linear(in_features=9*9*32,out_features=256)
        self.fc2=nn.Linear(in_features=256,out_features=ACTION_NUM)
        self.action_num=ACTION_NUM
    def forward(self,input):
        output=F.relu(self.conv1(input))
        output=F.relu(self.conv2(output))
        output=output.view(-1,9*9*32)
        output=F.relu(self.fc1(output))
        output=self.fc2(output)
        return output
    def net_copy(self):
        return copy.deepcopy(self)
    def select_action(self,input):
        '''
        parameters
        ----------
        input : {Tensor} of shape torch.Size([4,84,84])

        Return
        ------
        action_button , action_onehot : {int} , {Tensor}
        '''
        input=Variable(input.unsqueeze(0))
        output=self.forward(input)
        action_index=output.data.max(1)[1][0]
        if action_index==0: return 0,Tensor([1,0,0,0,0,0]) # action_button , action_onehot
        elif action_index==1: return 3,Tensor([0,1,0,0,0,0])
        elif action_index==2: return 7,Tensor([0,0,1,0,0,0])
        elif action_index==3: return 11,Tensor([0,0,0,1,0,0])
        elif action_index==4: return 4,Tensor([0,0,0,0,1,0])
        elif action_index==5: return 8,Tensor([0,0,0,0,0,1])

    def update(self,samples,loss_func,optim_func,learn_rate,target_net,BATCH_SIZE,GAMMA):
        '''update the value network one step

        Parameters
        ----------
        samples: {namedtuple}
            Transition(obs4=(o1,o2,...),act=(a1,a2,...),
            next_ob=(no1,no2,...),reward=(r1,r2,...),done=(d1,d2,...))
        loss: string
            the loss function of the network
            e.g. 'nn.MSELoss'
        optim: string
            the optimization function of the network
            e.g. 'optim.SGD'
        learn_rate: float
            the learing rate of the optimizer

        Functions
        ---------
            update the network one step
        '''
        obs4_batch=Variable(torch.cat(samples.obs4)) # ([BATCH,4,84,84])
        next_obs4_batch=Variable(torch.cat(samples.next_obs4)) # ([BATCH,4,84,84])
        action_batch=Variable(torch.cat(samples.act)) # ([BATCH,6])
        done_batch=samples.done # {tuple} of bool,len=BATCH
        reward_batch=torch.cat(samples.reward) # ([BATCH,1])
        ### compute the target Q(s,a) value ###
        value_batch=target_net(next_obs4_batch)
        target=Variable(torch.zeros(BATCH_SIZE).type(Tensor))
        for i in range(BATCH_SIZE):
            if done_batch[i]==False:
                target[i]=reward_batch[i]+GAMMA*Tensor.max(value_batch.data[i])
            elif done_batch[i]==True:
                target[i]=reward_batch[i]
        ### compute the current net output value ###
        output_all=self.forward(obs4_batch)*action_batch
        output=output_all.sum(dim=1) # {Variable contain FloatTensor}
        criterion=loss_func()
        optimizer=optim_func(self.parameters(),lr=learn_rate)
        loss=criterion(output,target)
        #print('loss=\n',loss)
        #print('output=\n',output)
        #print('target=\n',target)
        optimizer.zero_grad()# set gradients of parameters to be optimized to zero
        loss.backward()
        optimizer.step()


