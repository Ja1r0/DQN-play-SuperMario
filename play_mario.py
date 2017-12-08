import numpy as np
import random
import math
import cv2
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from net_pytorch import dqn_net
from replay_memory import replay_memory
from data import env
import matplotlib
# if gpu is to be used

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

Tensor = FloatTensor

def ob_process(frame):
    '''
    Parameters
    ----------
    frame: {ndarray} of shape (90,90)

    Returns
    -------
    frame: {Tensor} of shape torch.Size([1,84,84])
    '''
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame=frame.astype('float64')
    frame=torch.from_numpy(frame)
    frame=frame.unsqueeze(0).type(Tensor)
    return frame

def plot_graph(episode_reward):
    reward_list = torch.Tensor(episode_reward)
    plt.figure(1)
    plt.clf()
    plt.title('Episode Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    # 最近100个episode的total reward的平均值 #
    if len(reward_list)>=100:
        means=reward_list.unfold(0,100,1).mean(1).view(-1)
        means=torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())
    else:
        plt.plot(reward_list.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def learn(env,
        MAX_EPISODE,
        EPS_START,
        EPS_END,
        EPS_DECAY,
        ACTION_NUM,
        REPLAY_MEMORY_CAPACITY,
        BATCH_SIZE,
        LOSS_FUNCTION,
        OPTIM_METHOD,
        LEARNING_RATE,
        GAMMA,
        NET_COPY_STEP,
        PATH
        ):
    ### initialization ###
    action_space=[(0,Tensor([1,0,0,0,0,0])),
                  (3,Tensor([0,1,0,0,0,0])),
                  (7,Tensor([0,0,1,0,0,0])),
                  (11,Tensor([0,0,0,1,0,0])),
                  (4,Tensor([0,0,0,0,1,0])),
                  (8,Tensor([0,0,0,0,0,1]))]
    # (action_button , action_onehot)
    # 以上动作分别为：不动、左走、右走、跳、左跳、右跳
    value_net = dqn_net(ACTION_NUM)
    target_net=dqn_net(ACTION_NUM)
    if torch.cuda.is_available():
       value_net.cuda()
       target_net.cuda()
    if os.listdir(PATH):
        value_net.load_state_dict(torch.load(PATH))
    buffer=replay_memory(REPLAY_MEMORY_CAPACITY)
    env.reset()
    obs,_,_,_,_,_,_=env.step(0)
    obs=ob_process(obs)
    obs4=torch.cat(([obs,obs,obs,obs]),dim=0) # {Tensor} of shape torch.Size([4,84,84])
    judge_distance=0
    episode_total_reward = 0
    epi_total_reward_list=[]
    # counters #
    time_step=0
    update_times=0
    episode_num=0
    history_distance=200
    while episode_num <= MAX_EPISODE:
        ### choose an action with epsilon-greedy ###
        prob = random.random()
        threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * episode_num / EPS_DECAY)
        if prob <= threshold:
            action_index = np.random.randint(6)
            action_button = action_space[action_index][0] # {int}
            action_onehot = action_space[action_index][1] # {Tensor}
        else:
            action_button, action_onehot = value_net.select_action(obs4)
        ### do one step ###
        obs_next, reward, done, _, max_distance, _, now_distance = env.step(action_button)
        obs_next = ob_process(obs_next)
        obs4_next = torch.cat(([obs4[1:, :, :],obs_next]),dim=0)
        buffer.add(obs4.unsqueeze(0), action_onehot.unsqueeze(0), obs4_next.unsqueeze(0), Tensor([reward]).unsqueeze(0), done)
        episode_total_reward +=reward
        if now_distance <= history_distance:
            judge_distance+=1
        else:
            judge_distance=0
            history_distance=max_distance
        '''the transition added to buffer
        obs4: {ndarray} size (4,84,84)
        action: {list} size 6 e.g. [1,0,0,0,0,0] one hot list
        obs_next: {ndarray} size (84,84)
        reward: {int}
        done: {bool}
        '''
        ### go to the next state ###
        if done == False:
            obs4 = obs4_next
            time_step += 1
        elif done == True or judge_distance > 50:
            env.reset()
            obs, _, _, _, _, _, _ = env.step(0)
            obs = ob_process(obs)
            obs4 = torch.cat(([obs, obs, obs, obs]), dim=0)
            episode_num += 1
            history_distance = 200
            # plot graph #
            epi_total_reward_list.append(episode_total_reward)
            plot_graph(epi_total_reward_list)
            print('episode %d total reward=%.2f'%(episode_num,episode_total_reward))
            episode_total_reward = 0
        ### do one step update ###
        if len(buffer) == buffer.capacity and time_step % 4 == 0:
            batch_transition = buffer.sample(BATCH_SIZE)
            '''{Transition}
            0:{tuple} of {Tensor}-shape-torch.Size([4,84,84])
            1:{tuple} of {Tensor}-shape-torch.Size([6])
            2:{tuple} of {Tensor}-shape-torch.Size([4,84,84])
            3:{tuple} of {int}   
            4:{tuple} of {bool}        
            '''
            value_net.update(samples=batch_transition, loss_func=LOSS_FUNCTION,
                             optim_func=OPTIM_METHOD, learn_rate=LEARNING_RATE,
                             target_net=target_net, BATCH_SIZE=BATCH_SIZE,
                             GAMMA=GAMMA)
            update_times += 1
            ### copy value net parameters to target net ###
            if update_times % NET_COPY_STEP == 0:
                target_net.load_state_dict(value_net.state_dict())

    torch.save(value_net.state_dict(),PATH)

if __name__=='__main__':
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    learn(env=env.Env(),
        MAX_EPISODE=100,
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=200,
        ACTION_NUM=6,
        REPLAY_MEMORY_CAPACITY=100,
        BATCH_SIZE=10,
        LOSS_FUNCTION=nn.MSELoss,
        OPTIM_METHOD=optim.SGD,
        LEARNING_RATE=1e-6,
        GAMMA=0.99,
        NET_COPY_STEP=10,
        PATH='./model_saved/'
        )







