from agent_dir.agent import Agent
import scipy
import numpy as np

import os
import keras
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, Adamax, RMSprop
from keras import backend as K

MAX_TIMESTEP = 1000
MAX_EP = 3000
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))


np.random.seed(1)
tf.set_random_seed(1)

# Reference: https://github.com/mkturkcan/Keras-Pong/blob/master/keras_pong.py
def discount_rewards(r):
    gamma=0.99
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def categorical_crossentropy(target, output):
    _epsilon =  tf.convert_to_tensor(10e-8, dtype=output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return - tf.reduce_sum(target * tf.log(output),axis=len(output.get_shape()) - 1)

class Agent_PGC(Agent):
    def __init__(self, env, args):
        super(Agent_PGC,self).__init__(env)
    
        self.log_path = './pgc.log'
        self.model_path = './pong_model_cart_checkpoint.h5'
        if args.test_pg:
            print('loading trained model')
            self.model = load_model('./pong_model_cart_checkpoint.h5')
        else:
            self.env = env
            self.actions_avialbe = env.action_space.n
            self.feature_dim = env.observation_space.shape[0]
            self.learning_rate  = 1e-3

            model = Sequential()
            model.add(Dense(20,activation='relu',input_shape=(self.feature_dim,)))
            model.add(Dense(self.actions_avialbe,activation='softmax'))

            opt = Adam(lr=self.learning_rate)
            model.compile(loss=categorical_crossentropy, optimizer=opt)
            self.model = model

    def init_game_setting(self):
        self.prev_x = None


    def train(self):
        # Init
        log = open(self.log_path,'w')
        log.write('reward,avg_reward\n')
        batch_size = 1 
        frames, prob_actions, dlogps, drs =[], [], [], []
        tr_x, tr_y = [],[]
        avg_reward = []
        reward_sum = 0
        t = 0
        ep_number = 0
        prev_x = None
        observation = self.env.reset()        
        # Training progress
        while True:
            
            aprob = self.model.predict(observation.reshape((1,self.feature_dim)), batch_size=1).flatten()
            frames.append(observation)
            prob_actions.append(aprob)
            action = np.random.choice(self.actions_avialbe, 1, p=aprob.reshape((self.actions_avialbe)))[0]
            y = np.zeros([self.actions_avialbe])
            y[action] = 1
            observation, reward, done, info = self.env.step(action)

            reward_sum += reward
            drs.append(reward) 
            dlogps.append(np.array(y).astype('float32') - aprob)

            t += 1

            if done or t> 1000:
                ep_number +=1
                ep_x = np.vstack(frames)
                ep_dlogp = np.vstack(dlogps)
                ep_reward = np.vstack(drs)
                # Discount and normalize rewards
                discounted_ep_reward = discount_rewards(ep_reward)
                discounted_ep_reward -= np.mean(discounted_ep_reward)
                discounted_ep_reward /= np.std(discounted_ep_reward)
                ep_dlogp *= discounted_ep_reward

                # Store current episode into training batch
                tr_x.append(ep_x)
                tr_y.append(ep_dlogp)
                frames, dlogps, drs =[], [], []
                if ep_number % batch_size == 0:
                    input_tr_y = prob_actions + self.learning_rate * np.squeeze(np.vstack(tr_y))
                    self.model.train_on_batch(np.vstack(tr_x).reshape(-1,self.feature_dim), input_tr_y)
                    tr_x,tr_y,prob_actions = [],[],[]
                    # Checkpoint
                    os.remove(self.model_path) if os.path.exists(self.model_path) else None
                    self.model.save(self.model_path)

                avg_reward.append(float(reward_sum))
                if len(avg_reward)>30: avg_reward.pop(0)
                print('Epsidoe {:} reward {:.2f}, Last 30ep Avg. rewards {:.2f}.'.format(ep_number,reward_sum,np.mean(avg_reward)))
                print('{:.4f},{:.4f}'.format(reward_sum,np.mean(avg_reward)),end='\n',file=log,flush=True)
                reward_sum = 0
                t = 0
                observation = self.env.reset()
            if ep_number>3000:
                break



    def make_action(self, observation, test=True):
        """

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        cur_x = prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(cur_x.shape)
        self.prev_x = cur_x
        aprob = self.model.predict(x.reshape((1,80,80,1)), batch_size=1).flatten()

        return np.argmax(aprob)

