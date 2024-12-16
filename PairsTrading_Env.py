import gym
import os
import numpy as np
import torch



class PairsTradingEnv(gym.Env):

    def __init__(self, window_size, pair1_price, pair2_price,
                 t_z_score, f_average, f_beta):
        super(PairsTradingEnv, self).__init__()
        self.window_size = window_size
        self.pair1_price = pair1_price
        self.pair2_price = pair2_price
        self.current_cash = 10000

        self.z_score = t_z_score
        self.average = f_average
        self.shares = f_beta

        self.action_size = 6
        self.trade_boundaries = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        self.stop_loss_boundaries = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.current_trade_boundary = None
        self.current_stop_loss_boundary = None

        self.trade_open = False
        self.done = False # when it reaches the last day of trading window

        self.n = len(pair1_price)
        self.current_step = 0
        self.profit = 0
        self.c = 0

        self.ticker1_operate = []
        self.ticker2_operate = []

    def get_current_price(self):
        return [self.pair1_price[self.current_step], self.pair2_price[self.current_step]]

    def get_next_price(self):
        return [self.pair1_price[self.current_step+1], self.pair2_price[self.current_step+1]]

    def get_current_observation(self):
        return self.z_score[self.current_step]

    def get_next_observation(self):
        return self.z_score[self.current_step+1]

    def weights_calculate(self, v_at, v_bt):
        s_atp = self.get_next_price()[0]
        s_btp = self.get_next_price()[1]
        s_at = self.get_current_price()[0]
        s_b = self.get_current_price()[1]
        weight = v_at * (s_atp-s_at)/s_at + v_bt*(s_btp-s_b)/s_b
        return abs(weight)


    def step(self, action):

        reward = 0
        self.c += 1
        self.current_trade_boundary = self.trade_boundaries[action]
        self.current_stop_loss_boundary = self.stop_loss_boundaries[action]
        current_z_score = self.get_current_observation()
        p1_price = 0
        p2_price = 0
        p1_direction = 0
        p2_direction = 0
        # print('this is current z score of trading signal:', current_z_score,
        #       "it is times:",self.c,
        #       "done?",self.done,
        #       "trade open?", self.trade_open,
        #       "value we have", self.current_cash)
        if not self.trade_open and not self.done:
            if self.current_trade_boundary <= current_z_score < self.current_stop_loss_boundary:
                self.trade_open = True # ticker1 is undervalued, ticker2 is overvalued
                p1_direction = 1
                p2_direction = -1
                self.ticker1_operate.append([self.pair1_price[self.current_step], self.shares, p1_direction])
                self.ticker2_operate.append([self.pair2_price[self.current_step], 1, p2_direction])

                self.current_cash = self.current_cash + 1 * self.pair2_price[self.current_step] * (1-0.005)\
                                    - self.shares * self.pair1_price[self.current_step] * (1+0.005)

            elif -self.current_stop_loss_boundary < current_z_score <= -self.current_trade_boundary:
                self.trade_open = True
                p1_direction = -1
                p2_direction = 1
                self.ticker1_operate.append([self.pair1_price[self.current_step], self.shares, p1_direction])
                self.ticker2_operate.append([self.pair2_price[self.current_step], 1, p2_direction])

                self.current_cash = self.current_cash + 1 * self.pair1_price[self.current_step] * (1 - 0.005) \
                                    - self.shares * self.pair2_price[self.current_step] * (1 + 0.005)

            p1_price = self.pair1_price[self.current_step]
            p2_price = self.pair2_price[self.current_step]
        elif self.trade_open:

            if abs(self.current_trade_boundary) <= abs(current_z_score) < abs(self.current_stop_loss_boundary):
                if self.done:
                    # Exit
                    self.trade_open = False
                    p1_price, p1_share, p1_direction = self.ticker1_operate.pop()
                    p2_price, p2_share, p2_direction = self.ticker2_operate.pop()
                    if p1_direction == 1: # ticker1 is 1 unit
                        self.current_cash = self.current_cash + (-1)*p1_price*p1_share*p1_direction*(1-0.005)\
                                            + (-1)*p2_share*p2_share*p2_direction*(1+0.005)


                    elif p1_direction == -1:
                        self.current_cash = self.current_cash + (-1)*p1_price*p1_share*p1_direction*(1+0.005)\
                                            + (-1)*p2_share*p2_share*p2_direction*(1-0.005)
                    reward = -500 * self.weights_calculate(1, self.shares)

            elif abs(current_z_score) <= abs(self.current_trade_boundary):
                # Close
                self.trade_open = False
                p1_price, p1_share, p1_direction = self.ticker1_operate.pop()
                p2_price, p2_share, p2_direction = self.ticker2_operate.pop()
                if p1_direction == 1:
                    self.current_cash = self.current_cash + (-1)*p1_price*p1_share*p1_direction*(1-0.005)\
                                        + (-1)*p2_share*p2_share*p2_direction*(1+0.005)
                elif p1_direction == -1:
                    self.current_cash = self.current_cash + (-1)*p1_price*p1_share*p1_direction*(1+0.005)\
                                        + (-1)*p2_share*p2_share*p2_direction*(1-0.005)

                reward = 1000 * self.weights_calculate(1, self.shares)

            elif abs(current_z_score) > abs(self.current_stop_loss_boundary):
                # stop-loss
                self.trade_open = False
                p1_price, p1_share, p1_direction = self.ticker1_operate.pop()
                p2_price, p2_share, p2_direction = self.ticker2_operate.pop()
                if p1_direction == 1:
                    self.current_cash = self.current_cash + (-1) * p1_price * p1_share * p1_direction * (1 - 0.005) \
                                        + (-1) * p2_share * p2_share * p2_direction * (1 + 0.005)
                elif p1_direction == -1:
                    self.current_cash = self.current_cash + (-1) * p1_price * p1_share * p1_direction * (1 + 0.005) \
                                        + (-1) * p2_share * p2_share * p2_direction * (1 - 0.005)

                reward = -1000 * self.weights_calculate(1, self.shares)

        self.current_step += 1
        self.done = self.current_step >= self.n - 1
        next_state = self.update()

        return next_state, reward, self.done, self.current_cash, current_z_score, p1_price, p1_direction, \
                p2_price, p2_direction

    def reset(self):
        self.trade_open = False
        self.done = False
        self.current_step = 0
        self.profit = 0
        self.ticker1_operate = []
        self.ticker2_operate = []

        obs = np.array([self.pair1_price[self.current_step],
                        self.pair2_price[self.current_step],
                        self.z_score[self.current_step]], dtype=np.float32)

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        return obs_tensor

    def soft_reset(self):
        self.current_step = 0
        self.current_cash = 10000
        self.ticker1_operate = []
        self.ticker2_operate = []

        obs = np.array([self.pair1_price[self.current_step],
                        self.pair2_price[self.current_step],
                        self.z_score[self.current_step]], dtype=np.float32)

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        return obs_tensor


    def update(self):
        obs = np.array([self.pair1_price[self.current_step-1],
                        self.pair2_price[self.current_step-1],
                        self.z_score[self.current_step-1]], dtype=np.float32)

        return obs



















