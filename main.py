import statsmodels.api as sm
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
from datetime import date
import torch
import csv
from DeepQNetwork import *
from PairsTrading_Env import *
from ReplayBuffer import *

def all_seed(seed = 11):
    "seed"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def data_processing(data, col_name, split_date):
    piece = data[col_name]
    piece.index = piece.index.date
    piece_train = piece[piece.index <= split_date]
    piece_test = piece[piece.index > split_date]
    piece_train.index = pd.DatetimeIndex(piece_train.index).to_period('D')
    piece_test.index = pd.DatetimeIndex(piece_test.index).to_period('D')

    return piece_train, piece_test
def linear_regression(data1, data2):
    X = sm.add_constant(data1)
    y = data2
    model_OLS = sm.OLS(y, X).fit()
    residual = model_OLS.resid
    z_score = (residual - residual.mean()) / residual.std()
    average = z_score.mean()
    beta = model_OLS.params[1]  # hedge ratio
    return residual, z_score, average, beta

if __name__ == '__main__':

    cfg = {
        'n_actions': 6,  # 假设env已定义并有action_space属性
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_end': 0.0000001,
        'epsilon_decay': 0.9999,
        'batch_size': 64,
        'lr': 0.001,
        'train_eps': 20,
        'ep_max_steps': 10,
        'target_update': 10,
        'test_eps': 8,
        'buffer_size': 50
    }

    # window size = 15
    ticker1 = 'MSFT'
    ticker2 = 'JPM'
    start_date = '1990-01-02'
    end_date = '2024-10-31'
    current = 0# this will change to i next in the for loop
    multi_data =  yf.download([ticker1, ticker2], start=start_date, end=end_date)
    split_date = datetime.date(2015, 12, 31)
    # train: adj_train_price, adj_train_vol
    adj_train_price, adj_test_price = data_processing(multi_data, 'Adj Close', split_date)


    # diff to use: train
    p1_adj_train_price = adj_train_price[ticker1].values
    log_diff_p1 = np.diff(np.log(p1_adj_train_price))# the 1st element is not nan
    log_diff_p1 = np.insert(log_diff_p1, 0, 0)
    p2_adj_train_price = adj_train_price[ticker2].values
    log_diff_p2 = np.diff(np.log(p2_adj_train_price))# the 1st element is not nan
    log_diff_p2 = np.insert(log_diff_p2, 0, 0)

    # diff to use: test
    p1_adj_test_price = adj_test_price[ticker1].values
    log_diff_p1_test = np.diff(np.log(p1_adj_test_price))
    log_diff_p1_test = np.insert(log_diff_p1_test, 0, 0)
    p2_adj_test_price = adj_test_price[ticker2].values
    log_diff_p2_test = np.diff(np.log(p2_adj_test_price))
    log_diff_p2_test = np.insert(log_diff_p2_test, 0, 0)

    # formation_window, window_size
    formation_window = 120
    window_size = 60
    model = DQN(state_size=3, action_size=cfg['n_actions'])
    memory = ReplayMemory(cfg['buffer_size'])
    agent = Agent(model, memory, cfg, is_eval=False)

    print('Starting Training!')
    all_seed(seed=11)
    traing_dict = {}
    for e in range(cfg['train_eps']):
        print("Running episode: " + str(e+1) + "/" + str(cfg['train_eps']))

        ep_reward = 0
        ep_step = 0
        ep_profit = 0
        #ii = 0
        for steps in range(0, len(adj_train_price) - formation_window - window_size, formation_window - window_size):
            #ii += 1
            window_profit = 0
            #print(steps, ii)
            # formation_window data
            formation1_price = log_diff_p1[steps:steps+formation_window]
            formation2_price = log_diff_p2[steps:steps+formation_window]
            _, _, f_average, f_beta = linear_regression(formation1_price, formation2_price)

            # trading window
            trading1_price = log_diff_p1[steps+formation_window:steps+formation_window+window_size]
            trading2_price = log_diff_p2[steps+formation_window:steps+formation_window+window_size]
            # t_residual is the spread
            _, t_z_score, _, _ = linear_regression(trading1_price, trading2_price)

            pair1_price = p1_adj_train_price[steps+formation_window:steps+formation_window+window_size]
            pair2_price = p2_adj_train_price[steps+formation_window:steps+formation_window+window_size]

            env = PairsTradingEnv(window_size, pair1_price, pair2_price, t_z_score, f_average, f_beta)
            state = env.soft_reset()
            # ticker1_operate = []
            # ticker2_operate = []
            for _ in range(window_size):
                action = agent.action(state)
                next_state, reward, done, cash,z, p1_price, p1_direction,\
                    p2_price, p2_direction = env.step(action)

                if e == cfg['train_eps']-1:
                    with open('training.csv','a',newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['cash','p1_price','p1_direction','p2_price','p2_direction'])
                        writer.writerow([cash,p1_price,p1_direction,p2_price,p2_direction])

                if done:
                    break
                state = next_state
                agent.update()
                ep_reward += reward

            window_profit += cash - 10000

        ep_profit += window_profit

        if (e + 1) % cfg['target_update'] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (e + 1) % 1 == 0:
            print(f"Episode: {e+1}/{cfg['train_eps']}, Reward {ep_reward:.2f},"
                  f" Profit{ep_profit:.2f}, Epsilon:{agent.epsilon:.3f}, Action:{action}")

        if e == cfg['train_eps'] - 1:
            torch.save({
                'epoch': e,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'loss': ep_reward,
            }, f'checkpoint_{e + 1}.pth')

        env.reset()
    print('Training Finished!')
    env.close()

    checkpoint = torch.load('checkpoint_20.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    memory = ReplayMemory(cfg['buffer_size'])
    agent = Agent(model, memory, cfg, is_eval=True)

    print('Starting Testing!')
    for e in range(cfg['test_eps']):
        print("Running episode: " + str(e+1) + "/" + str(cfg['test_eps']))

        ep_reward = 0
        ep_step = 0
        ep_profit = 0
        #ii = 0
        for steps in range(0, len(adj_test_price) - formation_window - window_size, formation_window - window_size):
            #ii += 1
            window_profit = 0
            #print(steps, ii)
            # formation_window data
            formation1_price = log_diff_p1_test[steps:steps+formation_window]
            formation2_price = log_diff_p2_test[steps:steps+formation_window]
            _, _, f_average, f_beta = linear_regression(formation1_price, formation2_price)

            # trading window
            trading1_price = log_diff_p1_test[steps+formation_window:steps+formation_window+window_size]
            trading2_price = log_diff_p2_test[steps+formation_window:steps+formation_window+window_size]
            # t_residual is the spread
            _, t_z_score, _, _ = linear_regression(trading1_price, trading2_price)

            pair1_price = p1_adj_train_price[steps+formation_window:steps+formation_window+window_size]
            pair2_price = p2_adj_train_price[steps+formation_window:steps+formation_window+window_size]

            env = PairsTradingEnv(window_size, pair1_price, pair2_price, t_z_score, f_average, f_beta)
            state = env.soft_reset()
            # ticker1_operate = []
            # ticker2_operate = []
            for _ in range(window_size):
                action = agent.predict_action(state)
                next_state, reward, done, cash, z, p1_price, p1_direction, \
                    p2_price, p2_direction = env.step(action)

                if e == cfg['test_eps']-1:
                    with open('test.csv','a',newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['cash','p1_price','p1_direction','p2_price','p2_direction'])
                        writer.writerow([cash,p1_price,p1_direction,p2_price,p2_direction])

                if done:
                    break
                state = next_state
                agent.update()
                ep_reward += reward

            window_profit += cash - 10000

        ep_profit += window_profit

        # if (e + 1) % cfg['target_update'] == 0:
        #     agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (e + 1) % 1 == 0:
            print(f"Episode: {e+1}/{cfg['test_eps']}, Reward {ep_reward:.2f},"
                  f" Profit{ep_profit:.2f}, Action:{action}")

        env.reset()
    print('Testing Finished!')
    env.close()

