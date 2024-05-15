import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


Index = 0
k_jam = 0.145
w = 5
Num = 1
car_length = 5.5
bound = [0.001, 0.5]

# number of lanes
Lane = [3, 4, 3, 3, 3.32]
Color = ['#fc6f68', '#ffda43', '#63e5b3', '#7079de', '#b883d4']
plt.rcParams['font.sans-serif'] = ['Times New Roman']


def MFD(lambda_use, uf, k_jam, Q, w, k):
    """

    :param lambda_use: smoothing parameter λ
    :param uf: free-flow speed
    :param k_jam: jam density
    :param Q: intersection capacity
    :param w: backward wave speed
    :param k: density
    :return: predict Q
    """

    flow = -lambda_use * np.log(np.exp(-uf * k / lambda_use) + np.exp(-Q / lambda_use) +
                                np.exp(-(k_jam - k) * w / lambda_use))

    return flow


def uMFD(uf, k_jam, Q, w, k):
    """

    :param lambda_use: smoothing parameter λ
    :param uf: free-flow speed
    :param k_jam: jam density
    :param Q: intersection capacity
    :param w: backward wave speed
    :param k: density
    :return: predict Q
    """

    flow = np.min(np.vstack((uf * k, np.ones(len(k)) * Q, (k_jam - k) * w)), axis=0)

    return flow


def find_lambda_use(bound, k_real, flow_real, uf, k_jam, Q, w):
    """

    :param bound: range of λ
    :param k_real: real density
    :param flow_real: real flow
    :param uf: free-flow speed
    :param k_jam: jam density
    :param Q: intersection capacity
    :param w: backward wave speed
    :return: predict Q
    """

    loss = 10000
    lambda_now = 0
    for lambda_use in np.arange(bound[0], bound[1], 0.001):
        flow_pre = MFD(lambda_use, uf, k_jam, Q, w, k_real)
        loss_now = np.sum(np.square(flow_real - flow_pre))
        if loss_now < loss:
            lambda_now = lambda_use
            loss = loss_now
    return lambda_now, loss


# switch to veh/m
k = np.asarray(pd.read_csv('./data/region_data/b_reg_full_' + str(Index) + '.csv').iloc[:, 3317:3461]) / car_length / 100
# switch to veh/s
flow = np.asarray(pd.read_csv('./data/region_data/q_reg_full_' + str(Index) + '.csv').iloc[:, 3317:3461]) / 600 / Lane[Index]
# switch to m/s
speed = np.asarray(pd.read_csv('./data/region_data/v_reg_full_' + str(Index) + '.csv').iloc[:, 3317:3461]) / 3.6

k_mean = np.mean(k, axis=0)
flow_mean = np.mean(flow, axis=0)
speed_mean = np.mean(speed, axis=0)

Q1 = np.max(flow_mean)
Q = np.max(flow)
uf = np.mean(speed_mean[np.argsort(speed_mean)[-Num:]])

print('Q=' + str(Q))
print('Q1=' + str(Q1))
print('uf=' + str(uf))
print('max: ' + str(np.max(k_mean)))
print("="*10)

loss = 10000
lambda_out = 0
w_out = 0
k_jam_out = 0

for k_jam_now in k_jam:
    for w_now in w:
        lambda_now, loss_now = find_lambda_use(bound, k_mean, flow_mean, uf, k_jam_now, Q, w_now / 3.6)
        if loss_now < loss:
            loss = loss_now
            lambda_out = lambda_now
            w_out = w_now
            k_jam_out = k_jam_now
            print('tmp_lambda: ' + str(lambda_out))
            print('tmp_w: ' + str(w_out))
            print('tmp_k_jam: ' + str(k_jam_out))
            print("="*10)

print('output_lambda: ' + str(lambda_out))
print('output_w: ' + str(w_out))
print('output_k_jam: ' + str(k_jam_out))
print("="*10)

# visualization
fig, axs = plt.subplots(1, 1, dpi=900)
fig.set_size_inches(3000 / 900, 3000 / 900)
plt.subplots_adjust(top=0.95, bottom=0.10, right=0.95, left=0.15, hspace=0, wspace=0)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)

plt.tick_params(labelsize=12, width=0.2, direction='in')

axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)

plt.grid(axis='y', linewidth=0.2)

k = np.arange(0, k_jam_out, 0.001)
Q_bound = uMFD(uf, k_jam_out, Q, w_out / 3.6, k)
Q_pre = MFD(lambda_out, uf, k_jam_out, Q, w_out / 3.6, k)
plt.plot(k, Q_pre, color='b', linewidth=0.5)
plt.scatter(k_mean, flow_mean, color=Color[Index], s=5)
plt.plot(k, Q_bound, color='k', linewidth=0.8, label = str(Q))
plt.legend()

plt.xlim(0, 0.15)
plt.ylim(0, 0.4)

plt.xticks(np.arange(0, 0.151, 0.03))
plt.yticks(np.arange(0, 0.401, 0.08))

plt.savefig('./figure/figure' + str(Index) + '.png', dpi=600)
plt.close()
