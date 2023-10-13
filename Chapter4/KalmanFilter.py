import numpy as np
from matplotlib import pyplot as plt

# 1次元ランダムウォークを生成
def generate_1d_random_walk(length, std_latent, std_observation):
    """
    1次元ランダムウォークを生成
    
    Parameters
    ----------
    length: int
        ランダムウォークのデータ超
    std_latent: float
        状態変数に印加するホワイトノイズの強度
    std_onbsevation: float
        観測変数に印加するホワイトノイズの強度 
        
    Returns
    ----------
    X: ndarray
        状態変数の系列
    Y: ndarray
        観測変数の系列
    """
    
    X = np.zeros(length)
    Y = np.zeros(length)
    W_x = np.random.randn(length-1)
    W_y = np.random.randn(length)
    for i in range(length-1):
        X[i+1] = X[i] + std_latent * W_x[i]
    Y = X + std_observation * W_y
    return X, Y

# カルマンフィルタ
def kalman_filter(y, F, G, H, Q, R, x_0, P_0):
    """
    線形状態空間モデルに対するカルマンフィルタ

    線形状態空間モデル:
        x_{t+1} = F * x_t + G * w_t,  w_t ~ N(0, Q),
        y_{t+1} = H * x_{t+1} + v_{t+1},  v_{t+1} ~ N(0, R)
    
    Parameters
    ----------
    y: ndarray
        観測データ
    F: float
        システム行列
    G: float
        駆動行列
    H: float
        観測行列
    Q: float
        システムノイズの共分散行列
    R: float
        観測ノイズの共分散行列
    x_0: float
        状態変数の初期値
    P_0: float
        濾過行列の初期値
    
    Returns
    ----------
    flt_x: ndarray
        推定された状態変数
    flt_P: ndarray
        濾過行列
    """
    
    flt_x = np.zeros(len(y))
    flt_P = np.zeros(len(y))
    pred_x = x_0
    pred_P = P_0
    for i in range(len(y)):
        # フィルタリング
        K = pred_P * H / (H * pred_P * H + R)
        flt_x[i] = pred_x + K * (y[i] - H * pred_x)
        flt_P[i] = pred_P - K * H * pred_P
        # 時間発展
        pred_x = F * flt_x[i]
        pred_P = F * flt_P[i] * F + G * Q *G
    return flt_x, flt_P

# カルマンフィルタの実行
if __name__ == '__main__':
    print("1次元ランダムウォークに対してカルマンフィルタを実行")
    x, y = generate_1d_random_walk(1000, 1, 1)
    filtered_x, filtered_P  = kalman_filter(y, F=1, G=1, H=1, Q=1, R=1, x_0=0, P_0=1)
    plt.figure(figsize=(6.5, 4.0))
    plt.plot(x, alpha=0.6, label='State variable')
    plt.plot(filtered_x, alpha=0.6, label='Filtered variable')
    plt.xlim((0, len(x)))
    plt.xlabel("Time step")
    plt.ylabel("State variable")
    plt.legend(loc="upper right")
    plt.show()