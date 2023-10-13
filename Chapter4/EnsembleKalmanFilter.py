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

# アンサンブルカルマンフィルタ
def ensemble_kalman_filter(y, M, F=1, G=1, H=1, Q=1, R=1, x0=0, P0=0.1):
    """
    線形状態空間モデルに対するアンサンブルカルマンフィルタ

    線形状態空間モデル
    X_{t+1} = F * X_t + G * w_t,  w_t ~ N(0, Q),
    Y_{t+1} = H * X_{t+1} + v_{t+1},  v_{t+1} ~ N(0, R)
    
    Parameters
    ----------
    y: ndarray
        観測データ
    M: int
        アンサンブル数
    F, G, H, Q, R: float
        線形状態空間モデルのパラメータ
    x0: float
        初期分布の丙吉
    P0: float
        初期分布の共分散行列
    
    Return
    ----------
    estimated_data: ndarray
        推定値
    """
    estimated_data = []
    for i in range(len(y)):
        if i == 0:
            pred_x = x0 + P0 * np.random.randn(M)
        # フィルタリング
        pred_y = H * pred_x + R * np.random.randn(M)
        tilde_x = pred_x - np.mean(pred_x)
        tilde_y = pred_y - np.mean(pred_y)
        V = tilde_y @ tilde_y.T / (M-1)
        U = tilde_x @ tilde_y.T / (M-1)
        K = U / V
        flt_x = pred_x + K * (y[i] - pred_y)
        estimated_data.append(np.mean(flt_x))
        # 1期先予測
        pred_x = F * flt_x + G * Q * np.random.randn(M)
    return estimated_data

# アンサンブルカルマンフィルタの実行
if __name__ == '__main__':
    print("1次元ランダムウォークに対してアンサンブルカルマンフィルタを実行")
    x, y = generate_1d_random_walk(1000, 1, 1)
    enkf_x = ensemble_kalman_filter(y, M=50, F=1, G=1, H=1, Q=1, R=1, x0=0, P0=0.01)
    plt.figure(figsize=(6.5, 4.0))
    plt.plot(x, alpha=0.6, label='State variable')
    plt.plot(enkf_x, alpha=0.6, label='Filtered variable')
    plt.xlim((0, len(x)))
    plt.xlabel("Time step")
    plt.ylabel("State variable")
    plt.legend(loc="upper right")
    plt.show()