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

# 粒子フィルタ
def particle_filter(y, N, F, G, H, Q, R, x_0, P_0):
    """
    線形状態空間モデルに対する粒子フィルタ

    線形状態空間モデル:
        x_{t+1} = F * x_t + G * w_t,  w_t ~ N(0, Q),
        y_{t+1} = H * x_{t+1} + v_{t+1},  v_t ~ N(0, R),
        x_{0|-1} ~ N(x_0, P_0)
    
    Parameters
    ----------
    y: ndarray
        観測データ
    N: int
        粒子数
    F, G, H, Q, R: float
        線形状態空間モデルのパラメータ
    x_0: float
        x_{0|-1}の平均値
    P_0: float
        x_{0|-1}の共分散行列
    
    Returns
    ----------
    filtered_data: ndarray
        推定値
    """
    
    # 観測方程式の尤度関数
    def observed_likelihood(observation, latent):
        """
        観測方程式の尤度関数(正規分布)
        
        Parameters
        ----------
        observation: float
            観測量
        latent: ndarray
            状態量
        
        Return
        ----------
        normalized_alpha: ndarray
            規格化された尤度(リサンプリングの重み)
        """
        likelihood = np.exp(-(observation - H * latent)**2 / (2 * R**2)) / np.sqrt(2 * np.pi * R**2)
        return likelihood / np.sum(likelihood)
    
    # リサンプリング
    def resampling(random_data, weights, layer=True):
        """
        所与の重みでのリサンプリング
        
        Parameters
        ----------
        random_data: ndarray
            リサンプリング対象データ
        weights: ndarray
            重み係数
        layer: boolen
            層化リサンプリングの実施フラグ
        
        Return
        ----------
        resampled_data: ndarray
            リサンプリング後のデータ
        """
        L = len(random_data)
        resampled_data = np.zeros(L)
        if layer == True:
            xi = np.array([(i+0.5)/L for i in range(L)])
        else:
            xi = np.random.uniform(size=L)
        for j in range(L):
            for i in range(L):
                if i == 0:
                    if 0 < xi[j] <= weights[0]:
                        resampled_data[j] = random_data[0]
                        break
                else:
                    if np.sum(weights[:i-1]) < xi[j] <= np.sum(weights[:i]):
                        resampled_data[j] = random_data[i]
                        break
        return resampled_data
    
    # 状態変数の時間発展
    def system_evolution(recent_data):
        """
        状態方程式にしたがって粒子を時間発展
        
        Parameter
        ----------
        recent_data: ndarray
            現在時刻の状態変数
        
        Return
        ----------
            1期先の状態変数
        """
        return F * recent_data + Q * np.random.randn(len(recent_data))

    # フィルタリング
    filtered_data = []
    for i in range(len(y)):
        if i == 0:
            pred_x = x_0 + P_0 * np.random.randn(N)
        normalized_alpha = observed_likelihood(y[i], pred_x)
        flt_x = resampling(pred_x, normalized_alpha, layer=True)
        pred_x = system_evolution(flt_x)
        filtered_data.append(list(flt_x))
    return np.array(filtered_data).reshape(-1, N)

# 粒子フィルタの実行
if __name__ == '__main__':
    print("1次元ランダムウォークに対して粒子フィルタを実行")
    x, y = generate_1d_random_walk(1000, 1, 1)
    pf_x = particle_filter(y, N=50, F=1, G=1, H=1, Q=1, R=1, x_0=0, P_0=10)
    pf_x_mean = pf_x.mean(axis=1)
    plt.figure(figsize=(6.5, 4.0))
    plt.plot(x, alpha=0.6, label='State variable')
    plt.plot(pf_x_mean, alpha=0.6, label='Filtered variable')
    plt.xlim((0, len(x)))
    plt.xlabel("Time step")
    plt.ylabel("State variable")
    plt.legend(loc="upper right")
    plt.show()