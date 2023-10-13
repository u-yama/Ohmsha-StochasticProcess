import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t as StudentT

# RBFカーネル
def rbf_kernel(x1, x2, alpha=0.5, length_scale=1.0):
    """
    Radial Basis Function (RBF) kernel

    Parameters
    ----------
    x1: ndarry
        カーネル行列の第一引数
    x2: ndarry
        カーネル行列の第二引数
    alpha: float
        RBFカーネルのパラメータα
    length_scale: float
        RBFカーネルのパラメータl
    
    Returns
    -----------
        RBFカーネルを成分に持つカーネル行列
    """
    difference = np.expand_dims(x1, 1) - np.expand_dims(x2, 0)
    square_distance = difference ** 2
    return np.exp(-alpha * square_distance / length_scale**2)

# スチューデントのt過程回帰
def student_t_process_regression(x, y, x_new, nu=0.8, alpha=0.5, length_scale=1.0, noise_intensity=1e-10):
    """
    スチューデントのt過程回帰による予測分布を求める

    Parameters
    ----------
    x: ndarry
        学習データ(説明変数)
    y: ndarry
        学習データ(目的変数)
    x_new: ndarry
        予測対象の説明変数
    nu: float
        スチューデントのt分布の自由度
    alpha: float
        RBFカーネルのパラメータα
    length_scale: float
        RBFカーネルのパラメータl
    noise_intensiry: float
        回帰におけるノイズ強度

    Returns
    y_pred: ndarry
        平均の予測値
    std_pred: ndarry
        標準偏差の予測値
    ----------
    """
    K = rbf_kernel(x, x, alpha, length_scale) + noise_intensity * np.eye(len(x))
    K_inv = np.linalg.inv(K)
    
    K_s = rbf_kernel(x, x_new, alpha, length_scale)
    K_ss = rbf_kernel(x_new, x_new, alpha, length_scale)
    
    D = len(y)
    distance = y.dot(K_inv).dot(y)
    cov_gain = (nu + distance) / (nu + D)
    
    y_pred = K_s.T.dot(K_inv).dot(y)
    cov_pred = cov_gain * (K_ss - K_s.T.dot(K_inv).dot(K_s))
    std_pred = np.sqrt(np.diag(cov_pred))
    nu_pred = nu + D
    
    return y_pred, std_pred, nu_pred


# スチューデントのt過程回帰による予測
if __name__ == '__main__':
    x = np.linspace(0, 2*np.pi)
    y = np.sin(x) + 10e-5 * StudentT.rvs(0.8, size=len(x))
    split_position = -10
    x_known = x[:split_position]
    x_unknown = x[split_position:]
    y_known = y[:split_position]
    y_unknown = y[split_position:]

    y_pred, sigma_pred, _ = student_t_process_regression(x_known, y_known, x_unknown, noise_intensity=1e-8)
    plt.plot(x_known, y_known, color="C0", label="Learning data")
    plt.plot(x_unknown, y_unknown, color="C1", label="True data")
    plt.plot(x_unknown, y_pred, color="C2", label="Prediction curve")
    plt.fill_between(x_unknown, y_pred-sigma_pred, y_pred+sigma_pred, color="C2", alpha=0.4, label="Prediction uncertainity")
    plt.xlim((x[0], x[-1]))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc="upper right")
    plt.show()