from numba_stats import uniform, truncnorm, truncexpon, crystalball
import numpy as np


# Define the truncated distributions with given parameters using numba_stats
def g_s(x, beta, m, mu, sigma):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0., 5.) # Truncate x to [0, 5]
    num = crystalball.pdf(x.reshape(-1), beta=beta, m=m, loc=mu, scale=sigma).reshape(x.shape)
    denom = crystalball.cdf(5., beta=beta, m=m, loc=mu, scale=sigma) - crystalball.cdf(0., beta=beta, m=m, loc=mu, scale=sigma)
    return num / denom

def h_s(x, lmda, xmin=0., xmax=10.):
    x = np.asarray(x, dtype=np.float64)
    return truncexpon.pdf(x, xmin=xmin, xmax=xmax, loc=0., scale=1/lmda)

def g_b(x):
    x = np.asarray(x, dtype=np.float64)
    return uniform.pdf(x, a = 0., w = 5.)

def h_b(x, mu_b, sigma_b):
    x = np.asarray(x, dtype=np.float64)
    return truncnorm.pdf(x, xmin=0., xmax=10., loc=mu_b, scale=sigma_b)

# Joint density function
def s_xy(y, x, beta, m, mu, sigma, lmda):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return g_s(x, beta, m, mu, sigma) * h_s(y, lmda)

def b_xy(y, x, mu_b, sigma_b):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return g_b(x) * h_b(y, mu_b, sigma_b)

def f_xy(y, x, mu, sigma, beta, m, f, lmda, mu_b, sigma_b):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return f * g_s(x, beta, m, mu, sigma) * h_s(y, lmda) + (1. - f) * g_b(x) * h_b(y, mu_b, sigma_b)