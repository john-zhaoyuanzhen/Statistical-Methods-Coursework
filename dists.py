from scipy.stats import uniform, truncnorm, truncexpon, crystalball
import numpy as np


# Define the truncated distributions with given parameters
def g_s(x, beta, m, mu, sigma):
    x = np.clip(x, 0, 5) # Truncate x to [0, 5]
    cb = crystalball(beta=beta, m=m, loc=mu, scale=sigma)
    return cb.pdf(x) / (cb.cdf(5) - cb.cdf(0))

def h_s(x, lmda):
    return truncexpon.pdf(x, b=10*lmda, loc=0, scale=1/lmda)

def g_b(x):
    return uniform.pdf(x, scale=5)

def h_b(x, mu_b, sigma_b):
    return truncnorm.pdf(x, a=(0-mu_b)/sigma_b, b=(10-mu_b)/sigma_b, loc=mu_b, scale=sigma_b)

# Joint density function
def s_xy(y, x, beta, m, mu, sigma, lmda):
    return g_s(x, beta, m, mu, sigma) * h_s(y, lmda)

def b_xy(y, x, mu_b, sigma_b):
    return g_b(x) * h_b(y, mu_b, sigma_b)

def f_xy(y, x, mu, sigma, beta, m, f, lmda, mu_b, sigma_b):
    return f * g_s(x, beta, m, mu, sigma) * h_s(y, lmda) + (1 - f) * g_b(x) * h_b(y, mu_b, sigma_b)