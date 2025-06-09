import numpy as np

def logpdf(x, mean, cov):
    """Minimal logpdf for multivariate normal."""
    k = len(mean)
    x_m = x - mean
    return -0.5 * (np.log(np.linalg.det(cov)) + k * np.log(2 * np.pi) + x_m.T @ np.linalg.inv(cov) @ x_m) 