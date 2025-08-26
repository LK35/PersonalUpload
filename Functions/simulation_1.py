import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.stats import t, multivariate_normal, multivariate_t



def simulate_univariate_state_space(T, K, c, phi, Q, R, beta=None, seed=None, use_intercept=True):
    """
    Simulate a univariate linear Gaussian state-space model with optional intercept in regressors.

    Model:
        Observation: y_t = mu_t + X_t @ beta + epsilon_t, epsilon_t ~ N(0, R)
        State:      mu_{t+1} = c + phi * mu_t + eta_t, eta_t ~ N(0, Q)

    Parameters:
    -----------
    T : int
        Number of time points.
    K : int
        Number of regressors (including intercept if use_intercept=True).
    c : float
        State intercept.
    phi : float
        AR(1) coefficient in state equation.
    Q : float
        Variance of state noise.
    R : float
        Variance of observation noise.
    beta : ndarray (K,) or None
        Regressor coefficients. If use_intercept=True, first element is intercept.
    seed : int or None
        Random seed.
    use_intercept : bool
        Whether to include a column of 1s in regressors.

    Returns:
    --------
    y : ndarray (T,)
        Simulated observations.
    mu : ndarray (T,)
        Simulated latent state.
    X : ndarray (T, K) or None
        Simulated regressors (including intercept column if applicable).
    """
    if seed is not None:
        np.random.seed(seed)

    if beta is not None:
        beta = np.asarray(beta)
        if use_intercept:
            K_eff = len(beta) - 1
            X_raw = np.random.normal(size=(T, K_eff))
            X = np.ones((T, len(beta)))
            X[:, 1:] = X_raw
        else:
            X = np.random.normal(size=(T, len(beta)))
    else:
        X = None

    mu = np.zeros(T)
    y = np.zeros(T)

    mu[0] = c / (1 - phi)

    for t in range(T):
        x_beta = X[t] @ beta if beta is not None else 0.0
        y[t] = mu[t] + x_beta + np.random.normal(scale=np.sqrt(R))

        if t < T - 1:
            mu[t + 1] = c + phi * mu[t] + np.random.normal(scale=np.sqrt(Q))

    return y.reshape(-1,1), mu.reshape(-1,1), X


from scipy.stats import multivariate_t

def simulate_multivariate_state_space(
    T, N, c, Phi, Q, R, beta=None, seed=None, use_intercept=True,
    t_noise=False, nu=10
):
    """
    Simulate a multivariate linear state-space model with optional t-distributed observation noise.

    Parameters
    ----------
    ... (as before) ...
    t_noise : bool
        If True, use multivariate t-distributed observation noise. Else, Gaussian.
    nu : float
        Degrees of freedom for t-distribution (if t_noise=True).
    """
    if seed is not None:
        np.random.seed(seed)
    
    Phi = np.asarray(Phi).reshape(N, N)
    Q = np.asarray(Q).reshape(N, N)
    R = np.asarray(R).reshape(N, N)

    if beta is not None:
        beta = np.asarray(beta)
        K = beta.shape[0]
        if use_intercept:
            K_eff = K - 1
            X_raw = np.random.normal(size=(T, K_eff))
            X = np.ones((T, K))
            X[:, 1:] = X_raw
        else:
            X = np.random.normal(size=(T, K))
    else:
        X = None

    mu = np.zeros((T, N))
    y = np.zeros((T, N))

    mu[0] = np.linalg.solve(np.eye(N) - Phi, c)

    for t in range(T):
        if t < T - 1:
            mu[t + 1] = c + Phi @ mu[t] + np.random.multivariate_normal(np.zeros(N), Q)
        # --- Observation noise: Gaussian or Student-t ---
        if beta is not None:
            signal = mu[t] + X[t] @ beta
        else:
            signal = mu[t]
        if t_noise:
            obs_noise = multivariate_t.rvs(loc=np.zeros(N), shape=R, df=nu)
        else:
            obs_noise = np.random.multivariate_normal(np.zeros(N), R)
        y[t] = signal + obs_noise

    return y, mu, X
