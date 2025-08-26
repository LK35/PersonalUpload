import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from numpy.linalg import inv, slogdet
from scipy.optimize import minimize
from scipy.stats import t, multivariate_normal, multivariate_t
from tqdm import tqdm


def standardize_X(X, per_target=False):
    """
    Standardize X (or list of Xs).
    Returns standardized X, mean(s), std(s).
    """
    if not per_target:
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0, ddof=0)
        std_X[std_X == 0] = 1.0  # Avoid divide by zero
        Xs = (X - mean_X) / std_X
        return Xs, mean_X, std_X
    else:
        Xs, means, stds = [], [], []
        for Xi in X:
            mean_i = np.mean(Xi, axis=0)
            std_i = np.std(Xi, axis=0, ddof=0)
            std_i[std_i == 0] = 1.0
            Xs.append((Xi - mean_i) / std_i)
            means.append(mean_i)
            stds.append(std_i)
        return Xs, means, stds


def build_kf_initial_params(N, K=None, K_list=None, X_shared=True, estimate_omega=True):
    if estimate_omega:
        omega_init = np.zeros(N)
    else:
        omega_init = np.array([])

    Phi_init = 0.1 * np.eye(N).flatten()

    if X_shared and K is not None:
        beta_init = np.zeros(K * N)
    elif not X_shared and K_list is not None:
        beta_init = np.concatenate([np.zeros(Kj) for Kj in K_list])
    else:
        beta_init = np.array([])

    L_Q_elements = np.ones(N * (N + 1) // 2)
    L_R_elements = np.ones(N * (N + 1) // 2)
    params = [Phi_init, beta_init, L_Q_elements, L_R_elements]
    if estimate_omega:
        params = [omega_init] + params
    return np.concatenate(params)


def build_kf_param_bounds(N, K=None, K_list=None, X_shared=True, estimate_omega=True):
    bounds = []

    if estimate_omega:
        bounds += [(-10, 10)] * N

    bounds += [(-10, 10)] * (N * N)  # Phi
    
    if X_shared and K is not None:
        bounds += [(-100000, 100000)] * (K * N)
    elif not X_shared and K_list is not None:
        for Kj in K_list:
            bounds += [(-100000, 100000)] * Kj
    
    bounds += [(0.001, 10)] * (N * (N + 1) // 2)  # L_Q
    bounds += [(0.001, 10)] * (N * (N + 1) // 2)  # L_R

    return bounds

def stabilize_phi(Phi, eps=1e-3):
    eigvals = np.linalg.eigvals(Phi)
    maxabs = np.max(np.abs(eigvals))
    if maxabs >= 1:
        Phi = Phi / (maxabs + eps)
    return Phi

def stabilize_pd(M, min_eig=1e-8):
    eigvals = np.linalg.eigvals(M)
    minval = np.min(eigvals)
    if minval < min_eig:
        M = M + np.eye(M.shape[0]) * (min_eig - minval)
    return M


def kalman_multivariate_loglik(params, y, X=None, N=None, X_shared=True, estimate_omega=True):
    T, N_y = y.shape
    N = N or N_y
    idx = 0

    # Omega
    if estimate_omega:
        omega = params[idx:idx+N]
        idx += N
    else:
        omega = np.zeros(N)

    Phi = params[idx:idx+N*N].reshape(N, N)
    idx += N*N
    Phi = stabilize_phi(Phi)  

    # X and beta for per-target
    if X is not None:
        if X_shared:
            K = X.shape[1]
            beta = params[idx:idx+K*N].reshape(K, N)
            idx += K*N
        else:
            K_list = [X[j].shape[1] for j in range(N)]
            beta_list = []
            for j in range(N):
                beta_j = params[idx:idx+K_list[j]]
                beta_list.append(beta_j)
                idx += K_list[j]
    else:
        beta_list = None
        beta = None

    L_Q_elements = params[idx:idx+N*(N+1)//2]
    idx += N*(N+1)//2

    L_R_elements = params[idx:idx+N*(N+1)//2]

    L_Q = np.zeros((N, N))
    L_Q[np.tril_indices(N)] = L_Q_elements
    Q = L_Q @ L_Q.T + 1e-6*np.eye(N)
    Q = stabilize_pd(Q)

    L_R = np.zeros((N, N))
    L_R[np.tril_indices(N)] = L_R_elements
    R = L_R @ L_R.T + 1e-6*np.eye(N)
    R = stabilize_pd(R)

    mu_pred = omega.copy() if estimate_omega else np.zeros(N)
    P_pred = np.eye(N)*1000

    loglik = 0.0

    for t in range(T):
        if X is not None:
            if X_shared:
                y_pred = mu_pred + (X[t] @ beta if beta is not None else 0.0)
            else:
                y_pred = np.array([mu_pred[j] + X[j][t] @ beta_list[j] for j in range(N)])
        else:
            y_pred = mu_pred

        S = P_pred + R
        resid = y[t] - y_pred

        try:
            inv_S = np.linalg.inv(S)
            ll_t = -0.5 * (np.log(np.linalg.det(S)) + resid @ inv_S @ resid + N*np.log(2*np.pi))
        except np.linalg.LinAlgError:
            return 1e12
        loglik += ll_t

        K_gain = P_pred @ inv_S
        mu_upd = mu_pred + K_gain @ resid
        P_upd = (np.eye(N) - K_gain) @ P_pred

        # Stationary: omega + Phi @ (mu_upd - omega)
        # Random walk: Phi @ mu_upd (since omega=0, Phi can be identity)
        mu_pred = omega + Phi @ (mu_upd - omega) if estimate_omega else Phi @ mu_upd
        P_pred = Phi @ P_upd @ Phi.T + Q

    return -loglik



def kalman_filter_multivariate(y, omega, Phi, Q, R, X=None, beta=None, X_shared=True, beta_list=None):
    T, N = y.shape
    mu_predicted = np.zeros((T, N))
    mu_filtered = np.zeros((T, N))
    P_filtered = np.zeros((T, N, N))
    kalman_gain = np.zeros((T, N, N))


    mu_pred = omega.copy()
    P_pred = np.eye(N)*1000

    for i in range(T):
        mu_predicted[i] = mu_pred  # Save predicted state before seeing y[i]
        if X is not None:
            if X_shared:
                y_pred = mu_pred + (X[i] @ beta if beta is not None else 0.0)
            else:
                y_pred = np.array([mu_pred[j] + X[j][i] @ beta_list[j] for j in range(N)])
        else:
            y_pred = mu_pred

        S = P_pred + R
        resid = y[i] - y_pred
        inv_S = np.linalg.inv(S)
        K_gain = P_pred @ inv_S

        mu_upd = mu_pred + K_gain @ resid
        mu_filtered[i] = mu_upd  # Save filtered state after update
        P_upd = (np.eye(N) - K_gain) @ P_pred
        P_filtered[i] = P_upd
        kalman_gain[i] = K_gain

        mu_pred = omega + Phi @ (mu_upd - omega)
        P_pred = Phi @ P_upd @ Phi.T + Q

    return mu_predicted, mu_filtered, P_filtered, kalman_gain

def multivariate_KF_with_estimation(
    y, 
    X=None, 
    initial_params=None, 
    verbose=True, 
    max_iter=10000, 
    X_shared=True,
    optim_method: str = "L-BFGS-B",
    ftol = 1e-9, 
    gtol = 1e-6, 
    optim_bounds = None,
    standardize: bool = False,
    burn_in: float = 0.0,  # Proportion of sample to burn 
    estimate_omega: bool = True
):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if X is not None:
        if X_shared and X.ndim == 1:
            X = X.reshape(-1, 1)
    T, N = y.shape

    # --- Standardization logic ---
    scaler_info = {}
    if standardize and X is not None:
        if X_shared:
            Xs, mean_X, std_X = standardize_X(X)
            X_input = Xs
            scaler_info["mean_X"] = mean_X
            scaler_info["std_X"] = std_X
            K = X_input.shape[1]
            K_list = None
        else:
            Xs, means_X, stds_X = standardize_X(X, per_target=True)
            X_input = Xs
            scaler_info["mean_X"] = means_X
            scaler_info["std_X"] = stds_X
            K = None
            K_list = [X_input[j].shape[1] for j in range(N)]
    else:
        X_input = X
        if X_shared and X is not None:
            K = X.shape[1]
            K_list = None
        elif not X_shared and X is not None:
            K = None
            K_list = [X[j].shape[1] for j in range(N)]
        else:
            K = None
            K_list = None

    # -- Build parameter vector and bounds --
    if initial_params is not None:
        init_params = initial_params
    else:
        init_params = build_kf_initial_params(N, K, K_list, X_shared, estimate_omega)
    if optim_bounds is None:
        bounds = build_kf_param_bounds(N, K, K_list, X_shared, estimate_omega)
    else:
        bounds = optim_bounds

    # -- Optimize --
    res = minimize(
        kalman_multivariate_loglik,
        init_params,
        args=(y, X_input, N, X_shared, estimate_omega),
        bounds=bounds,
        method=optim_method,
        options={
            'maxiter': max_iter,
            'maxfun': 20000,
            'ftol': ftol,
            'gtol': gtol,
            'disp': True
        }
    )
    if not res.success:
        if 'exceeds limit' in res.message.lower() and np.isfinite(res.fun):
            print("Warning: Optimization stopped at function/iteration limit, using last parameters.")
        else:
            raise RuntimeError("Kalman filter optimization failed: " + res.message)

    # -- Parse parameters --
    idx = 0
    if estimate_omega:
        omega_est = res.x[idx:idx + N]; idx += N
    else:
        omega_est = np.zeros(N)
    Phi_est = res.x[idx:idx + N * N].reshape(N, N); idx += N * N
    if X_input is not None:
        if X_shared:
            beta_est = res.x[idx:idx + K * N].reshape(K, N); idx += K * N
            beta_list_est = None
        else:
            beta_list_est = []
            for Kj in K_list:
                beta_j = res.x[idx:idx+Kj]
                beta_list_est.append(beta_j)
                idx += Kj
            beta_est = None
    else:
        beta_list_est = None
        beta_est = None
    L_Q_elements_est = res.x[idx:idx + N * (N + 1) // 2]; idx += N * (N + 1) // 2
    L_R_elements_est = res.x[idx:idx + N * (N + 1) // 2]
    L_Q_est = np.zeros((N, N))
    L_Q_est[np.tril_indices(N)] = L_Q_elements_est
    Q_est = L_Q_est @ L_Q_est.T + 1e-6 * np.eye(N)
    L_R_est = np.zeros((N, N))
    L_R_est[np.tril_indices(N)] = L_R_elements_est
    R_est = L_R_est @ L_R_est.T + 1e-6 * np.eye(N)

    # Filtering
    if X_input is not None:
        if X_shared:
            mu_predicted, mu_filtered, P_filtered, kalman_gain = kalman_filter_multivariate(
                y, omega_est, Phi_est, Q_est, R_est, X_input, beta_est
            )
        else:
            mu_predicted, mu_filtered, P_filtered, kalman_gain = kalman_filter_multivariate(
                y, omega_est, Phi_est, Q_est, R_est, X_input, None, X_shared, beta_list_est
            )
    else:
        mu_predicted, mu_filtered, P_filtered, kalman_gain = kalman_filter_multivariate(
            y, omega_est, Phi_est, Q_est, R_est
        )

    # Likelihood and info metrics
    final_neg_loglik = kalman_multivariate_loglik(
        res.x, y, X_input, N, X_shared, estimate_omega
    )
    loglik = -final_neg_loglik
    k = len(res.x)
    n_obs = T * N
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n_obs) - 2 * loglik

    if burn_in > 0.0:
        burn_t = int(np.floor(burn_in * T))
        y_cut = y[burn_t:]
        if X_input is not None:
            if X_shared:
                X_cut = X_input[burn_t:]
            else:
                X_cut = [X_input[j][burn_t:] for j in range(N)]
        else:
            X_cut = None
        T_cut = y_cut.shape[0]
        cut_loglik = -kalman_multivariate_loglik(
            res.x, y_cut, X_cut, N, X_shared, estimate_omega
        )
        cut_n_obs = T_cut * N
        cut_aic = 2 * k - 2 * cut_loglik
        cut_bic = k * np.log(cut_n_obs) - 2 * cut_loglik
    else:
        cut_loglik = loglik
        cut_aic = aic
        cut_bic = bic

    if verbose:
        print("✅ Kalman estimation completed successfully.")
        print("Estimated parameters:")
        print(f"omega (unconditional mean): {omega_est}")
        print(f"Phi (persistence matrix): \n{Phi_est}")
        if X_input is not None:
            if X_shared:
                print(f"beta: \n{beta_est}")
            else:
                for j, beta_j in enumerate(beta_list_est):
                    print(f"beta[{j}]: {beta_j}")
        print(f"Q (state noise covariance): \n{Q_est}")
        print(f"R (observation noise covariance): \n{R_est}")
        print("Log-likelihood:", loglik)
        print("AIC:", aic)
        print("BIC:", bic)
        if burn_in > 0.0:
            print(f"Post-burn-in (after {burn_t} obs):")
            print("Cut log-likelihood:", cut_loglik)
            print("Cut AIC:", cut_aic)
            print("Cut BIC:", cut_bic)

    return {
        'omega': omega_est,
        'Phi': Phi_est,
        'beta': beta_est if (X_input is not None and X_shared) else beta_list_est,
        'Q': Q_est,
        'R': R_est,
        'mu_predicted': mu_predicted,
        'mu_filtered': mu_filtered,
        'P_filtered': P_filtered,
        'kalman_gain': kalman_gain,
        'loglik': loglik,
        'aic': aic,
        'bic': bic,
        'cut_loglik': cut_loglik,
        'cut_aic': cut_aic,
        'cut_bic': cut_bic,
        'optimization_result': res
    }

