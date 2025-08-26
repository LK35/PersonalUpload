import numpy as np
from numpy.linalg import inv, pinv, slogdet
from scipy.optimize import minimize
from scipy.special import gammaln
from numpy import tril_indices, triu_indices
from typing import List, Literal, Optional, Union

def student_t_logpdf(y, mu, Omega, nu, log_output=True, min_det=1e-6):
    R = Omega.shape[0]
    diff = y - mu
    try:
        sign, logdet_Omega = slogdet(Omega)
        if sign <= 0 or logdet_Omega < np.log(min_det):
            logdet_Omega = np.log(min_det)
        L = np.linalg.cholesky(Omega + 1e-8 * np.eye(R))
        inv_sqrt_Omega = inv(L.T)
    except np.linalg.LinAlgError:
        return -1e12
    eps = inv_sqrt_Omega @ diff
    mahal = np.clip(eps.T @ eps, a_min=1e-8, a_max=1e8)
    w = 1 + (1 / nu) * mahal
    log_pdf = (
        gammaln((nu + R) / 2)
        - gammaln(nu / 2)
        - (R / 2) * np.log(np.pi)
        - (R / 2) * np.log(nu)
        - 0.5 * logdet_Omega
        - ((nu + R) / 2) * np.log(w)
    )
    if not np.isfinite(log_pdf):
        return -1e12
    return log_pdf if log_output else np.exp(log_pdf)

def get_param_size(structure, R):
    return {
        "scalar": 1,
        "diagonal": R,
        "full": R * R,
        "lower": R * (R + 1) // 2,
        "upper": R * (R + 1) // 2
    }[structure]



def build_initial_params(N, P_or_Plist, phi_type, kappa_type, fix_nu=None, X_shared=True, estimate_omega=True, verbose=False):
    init_phi = {
        "scalar": np.array([0.5]),
        "diagonal": 0.5 * np.ones(N),
        "full": 0.5 * np.eye(N).flatten(),
        "lower": 0.5 * np.ones(N * (N + 1) // 2),
        "upper": 0.5 * np.ones(N * (N + 1) // 2)
    }[phi_type]
    init_lambda = np.zeros(N)
    init_kappa = {
        "scalar": np.array([0.5]),
        "diagonal": 0.5 * np.ones(N),
        "full": 0.5 * np.eye(N).flatten(),
        "lower": 0.5 * np.ones(N * (N + 1) // 2),
        "upper": 0.5 * np.ones(N * (N + 1) // 2)
    }[kappa_type]
    if estimate_omega:
        init_omega = np.zeros(N)
    else:
        init_omega = np.array([])  # empty
    if X_shared:
        P = P_or_Plist
        init_beta = np.zeros((P, N)).flatten()
    else:
        P_list = P_or_Plist
        init_beta = np.concatenate([np.zeros(Pi) for Pi in P_list])
    init_nu = [] if fix_nu is not None else [5.0]
    # Only include omega if estimating
    init_params = np.concatenate([init_phi, init_omega, init_beta, init_lambda, init_kappa, init_nu])
    if verbose:
        print("Initial parameters:")
        print("Phi:", init_phi)
        print("Omega:", init_omega)
        print("Beta:", init_beta)
        print("Lambda:", init_lambda)
        print("Kappa:", init_kappa)
        print("Nu:", init_nu if fix_nu is None else fix_nu)
    return init_params


def build_gas_param_bounds(
    R, P_or_Plist, phi_type, kappa_type, fix_nu=None, X_shared=True, cL=None, estimate_omega=True
):
    bounds = []
    # Phi
    if phi_type == "full":
        bounds += [(-0.995, 0.995)] * (R * R)
    elif phi_type == "diagonal":
        bounds += [(-0.995, 0.995)] * R
    elif phi_type == "scalar":
        bounds += [(-0.995, 0.995)]
    elif phi_type in ["lower", "upper"]:
        bounds += [(-0.995, 0.995)] * (R * (R + 1) // 2)
    # omega (only if estimated)
    if estimate_omega:
        bounds += [(-10, 10)] * R
    # Beta
    if X_shared:
        P = P_or_Plist
        bounds += [(-100000, 100000)] * (P * R)
    else:
        for Pi in P_or_Plist:
            bounds += [(-100000, 100000)] * Pi
    # Lambda (scale)
    cL = cL or R
    bounds += [(-10, 10)] * cL
    # # Kappa
    if kappa_type == "full":
        # Diagonal strictly positive, off-diagonal can be negative
        for i in range(R):
            for j in range(R):
                if i == j:
                    bounds.append((0, 10))
                else:
                    bounds.append((-10, 10))
    elif kappa_type == "diagonal":
        bounds += [(0, 10)] * R
    elif kappa_type == "scalar":
        bounds += [(0, 10)]
    elif kappa_type in ["lower", "upper"]:
        # Bounds for vectorized lower/upper-triangular Kappa
        idxs = np.tril_indices(R) if kappa_type == "lower" else np.triu_indices(R)
        for i, j in zip(*idxs):
            if i == j:
                bounds.append((0, 10))
            else:
                bounds.append((-10, 10))

    # nu
    if fix_nu is None:
        bounds += [(5.0, 500.0)]
    return bounds

def stabilize_phi(Phi, eps=1e-3):
    """Force the largest eigenvalue inside unit circle, unless already strictly inside."""
    eigvals = np.linalg.eigvals(Phi)
    maxabs = np.max(np.abs(eigvals))
    if maxabs >= 1:
        Phi = Phi / (maxabs + eps)
    return Phi

def stabilize_pd(M, min_eig=1e-8):
    """Ensure symmetric matrix is positive definite by bumping up diagonal if needed."""
    eigvals = np.linalg.eigvals(M)
    minval = np.min(eigvals)
    if minval < min_eig:
        M = M + np.eye(M.shape[0]) * (min_eig - minval)
    return M


def clip_resid(resid, clip_val=1e6):
    return np.clip(resid, -clip_val, clip_val)


def extract_matrix(params, idx, N, mat_type):
    if mat_type == "full":
        mat = params[idx:idx + N * N].reshape(N, N)
        idx += N * N
    elif mat_type == "diagonal":
        diag_vec = params[idx:idx + N]
        mat = np.diag(diag_vec)
        idx += N
    elif mat_type == "scalar":
        scalar_val = params[idx]
        mat = np.eye(N) * scalar_val
        idx += 1
    elif mat_type == "lower":
        tril_idx = np.tril_indices(N)
        L = np.zeros((N, N))
        L[tril_idx] = params[idx:idx + len(tril_idx[0])]
        mat = L
        idx += len(tril_idx[0])
    elif mat_type == "upper":
        triu_idx = np.triu_indices(N)
        U = np.zeros((N, N))
        U[triu_idx] = params[idx:idx + len(triu_idx[0])]
        mat = U
        idx += len(triu_idx[0])
    else:
        raise ValueError(f"Invalid matrix type: {mat_type}")


    return mat, idx


def standardize_X(X, mean=None, std=None, per_target=False):
    """
    Standardizes X to zero mean and unit variance.
    If mean/std provided, applies them. If not, computes from X.
    Returns X_standardized, mean, std.
    - X: np.ndarray or list of np.ndarray (for per-target)
    - per_target: whether X is a list (per target) or shared
    """
    if not per_target:
        if mean is None:
            mean = X.mean(axis=0)
        if std is None:
            std = X.std(axis=0, ddof=0)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        Xs = (X - mean) / std
        return Xs, mean, std
    else:
        # X is a list of [X0, ..., XR-1]
        means, stds, Xs = [], [], []
        for Xi in X:
            m = Xi.mean(axis=0)
            s = Xi.std(axis=0, ddof=0)
            s = np.where(s == 0, 1, s)
            Xs.append((Xi - m) / s)
            means.append(m)
            stds.append(s)
        return Xs, means, stds

def gas_general_filter(
    params, Y, X, cL, cK, fix_nu=None,
    kappa_type="diagonal", phi_type="full",
    X_shared=True, P_list=None, estimate_omega=True
):
    if not np.isfinite(params).all():
        print("Non-finite parameters at start:", params)
        return 1e12  # Large penalty for optimizer
    T, R = Y.shape
    idx = 0
    # Phi structure
    if phi_type == "diagonal":
        diag_phi = params[idx:idx + R]; idx += R
        Phi = np.diag(diag_phi)
    elif phi_type == "full":
        Phi = params[idx:idx + R * R].reshape(R, R); idx += R * R
    elif phi_type == "scalar":
        phi_scalar = params[idx]; idx += 1
        Phi = np.eye(R) * phi_scalar
    elif phi_type == "lower":
        L = np.zeros((R, R))
        tril_idx = tril_indices(R)
        L[tril_idx] = params[idx:idx + len(tril_idx[0])]; idx += len(tril_idx[0])
        Phi = L
    elif phi_type == "upper":
        U = np.zeros((R, R))
        triu_idx = triu_indices(R)
        U[triu_idx] = params[idx:idx + len(triu_idx[0])]; idx += len(triu_idx[0])
        Phi = U
    else:
        raise ValueError("phi_type must be 'full', 'diagonal', 'scalar', 'lower', or 'upper'")
    
    Phi = stabilize_phi(Phi)
    
    
    if estimate_omega:
        omega = params[idx:idx + R]; idx += R
    else:
        omega = np.zeros(R)
    if X_shared:
        P = X.shape[1]
        beta = params[idx:idx + P * R].reshape(P, R); idx += P * R
    else:
        beta = []
        for i, Pi in enumerate(P_list):
            beta_i = params[idx:idx + Pi]
            beta.append(beta_i)
            idx += Pi
    lambda_vec = params[idx:idx + cL]; idx += cL
    # Kappa
    if kappa_type == "diagonal":
        kappa_vec = params[idx:idx + R]; idx += R
        Kappa = np.diag(kappa_vec)
    elif kappa_type == "full":
        kappa_vec = params[idx:idx + R * R]; idx += R * R
        Kappa = kappa_vec.reshape(R, R)
    elif kappa_type == "scalar":
        kappa_scalar = params[idx]; idx += 1
        Kappa = np.eye(R) * kappa_scalar
    elif kappa_type == "lower":
        L = np.zeros((R, R))
        tril_idx = tril_indices(R)
        L[tril_idx] = params[idx:idx + len(tril_idx[0])]; idx += len(tril_idx[0])
        Kappa = L
    elif kappa_type == "upper":
        U = np.zeros((R, R))
        triu_idx = triu_indices(R)
        U[triu_idx] = params[idx:idx + len(triu_idx[0])]; idx += len(triu_idx[0])
        Kappa = U
    else:
        raise ValueError("kappa_type must be 'diagonal', 'full', 'scalar', 'lower', or 'upper'")
    nu = fix_nu if fix_nu is not None else params[idx]; idx += 1
    Lambda = np.diag(np.exp(2 * lambda_vec))
    Lambda = stabilize_pd(Lambda)  # Ensure positive definiteness
    Lambda_inv = pinv(Lambda)
    mu = np.zeros((T + 1, R))
    mu[0] = omega.copy() if estimate_omega else np.zeros(R)
    loglik = 0.0
    for t in range(T):
        if X_shared:
            x_beta = X[t] @ beta
        else:
            x_beta = np.array([X[j][t] @ beta[j] for j in range(R)])
        resid = Y[t] - x_beta - mu[t]
        alpha_t = 1 + (1 / nu) * (resid.T @ Lambda_inv @ resid)
        u_t = (Lambda_inv @ resid) / alpha_t
        mu[t + 1] = omega + Phi @ (mu[t] - omega) + Kappa @ u_t if estimate_omega else Phi @ mu[t] + Kappa @ u_t
        loglik += student_t_logpdf(Y[t], x_beta + mu[t], Lambda, nu)
        if not np.isfinite(loglik):
            print(f"Non-finite log_pdf at t={t}: {loglik}, resid={resid}, mu={mu[t]}, params={params}")
            return 1e12
    return -loglik

def gas_filter_eval(
    params, Y, X, cL, cK, fix_nu=None,
    kappa_type="diagonal", phi_type="full",
    X_shared=True, P_list=None, estimate_omega=True
):
    T, R = Y.shape
    idx = 0
    if phi_type == "full":
        Phi = params[idx:idx + R * R].reshape(R, R); idx += R * R
    elif phi_type == "diagonal":
        phi_diag = params[idx:idx + R]; idx += R
        Phi = np.diag(phi_diag)
    elif phi_type == "scalar":
        phi_scalar = params[idx]; idx += 1
        Phi = np.eye(R) * phi_scalar
    elif phi_type == "lower":
        L = np.zeros((R, R))
        tril_idx = np.tril_indices(R)
        L[tril_idx] = params[idx:idx + len(tril_idx[0])]; idx += len(tril_idx[0])
        Phi = L
    elif phi_type == "upper":
        U = np.zeros((R, R))
        triu_idx = np.triu_indices(R)
        U[triu_idx] = params[idx:idx + len(triu_idx[0])]; idx += len(triu_idx[0])
        Phi = U
    else:
        raise ValueError("Invalid phi_type")
    
    Phi = stabilize_phi(Phi)

    if estimate_omega:
        omega = params[idx:idx + R]; idx += R
    else:
        omega = np.zeros(R)
    if X_shared:
        P = X.shape[1]
        beta = params[idx:idx + P * R].reshape(P, R); idx += P * R
    else:
        beta = []
        for i, Pi in enumerate(P_list):
            beta_i = params[idx:idx + Pi]; idx += Pi
            beta.append(beta_i)
    lambda_vec = params[idx:idx + cL]; idx += cL
    Lambda = np.diag(np.exp(2 * lambda_vec))
    Lambda = stabilize_pd(Lambda)  # Ensure positive definiteness
    Lambda_inv = pinv(Lambda)
    if kappa_type == "diagonal":
        kappa_vec = params[idx:idx + R]; idx += R
        Kappa = np.diag(kappa_vec)
    elif kappa_type == "full":
        kappa_vec = params[idx:idx + R * R]; idx += R * R
        Kappa = kappa_vec.reshape(R, R)
    elif kappa_type == "scalar":
        kappa_scalar = params[idx]; idx += 1
        Kappa = np.eye(R) * kappa_scalar
    elif kappa_type == "lower":
        L = np.zeros((R, R))
        tril_idx = np.tril_indices(R)
        L[tril_idx] = params[idx:idx + len(tril_idx[0])]; idx += len(tril_idx[0])
        Kappa = L
    elif kappa_type == "upper":
        U = np.zeros((R, R))
        triu_idx = np.triu_indices(R)
        U[triu_idx] = params[idx:idx + len(triu_idx[0])]; idx += len(triu_idx[0])
        Kappa = U
    else:
        raise ValueError("Invalid kappa_type")
    nu = fix_nu if fix_nu is not None else params[idx]
    mu = np.zeros((T + 1, R))
    mu[0] = omega.copy() if estimate_omega else np.zeros(R)
    u = np.zeros((T, R))
    resid = np.zeros((T, R))
    mu_predicted = np.zeros((T, R))
    mu_filtered = np.zeros((T, R))
    for t in range(T):
        mu_predicted[t] = mu[t]
        if X_shared:
            x_beta = X[t] @ beta
        else:
            x_beta = np.array([X[j][t] @ beta[j] for j in range(R)])
        resid[t] = Y[t] - x_beta - mu[t]
        alpha_t = 1 + (1 / nu) * (resid[t].T @ Lambda_inv @ resid[t])
        u[t] = (Lambda_inv @ resid[t]) / alpha_t
        mu[t + 1] = omega + Phi @ (mu[t] - omega) + Kappa @ u[t] if estimate_omega else Phi @ mu[t] + Kappa @ u[t]
        mu_filtered[t] = mu[t + 1]
    return mu_filtered, mu_predicted, u, resid



def estimate_and_filter_gas(
    y,
    X: Union[np.ndarray, List[np.ndarray]],
    phi_type: Literal["full", "diagonal", "scalar", "lower", "upper"] = "full",
    kappa_type: Literal["full", "diagonal", "scalar", "lower", "upper"] = "diagonal",
    fix_nu: Optional[float] = None,
    X_shared: bool = True,
    P_list: Optional[List[int]] = None,
    standardize: bool = False,
    verbose: bool = True,
    optim_method: str = "L-BFGS-B",
    maxiter = 10000,
    maxfun = 20000,
    ftol = 1e-9,
    gtol = 1e-6,
    optim_bounds = None,
    burn_in: float = 0.0,  # Proportion of sample to drop for cut metrics (e.g. 0.1 for 10%)
    estimate_omega: bool = True
):
    """
    y: ndarray (T, R)
    X: ndarray (T, P) for shared OR list of [X_0, ..., X_{R-1}] with shapes (T, P_i) for separate
    X_shared: True if shared regressors, False for separate
    P_list: required if X_shared=False, e.g., [P0, P1, ...]
    standardize: whether to standardize X before estimation
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    T, R = y.shape

    # --- Standardization logic ---
    scaler_info = {}
    if standardize:
        if X_shared:
            Xs, mean_X, std_X = standardize_X(X)
            X_input = Xs
            scaler_info["mean_X"] = mean_X
            scaler_info["std_X"] = std_X
        else:
            Xs, means_X, stds_X = standardize_X(X, per_target=True)
            X_input = Xs
            scaler_info["mean_X"] = means_X
            scaler_info["std_X"] = stds_X
    else:
        X_input = X

    if X_shared:
        if X_input.ndim == 1:
            X_input = X_input.reshape(-1, 1)
        P = X_input.shape[1]
        P_or_Plist = P
    else:
        assert isinstance(X_input, list) and len(X_input) == R, "X must be list of length R"
        assert P_list is not None and len(P_list) == R
        P_or_Plist = P_list

    cL = R
    cK = get_param_size(kappa_type, R)
    cPhi = get_param_size(phi_type, R)
    init_params = build_initial_params(R, P_or_Plist, phi_type, kappa_type, fix_nu, X_shared=X_shared, estimate_omega=estimate_omega)

    if optim_bounds is None:
        bounds = build_gas_param_bounds(
            R=R,
            P_or_Plist=P_or_Plist,
            phi_type=phi_type,
            kappa_type=kappa_type,
            fix_nu=fix_nu,
            X_shared=X_shared,
            cL=cL,
            estimate_omega=estimate_omega
        )
    else:
        bounds = optim_bounds

    # --- Optimization ---
    if optim_method == "2stage":
        res_powell = minimize(
            gas_general_filter,
            init_params,
            args=(y, X_input, cL, cK, fix_nu, kappa_type, phi_type, X_shared, P_list, estimate_omega),
            method="Powell",
            options={'maxiter': 3000, 'disp': verbose},
            bounds=bounds
        )
        res = minimize(
            gas_general_filter,
            res_powell.x,
            args=(y, X_input, cL, cK, fix_nu, kappa_type, phi_type, X_shared, P_list, estimate_omega),
            method="L-BFGS-B",
            options={
                'maxiter': maxiter,
                'maxfun': maxfun,
                'ftol': ftol,
                'gtol': gtol,
                'disp': True
            },
            bounds=bounds
        )
    else:
        res = minimize(
            gas_general_filter,
            init_params,
            args=(y, X_input, cL, cK, fix_nu, kappa_type, phi_type, X_shared, P_list, estimate_omega),
            method=optim_method,
            options={
                'maxiter': maxiter,
                'maxfun': maxfun,
                'ftol': ftol,
                'gtol': gtol,
                'disp': True
            },
            bounds=bounds
        )

    if not res.success:
        if 'exceeds limit' in res.message.lower() and np.isfinite(res.fun):
            print("Warning: Optimization stopped at function/iteration limit, using last parameters.")
        else:
            raise RuntimeError(f"Estimation failed: {res.message}")

    est = res.x
    idx = 0
    Phi_est, idx = extract_matrix(est, idx, R, phi_type)
    Phi = stabilize_phi(Phi_est)

    if estimate_omega:
        omega_est = est[idx:idx + R]; idx += R
    else:
        omega_est = np.zeros(R)

    if X_shared:
        P = X_input.shape[1]
        beta_est = est[idx:idx + P * R].reshape(P, R); idx += P * R
    else:
        beta_est = []
        for i, Pi in enumerate(P_list):
            beta_i = est[idx:idx + Pi]; idx += Pi
            beta_est.append(beta_i)

    lambda_est = est[idx:idx + cL]; idx += cL
    Omega_est = np.diag(np.exp(2 * lambda_est))
    Kappa_est, idx = extract_matrix(est, idx, R, kappa_type)
    nu_est = fix_nu if fix_nu is not None else est[idx]

    mu_filtered, mu_predicted, u, resid = gas_filter_eval(
        params=est,
        Y=y,
        X=X_input,
        cL=cL,
        cK=cK,
        fix_nu=fix_nu,
        kappa_type=kappa_type,
        phi_type=phi_type,
        X_shared=X_shared,
        P_list=P_list,
        estimate_omega=estimate_omega
    )

    final_neg_loglik = gas_general_filter(
        est, y, X_input, cL, cK, fix_nu, kappa_type, phi_type, X_shared, P_list, estimate_omega
    )
    loglik = -final_neg_loglik
    k = len(est)
    n_obs = T * R
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n_obs) - 2 * loglik

    if burn_in > 0.0:
        burn_t = int(np.floor(burn_in * T))
        y_cut = y[burn_t:]
        if X_shared:
            X_cut = X[burn_t:]
        else:
            X_cut = [X[j][burn_t:] for j in range(R)]
        T_cut = y_cut.shape[0]
        cut_loglik = -gas_general_filter(
            est, y_cut, X_cut, cL, cK, fix_nu, kappa_type, phi_type, X_shared, P_list, estimate_omega
        )
        cut_n_obs = T_cut * R
        cut_aic = 2 * k - 2 * cut_loglik
        cut_bic = k * np.log(cut_n_obs) - 2 * cut_loglik
    else:
        cut_loglik = loglik
        cut_aic = aic
        cut_bic = bic



    if verbose:
        print("Estimated omega:\n", omega_est)
        print("Estimated Phi:\n", Phi_est)
        print("Estimated beta:\n", beta_est)
        print("Estimated Omega (from lambda):\n", Omega_est)
        print("Estimated Kappa:\n", Kappa_est)
        print("Estimated nu:\n", nu_est)
        print("Log-likelihood:", loglik)
        print("AIC:", aic)
        print("BIC:", bic)
        if burn_in > 0.0:
            print(f"Post-burn-in (after {burn_t} obs):")
            print("Cut log-likelihood:", cut_loglik)
            print("Cut AIC:", cut_aic)
            print("Cut BIC:", cut_bic)

    return {
        "Phi": Phi_est,
        "omega": omega_est,
        "beta": beta_est,
        "Omega": Omega_est,
        "Kappa": Kappa_est,
        "nu": nu_est,
        "mu_filtered": mu_filtered,
        "mu_predicted": mu_predicted,
        "u": u,
        "resid": resid,
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
        "cut_loglik": cut_loglik,
        "cut_aic": cut_aic,
        "cut_bic": cut_bic,
        "optimization_result": res
    }
