import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider


def make_control_input(t, dt, fast_maneuvers=False):
    n = len(t)
    u = np.zeros(n)
    if fast_maneuvers:
        step_duration = 1.25
        values = [0.0, 3.0, -3.0, 2.0, -2.0, 1.5, -1.5, 0.0]
    else:
        step_duration = 2.5
        values = [0.0, 2.0, 0.0, -2.0, 1.0, -1.0]

    for i, val in enumerate(values):
        start_index = int((i * step_duration) / dt)
        end_index = int(((i + 1) * step_duration) / dt)
        if start_index >= n:
            break
        u[start_index:min(end_index, n)] = val
    return u


def integrate_state(u, dt):
    n = len(u)
    x_true = np.zeros(n)
    v_true = np.zeros(n)
    for k in range(1, n):
        v_true[k] = v_true[k - 1] + u[k - 1] * dt
        x_true[k] = x_true[k - 1] + v_true[k - 1] * dt
    return x_true, v_true


def build_signals(
    t,
    dt,
    meas_noise,
    u_noise,
    drop_rand,
    outlier_rand,
    outlier_sign,
    dropout_rate,
    outlier_rate,
    outlier_scale,
    noisy_u,
    fast_maneuvers,
):
    u_true = make_control_input(t, dt, fast_maneuvers=fast_maneuvers)
    x_true, _ = integrate_state(u_true, dt)

    z = x_true + meas_noise
    if dropout_rate > 0.0:
        z = z.copy()
        z[drop_rand < dropout_rate] = np.nan

    if outlier_rate > 0.0 and outlier_scale > 0.0:
        z = z.copy()
        mask = (outlier_rand < outlier_rate) & np.isfinite(z)
        z[mask] = z[mask] + outlier_sign[mask] * outlier_scale

    u_for_kf = u_true.copy()
    if noisy_u:
        u_for_kf = u_for_kf + 0.8 * u_noise

    return u_true, u_for_kf, x_true, z


def build_noises(n, seed=42):
    rng = np.random.default_rng(seed)
    meas_noise = rng.normal(0.0, 1.0, size=n)
    u_noise = rng.normal(0.0, 1.0, size=n)
    drop_rand = rng.random(n)
    outlier_rand = rng.random(n)
    outlier_sign = rng.choice(np.array([-1.0, 1.0]), size=n)
    return meas_noise, u_noise, drop_rand, outlier_rand, outlier_sign


def run_exponential(z, alpha, beta):
    # Second-order exponential smoothing (level + trend)
    x_exp = np.zeros_like(z)
    trend = np.zeros_like(z)
    x_exp[0] = z[0] if not np.isnan(z[0]) else 0.0
    for k in range(1, len(z)):
        x_prev = x_exp[k - 1]
        b_prev = trend[k - 1]
        x_pred = x_prev + b_prev
        if np.isnan(z[k]):
            x_exp[k] = x_pred
            trend[k] = b_prev
        else:
            x_exp[k] = alpha * z[k] + (1.0 - alpha) * x_pred
            trend[k] = beta * (x_exp[k] - x_prev) + (1.0 - beta) * b_prev
    return x_exp


def run_alpha_beta(z, dt, alpha_ab, beta_ab):
    x_ab = np.zeros_like(z)
    v_ab = np.zeros_like(z)
    for k in range(1, len(z)):
        x_pred = x_ab[k - 1] + v_ab[k - 1] * dt
        v_pred = v_ab[k - 1]
        if np.isnan(z[k]):
            x_ab[k] = x_pred
            v_ab[k] = v_pred
        else:
            resid = z[k] - x_pred
            x_ab[k] = x_pred + alpha_ab * resid
            v_ab[k] = v_pred + (beta_ab / dt) * resid
    return x_ab


def run_kalman(z, u, dt, sigma_w, sigma_v, use_control=True):
    n = len(z)
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.5 * dt**2], [dt]])
    h = np.array([[1.0, 0.0]])
    # Discrete Q for white-acceleration process noise:
    # x_k+1 = A x_k + B u_k + G w_k, w_k ~ N(0, sigma_w^2)
    # with G = [0.5*dt^2, dt]^T.
    q = (sigma_w**2) * np.array(
        [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]]
    )
    r = sigma_v**2 * np.array([[1.0]])

    x_kf = np.zeros((2, n))
    k_hist = np.zeros((2, n))
    p = np.eye(2)

    for k in range(1, n):
        u_k = u[k - 1] if use_control else 0.0
        x_pred = a @ x_kf[:, k - 1] + b.flatten() * u_k
        p_pred = a @ p @ a.T + q

        if np.isnan(z[k]):
            x_kf[:, k] = x_pred
            p = p_pred
            k_hist[:, k] = np.nan
            continue

        s = h @ p_pred @ h.T + r
        k_gain = p_pred @ h.T @ np.linalg.inv(s)
        resid = z[k] - (h @ x_pred)
        x_kf[:, k] = x_pred + (k_gain.flatten() * resid)
        k_hist[:, k] = k_gain.flatten()
        p = (np.eye(2) - k_gain @ h) @ p_pred

    return x_kf[0], k_hist


def rmse(x_true, x_est):
    return np.sqrt(np.mean((x_true - x_est) ** 2))


def mae(x_true, x_est):
    return np.mean(np.abs(x_true - x_est))


def estimate_lag_seconds(x_true, x_est, dt, max_lag_seconds=2.0):
    max_lag_samples = int(max_lag_seconds / dt)

    # Use only points where both signals are valid.
    valid_points = np.isfinite(x_true) & np.isfinite(x_est)
    if np.sum(valid_points) < 3:
        return np.nan

    x_true_valid = x_true[valid_points]
    x_est_valid = x_est[valid_points]

    # Correlate changes (first differences), not raw position.
    # Raw position trend dominates correlation and often forces zero-lag.
    a = np.diff(x_true_valid)
    b = np.diff(x_est_valid)
    a = a - np.mean(a)
    b = b - np.mean(b)

    if np.allclose(a, 0.0) or np.allclose(b, 0.0):
        return 0.0

    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    valid_lags = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    best_lag_samples = lags[valid_lags][np.argmax(corr[valid_lags])]

    # Positive lag means estimate is behind true signal.
    return -best_lag_samples * dt


def peak_overshoot(x_true, x_est):
    return np.max(x_est - x_true)


def build_metrics_text(x_true, x_exp, x_ab, x_kf, dt):
    return (
        "RMSE / MAE / Lag(s) / Overshoot\n"
        f"Exp2: {rmse(x_true, x_exp):.3f} / {mae(x_true, x_exp):.3f} / "
        f"{estimate_lag_seconds(x_true, x_exp, dt):.2f} / {peak_overshoot(x_true, x_exp):.3f}\n"
        f"AB:   {rmse(x_true, x_ab):.3f} / {mae(x_true, x_ab):.3f} / "
        f"{estimate_lag_seconds(x_true, x_ab, dt):.2f} / {peak_overshoot(x_true, x_ab):.3f}\n"
        f"KF:   {rmse(x_true, x_kf):.3f} / {mae(x_true, x_kf):.3f} / "
        f"{estimate_lag_seconds(x_true, x_kf, dt):.2f} / {peak_overshoot(x_true, x_kf):.3f}"
    )


def main():
    dt = 0.2
    duration = 15.0
    t = np.arange(0.0, duration, dt)
    meas_noise, u_noise, drop_rand, outlier_rand, outlier_sign = build_noises(
        len(t), seed=42
    )

    # Initial slider values
    dt_0 = dt
    alpha_exp_0 = 0.2
    beta_exp_0 = 0.1
    sigma_w_0 = 0.1
    sigma_v_0 = 1.0
    alpha_ab_0 = 0.2
    beta_ab_0 = 0.1
    dropout_rate_0 = 0.0
    outlier_rate_0 = 0.0
    outlier_scale_0 = 8.0

    noisy_u_0 = False
    fast_maneuvers_0 = False

    u_true, u_for_kf, x_true, z = build_signals(
        t,
        dt,
        meas_noise,
        u_noise,
        drop_rand,
        outlier_rand,
        outlier_sign,
        dropout_rate_0,
        outlier_rate_0,
        outlier_scale_0,
        noisy_u_0,
        fast_maneuvers_0,
    )
    x_exp = run_exponential(z, alpha_exp_0, beta_exp_0)
    x_ab = run_alpha_beta(z, dt, alpha_ab_0, beta_ab_0)
    x_kf, k_hist = run_kalman(z, u_for_kf, dt, sigma_w_0, sigma_v_0)

    fig = plt.figure(figsize=(12, 7))
    ax_plot = fig.add_axes([0.36, 0.10, 0.63, 0.84])

    line_true, = ax_plot.plot(t, x_true, lw=2.0, label="True position")
    line_meas, = ax_plot.plot(t, z, ".", ms=4, alpha=0.6, label="Measurements")
    line_exp, = ax_plot.plot(t, x_exp, lw=2.0, label="Exponential filter (2nd order)")
    line_ab, = ax_plot.plot(t, x_ab, lw=2.0, label="Alpha-beta filter")
    line_kf, = ax_plot.plot(t, x_kf, lw=2.0, label="Kalman filter (with u)")

    ax_plot.set_title("Filter Comparison")
    ax_plot.set_xlabel("Time [s]")
    ax_plot.set_ylabel("Position")
    ax_plot.set_xlim(0.0, duration)
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend(loc="upper left")

    valid_k = np.where(~np.isnan(k_hist[0]))[0]
    if len(valid_k) > 0:
        last_idx = valid_k[-1]
        kx_last = k_hist[0, last_idx]
        kv_last = k_hist[1, last_idx]
    else:
        kx_last = np.nan
        kv_last = np.nan

    kf_gain_text = ax_plot.text(
        0.02,
        0.7,
        (
            f"KF gains\n"
            f"Kx(last)={kx_last:.4f}\n"
            f"Kv(last)={kv_last:.4f}"
        ),
        transform=ax_plot.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    metrics_text = ax_plot.text(
        0.4,
        0.9,
        build_metrics_text(x_true, x_exp, x_ab, x_kf, dt),
        transform=ax_plot.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    slider_specs = [
        ("dt", 0.02, 0.50, dt_0, "dt [s]"),
        ("alpha_exp", 0.001, 1.0, alpha_exp_0, "Alpha (Exp2)"),
        ("beta_exp", 0.001, 1.0, beta_exp_0, "Beta (Exp2)"),
        ("alpha_ab", 0.001, 1.0, alpha_ab_0, "Alpha (AB)"),
        ("beta_ab", 0.001, 1.0, beta_ab_0, "Beta (AB)"),
        ("sigma_w", 0.001, 2.0, sigma_w_0, "Sigma w (KF)"),
        ("sigma_v", 0.01, 5.0, sigma_v_0, "Sigma v (KF)"),
        ("dropout_rate", 0.0, 0.9, dropout_rate_0, "Dropout rate"),
        ("outlier_rate", 0.0, 0.5, outlier_rate_0, "Outlier rate"),
        ("outlier_scale", 0.0, 20.0, outlier_scale_0, "Outlier amp"),
    ]

    sliders = {}
    top = 0.90
    height = 0.04
    gap = 0.05

    for idx, (key, vmin, vmax, vinit, label) in enumerate(slider_specs):
        y = top - idx * gap
        ax_slider = fig.add_axes([0.07, y, 0.22, height])
        sliders[key] = Slider(
            ax=ax_slider,
            label=label,
            valmin=vmin,
            valmax=vmax,
            valinit=vinit,
        )

    # y, x, width, height for the KF option checkbox axes
    # from the bottom left corner of the figure
    ax_check = fig.add_axes([0.06, 0.34, 0.22, 0.08])
    kf_u_checkbox = CheckButtons(ax_check, ["Use u in KF"], [True])
    ax_check.set_title("KF option", fontsize=10)

    # y, x, width, height for the scenario checkbox axes
    # from the bottom left corner of the figure
    ax_scenario = fig.add_axes([0.06, 0.14, 0.22, 0.12])
    scenario_checkbox = CheckButtons(
        ax_scenario,
        ["Noisy u for KF", "Fast maneuvers"],
        [noisy_u_0, fast_maneuvers_0],
    )
    ax_scenario.set_title("Stress scenarios", fontsize=10)

    def update(_):
        dt_local = sliders["dt"].val
        t_local = np.arange(0.0, duration, dt_local)
        (
            meas_noise_local,
            u_noise_local,
            drop_rand_local,
            outlier_rand_local,
            outlier_sign_local,
        ) = build_noises(len(t_local), seed=42)

        alpha_exp = sliders["alpha_exp"].val
        beta_exp = sliders["beta_exp"].val
        sigma_w = sliders["sigma_w"].val
        sigma_v = sliders["sigma_v"].val
        alpha_ab = sliders["alpha_ab"].val
        beta_ab = sliders["beta_ab"].val
        dropout_rate = sliders["dropout_rate"].val
        outlier_rate = sliders["outlier_rate"].val
        outlier_scale = sliders["outlier_scale"].val
        use_control = kf_u_checkbox.get_status()[0]

        scenario_state = scenario_checkbox.get_status()
        noisy_u = scenario_state[0]
        fast_maneuvers = scenario_state[1]

        _, u_for_kf_local, x_true_local, z_local = build_signals(
            t_local,
            dt_local,
            meas_noise_local,
            u_noise_local,
            drop_rand_local,
            outlier_rand_local,
            outlier_sign_local,
            dropout_rate,
            outlier_rate,
            outlier_scale,
            noisy_u,
            fast_maneuvers,
        )

        x_exp_new = run_exponential(z_local, alpha_exp, beta_exp)
        x_ab_new = run_alpha_beta(z_local, dt_local, alpha_ab, beta_ab)
        x_kf_new, k_hist_new = run_kalman(
            z_local, u_for_kf_local, dt_local, sigma_w, sigma_v, use_control
        )

        line_true.set_xdata(t_local)
        line_meas.set_xdata(t_local)
        line_exp.set_xdata(t_local)
        line_ab.set_xdata(t_local)
        line_kf.set_xdata(t_local)
        line_true.set_ydata(x_true_local)
        line_meas.set_ydata(z_local)
        line_exp.set_ydata(x_exp_new)
        line_ab.set_ydata(x_ab_new)
        line_kf.set_ydata(x_kf_new)

        valid_k_local = np.where(~np.isnan(k_hist_new[0]))[0]
        if len(valid_k_local) > 0:
            last_idx_local = valid_k_local[-1]
            kx_last_local = k_hist_new[0, last_idx_local]
            kv_last_local = k_hist_new[1, last_idx_local]
        else:
            kx_last_local = np.nan
            kv_last_local = np.nan

        kf_gain_text.set_text(
            "KF gains\n"
            f"Kx(last)={kx_last_local:.4f}\n"
            f"Kv(last)={kv_last_local:.4f}"
        )
        metrics_text.set_text(
            build_metrics_text(x_true_local, x_exp_new, x_ab_new, x_kf_new, dt_local)
        )
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)
    kf_u_checkbox.on_clicked(update)
    scenario_checkbox.on_clicked(update)

    plt.show()


if __name__ == "__main__":
    main()
