"""
Hedge-fund style risk analytics on daily Rs P&L (and minute-level equity).

All ratios use daily returns = daily P&L / starting capital (fixed lots, no compounding),
252 trading days a year. Sharpe/Sortino subtract a risk-free rate (default 6.5% p.a.).
"""
import numpy as np
import pandas as pd

ANN = 252


def drawdown(equity):
    eq = np.asarray(equity, float)
    peak = np.maximum.accumulate(eq)
    return eq - peak, peak


def underwater_days(dd):
    """Longest run of consecutive days below a previous peak."""
    best = cur = 0
    for x in dd:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best


def summary(daily_pnl, capital, rf=0.065, intraday=None, trades=None):
    """daily_pnl: Series (index = date) of Rs P&L. intraday: optional (days x 375) equity array."""
    x = daily_pnl.to_numpy(float)
    n = len(x)
    ret = x / capital
    eq = capital + np.cumsum(x)
    dd, peak = drawdown(np.concatenate([[capital], eq]))
    dd = dd[1:]
    years = n / ANN
    total = x.sum()
    ann_ret = ret.mean() * ANN
    vol = ret.std(ddof=1) * np.sqrt(ANN) if n > 1 else np.nan
    dn = ret[ret < 0]
    dvol = np.sqrt((np.minimum(ret - rf / ANN, 0) ** 2).mean()) * np.sqrt(ANN) if n else np.nan
    sharpe = (ann_ret - rf) / vol if vol and vol > 0 else np.nan
    sortino = (ann_ret - rf) / dvol if dvol and dvol > 0 else np.nan
    maxdd = dd.min()
    cagr = (eq[-1] / capital) ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else np.nan
    calmar = ann_ret / (abs(maxdd) / capital) if maxdd < 0 else np.nan
    var95, var99 = np.percentile(x, 5), np.percentile(x, 1)
    cvar95 = x[x <= var95].mean() if (x <= var95).any() else np.nan
    cvar99 = x[x <= var99].mean() if (x <= var99).any() else np.nan
    wk = daily_pnl.copy()
    wk.index = pd.to_datetime(wk.index)
    weekly = wk.resample("W").sum()
    monthly = wk.groupby(wk.index.to_period("M")).sum()
    gains, losses = x[x > 0].sum(), -x[x < 0].sum()
    q95, q5 = np.percentile(x, 95), np.percentile(x, 5)
    out = dict(
        total_pnl=total, total_return_pct=100 * total / capital, ann_return_pct=100 * ann_ret,
        cagr_pct=100 * cagr, ann_vol_pct=100 * vol, sharpe=sharpe, sortino=sortino, calmar=calmar,
        max_dd=maxdd, max_dd_pct=100 * maxdd / capital, longest_underwater_days=underwater_days(dd),
        avg_dd_pct=100 * dd[dd < 0].mean() / capital if (dd < 0).any() else 0.0,
        ulcer_pct=100 * np.sqrt(np.mean((dd / (capital + np.maximum.accumulate(eq) - capital)) ** 2)),
        var95=var95, var99=var99, cvar95=cvar95, cvar99=cvar99,
        worst_day=x.min(), best_day=x.max(), worst_week=weekly.min(), worst_month=monthly.min(),
        best_month=monthly.max(), pos_months=int((monthly > 0).sum()), months=len(monthly),
        win_day_pct=100 * (x > 0).mean(), profit_factor=gains / losses if losses > 0 else np.inf,
        skew=pd.Series(x).skew(), kurtosis=pd.Series(x).kurt(),
        tail_ratio=abs(q95 / q5) if q5 != 0 else np.nan,
        gain_to_pain=total / losses if losses > 0 else np.inf, days=n,
    )
    if intraday is not None:
        flat = intraday.reshape(-1)
        idd, _ = drawdown(flat)
        out["intraday_max_dd"] = idd.min()
        out["intraday_max_dd_pct"] = 100 * idd.min() / capital
        out["intraday_low_equity"] = flat.min()
        # worst peak-to-trough within a single day
        day_dd = intraday - np.maximum.accumulate(intraday, axis=1)
        out["worst_intraday_swing"] = day_dd.min()
    if trades is not None and len(trades):
        t = trades["net_inr"]
        w, l = t[t > 0], t[t <= 0]
        out.update(trades=len(t), trade_win_pct=100 * (t > 0).mean(),
                   avg_win=w.mean() if len(w) else 0.0, avg_loss=l.mean() if len(l) else 0.0,
                   payoff=(w.mean() / abs(l.mean())) if len(w) and len(l) and l.mean() != 0 else np.nan,
                   expectancy=t.mean(), sl_hit_pct=100 * (trades["reason"] == "SL").mean())
    return out


def drawdown_series(daily_pnl, capital):
    eq = capital + daily_pnl.cumsum()
    dd, peak = drawdown(np.concatenate([[capital], eq.to_numpy()]))
    return pd.Series(dd[1:], index=daily_pnl.index), eq


def monthly_table(daily_pnl, capital):
    s = daily_pnl.copy()
    s.index = pd.to_datetime(s.index)
    m = s.groupby(s.index.to_period("M")).sum() / capital * 100
    tab = pd.DataFrame({"y": [p.year for p in m.index], "m": [p.month for p in m.index], "v": m.values}).pivot(index="y", columns="m", values="v")
    tab["Year"] = tab.sum(axis=1)
    return tab


def rolling_sharpe(daily_pnl, capital, window=60, rf=0.065):
    r = daily_pnl / capital
    mu = r.rolling(window).mean() * ANN
    sd = r.rolling(window).std() * np.sqrt(ANN)
    return (mu - rf) / sd


def risk_contribution(daily_df, cols):
    """Share of portfolio variance and of CVaR(95) coming from each strategy."""
    X = daily_df[cols]
    tot = X.sum(axis=1)
    cov = X.apply(lambda c: np.cov(c, tot)[0, 1])
    share_var = cov / tot.var()
    cut = np.percentile(tot, 5)
    tail = tot <= cut
    share_cvar = X[tail].mean() / tot[tail].mean() if tail.any() else X.mean() * np.nan
    stand_vol = X.std()
    div_ratio = stand_vol.sum() / tot.std() if tot.std() > 0 else np.nan
    return pd.DataFrame({"pnl": X.sum(), "daily_vol": stand_vol, "share_of_variance": share_var,
                         "share_of_tail_loss": share_cvar}), div_ratio


def monte_carlo(daily_pnl, capital, n_paths=3000, block=5, seed=7):
    """Block bootstrap of daily P&L (keeps short-term clustering). Returns dict of arrays."""
    rng = np.random.default_rng(seed)
    x = daily_pnl.to_numpy(float)
    n = len(x)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :n]
    paths = x[idx]
    eq = capital + np.cumsum(paths, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.full((n_paths, 1), capital), eq], axis=1), axis=1)[:, 1:]
    mdd = (eq - peak).min(axis=1)
    return dict(final=eq[:, -1] - capital, max_dd=mdd, curves=eq)


def stress_table(daily_df, capital, margin_series=None):
    """Replay the worst historical days at 1x / 2x / 3x on top of the current book."""
    tot = daily_df["TOTAL"]
    worst = tot.nsmallest(5)
    rows = []
    for d, v in worst.items():
        rows.append(dict(date=d, pnl_1x=v, pnl_2x=2 * v, pnl_3x=3 * v,
                         pct_capital_3x=100 * 3 * v / capital))
    return pd.DataFrame(rows)


def suggest_lots(unit_daily, current_lots, rf=0.065):
    """Suggested lot splits at the SAME total daily volatility as the current book.
    unit_daily: DataFrame (days x strategies) of Rs P&L at 1 lot each."""
    cols = list(unit_daily.columns)
    cov = unit_daily.cov().to_numpy()
    mu = unit_daily.mean().to_numpy()
    cur = np.array([current_lots[c] for c in cols], float)
    target_vol = np.sqrt(cur @ cov @ cur)

    def scale(w):
        w = np.maximum(w, 0)
        v = np.sqrt(w @ cov @ w)
        return w * (target_vol / v) if v > 0 else w

    inv_vol = scale(1 / np.sqrt(np.diag(cov)))
    # equal risk contribution (iterative)
    w = np.ones(len(cols))
    for _ in range(500):
        mrc = cov @ w
        rc = w * mrc
        w = w * (rc.mean() / rc) ** 0.5
        w = w / w.sum()
    erc = scale(w)
    # max Sharpe, long-only: projected gradient on a simplex
    best, ws = -9, None
    rng = np.random.default_rng(1)
    for _ in range(20000):
        c = rng.dirichlet(np.ones(len(cols)))
        sh = (c @ mu) / np.sqrt(c @ cov @ c)
        if sh > best:
            best, ws = sh, c
    ms = scale(ws)
    mv = None
    best = 1e18
    for _ in range(20000):
        c = rng.dirichlet(np.ones(len(cols)))
        v = c @ cov @ c
        if v < best:
            best, mv = v, c
    mv = scale(mv)
    out = pd.DataFrame({"current": cur, "inverse_vol": inv_vol, "risk_parity": erc,
                        "max_sharpe": ms, "min_variance": mv}, index=cols).round(1)
    return out


# ---------------------------------------------------------------------------------- benchmark comparison
def relative_stats(p, b, rf=0.065):
    """Portfolio vs benchmark from aligned DAILY return arrays (fractions). Returns a dict."""
    p, b = np.asarray(p, float), np.asarray(b, float)
    n = len(p)
    rfd = rf / ANN
    var_b = np.var(b, ddof=1)
    beta = np.cov(p, b, ddof=1)[0, 1] / var_b if var_b > 0 else np.nan
    alpha = ((p - rfd).mean() - beta * (b - rfd).mean()) * ANN
    corr = np.corrcoef(p, b)[0, 1] if n > 2 else np.nan
    act = p - b
    te = act.std(ddof=1) * np.sqrt(ANN)
    up, dn = b > 0, b < 0
    up_cap = p[up].mean() / b[up].mean() if up.any() else np.nan
    dn_cap = p[dn].mean() / b[dn].mean() if dn.any() else np.nan
    return dict(beta=beta, alpha_ann_pct=100 * alpha, corr=corr, r2=corr ** 2 if n > 2 else np.nan,
                tracking_error_pct=100 * te, info_ratio=(act.mean() * ANN) / te if te > 0 else np.nan,
                up_capture=up_cap, down_capture=dn_cap)


def series_stats(ret, rf=0.065, compound=True):
    """Compact stats from a daily return array: total, CAGR, vol, Sharpe, Sortino, max DD, Calmar, best/worst day.
    compound=False treats returns as P&L on a fixed starting capital (no reinvestment), like the portfolio itself."""
    r = np.asarray(ret, float)
    n = len(r)
    eq = np.cumprod(1 + r) if compound else 1 + np.cumsum(r)
    dd = eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1
    years = n / ANN
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(ANN)
    ex = r.mean() * ANN - rf
    dvol = np.sqrt((np.minimum(r - rf / ANN, 0) ** 2).mean()) * np.sqrt(ANN)
    return dict(total_pct=100 * (eq[-1] - 1), cagr_pct=100 * cagr, vol_pct=100 * vol,
                sharpe=ex / vol if vol > 0 else np.nan, sortino=ex / dvol if dvol > 0 else np.nan,
                max_dd_pct=100 * dd.min(), calmar=(r.mean() * ANN) / abs(dd.min()) if dd.min() < 0 else np.nan,
                best_day_pct=100 * r.max(), worst_day_pct=100 * r.min(), up_days_pct=100 * (r > 0).mean())
