"""
Portfolio layer: turns per-share trade logs (results/trades_*.csv) into rupee P&L for
any combination of lots, hedge on/off, slippage, transaction costs and a daily stop.
Everything here is cheap to recompute, so the dashboard re-runs it on every click.

Conventions
-----------
* P&L is per SHARE in the trade logs; rupee P&L = pts * lot_size * lots.
* Slippage is always AGAINST you: the short leg sells lower / buys back higher, the
  hedge buys higher / sells lower. `slip_sl_extra` is added on top for stop-loss exits
  (stops fill worse than normal exits). `slip_pct` is a % of the price per fill.
* Costs (optional): brokerage per order, STT on option SELL premium (0.10% before 1-Apr-2026, 0.15% from then), exchange
  transaction charges + SEBI fee on turnover, stamp duty on BUY premium, GST on
  (brokerage + exchange + SEBI). Rates are inputs - check them against current rules.
* Equity base is the starting capital (lots are fixed, nothing is reinvested).
"""
import json
import os
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

MINUTES = 375                                   # 1-minute bars per session, 09:15-15:29

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
_MANIFEST = os.path.join(RESULTS, "manifest.json")
# Optional results/manifest.json: {"strategies": [display names], "files": {display name: log-file stem}}
# lets a published copy use neutral names and neutral file names. Without it the internal names are used.
MANIFEST = json.load(open(_MANIFEST)) if os.path.exists(_MANIFEST) else None
STRATEGIES = MANIFEST["strategies"] if MANIFEST else ["Strategy A", "Strategy B", "Strategy C"]


@dataclass
class Settings:
    lots: dict = field(default_factory=lambda: {s: 24 for s in STRATEGIES})
    lot_size: int = 65
    capital: float = 1e7
    hedge: bool = True                 # include hedge-leg P&L (and hedge costs / margin benefit)
    slip_in: float = 0.0               # pts per fill, short-leg entry
    slip_out: float = 0.0              # pts per fill, short-leg exit
    slip_sl_extra: float = 0.0         # extra pts on short-leg SL exits
    slip_hedge: float = 0.0            # pts per fill, hedge buy and sell
    slip_pct: float = 0.0              # % of price per fill (all legs), on top of points
    costs: bool = False
    brokerage: float = 20.0            # Rs per order
    stt_pct: float = 0.10              # % of premium on option SELL turnover, trades BEFORE stt_change
    stt_pct_new: float = 0.15          # ... trades ON/AFTER stt_change (STT on options raised from 1 Apr 2026)
    stt_change: str = "2026-04-01"
    exch_pct: float = 0.03503          # % of turnover (NSE options)
    sebi_pct: float = 0.0001           # % of turnover
    stamp_pct: float = 0.003           # % of BUY turnover
    gst_pct: float = 18.0              # % on brokerage + exchange + SEBI
    margin_hedged: float = 150000.0    # Rs per lot, short + hedge (assumption - use your broker's calculator)
    margin_unhedged: float = 250000.0  # Rs per lot, naked short (assumption)
    daily_stop: float = 0.0            # Rs; stop NEW entries once the day's realised P&L <= -this (0 = off)

    def with_(self, **kw):
        return replace(self, **kw)


def _log_name(s):
    return MANIFEST["files"][s] if MANIFEST else s


def load_trades():
    out = {}
    for s in STRATEGIES:
        df = pd.read_csv(os.path.join(RESULTS, f"trades_{_log_name(s)}.csv"))
        df["date"] = pd.to_datetime(df["date"]).dt.date
        out[s] = df
    return out


def load_mtm():
    out = {}
    for s in STRATEGIES:
        z = np.load(os.path.join(RESULTS, f"mtm_{_log_name(s)}.npz"))
        out[s] = (z["sell"], z["hedge"])
    return out


def trading_days(trades):
    days = sorted({d for df in trades.values() for d in df["date"]})
    return days


def price_trades(df, st, lots):
    """Add net P&L columns (per share and Rs) to a trade log."""
    d = df.copy()
    has_h = st.hedge & (d["hedge_note"] != "no_hedge")
    p = st.slip_pct / 100.0
    is_sl = (d["reason"] == "SL").astype(float)
    # --- short leg (sell high-ish... fill lower, buy back higher)
    in_slip = st.slip_in + p * d["sell_entry"]
    out_slip = st.slip_out + st.slip_sl_extra * is_sl + p * d["sell_exit"]
    d["sell_net_pts"] = d["sell_pts"] - in_slip - out_slip
    # --- hedge leg (long)
    h_slip = 2 * st.slip_hedge + p * (d["hedge_entry"].fillna(0) + d["hedge_exit"].fillna(0))
    d["hedge_net_pts"] = np.where(has_h, d["hedge_pts"] - h_slip, 0.0)
    d["net_pts"] = d["sell_net_pts"] + d["hedge_net_pts"]
    qty = lots * st.lot_size
    d["qty"] = qty
    d["gross_inr"] = (d["sell_pts"] + np.where(has_h, d["hedge_pts"], 0.0)) * qty
    cost = np.zeros(len(d))
    if st.costs and qty > 0:
        sell_to = d["sell_entry"] * qty                         # sell leg: SELL at entry
        buy_to = d["sell_exit"] * qty                           # ... BUY at exit
        h_buy = np.where(has_h, d["hedge_entry"].fillna(0) * qty, 0.0)
        h_sell = np.where(has_h, d["hedge_exit"].fillna(0) * qty, 0.0)
        turnover = sell_to + buy_to + h_buy + h_sell
        orders = 2 + 2 * has_h.astype(int)
        brokerage = st.brokerage * orders
        exch = turnover * (st.exch_pct + st.sebi_pct) / 100
        gst = (brokerage + exch) * st.gst_pct / 100
        new = pd.to_datetime(d["date"]) >= pd.Timestamp(st.stt_change)
        stt_rate = np.where(new.to_numpy(), st.stt_pct_new, st.stt_pct)
        stt = (sell_to + h_sell) * stt_rate / 100
        stamp = (buy_to + h_buy) * st.stamp_pct / 100
        cost = brokerage + exch + gst + stt + stamp
    d["cost_inr"] = cost
    d["net_inr"] = d["net_pts"] * qty - cost
    return d


def apply_daily_stop(priced, stop):
    """Drop trades that would ENTER after the day's realised (net) P&L has hit -stop."""
    if not stop or stop <= 0:
        priced["kept"] = True
        return priced
    keep = pd.Series(True, index=priced.index)
    for day, g in priced.groupby("date"):
        ev = g.sort_values("exit_g")
        cum = ev["net_inr"].cumsum()
        hit = ev.loc[cum <= -stop, "exit_g"]
        if len(hit):
            stop_g = hit.iloc[0]
            keep.loc[g.index[g["entry_g"] > stop_g]] = False
    priced["kept"] = keep
    return priced


def build(trades, st, mtm=None):
    """Price every strategy and assemble portfolio results.

    Returns dict with: trades (priced, kept only), daily (DataFrame days x strategies + TOTAL, Rs),
    margin (DataFrame per day: peak lots, peak margin), intraday (equity per minute, if mtm given).
    """
    days = trading_days(trades)
    priced_all, daily = [], {}
    for s in STRATEGIES:
        lots = st.lots.get(s, 0)
        p = price_trades(trades[s], st, lots)
        p["strategy"] = s
        p["lots"] = lots
        priced_all.append(p)
    pr = pd.concat(priced_all, ignore_index=True)
    pr["_key"] = np.arange(len(pr))
    pr = apply_daily_stop(pr, st.daily_stop * 1.0)
    pk = pr[pr["kept"] & (pr["lots"] > 0)]
    dd = pk.groupby(["date", "strategy"])["net_inr"].sum().unstack(fill_value=0.0)
    daily = pd.DataFrame(0.0, index=pd.Index(days, name="date"), columns=STRATEGIES)
    daily.loc[dd.index, dd.columns] = dd
    daily["TOTAL"] = daily[STRATEGIES].sum(axis=1)
    out = dict(trades=pk.drop(columns=["_key"]), daily=daily, all_trades=pr)
    out["margin"] = margin_profile(pk, st, days)
    if mtm is not None:
        out["intraday"] = intraday_equity(pr, st, days, mtm, daily)
    return out


def margin_profile(pk, st, days):
    """Peak concurrent short lots and margin per day (sweep over entry/exit events)."""
    per_lot = st.margin_hedged if st.hedge else st.margin_unhedged
    rows = []
    for day, g in pk.groupby("date"):
        ev = np.concatenate([np.column_stack([g["entry_g"], g["lots"]]),
                             np.column_stack([g["exit_g"], -g["lots"]])])
        ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]          # exits (negative) before entries at the same minute
        run = np.cumsum(ev[:, 1])
        rows.append((day, run.max()))
    m = pd.DataFrame(rows, columns=["date", "peak_lots"]).set_index("date").reindex(days).fillna(0.0)
    m["peak_margin"] = m["peak_lots"] * per_lot
    return m


def intraday_equity(pr, st, days, mtm, daily):
    """Minute-level equity in Rs, shape (n_days, 375): previous close equity + today's
    realised P&L + open mark-to-market. Row r of a strategy's MTM array is trade_id r+1."""
    didx = {d: i for i, d in enumerate(days)}
    day_pnl = np.zeros((len(days), MINUTES))
    live = pr[pr["kept"] & (pr["lots"] > 0)]
    for s in STRATEGIES:
        sell, hedge = mtm[s]
        for t in live[live["strategy"] == s].itertuples(index=False):
            d, r = didx[t.date], int(t.trade_id) - 1
            e = int(t.entry_g) % MINUTES
            x = int(t.exit_g) % MINUTES
            m = np.nan_to_num(sell[r, e:x], nan=0.0).astype(float)
            if st.hedge and t.hedge_note != "no_hedge":
                m = m + np.nan_to_num(hedge[r, e:x], nan=0.0)
            day_pnl[d, e:x] += m * t.qty
            day_pnl[d, x:] += t.net_inr
    prev = np.concatenate([[0.0], daily["TOTAL"].cumsum().to_numpy()[:-1]])
    return st.capital + prev[:, None] + day_pnl
