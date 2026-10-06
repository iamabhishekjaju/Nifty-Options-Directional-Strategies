import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optlab import portfolio as P, risk as R              # noqa: E402

FIRST = P.STRATEGIES[0]


@pytest.fixture(scope="module")
def book():
    return P.load_trades(), P.load_mtm()


def test_daily_equals_trade_sum_and_intraday_ties_out(book):
    tr, mt = book
    s = P.Settings().with_(slip_in=0.4, slip_out=0.3, slip_hedge=0.2, costs=True)
    r = P.build(tr, s, mt)
    assert r["daily"]["TOTAL"].sum() == pytest.approx(r["trades"]["net_inr"].sum())
    assert r["intraday"][-1, -1] == pytest.approx(s.capital + r["daily"]["TOTAL"].sum())
    assert np.allclose(r["intraday"][:, -1], s.capital + r["daily"]["TOTAL"].cumsum().to_numpy())


def test_slippage_is_linear_and_hurts(book):
    tr, _ = book
    s0 = P.Settings(hedge=False)
    one = {k: 1 for k in P.STRATEGIES}
    a = P.build(tr, s0.with_(lots=one))["daily"]["TOTAL"].sum()
    b = P.build(tr, s0.with_(lots=one, slip_in=1.0, slip_out=1.0))["daily"]["TOTAL"].sum()
    n = sum(len(v) for v in tr.values())
    assert a - b == pytest.approx(2 * n * s0.lot_size)


def test_hedge_toggle_and_lots_scale(book):
    tr, _ = book
    one = {k: 1 for k in P.STRATEGIES}
    on = P.build(tr, P.Settings(lots=one))["daily"]["TOTAL"].sum()
    off = P.build(tr, P.Settings(lots=one, hedge=False))["daily"]["TOTAL"].sum()
    hedge_pts = sum(v.loc[v.hedge_note != "no_hedge", "hedge_pts"].sum() for v in tr.values())
    assert on - off == pytest.approx(hedge_pts * 65)
    assert P.build(tr, P.Settings())["daily"]["TOTAL"].sum() == pytest.approx(24 * on)


def test_daily_stop_only_removes_trades(book):
    tr, _ = book
    assert len(P.build(tr, P.Settings().with_(daily_stop=100000.0))["trades"]) < len(P.build(tr, P.Settings())["trades"])


def test_stt_rate_switches_on_1_april_2026(book):
    tr, _ = book
    s = P.Settings(costs=True, brokerage=0.0, exch_pct=0.0, sebi_pct=0.0, stamp_pct=0.0, gst_pct=0.0, hedge=False)
    pr = P.price_trades(tr[FIRST], s, 1)
    cut = pd.Timestamp("2026-04-01").date()
    old, new = pr[pr["date"] < cut], pr[pr["date"] >= cut]
    assert len(old) and len(new)
    assert np.allclose(old["cost_inr"], old["sell_entry"] * 65 * 0.10 / 100)
    assert np.allclose(new["cost_inr"], new["sell_entry"] * 65 * 0.15 / 100)


def test_risk_summary_sanity():
    idx = pd.date_range("2025-01-01", periods=200, freq="B").date
    x = pd.Series(np.r_[np.full(100, 1000.0), np.full(100, -500.0)], index=idx)
    s = R.summary(x, 1e6)
    assert s["max_dd"] == pytest.approx(-50000.0) and s["total_pnl"] == pytest.approx(50000.0)
    assert s["longest_underwater_days"] == 100


def test_relative_stats_beta_and_series_stats():
    b = np.random.default_rng(3).normal(0.0004, 0.01, 400)
    s = R.relative_stats(2 * b, b)
    assert s["beta"] == pytest.approx(2.0) and s["corr"] == pytest.approx(1.0)
    assert R.series_stats(np.full(252, 0.001), compound=False)["total_pct"] == pytest.approx(25.2)


def test_trade_logs_are_sane(book):
    tr, _ = book
    for name, d in tr.items():
        assert (d["exit_g"] >= d["entry_g"]).all(), name
        assert np.allclose(d["sell_pts"], d["sell_entry"] - d["sell_exit"], atol=0.01)
