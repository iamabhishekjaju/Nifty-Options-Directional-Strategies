"""
Nifty Options Portfolio Lab - institutional-style portfolio & risk dashboard.

Run:  streamlit run app/dashboard.py

Everything is recomputed live from results/trades_*.csv (per-share trade logs with hedge legs):
lots, hedge, slippage, costs and stops are applied here, so nothing is re-backtested when a
control moves.
"""
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optlab import portfolio as P            # noqa: E402
from optlab import risk as R                 # noqa: E402

st.set_page_config(page_title="Nifty Options Portfolio Lab", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------------------ theme
LIGHT = dict(bg="#eef1f5", panel="#ffffff", ink="#0f172a", ink2="#475569", muted="#94a3b8", line="#e2e8f0",
             grid="#eef2f7", side="#f8fafc", inp="#ffffff", zebra="#f1f5f9", mid="#f1f5f9", total="#0f172a",
             pos="#1d4ed8", neg="#dc2626", good="#15803d", accent="#1d4ed8",
             top="linear-gradient(100deg,#0b1f3a 0%,#12325c 100%)",
             dc="#2a78d6", vw="#eb6834", rs="#1baf7a", shadow="0 1px 2px rgba(15,23,42,.04)")
DARK = dict(bg="#0a1120", panel="#111a2c", ink="#e6edf7", ink2="#a8b5c9", muted="#6b7a90", line="#22304a",
            grid="#1b2740", side="#0d1526", inp="#0a1120", zebra="#0f1829", mid="#1b2740", total="#e6edf7",
            pos="#4f8ff0", neg="#f05252", good="#4ade80", accent="#4f8ff0",
            top="linear-gradient(100deg,#050b17 0%,#0f2447 100%)",
            dc="#3987e5", vw="#d95926", rs="#199e70", shadow="0 1px 2px rgba(0,0,0,.35)")

with st.sidebar:
    st.markdown("### Appearance")
    dark = st.toggle("Dark mode", key="dark")
T = DARK if dark else LIGHT
STRATS = P.STRATEGIES
COL = {**dict(zip(STRATS, [T["dc"], T["vw"], T["rs"]])), "TOTAL": T["total"]}
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
.stApp, [data-testid="stAppViewContainer"] {{ background:{T['bg']}; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding: 0.9rem 1.6rem 2rem 1.6rem; max-width: 1760px; }}
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stCaptionContainer"] {{ color:{T['ink']}; }}
[data-testid="stCaptionContainer"] p, .stApp small {{ color:{T['ink2']}; }}
[data-testid="stSidebar"] {{ background:{T['side']}; border-right:1px solid {T['line']}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: .8rem; }}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {{ font-size:.72rem; letter-spacing:.09em;
   text-transform:uppercase; color:{T['ink2']}; font-weight:700; margin:.2rem 0 .4rem 0; }}
[data-testid="stSidebar"] label p {{ font-size:.78rem; color:{T['ink2']}; }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{ background:{T['panel']}; border:1px solid {T['line']}; border-radius:8px; margin-bottom:.55rem; }}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {{ background:{T['panel']}; border-radius:8px; }}
[data-testid="stExpander"] summary p {{ font-weight:600; font-size:.82rem; color:{T['ink']}; }}
/* widgets */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="input"] input, [data-testid="stNumberInput"] input {{
   background:{T['inp']} !important; color:{T['ink']} !important; }}
[data-baseweb="select"] > div {{ background:{T['inp']} !important; color:{T['ink']} !important; border-color:{T['line']} !important; }}
[data-baseweb="select"] svg {{ fill:{T['ink2']}; }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] {{ background:{T['panel']} !important; color:{T['ink']} !important; }}
[data-testid="stNumberInput"] button {{ background:{T['inp']} !important; color:{T['ink']} !important; border-color:{T['line']} !important; }}
.stButton button, .stDownloadButton button {{ background:{T['panel']}; color:{T['ink']}; border:1px solid {T['line']}; }}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stThumbValue"] {{ color:{T['ink2']}; }}
[data-testid="stTooltipIcon"] svg {{ stroke:{T['muted']}; }}
button[data-baseweb="tab"] {{ font-weight:600; font-size:.84rem; }}
button[data-baseweb="tab"] p {{ color:{T['ink2']}; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color:{T['accent']}; }}
[data-testid="stTabs"] [role="tablist"] {{ border-bottom:1px solid {T['line']}; }}
/* top bar */
.topbar {{ background:{T['top']}; color:#fff; border-radius:10px; padding:14px 22px;
   display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
.topbar .t1 {{ font-size:1.18rem; font-weight:700; letter-spacing:.01em; color:#fff; }}
.topbar .t2 {{ font-size:.76rem; color:#b6c4dc; margin-top:2px; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
.chip {{ font-size:.7rem; font-weight:600; padding:4px 10px; border-radius:999px; background:rgba(255,255,255,.10);
   border:1px solid rgba(255,255,255,.22); color:#e8eefb; white-space:nowrap; }}
.chip.on {{ background:rgba(34,197,94,.18); border-color:rgba(34,197,94,.5); color:#bbf7d0; }}
.chip.warn {{ background:rgba(245,158,11,.18); border-color:rgba(245,158,11,.55); color:#fde68a; }}
/* KPI strip */
.kpis {{ display:grid; grid-template-columns:repeat(8,minmax(0,1fr)); gap:10px; margin-bottom:12px; }}
.kpi {{ background:{T['panel']}; border:1px solid {T['line']}; border-radius:10px; padding:11px 14px; box-shadow:{T['shadow']}; }}
.kpi .l {{ font-size:.64rem; letter-spacing:.08em; text-transform:uppercase; color:{T['ink2']}; font-weight:600; }}
.kpi .v {{ font-size:1.42rem; font-weight:700; color:{T['ink']}; margin-top:3px; font-variant-numeric:tabular-nums; letter-spacing:-.01em; }}
.kpi .s {{ font-size:.7rem; color:{T['muted']}; margin-top:1px; font-variant-numeric:tabular-nums; }}
.kpi .s.pos {{ color:{T['good']}; }} .kpi .s.neg {{ color:{T['neg']}; }}
/* panels */
[data-testid="stVerticalBlockBorderWrapper"] {{ background:{T['panel']}; border:1px solid {T['line']} !important; border-radius:10px;
   box-shadow:{T['shadow']}; }}
.ph {{ font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; font-weight:700; color:{T['ink2']}; margin:0 0 2px 0; }}
.ps {{ font-size:.74rem; color:{T['muted']}; margin:0 0 6px 0; }}
/* tables */
table.tbl {{ width:100%; border-collapse:collapse; font-size:.78rem; font-variant-numeric:tabular-nums; }}
table.tbl th {{ text-align:right; font-size:.64rem; letter-spacing:.07em; text-transform:uppercase; color:{T['ink2']}; font-weight:700;
   padding:6px 8px; border-bottom:1.5px solid {T['line']}; white-space:nowrap; position:sticky; top:0; background:{T['panel']}; }}
table.tbl th:first-child, table.tbl td:first-child {{ text-align:left; }}
table.tbl td {{ text-align:right; padding:6px 8px; border-bottom:1px solid {T['zebra']}; color:{T['ink']}; white-space:nowrap; }}
table.tbl tr:last-child td {{ border-bottom:none; }}
table.tbl td.neg {{ color:{T['neg']}; }} table.tbl td.pos {{ color:{T['good']}; }}
.scroll {{ max-height:520px; overflow:auto; }}
.stat {{ display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid {T['zebra']}; font-size:.78rem; }}
.stat:last-child {{ border-bottom:none; }}
.stat span:first-child {{ color:{T['ink2']}; }} .stat span:last-child {{ font-weight:600; font-variant-numeric:tabular-nums; color:{T['ink']}; }}
.sh {{ font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; font-weight:700; color:{T['accent']}; margin:2px 0 6px 0;
   padding-bottom:4px; border-bottom:2px solid {T['line']}; }}
.foot {{ color:{T['muted']}; font-size:.7rem; text-align:center; margin-top:18px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------ helpers
def inr(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    s = "-" if x < 0 else ("+" if signed else "")
    a = abs(x)
    if a >= 1e7:
        return f"{s}₹{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{s}₹{a / 1e5:.2f} L"
    return f"{s}₹{a:,.0f}"


def pct(x, d=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}%"


def kpi_html(items, cols=8):
    cells = "".join(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div>'
                    f'<div class="s {c}">{s}</div></div>' for l, v, s, c in items)
    return f'<div class="kpis" style="grid-template-columns:repeat({cols},minmax(0,1fr))">{cells}</div>'


def html_table(df, cls_cols=(), scroll=False):
    head = "".join(f"<th>{c}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = []
    for idx, r in df.iterrows():
        tds = [f"<td>{idx}</td>"]
        for c in df.columns:
            v = r[c]
            klass = ""
            if c in cls_cols and isinstance(v, str) and v.strip().startswith("-"):
                klass = ' class="neg"'
            elif c in cls_cols and isinstance(v, str) and v.strip().startswith("+"):
                klass = ' class="pos"'
            tds.append(f"<td{klass}>{v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    tbl = f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    return f'<div class="scroll">{tbl}</div>' if scroll else tbl


def stat_block(title, items):
    body = "".join(f'<div class="stat"><span>{a}</span><span>{b}</span></div>' for a, b in items)
    return f'<div class="sh">{title}</div>{body}'


@contextmanager
def panel(title, sub=None):
    with st.container(border=True):
        st.markdown(f'<div class="ph">{title}</div>' + (f'<div class="ps">{sub}</div>' if sub else ""),
                    unsafe_allow_html=True)
        yield


def style(fig, h=320, legend=True):
    fig.update_layout(
        height=h, margin=dict(l=4, r=4, t=6, b=4), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", font=dict(family=FONT, size=11, color=T["ink2"]), showlegend=legend,
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=10.5, color=T["ink2"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT, size=11)))
    fig.update_xaxes(showgrid=False, linecolor=T["line"], tickfont=dict(size=10), zeroline=False)
    fig.update_yaxes(gridcolor=T["grid"], zeroline=True, zerolinecolor=T["line"], linecolor=T["line"],
                     tickfont=dict(size=10))
    return fig


def show(fig, h=320, legend=True):
    st.plotly_chart(style(fig, h, legend), use_container_width=True, config={"displayModeBar": False})


DIV = lambda: [[0, T["neg"]], [0.5, T["mid"]], [1, T["pos"]]]


@st.cache_data(show_spinner=False)
def _load():
    return P.load_trades(), P.load_mtm()


@st.cache_data(show_spinner=False)
def _bench():
    b = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                                 "benchmark_nifty.csv"), parse_dates=["date"])
    b["date"] = b["date"].dt.date
    b["ret"] = b["close"].pct_change()          # close-to-close; day 1 has no prior close
    for c_ in ("dte", "iv_proxy"):
        if c_ not in b.columns:
            b[c_] = np.nan
    return b.set_index("date")


@st.cache_data(show_spinner=False, max_entries=96)
def _build(settings_json, d0, d1, strategies, with_intraday):
    s = P.Settings(**json.loads(settings_json))
    trades, mtm = _load()
    s.lots = {k: (v if k in strategies else 0) for k, v in s.lots.items()}
    f = {k: v[(v["date"] >= d0) & (v["date"] <= d1)] for k, v in trades.items()}
    return P.build(f, s, mtm if with_intraday else None)


def run(s, d0, d1, strategies, intraday=True):
    return _build(json.dumps(asdict(s), sort_keys=True), d0, d1, tuple(strategies), intraday)


# ------------------------------------------------------------------------------ sidebar
trades0, _ = _load()
all_days = sorted({d for v in trades0.values() for d in v["date"]})

with st.sidebar:
    st.markdown("### Portfolio controls")
    with st.expander("Book & period", expanded=True):
        sel = st.multiselect("Strategies in the book", STRATS, default=STRATS,
                             help="One strategy = stand-alone view; several = combined book.")
        dr = st.slider("Period", min_value=all_days[0], max_value=all_days[-1],
                       value=(all_days[0], all_days[-1]), format="DD MMM YY")
    with st.expander("Capital & sizing", expanded=True):
        capital = st.number_input("Starting capital (₹)", min_value=1e6, value=1e7, step=1e6, format="%.0f")
        lot_size = st.number_input("Lot size (qty)", min_value=1, value=65, step=1)
        lots = {}
        c = st.columns(3)
        for i, s_ in enumerate(STRATS):
            lots[s_] = c[i].number_input(f"{s_}", min_value=0, value=24, step=1, key=f"lots_{s_}",
                                         help="Lots for this strategy")
    with st.expander("Hedge", expanded=True):
        hedge = st.toggle("Include hedge legs", value=True,
                          help="ON: the hedge bought with every short is included (its P&L, slippage and costs). "
                               "OFF: short legs only.")
    with st.expander("Slippage", expanded=True):
        sell_slip = st.slider("Short leg: slippage per trade (pts)", 0.0, 3.0, 0.0, step=0.05,
                              help="Total for one round trip, split equally between entry and exit. "
                                   "0.2 means 0.1 on the entry and 0.1 on the exit.")
        hedge_slip = st.slider("Hedge legs: slippage per trade (pts)", 0.0, 3.0, 0.0, step=0.05,
                               help="Total for one round trip (buy and sell), split equally.")
    with st.expander("Transaction costs", expanded=False):
        costs = st.toggle("Include transaction costs", value=True)
        brokerage = st.number_input("Brokerage ₹ / order", value=20.0)
        c1_, c2_ = st.columns(2)
        stt_old = c1_.number_input("STT % before 1 Apr 2026", value=0.10, format="%.3f")
        stt_new = c2_.number_input("STT % from 1 Apr 2026", value=0.15, format="%.3f")
        exch = st.number_input("Exchange charge % of turnover", value=0.03503, format="%.5f")
        stamp = st.number_input("Stamp duty % on buy premium", value=0.003, format="%.4f")
        gst = st.number_input("GST % on brokerage + exchange", value=18.0)
    with st.expander("Risk rules", expanded=False):
        dstop = st.number_input("Daily stop (₹): no new entries once day P&L ≤ -X (0 = off)", min_value=0.0,
                                value=0.0, step=50000.0, format="%.0f")
        rf = st.number_input("Risk-free rate % p.a.", value=6.5, step=0.25) / 100

if not sel:
    st.warning("Select at least one strategy.")
    st.stop()

# slippage inputs are "per trade, round trip": half on entry, half on exit
S = P.Settings(lots=lots, lot_size=int(lot_size), capital=float(capital), hedge=hedge, slip_in=sell_slip / 2,
               slip_out=sell_slip / 2, slip_sl_extra=0.0, slip_hedge=hedge_slip / 2, slip_pct=0.0, costs=costs,
               brokerage=brokerage, stt_pct=stt_old, stt_pct_new=stt_new, exch_pct=exch, stamp_pct=stamp,
               gst_pct=gst, daily_stop=dstop)
d0, d1 = dr
res = run(S, d0, d1, sel)
daily, trd, mar, eq_i = res["daily"], res["trades"], res["margin"], res["intraday"]
live = [s_ for s_ in STRATS if s_ in sel and lots[s_] > 0]
if not live:
    st.warning("All selected strategies have 0 lots.")
    st.stop()
summ = R.summary(daily["TOTAL"], S.capital, rf=rf, intraday=eq_i, trades=trd)
x_day = daily["TOTAL"]

# ------------------------------------------------------------------------------ header + KPI strip
slip_on = sell_slip > 0 or (hedge and hedge_slip > 0)
chips = [f'<span class="chip">{d0:%d %b %Y} → {d1:%d %b %Y}</span>', f'<span class="chip">{summ["days"]} trading days</span>',
         f'<span class="chip {"on" if hedge else "warn"}">Hedge {"ON" if hedge else "OFF"}</span>',
         f'<span class="chip {"warn" if slip_on else "on"}">Slippage {sell_slip:.2f} / {hedge_slip:.2f} pts</span>',
         f'<span class="chip {"on" if costs else "warn"}">Costs {"ON" if costs else "OFF"}</span>',
         f'<span class="chip">Capital {inr(S.capital)}</span>']
st.markdown(f'<div class="topbar"><div><div class="t1">Nifty Options Portfolio Lab</div>'
            f'<div class="t2">Systematic weekly index-option strategies  |  1-min exchange data  |  net of hedge, '
            f'slippage and costs as set  |  historical simulation, not advice</div></div>'
            f'<div class="chips">{"".join(chips)}</div></div>', unsafe_allow_html=True)

sgn = lambda v: "pos" if v > 0 else ("neg" if v < 0 else "")
n_up = int((x_day > 0).sum())
st.markdown(kpi_html([
    ("Net P&L", inr(summ["total_pnl"]), f"{summ['total_return_pct']:+.1f}% of capital", sgn(summ["total_pnl"])),
    ("CAGR", pct(summ["cagr_pct"]), f"vol {pct(summ['ann_vol_pct'])} p.a.", ""),
    ("Sharpe", f"{summ['sharpe']:.2f}", f"Sortino {summ['sortino']:.2f}", ""),
    ("Calmar", f"{summ['calmar']:.1f}" if not np.isnan(summ["calmar"]) else "-", f"daily PF {summ['profit_factor']:.2f}", ""),
    ("Max drawdown", pct(summ["max_dd_pct"]), f"closed-day · {inr(summ['max_dd'])}", "neg"),
    ("Intraday max DD", pct(summ["intraday_max_dd_pct"]), "marked to 1-min", "neg"),
    ("Trade win rate", pct(summ["trade_win_pct"], 1), f"{summ['trades']:,} trades · payoff {summ['payoff']:.2f}", ""),
    ("Day win rate", pct(summ["win_day_pct"], 1), f"{n_up} of {summ['days']} days", ""),
]), unsafe_allow_html=True)

tab_ov, tab_st, tab_rk, tab_bm, tab_rg, tab_sc, tab_tr = st.tabs(
    ["Overview", "Strategies", "Risk", "Benchmark", "Regimes", "Scenarios", "Trades"])

# ------------------------------------------------------------------------------ overview
with tab_ov:
    left, right = st.columns([2.05, 1])
    with left:
        with panel("Performance", "Cumulative net P&L (₹ lakh) and drawdown from peak (% of capital)"):
            cum = daily.cumsum() / 1e5
            dd_s, _ = R.drawdown_series(daily["TOTAL"], S.capital)
            f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
            for s_ in live:
                f.add_scatter(x=cum.index, y=cum[s_], name=s_, line=dict(color=COL[s_], width=1.3), row=1, col=1,
                              hovertemplate="%{y:.1f} L")
            f.add_scatter(x=cum.index, y=cum["TOTAL"], name="Portfolio", line=dict(color=T["total"], width=2.4), row=1,
                          col=1, hovertemplate="%{y:.1f} L")
            f.add_scatter(x=dd_s.index, y=100 * dd_s / S.capital, name="Drawdown %", fill="tozeroy",
                          line=dict(color=T["neg"], width=1), fillcolor="rgba(220,38,38,0.20)", row=2, col=1,
                          hovertemplate="%{y:.2f}%")
            f.update_yaxes(title_text="₹ lakh", title_font=dict(size=10), row=1, col=1)
            f.update_yaxes(title_text="DD %", title_font=dict(size=10), row=2, col=1)
            show(f, 470)
    with right:
        with panel("Book snapshot", "Contribution and stand-alone quality of each sleeve"):
            rows = {}
            tot = daily["TOTAL"].sum()
            for s_ in live:
                sm = R.summary(daily[s_], S.capital, rf=rf, trades=trd[trd["strategy"] == s_])
                rows[s_] = {"Lots": lots[s_], "Net P&L": inr(sm["total_pnl"]),
                            "Share": pct(100 * daily[s_].sum() / tot, 0) if tot else "-",
                            "Sharpe": f"{sm['sharpe']:.2f}", "Max DD": pct(sm["max_dd_pct"])}
            rows["Portfolio"] = {"Lots": sum(lots[s_] for s_ in live), "Net P&L": inr(summ["total_pnl"]), "Share": "100%",
                                 "Sharpe": f"{summ['sharpe']:.2f}", "Max DD": pct(summ["max_dd_pct"])}
            snap = pd.DataFrame(rows).T
            snap.index.name = "Sleeve"
            st.markdown(html_table(snap, cls_cols=("Max DD",)), unsafe_allow_html=True)
            alloc = go.Figure(go.Pie(labels=live, values=[lots[s_] for s_ in live], hole=0.62, sort=False,
                                     marker=dict(colors=[COL[s_] for s_ in live], line=dict(color=T["panel"], width=2)),
                                     textinfo="percent", textfont=dict(size=11, color="#fff"),
                                     hovertemplate="%{label}: %{value} lots<extra></extra>"))
            alloc.update_layout(height=190, margin=dict(l=0, r=0, t=6, b=0), showlegend=True,
                                legend=dict(orientation="v", x=0.0, y=0.5, font=dict(size=10.5, color=T["ink2"])),
                                paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT),
                                annotations=[dict(text="Lot<br>allocation", showarrow=False,
                                                  font=dict(size=10, color=T["ink2"]))])
            st.plotly_chart(alloc, use_container_width=True, config={"displayModeBar": False})

    with panel("Performance analytics", "Trade-level, day-level and tail statistics at the current settings"):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(stat_block("Trades", [
            ("Trades", f"{summ['trades']:,}"), ("Win rate", pct(summ["trade_win_pct"], 1)),
            ("SL-hit rate", pct(summ["sl_hit_pct"], 1)),
            ("Avg win / avg loss", f"{inr(summ['avg_win'])} / {inr(summ['avg_loss'])}"),
            ("Payoff ratio", f"{summ['payoff']:.2f}"), ("Expectancy / trade", inr(summ["expectancy"]))]),
            unsafe_allow_html=True)
        c2.markdown(stat_block("Days", [
            ("Winning days", f"{n_up} of {summ['days']} · {pct(summ['win_day_pct'], 1)}"),
            ("Best / worst day", f"{inr(summ['best_day'])} / {inr(summ['worst_day'])}"),
            ("Average day", inr(x_day.mean())), ("Profit factor (daily)", f"{summ['profit_factor']:.2f}"),
            ("Positive months", f"{summ['pos_months']} of {summ['months']}"),
            ("Best / worst month", f"{inr(summ['best_month'])} / {inr(summ['worst_month'])}")]), unsafe_allow_html=True)
        c3.markdown(stat_block("Drawdown", [
            ("Max DD (closed-day)", f"{pct(summ['max_dd_pct'])}  ·  {inr(summ['max_dd'])}"),
            ("Max DD (intraday)", f"{pct(summ['intraday_max_dd_pct'])}  ·  {inr(summ['intraday_max_dd'])}"),
            ("Worst intraday swing", inr(summ["worst_intraday_swing"])),
            ("Longest underwater", f"{summ['longest_underwater_days']} days"),
            ("Average drawdown", pct(summ["avg_dd_pct"], 2)), ("Ulcer index", pct(summ["ulcer_pct"], 2))]),
            unsafe_allow_html=True)
        c4.markdown(stat_block("Tail & shape", [
            ("VaR 95% (1 day)", inr(summ["var95"])), ("CVaR 95%", inr(summ["cvar95"])),
            ("VaR 99% (1 day)", inr(summ["var99"])), ("Tail ratio p95/p5", f"{summ['tail_ratio']:.2f}"),
            ("Skew / ex. kurtosis", f"{summ['skew']:.2f} / {summ['kurtosis']:.2f}"),
            ("Worst week", inr(summ["worst_week"]))]), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        tab = R.monthly_table(daily["TOTAL"], S.capital)
        yrs = "  ·  ".join(f"{int(y)}: {tab.loc[y, 'Year']:+.1f}%" for y in tab.index)
        with panel("Monthly returns", f"% of starting capital  |  {yrs}"):
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            z = tab.reindex(columns=list(range(1, 13)))
            fh = go.Figure(go.Heatmap(z=z.values, x=months, y=[str(int(y)) for y in z.index], zmid=0,
                                      colorscale=DIV(),
                                      text=np.where(np.isnan(z.values), "", np.round(z.values, 1).astype(str)),
                                      texttemplate="%{text}", textfont=dict(size=10), showscale=False,
                                      xgap=2, ygap=2, hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>"))
            fh.update_yaxes(autorange="reversed", type="category")
            show(fh, 230, legend=False)
    with c2:
        with panel("P&L bridge", "From ideal short-leg fills to net result (₹ lakh)"):
            base = S.with_(hedge=False, slip_in=0, slip_out=0, slip_sl_extra=0, slip_hedge=0, slip_pct=0, costs=False)
            a = run(base, d0, d1, sel, False)["daily"]["TOTAL"].sum()
            b_ = run(base.with_(hedge=hedge), d0, d1, sel, False)["daily"]["TOTAL"].sum()
            c_ = run(S.with_(costs=False), d0, d1, sel, False)["daily"]["TOTAL"].sum()
            d_ = daily["TOTAL"].sum()
            steps = [("Short legs<br>ideal fills", a, "absolute"), ("Hedge<br>legs", b_ - a, "relative"),
                     ("Slippage", c_ - b_, "relative"), ("Costs", d_ - c_, "relative"), ("Net P&L", d_, "total")]
            fw = go.Figure(go.Waterfall(x=[s[0] for s in steps], measure=[s[2] for s in steps],
                                        y=[s[1] / 1e5 for s in steps], connector=dict(line=dict(color=T["line"])),
                                        increasing=dict(marker=dict(color=T["pos"])),
                                        decreasing=dict(marker=dict(color=T["neg"])),
                                        totals=dict(marker=dict(color=T["total"])), texttemplate="%{y:.1f}",
                                        textfont=dict(size=10), hovertemplate="%{y:.2f} L<extra></extra>"))
            show(fw, 230, legend=False)

    with panel("Monthly P&L by strategy", "₹ lakh, stacked - which sleeve earned each month"):
        t2 = trd.copy()
        t2["month"] = pd.to_datetime(t2["date"]).dt.to_period("M").astype(str)
        mo = t2.groupby(["month", "strategy"])["net_inr"].sum().unstack(fill_value=0) / 1e5
        fm = go.Figure()
        for s_ in live:
            fm.add_bar(x=mo.index, y=mo[s_], name=s_, marker_color=COL[s_])
        fm.update_layout(barmode="relative", bargap=0.25)
        show(fm, 250)

# ------------------------------------------------------------------------------ strategies
with tab_st:
    with panel("Strategy scorecard", "Stand-alone statistics of each sleeve at the lots and execution set in the sidebar"):
        rows = {}
        for s_ in live:
            t = trd[trd["strategy"] == s_]
            sm = R.summary(daily[s_], S.capital, rf=rf, trades=t)
            rows[s_] = {"Lots": lots[s_], "Trades": f"{sm['trades']:,}", "Win %": f"{sm['trade_win_pct']:.1f}",
                        "SL hit %": f"{sm['sl_hit_pct']:.1f}", "Avg pts (net)": f"{t['net_pts'].mean():.2f}",
                        "Payoff": f"{sm['payoff']:.2f}", "Profit factor": f"{sm['profit_factor']:.2f}",
                        "Net P&L": inr(sm["total_pnl"]), "Hedge P&L": inr((t["hedge_net_pts"] * t["qty"]).sum()),
                        "Costs": inr(t["cost_inr"].sum()), "Sharpe": f"{sm['sharpe']:.2f}",
                        "Max DD": pct(sm["max_dd_pct"]), "Worst day": inr(sm["worst_day"])}
        sc = pd.DataFrame(rows).T
        sc.index.name = "Strategy"
        st.markdown(html_table(sc, cls_cols=("Max DD", "Worst day", "Hedge P&L")), unsafe_allow_html=True)

    t2 = trd.copy()
    t2["hour"] = t2["entry_time"].str[:2] + ":00"
    c1, c2 = st.columns(2)

    def grp(df, col):
        g = df.groupby([col, "strategy"])["net_inr"].sum().unstack(fill_value=0) / 1e5
        return g.reindex(columns=live)

    def bars(title, sub, g, h=260):
        with panel(title, sub):
            fb = go.Figure()
            for s_ in live:
                fb.add_bar(x=g.index.astype(str), y=g[s_], name=s_, marker_color=COL[s_])
            fb.update_layout(barmode="group", bargap=0.25)
            show(fb, h)

    with c1:
        bars("P&L by entry hour", "₹ lakh", grp(t2, "hour"))
    with c2:
        bars("P&L by exit type", "₹ lakh", grp(t2, "reason"))
    c3, c4 = st.columns(2)
    with c3:
        bars("CE vs PE", "₹ lakh", grp(t2, "side"))
    with c4:
        with panel("Trade P&L distribution", "net points per share"):
            fd = go.Figure()
            for s_ in live:
                fd.add_histogram(x=t2[t2["strategy"] == s_]["net_pts"], name=s_, marker_color=COL[s_], opacity=0.7,
                                 xbins=dict(size=2))
            fd.update_layout(barmode="overlay", hovermode="closest")
            show(fd, 260)

# ------------------------------------------------------------------------------ risk
with tab_rk:
    c1, c2 = st.columns(2)
    with c1:
        with panel("Daily P&L distribution", "₹ lakh with VaR markers"):
            fv = go.Figure(go.Histogram(x=x_day / 1e5, nbinsx=50, marker_color=T["ink2"], name="Daily P&L"))
            for v, nm in ((summ["var95"], "VaR 95"), (summ["var99"], "VaR 99")):
                fv.add_vline(x=v / 1e5, line=dict(color=T["neg"], dash="dash", width=1.4), annotation_text=nm,
                             annotation_font_size=10, annotation_font_color=T["ink2"])
            fv.update_layout(hovermode="closest")
            show(fv, 280, legend=False)
    with c2:
        with panel("Rolling 60-day Sharpe", "annualised, net of risk-free"):
            rs = R.rolling_sharpe(daily["TOTAL"], S.capital, 60, rf)
            fr = go.Figure(go.Scatter(x=rs.index, y=rs, line=dict(color=T["total"], width=1.6), name="60d Sharpe"))
            show(fr, 280, legend=False)

    if len(live) > 1:
        c3, c4 = st.columns(2)
        with c3:
            with panel("Correlation of daily P&L"):
                corr = daily[live].corr()
                fc = go.Figure(go.Heatmap(z=corr.values, x=live, y=live, zmin=-1, zmax=1, zmid=0, xgap=2, ygap=2,
                                          colorscale=DIV(), text=np.round(corr.values, 2), texttemplate="%{text}", showscale=False))
                fc.update_yaxes(autorange="reversed")
                show(fc, 260, legend=False)
        with c4:
            rc, dr_ = R.risk_contribution(daily, live)
            with panel("Risk contribution vs allocation", f"diversification ratio {dr_:.2f}  (1.0 = none)"):
                fb = go.Figure()
                fb.add_bar(x=live, y=100 * rc["share_of_variance"], name="Share of variance", marker_color=T["muted"])
                fb.add_bar(x=live, y=100 * rc["share_of_tail_loss"], name="Share of tail losses (worst 5% days)",
                           marker_color=T["neg"])
                fb.add_scatter(x=live, y=[100 * lots[s_] / sum(lots[q] for q in live) for s_ in live], mode="markers",
                               name="Share of lots", marker=dict(color=T["total"], size=10, symbol="diamond"))
                fb.update_layout(barmode="group", hovermode="closest")
                show(fb, 260)

    with panel("Concurrent exposure", "Most short lots open at the same time, by day"):
        fmg = go.Figure()
        fmg.add_scatter(x=mar.index, y=mar["peak_lots"], name="Peak concurrent lots", line=dict(color=T["ink2"], width=1.2),
                        fill="tozeroy", fillcolor="rgba(100,116,139,0.18)")
        show(fmg, 220, legend=False)
        st.caption(f"Peak concurrent short lots: {int(mar['peak_lots'].max())}  ·  average of the daily peaks: "
                   f"{mar['peak_lots'].mean():.0f}. Margin is not modelled.")

    c5, c6 = st.columns(2)
    with c5:
        with panel("Stress test", "Worst 5 historical days replayed at 1x / 2x / 3x"):
            stt_ = R.stress_table(daily, S.capital)
            tb = pd.DataFrame({"1x": stt_["pnl_1x"].map(inr), "2x": stt_["pnl_2x"].map(inr),
                               "3x": stt_["pnl_3x"].map(inr),
                               "3x % capital": stt_["pct_capital_3x"].round(1).map(lambda v: f"{v:.1f}%")})
            tb.index = stt_["date"].astype(str)
            tb.index.name = "Date"
            st.markdown(html_table(tb, cls_cols=("1x", "2x", "3x", "3x % capital")), unsafe_allow_html=True)
    mc = R.monte_carlo(daily["TOTAL"], S.capital)
    with c6:
        with panel("Monte-Carlo", "5-day block bootstrap, 3,000 paths of the same length"):
            pc_ = np.percentile
            mt = pd.DataFrame({"5th pct": [inr(pc_(mc["final"], 5)), inr(pc_(mc["max_dd"], 5))],
                               "Median": [inr(pc_(mc["final"], 50)), inr(pc_(mc["max_dd"], 50))],
                               "95th pct": [inr(pc_(mc["final"], 95)), inr(pc_(mc["max_dd"], 95))]},
                              index=["Final P&L", "Max drawdown"])
            mt.index.name = "Metric"
            st.markdown(html_table(mt), unsafe_allow_html=True)
            st.caption(f"P(final P&L < 0) = {100 * (mc['final'] < 0).mean():.1f}%  ·  "
                       f"P(max DD worse than 10% of capital) = {100 * (mc['max_dd'] < -0.10 * S.capital).mean():.1f}%")
    with panel("Equity paths", "History vs 150 resampled alternatives (₹ lakh)"):
        fmc = go.Figure()
        xs = np.arange(1, mc["curves"].shape[1] + 1)
        for cv in mc["curves"][:150]:
            fmc.add_scatter(x=xs, y=cv / 1e5, mode="lines", line=dict(color="rgba(100,116,139,0.14)", width=1),
                            hoverinfo="skip", showlegend=False)
        fmc.add_scatter(x=xs, y=(S.capital + np.cumsum(x_day.to_numpy())) / 1e5, name="Actual history",
                        line=dict(color=T["pos"], width=2.4))
        fmc.update_layout(hovermode="closest")
        show(fmc, 280, legend=False)


# ------------------------------------------------------------------------------ benchmark
with tab_bm:
    bm = _bench()
    idx = [d for d in bm.index if d0 <= d <= d1 and d >= daily.index[0]]
    bret = bm.loc[idx, "ret"]
    pnl = daily["TOTAL"].reindex(idx).fillna(0.0)
    pret = pnl / S.capital
    rfd = rf / 252
    if len(idx) < 30:
        st.info("Select a longer period to compare against the benchmark.")
    else:
        rel = R.relative_stats(pret.values, bret.values, rf)
        ps = R.series_stats(pret.values, rf, compound=False)
        ns = R.series_stats(bret.values, rf)
        cs = R.series_stats(np.full(len(idx), rfd), rf)
        st.markdown(kpi_html([
            ("Beta to Nifty", f"{rel['beta']:.2f}", "daily P&L vs Nifty move", ""),
            ("Alpha (ann.)", pct(rel["alpha_ann_pct"]), "over cash, after beta", sgn(rel["alpha_ann_pct"])),
            ("Correlation", f"{rel['corr']:.2f}", f"R² {rel['r2']:.2f}", ""),
            ("Info ratio", f"{rel['info_ratio']:.2f}", f"tracking err {pct(rel['tracking_error_pct'])}", ""),
            ("Excess CAGR", pct(ps["cagr_pct"] - ns["cagr_pct"]), f"vs Nifty {pct(ns['cagr_pct'])}",
             sgn(ps["cagr_pct"] - ns["cagr_pct"])),
            ("Up capture", f"{rel['up_capture']:.2f}x", "avg return on Nifty-up days", ""),
            ("Down capture", f"{rel['down_capture']:.2f}x", "negative = gains when Nifty falls", ""),
            ("Nifty 50", pct(ns["total_pct"]), f"price index, max DD {pct(ns['max_dd_pct'])}", sgn(ns["total_pct"])),
        ]), unsafe_allow_html=True)

        eq_p = S.capital + np.cumsum(pnl.values)                                  # fixed lots, no compounding
        eq_n = S.capital * np.cumprod(1 + bret.values)
        eq_c = S.capital * np.cumprod(1 + np.full(len(idx), rfd))
        dd = lambda e: 100 * (e / np.maximum.accumulate(np.concatenate([[S.capital], e]))[1:] - 1)
        c1, c2 = st.columns([2.05, 1])
        with c1:
            with panel("Growth of capital", f"{inr(S.capital)} invested at the start: portfolio vs Nifty 50 buy-and-hold vs cash at {rf * 100:.1f}%"):
                f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
                for name, e, colr, w in (("Portfolio", eq_p, T["total"], 2.4), ("Nifty 50", eq_n, T["vw"], 1.6),
                                         (f"Cash {rf * 100:.1f}%", eq_c, T["muted"], 1.2)):
                    f.add_scatter(x=idx, y=e / 1e7, name=name, line=dict(color=colr, width=w), row=1, col=1,
                                  hovertemplate="%{y:.2f} Cr")
                    f.add_scatter(x=idx, y=dd(e), name=name + " DD", line=dict(color=colr, width=1), row=2, col=1,
                                  showlegend=False, hovertemplate="%{y:.1f}%")
                f.update_yaxes(title_text="₹ Cr", title_font=dict(size=10), row=1, col=1)
                f.update_yaxes(title_text="DD %", title_font=dict(size=10), row=2, col=1)
                show(f, 440)
        with c2:
            with panel("Side by side", "Same period, daily data"):
                rows = {
                    "Total return": [pct(ps["total_pct"]), pct(ns["total_pct"]), pct(cs["total_pct"])],
                    "CAGR": [pct(ps["cagr_pct"]), pct(ns["cagr_pct"]), pct(cs["cagr_pct"])],
                    "Volatility": [pct(ps["vol_pct"]), pct(ns["vol_pct"]), pct(cs["vol_pct"])],
                    "Sharpe": [f"{ps['sharpe']:.2f}", f"{ns['sharpe']:.2f}", "-"],
                    "Sortino": [f"{ps['sortino']:.2f}", f"{ns['sortino']:.2f}", "-"],
                    "Max drawdown": [pct(ps["max_dd_pct"]), pct(ns["max_dd_pct"]), pct(cs["max_dd_pct"])],
                    "Calmar": [f"{ps['calmar']:.1f}", f"{ns['calmar']:.1f}", "-"],
                    "Best day": [pct(ps["best_day_pct"], 2), pct(ns["best_day_pct"], 2), "-"],
                    "Worst day": [pct(ps["worst_day_pct"], 2), pct(ns["worst_day_pct"], 2), "-"],
                    "Up days": [pct(ps["up_days_pct"], 0), pct(ns["up_days_pct"], 0), "-"],
                }
                tb = pd.DataFrame(rows, index=["Portfolio", "Nifty 50", "Cash"]).T
                tb.index.name = "Metric"
                st.markdown(html_table(tb, cls_cols=("Portfolio", "Nifty 50")), unsafe_allow_html=True)
                st.caption("Nifty = price index (dividends excluded). Portfolio returns are on the starting capital with "
                           "fixed lots; Nifty and cash compound.")

        c3, c4 = st.columns(2)
        with c3:
            with panel("Daily P&L vs Nifty move", "Each dot is a day; the line is the regression (slope = beta)"):
                x = 100 * bret.values
                y = 100 * pret.values
                fs = go.Figure()
                fs.add_scatter(x=x, y=y, mode="markers", name="Days", marker=dict(size=5, color=T["pos"], opacity=0.55),
                               hovertemplate="Nifty %{x:.2f}% · portfolio %{y:.2f}%<extra></extra>")
                xs_ = np.array([x.min(), x.max()])
                fs.add_scatter(x=xs_, y=rel["beta"] * xs_ + (y.mean() - rel["beta"] * x.mean()), mode="lines",
                               name=f"beta {rel['beta']:.2f}", line=dict(color=T["neg"], width=2))
                fs.update_xaxes(title_text="Nifty daily move (%)", title_font=dict(size=10))
                fs.update_yaxes(title_text="Portfolio daily return (% of capital)", title_font=dict(size=10))
                fs.update_layout(hovermode="closest")
                show(fs, 300)
        with c4:
            with panel("Portfolio by market regime", "Average daily P&L (₹ lakh) by Nifty daily move; label = number of days"):
                bins = [-np.inf, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, np.inf]
                labs = ["< -1.5%", "-1.5 to -0.75", "-0.75 to -0.25", "flat ±0.25", "0.25 to 0.75", "0.75 to 1.5", "> 1.5%"]
                bk = pd.cut(100 * bret, bins=bins, labels=labs)
                g = pnl.groupby(bk, observed=False).agg(["mean", "count"])
                fr = go.Figure(go.Bar(x=labs, y=(g["mean"] / 1e5).fillna(0),
                                      marker_color=[T["pos"] if v >= 0 else T["neg"] for v in g["mean"].fillna(0)],
                                      text=[f"n={int(c)}" for c in g["count"]], textposition="outside",
                                      hovertemplate="%{x}: %{y:.2f} L avg<extra></extra>"))
                fr.update_layout(hovermode="closest")
                show(fr, 300, legend=False)

        c5, c6 = st.columns(2)
        with c5:
            with panel("Rolling beta and correlation", "60-day window vs Nifty"):
                w = 60
                pr_, br_ = pret, bret
                rb = pr_.rolling(w).cov(br_) / br_.rolling(w).var()
                rc_ = pr_.rolling(w).corr(br_)
                fb = go.Figure()
                fb.add_scatter(x=idx, y=rb, name="Beta", line=dict(color=T["total"], width=1.6))
                fb.add_scatter(x=idx, y=rc_, name="Correlation", line=dict(color=T["vw"], width=1.4))
                show(fb, 280)
        with c6:
            m_p = pnl.groupby(pd.to_datetime(pd.Series(idx, index=idx)).dt.to_period("M")).sum() / S.capital * 100
            m_n = ((1 + bret).groupby(pd.to_datetime(pd.Series(idx, index=idx)).dt.to_period("M")).prod() - 1) * 100
            with panel("Monthly return: portfolio vs Nifty", "% (portfolio on starting capital)"):
                fm2 = go.Figure()
                fm2.add_bar(x=m_p.index.astype(str), y=m_p.values, name="Portfolio", marker_color=T["total"])
                fm2.add_bar(x=m_n.index.astype(str), y=m_n.values, name="Nifty 50", marker_color=T["vw"])
                fm2.update_layout(barmode="group", bargap=0.25)
                show(fm2, 280)

        c7, c8 = st.columns(2)
        shock = pd.DataFrame({"nifty": 100 * bret, "pnl": pnl, "ret": 100 * pret})

        def day_table(df_):
            out = pd.DataFrame({"Nifty move": df_["nifty"].map(lambda v: f"{v:+.2f}%"),
                                "Portfolio P&L": df_["pnl"].map(inr),
                                "Portfolio %": df_["ret"].map(lambda v: f"{v:+.2f}%")})
            out.index = [str(d) for d in df_.index]
            out.index.name = "Date"
            return out

        with c7:
            with panel("Nifty's 8 worst days", "How the book behaved when the market fell hardest"):
                st.markdown(html_table(day_table(shock.nsmallest(8, "nifty")), cls_cols=("Nifty move", "Portfolio P&L",
                                                                                         "Portfolio %")),
                            unsafe_allow_html=True)
        with c8:
            with panel("Nifty's 8 best days", "How the book behaved when the market rallied hardest"):
                st.markdown(html_table(day_table(shock.nlargest(8, "nifty")), cls_cols=("Nifty move", "Portfolio P&L",
                                                                                        "Portfolio %")),
                            unsafe_allow_html=True)

# ------------------------------------------------------------------------------ regimes
with tab_rg:
    bm_ = _bench()
    idx_r = [d for d in bm_.index if d0 <= d <= d1 and d >= daily.index[0]]
    meta = bm_.loc[idx_r].copy()
    dr_ = daily.reindex(idx_r).fillna(0.0)
    series_cols = live + ["TOTAL"]
    # trailing 10-day realised vol of Nifty, known BEFORE each day
    meta["rv10"] = (bm_["ret"].rolling(10).std() * np.sqrt(252) * 100).shift(1).reindex(idx_r)
    meta["wd"] = pd.to_datetime(pd.Series(idx_r, index=idx_r)).dt.day_name()

    def terciles(x, name):
        ok = x.dropna()
        if len(ok) < 30:
            return pd.Series(index=x.index, dtype=object), []
        q1, q2 = ok.quantile([1 / 3, 2 / 3])
        labs = [f"Low (<{q1:.1f}%)", f"Mid ({q1:.1f}-{q2:.1f}%)", f"High (>{q2:.1f}%)"]
        cat = pd.cut(x, bins=[-np.inf, q1, q2, np.inf], labels=labs)
        return cat.astype(object), labs

    def reg_table(cat, order):
        rows = {}
        for c in order:
            m = (cat == c).reindex(dr_.index).fillna(False).values
            ds = dr_.loc[m]
            if not len(ds):
                continue
            tot = ds["TOTAL"]
            rows[c] = {"Days": len(ds), "Net P&L": inr(tot.sum()), "Avg / day": inr(tot.mean()),
                       "Win days": pct(100 * (tot > 0).mean(), 0), "Best day": inr(tot.max()), "Worst day": inr(tot.min()),
                       **{s_: inr(ds[s_].sum()) for s_ in live}}
        out = pd.DataFrame(rows).T
        out.index.name = "Bucket"
        return out

    def reg_bars(cat, order, h=260):
        fb = go.Figure()
        labels = []
        for c in order:
            m = (cat == c).reindex(dr_.index).fillna(False).values
            labels.append(f"{c}<br>n={int(m.sum())}")
        for s_ in live:
            ys = []
            for c in order:
                m = (cat == c).reindex(dr_.index).fillna(False).values
                ys.append(dr_.loc[m, s_].mean() / 1e5 if m.any() else 0.0)
            fb.add_bar(x=labels, y=ys, name=s_, marker_color=COL[s_])
        fb.update_layout(barmode="group", bargap=0.25, hovermode="closest")
        show(fb, h)

    def reg_panel(title, sub, cat, order):
        with panel(title, sub):
            a_, b_ = st.columns([1, 1.25])
            with a_:
                reg_bars(cat, order)
            with b_:
                st.markdown(html_table(reg_table(cat, order), cls_cols=("Net P&L", "Avg / day", "Worst day"), scroll=True),
                            unsafe_allow_html=True)

    # ---- rolling quarterly returns
    with panel("Rolling quarterly returns", "Return over the trailing 63 trading days (about one quarter), % of starting capital"):
        roll = dr_.rolling(63).sum() / S.capital * 100
        c1, c2 = st.columns([1.6, 1])
        with c1:
            fr_ = go.Figure()
            for s_ in live:
                fr_.add_scatter(x=roll.index, y=roll[s_], name=s_, line=dict(color=COL[s_], width=1.2))
            fr_.add_scatter(x=roll.index, y=roll["TOTAL"], name="Portfolio", line=dict(color=T["total"], width=2.4))
            fr_.add_hline(y=0, line=dict(color=T["line"], width=1))
            fr_.update_yaxes(title_text="63-day return %", title_font=dict(size=10))
            show(fr_, 290)
        with c2:
            rq = roll["TOTAL"].dropna()
            if len(rq):
                st.markdown(stat_block("Rolling quarter, portfolio", [
                    ("Windows", f"{len(rq)}"), ("Positive", pct(100 * (rq > 0).mean(), 0)),
                    ("Best", pct(rq.max())), ("Median", pct(rq.median())), ("Worst", pct(rq.min())),
                    ("5th percentile", pct(rq.quantile(0.05)))]), unsafe_allow_html=True)
        cal = dr_.groupby(pd.to_datetime(pd.Series(idx_r, index=idx_r)).dt.to_period("Q")).sum() / S.capital * 100
        fq = go.Figure()
        for s_ in live:
            fq.add_bar(x=[f"{q.year} Q{q.quarter}" for q in cal.index], y=cal[s_], name=s_, marker_color=COL[s_])
        fq.add_scatter(x=[f"{q.year} Q{q.quarter}" for q in cal.index], y=cal["TOTAL"], mode="markers+lines",
                       name="Portfolio", line=dict(color=T["total"], width=1.5), marker=dict(size=7, symbol="diamond"))
        fq.update_layout(barmode="relative", bargap=0.3)
        fq.update_yaxes(title_text="% of capital", title_font=dict(size=10))
        st.caption("Calendar quarters (first and last may be partial)")
        show(fq, 250)

    # ---- volatility regimes
    iv_cat, iv_lab = terciles(meta["iv_proxy"], "iv")
    rv_cat, rv_lab = terciles(meta["rv10"], "rv")
    if iv_lab:
        reg_panel("Volatility regime: implied (ATM straddle proxy)",
                  "Days split into thirds by the annualised implied-vol proxy at 09:24 (known before trading starts)", iv_cat, iv_lab)
    if rv_lab:
        reg_panel("Volatility regime: realised (trailing 10-day Nifty)",
                  "Days split into thirds by the annualised realised volatility of the previous 10 sessions", rv_cat, rv_lab)

    # ---- weekday and expiry
    wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    reg_panel("By weekday", "Average P&L per day, by sleeve", meta["wd"], wd_order)
    exp_cat = pd.Series(np.where(meta["dte"] == 0, "Expiry day", "Non-expiry day"), index=meta.index)
    reg_panel("Expiry day vs other days", "Expiry day = last session of the current weekly contract", exp_cat,
              ["Expiry day", "Non-expiry day"])
    dte_lab = {0: "0 (expiry)", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5-6", 6: "5-6"}
    dte_cat = meta["dte"].map(dte_lab)
    reg_panel("By days to weekly expiry", "Calendar days until the current weekly contract expires", dte_cat,
              ["0 (expiry)", "1", "2", "3", "4", "5-6"])
    st.caption("Volatility proxies are derived from the option prices in the data (no VIX series). Bucket sizes are shown "
               "as n; small buckets are noisy. Two special weekend sessions are left out of the weekday table.")

# ------------------------------------------------------------------------------ scenarios
with tab_sc:
    with panel("Slippage ladder", "Round-trip slippage per trade, applied to the short leg and to the hedge; costs as set in the sidebar"):
        rows = {}
        for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            s2 = S.with_(slip_in=v / 2, slip_out=v / 2, slip_sl_extra=0.0, slip_hedge=(v / 2 if hedge else 0.0), slip_pct=0.0)
            r_ = run(s2, d0, d1, sel, False)["daily"]
            sm = R.summary(r_["TOTAL"], S.capital, rf=rf)
            rows[f"{v:.1f} pts"] = {**{s_: inr(r_[s_].sum()) for s_ in live}, "Portfolio": inr(r_["TOTAL"].sum()),
                                    "Sharpe": f"{sm['sharpe']:.2f}", "Max DD": pct(sm["max_dd_pct"])}
        lad = pd.DataFrame(rows).T
        lad.index.name = "Slippage / trade"
        st.markdown(html_table(lad, cls_cols=("Portfolio", "Max DD")), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        with panel("Break-even slippage", "Round-trip points per trade that would wipe out each strategy's profit (short leg only)"):
            p0 = run(S.with_(slip_in=0, slip_out=0, slip_sl_extra=0, slip_hedge=0, slip_pct=0), d0, d1, sel, False)["daily"]
            p1 = run(S.with_(slip_in=0.5, slip_out=0.5, slip_sl_extra=0, slip_hedge=0, slip_pct=0), d0, d1, sel, False)["daily"]
            be = {s_: (p0[s_].sum() / (p0[s_].sum() - p1[s_].sum()) if p0[s_].sum() > p1[s_].sum() else np.nan)
                  for s_ in live}
            fbe = go.Figure(go.Bar(x=list(be.values()), y=list(be.keys()), orientation="h",
                                   marker_color=[COL[s_] for s_ in be], text=[f"{v:.2f} pts" for v in be.values()],
                                   textposition="outside", hovertemplate="%{y}: %{x:.2f} pts<extra></extra>"))
            fbe.update_layout(hovermode="closest", margin=dict(r=60))
            show(fbe, 210, legend=False)
    with c2:
        with panel("Position-size ladder", "Same trades, lots × multiple"):
            rows = {}
            for mult in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                sm = R.summary(daily["TOTAL"] * mult, S.capital, rf=rf)
                rows[f"{mult:g}x"] = {"Net P&L": inr(sm["total_pnl"]), "Sharpe": f"{sm['sharpe']:.2f}",
                                      "Max DD": pct(sm["max_dd_pct"]), "Worst day": inr(sm["worst_day"]),
                                      "Peak lots": f"{int(mar['peak_lots'].max() * mult)}"}
            sz = pd.DataFrame(rows).T
            sz.index.name = "Lots"
            st.markdown(html_table(sz, cls_cols=("Max DD", "Worst day")), unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        with panel("Suggested lot splits", "Same total volatility as the current book; fitted in-sample, a balance reference not a forecast"):
            if len(live) >= 2:
                unit = run(S.with_(lots={k: 1 for k in STRATS}), d0, d1, sel, False)["daily"][live]
                sug = R.suggest_lots(unit, {k: lots[k] for k in live}, rf).T
                sug.index = ["Current", "Inverse-vol", "Risk-parity", "Max-Sharpe", "Min-variance"]
                sug.index.name = "Method"
                sug = sug.map(lambda v: f"{v:.1f}")
                st.markdown(html_table(sug), unsafe_allow_html=True)
            else:
                st.info("Select two or more strategies to see weight suggestions.")
    with c4:
        with panel("Hedge value", "Same book with and without hedge legs"):
            h_on = run(S.with_(hedge=True), d0, d1, sel, True)
            h_off = run(S.with_(hedge=False), d0, d1, sel, True)
            so = R.summary(h_on["daily"]["TOTAL"], S.capital, rf=rf, intraday=h_on["intraday"])
            sf = R.summary(h_off["daily"]["TOTAL"], S.capital, rf=rf, intraday=h_off["intraday"])
            hv = pd.DataFrame({
                "With hedge": [inr(so["total_pnl"]), f"{so['sharpe']:.2f}", pct(so["max_dd_pct"]),
                               pct(so["intraday_max_dd_pct"]), inr(so["worst_day"])],
                "Without hedge": [inr(sf["total_pnl"]), f"{sf['sharpe']:.2f}", pct(sf["max_dd_pct"]),
                                  pct(sf["intraday_max_dd_pct"]), inr(sf["worst_day"])]},
                index=["Net P&L", "Sharpe", "Max DD (closed)", "Max DD (intraday)", "Worst day"])
            hv.index.name = "Metric"
            st.markdown(html_table(hv), unsafe_allow_html=True)
            st.caption("The hedge costs a few points per trade here but caps the tail. This window has no crash day, "
                       "so its insurance value is under-represented.")

# ------------------------------------------------------------------------------ trades
with tab_tr:
    with panel("Trade blotter", "Every trade at the current lots, slippage, cost and hedge settings"):
        show_cols = [c_ for c_ in ["strategy", "date", "side", "sell_strike", "entry_time", "exit_time",
                                   "sell_entry", "sell_exit", "reason", "sell_pts", "hedge_entry",
                                   "hedge_exit", "hedge_pts", "net_pts", "qty", "cost_inr", "net_inr"] if c_ in trd.columns]
        blot = trd[show_cols].sort_values(["date", "entry_time"])
        f1, f2_, f3 = st.columns([1, 1, 1])
        sf_ = f1.multiselect("Strategy", live, default=live)
        reasons = sorted(trd["reason"].unique().tolist())
        rs_ = f2_.multiselect("Exit type", reasons, default=reasons)
        sub = blot[blot["strategy"].isin(sf_) & trd.loc[blot.index, "reason"].isin(rs_)].round(2)
        page_n = 150
        pages = max(1, int(np.ceil(len(sub) / page_n)))
        pg = f3.number_input(f"Page (of {pages})", min_value=1, max_value=pages, value=1, step=1)
        view = sub.iloc[(pg - 1) * page_n: pg * page_n].reset_index(drop=True)
        view.index = view.index + 1 + (pg - 1) * page_n
        view.index.name = "#"
        view = view.rename(columns={"strategy": "Strategy", "date": "Date", "side": "Side", "sell_strike": "Strike", "entry_time": "Entry", "exit_time": "Exit",
                                    "sell_entry": "Entry px", "sell_exit": "Exit px", "reason": "Exit type",
                                    "sell_pts": "Short pts", "hedge_entry": "Hedge in",
                                    "hedge_exit": "Hedge out", "hedge_pts": "Hedge pts", "net_pts": "Net pts",
                                    "qty": "Qty", "cost_inr": "Costs ₹", "net_inr": "Net ₹"}).astype(str)
        st.markdown(html_table(view, scroll=True), unsafe_allow_html=True)
        st.caption(f"{len(sub):,} trades match  ·  showing {len(view)}")
        st.download_button("Download filtered trades (CSV)", sub.to_csv(index=False), "trades_filtered.csv", "text/csv")

with st.expander("Notes & disclaimer"):
    st.markdown(
        "* **Data:** 1-minute exchange option data (Dec 2024 - Sep 2026). Hedge legs are priced from their own 1-minute bars.\n"
        "* **Slippage** is entered per trade as a round trip and split equally between entry and exit "
        "(0.2 means 0.1 on the entry and 0.1 on the exit); it always works against you. Hedge slippage is defined the same way.\n"
        "* **Costs** (optional): brokerage per order, STT on option sell premium (0.10% before 1 Apr 2026, 0.15% from that date), "
        "exchange charges, GST and stamp duty.\n"
        "* **Returns** are measured on the starting capital with fixed lots (nothing is reinvested). Drawdown is shown on closed-day "
        "P&L and marked to the minute.\n"
        "* **Exposure:** margin is not modelled; concurrent short lots are shown instead.\n"
        "* ~21 months in one market regime; results are a historical simulation, not a forecast and not investment advice.")
st.markdown('<div class="foot">Nifty Options Portfolio Lab · historical simulation for research and education · '
            'not investment advice</div>', unsafe_allow_html=True)
