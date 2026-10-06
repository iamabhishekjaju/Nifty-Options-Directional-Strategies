### How the numbers are produced

**Backtest** - each strategy replays 1-minute Upstox option data (Dec 2024 - Sep 2026) and writes a
per-share trade log. The hedge leg is priced from its own 1-minute bars (buy at the open of the entry
minute, sell at the open/close of the exit minute). Lots, slippage, costs and stops are applied *here in
the dashboard*, so nothing is re-backtested when you move a control.

**Fills** - entries/exits are filled at the rule price. Slippage is entered per trade as a round trip and split equally
between entry and exit (0.2 = 0.1 in + 0.1 out) and always works against you; hedge slippage is defined the same way.
**Costs** - brokerage, STT on option sell premium (0.10% before 1-Apr-2026, 0.15% from then), exchange charges, GST, stamp duty.

**Equity & drawdown** - closed-day drawdown uses daily P&L. Intraday drawdown marks every open trade
(short leg + hedge) to the 1-minute close, so it includes open losses. Returns are P&L / starting capital
(fixed lots, nothing reinvested).

**Exposure** - margin is not modelled; the dashboard shows concurrent short lots instead.

**Risk statistics** - Sharpe/Sortino use a 6.5% risk-free rate (editable), 252 days. VaR/CVaR are
historical on daily P&L. Monte-Carlo is a 5-day block bootstrap of daily P&L.

### Read this before trusting any number
* ~21 months, one market regime, **no crash/gap day** - tail statistics and the hedge's insurance value are
  under-represented.
* Strategy parameters were chosen on this same data (in-sample). The optimised DC variant is more so.
* The per-trade edge is only ~2.5-5 points; **a point or two of slippage changes the conclusion** - see the
  Sensitivity tab and calibrate with your own broker fills.
* Entry/exit at rule prices from 1-minute bars ignores queue position, spreads and partial fills.
* Data ends 15:29 while live square-off is ~15:39; end-of-day exits use the 15:29 close.
* Rates for STT / charges / margin are editable inputs - verify them against current rules.
* Historical simulation for research and education, not investment advice.
