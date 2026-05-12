#!/usr/bin/env python3
"""
==============================================================================
SuperTrend Hypothesis Tester  |  Gold XAUUSD
==============================================================================
HYPOTHESIS:
  When SuperTrend flips direction, what is the probability of:
    A) Riding the trend  (exit in profit at next ST flip)
    B) Hitting the SL    (exit at a loss at next ST flip)

WHAT THIS SCRIPT DOES:
  1. Loads OHLC CSV (flexible — handles MT5 / standard exports)
  2. Computes SuperTrend with RMA ATR (exact match to your MQL5 EA)
  3. Simulates every ST flip as a trade entry; ST band = SL; next ST flip = exit
  4. Full statistical validation (binomial test, t-test, bootstrap CI)
  5. Directional, session, volatility, prior-trend breakdowns
  6. HMM regime analysis — does market state predict outcome? (needs hmmlearn)
  7. Alpha discovery — grid-search filter combinations for edge
  8. Visual report saved as PNG

INSTALL:
  pip install pandas numpy matplotlib scipy hmmlearn

USAGE:
  Set CSV_PATH below, then:  python supertrend_hypothesis_tester.py
==============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import binomtest, chi2_contingency
from itertools import product as iproduct

warnings.filterwarnings('ignore')

# ── optional HMM ──────────────────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# ==============================================================================
# ▶  CONFIGURATION  — edit before running
# ==============================================================================
CSV_PATH      = r"C:\Users\Pandya Anand\Downloads\xau_usd_dataset_London-Strategic-Edge.csv"   # path to your exported CSV
ST_ATR_LEN    = 10                  # SuperTrend ATR period   (matches EA)
ST_FACTOR     = 1.5                 # SuperTrend multiplier   (matches EA)
HMM_STATES    = 3                   # HMM hidden states (2 or 3 recommended)
MIN_TRADES    = 30                  # minimum trades needed for any stat
ALPHA_MIN_N   = 20                  # minimum trades per filter cell for alpha
OUTPUT_PNG    = "supertrend_hypothesis_report.png"
# ==============================================================================


# ──────────────────────────────────────────────────────────────────────────────
#  1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """
    Flexible OHLC CSV loader.
    Handles MT5 export ('2024.01.15 08:00') and standard ISO formats.
    Normalises column names to lowercase open/high/low/close.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}\n"
                                f"  → Set CSV_PATH at the top of the script.")

    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Detect datetime column
    time_candidates = ['time', 'date', 'datetime', 'timestamp', 'date_time',
                       'gmt_time', 'local_time']
    time_col = next((c for c in time_candidates if c in df.columns),
                    df.columns[0])

    # Try common datetime formats
    parsed = False
    for fmt in ['%Y.%m.%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
                '%d/%m/%Y %H:%M', '%Y/%m/%d %H:%M', '%m/%d/%Y %H:%M']:
        try:
            df['datetime'] = pd.to_datetime(df[time_col], format=fmt)
            parsed = True
            break
        except Exception:
            continue
    if not parsed:
        df['datetime'] = pd.to_datetime(df[time_col], infer_datetime_format=True)

    # Normalise OHLC column names
    rename = {}
    for col in df.columns:
        cl = col.replace('<', '').replace('>', '')
        if cl in ('open',  'o'):                              rename[col] = 'open'
        elif cl in ('high', 'h'):                             rename[col] = 'high'
        elif cl in ('low',  'l'):                             rename[col] = 'low'
        elif cl in ('close','c'):                             rename[col] = 'close'
        elif cl in ('volume','vol','tick_volume','tickvol'):  rename[col] = 'volume'
    df = df.rename(columns=rename)

    required = ['open', 'high', 'low', 'close']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    keep = ['datetime', 'open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        keep.append('volume')
    df = df[keep].copy()
    df = df.sort_values('datetime').reset_index(drop=True)
    df[['open', 'high', 'low', 'close']] = \
        df[['open', 'high', 'low', 'close']].astype(float)

    print(f"  Loaded {len(df):,} bars")
    print(f"  Range : {df['datetime'].iloc[0]}  →  {df['datetime'].iloc[-1]}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  2. SUPERTREND (RMA ATR — exact match to your MQL5 EA)
# ──────────────────────────────────────────────────────────────────────────────

def calc_supertrend(df: pd.DataFrame,
                    atr_len: int  = 10,
                    factor:  float = 3.0) -> pd.DataFrame:
    """
    SuperTrend using Wilder's RMA for ATR.
    Matches the MQL5 EA logic bar-for-bar.

    Added columns:
        atr           — Wilder RMA of True Range
        upper_band    — raw upper ST band
        lower_band    — raw lower ST band
        st_direction  — -1 = bullish, +1 = bearish
        st_value      — the displayed ST line (lower for bull, upper for bear)
        st_flipped    — True on the first bar of a new direction
    """
    df   = df.copy()
    n    = len(df)
    alpha = 1.0 / atr_len

    # True Range
    prev_close   = df['close'].shift(1).fillna(df['close'])
    df['tr']     = np.maximum(
        df['high'] - df['low'],
        np.maximum(np.abs(df['high'] - prev_close),
                   np.abs(df['low']  - prev_close))
    )

    # Wilder RMA (vectorised seed, then loop — accurate even on long series)
    atr_arr  = np.empty(n)
    tr_vals  = df['tr'].values
    atr_arr[0] = tr_vals[0]
    for i in range(1, n):
        atr_arr[i] = alpha * tr_vals[i] + (1.0 - alpha) * atr_arr[i - 1]
    df['atr'] = atr_arr

    hl2_vals   = ((df['high'] + df['low']) / 2.0).values
    close_vals = df['close'].values
    upper_arr  = np.empty(n)
    lower_arr  = np.empty(n)
    dir_arr    = np.ones(n, dtype=int)   # +1 = bear, -1 = bull

    upper_arr[0] = hl2_vals[0] + factor * atr_arr[0]
    lower_arr[0] = hl2_vals[0] - factor * atr_arr[0]

    for i in range(1, n):
        bu = hl2_vals[i] + factor * atr_arr[i]
        bl = hl2_vals[i] - factor * atr_arr[i]
        pu = upper_arr[i - 1]
        pl = lower_arr[i - 1]

        upper_arr[i] = bu if (bu < pu or close_vals[i - 1] > pu) else pu
        lower_arr[i] = bl if (bl > pl or close_vals[i - 1] < pl) else pl

        prev_dir = dir_arr[i - 1]
        if   close_vals[i] > pu: dir_arr[i] = -1
        elif close_vals[i] < pl: dir_arr[i] =  1
        else:                    dir_arr[i]  = prev_dir

    df['upper_band']   = upper_arr
    df['lower_band']   = lower_arr
    df['st_direction'] = dir_arr
    df['st_value']     = np.where(dir_arr == -1, lower_arr, upper_arr)

    flipped            = np.zeros(n, dtype=bool)
    flipped[1:]        = dir_arr[1:] != dir_arr[:-1]
    df['st_flipped']   = flipped

    return df


# ──────────────────────────────────────────────────────────────────────────────
#  3. TRADE SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def simulate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    One trade per SuperTrend flip.

    Entry  : close of the flip bar
    SL     : ST band at entry bar (lower_band for long, upper_band for short)
    Exit   : close of the bar where ST flips back (next opposing flip)

    Outcome: 'win'  if exit P/L > 0  (trailing ST stop exited in profit)
             'loss' if exit P/L <= 0 (price reversed through ST = SL hit)

    Per-trade columns:
        direction, entry_price, sl_price, exit_price,
        sl_dist, trade_pnl, r_multiple,
        outcome, mfe, mae,           — in R units
        duration_bars, entry_hour, entry_dow,
        entry_atr, prior_bars
    """
    flip_idx = df.index[df['st_flipped']].tolist()
    # Skip very first bar if it was flagged (no prior bar for trade setup)
    if flip_idx and flip_idx[0] == 0:
        flip_idx = flip_idx[1:]

    trades = []

    for k, start_i in enumerate(flip_idx):
        direction   = df.loc[start_i, 'st_direction']   # -1=bull, +1=bear
        trade_dir   = 'long' if direction == -1 else 'short'

        entry_price = df.loc[start_i, 'close']
        sl_price    = (df.loc[start_i, 'lower_band'] if trade_dir == 'long'
                       else df.loc[start_i, 'upper_band'])
        sl_dist     = abs(entry_price - sl_price)
        if sl_dist < 1e-9:
            continue   # degenerate bar — skip

        # Next opposing flip = natural exit (SL or trailing-stop closure)
        future_flips = [f for f in flip_idx if f > start_i]
        if not future_flips:
            continue   # open trade at data end — skip (no confirmed outcome)

        end_i       = future_flips[0]
        exit_price  = df.loc[end_i, 'close']

        trade_pnl   = (exit_price - entry_price if trade_dir == 'long'
                       else entry_price - exit_price)
        r_multiple  = trade_pnl / sl_dist

        # MFE / MAE over the life of the trade (in R)
        in_trade = df.loc[start_i:end_i]
        if trade_dir == 'long':
            mfe = (in_trade['high'].max()  - entry_price) / sl_dist
            mae = (entry_price - in_trade['low'].min())   / sl_dist
        else:
            mfe = (entry_price - in_trade['low'].min())  / sl_dist
            mae = (in_trade['high'].max() - entry_price) / sl_dist

        prior_bars  = (start_i - flip_idx[k - 1]) if k > 0 else 0

        trades.append({
            'entry_idx'    : start_i,
            'exit_idx'     : end_i,
            'datetime'     : df.loc[start_i, 'datetime'],
            'direction'    : trade_dir,
            'entry_price'  : entry_price,
            'sl_price'     : sl_price,
            'exit_price'   : exit_price,
            'sl_dist'      : sl_dist,
            'trade_pnl'    : trade_pnl,
            'r_multiple'   : r_multiple,
            'outcome'      : 'win' if r_multiple > 0 else 'loss',
            'mfe'          : mfe,
            'mae'          : mae,
            'duration_bars': end_i - start_i,
            'entry_hour'   : df.loc[start_i, 'datetime'].hour,
            'entry_dow'    : df.loc[start_i, 'datetime'].dayofweek,
            'entry_atr'    : df.loc[start_i, 'atr'],
            'prior_bars'   : prior_bars,
        })

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        raise ValueError("No completed trades found — is the CSV long enough?")

    tdf['datetime'] = pd.to_datetime(tdf['datetime'])
    tdf['atr_pct']  = tdf['entry_atr'] / tdf['entry_price'] * 100.0

    # Quartile bins for volatility and prior-trend duration
    try:
        tdf['atr_q'] = pd.qcut(tdf['entry_atr'], 4,
                                labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
    except ValueError:
        tdf['atr_q'] = pd.qcut(tdf['entry_atr'], 4, duplicates='drop',
                                labels=False).astype(str)

    try:
        tdf['prior_q'] = pd.qcut(tdf['prior_bars'].clip(lower=0), 4,
                                  labels=['brief', 'short', 'medium', 'long'],
                                  duplicates='drop')
    except ValueError:
        tdf['prior_q'] = 'all'

    return tdf


# ──────────────────────────────────────────────────────────────────────────────
#  4. STATISTICS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _hline(char='─', width=68):
    return char * width

def print_header(title: str):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print('='*68)


def core_stats(tdf: pd.DataFrame, label: str = "ALL TRADES") -> dict:
    """
    Full statistical summary for a trade set.

    Tests:
        Binomial test   — H0: win rate = 50%  (one-sided, greater)
        One-sample t    — H0: mean R-multiple = 0
        Bootstrap CI    — 10,000 resamples of win rate
    """
    n       = len(tdf)
    wins    = (tdf['outcome'] == 'win').sum()
    losses  = n - wins
    wr      = wins / n

    r_wins   = tdf.loc[tdf['outcome'] == 'win',  'r_multiple']
    r_losses = tdf.loc[tdf['outcome'] == 'loss', 'r_multiple']

    avg_win  = r_wins.mean()   if len(r_wins)   > 0 else 0.0
    avg_loss = r_losses.mean() if len(r_losses) > 0 else 0.0
    payoff   = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
    ev       = wr * avg_win + (1 - wr) * avg_loss

    # Binomial test — is win rate > 50%?
    binom_res = binomtest(int(wins), n, p=0.5, alternative='greater')

    # One-sample t-test — is mean R significantly != 0?
    t_stat, t_pval = stats.ttest_1samp(tdf['r_multiple'], 0.0)

    # Bootstrap CI on win rate
    rng     = np.random.default_rng(42)
    boot_wr = np.array([
        (rng.choice(tdf['outcome'].values, size=n, replace=True) == 'win').mean()
        for _ in range(10_000)
    ])
    ci_lo, ci_hi = np.percentile(boot_wr, [2.5, 97.5])

    # R-multiple descriptive stats
    r_vals  = tdf['r_multiple']
    sharpe  = r_vals.mean() / r_vals.std() if r_vals.std() > 0 else 0.0

    d = {
        'label': label, 'n': n, 'wins': wins, 'losses': losses,
        'win_rate': wr, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'avg_win_r': avg_win, 'avg_loss_r': avg_loss,
        'payoff': payoff, 'ev_r': ev,
        'binom_pval': binom_res.pvalue,
        'ttest_pval': t_pval,
        'sharpe_r': sharpe,
        'med_r': r_vals.median(),
        'std_r': r_vals.std(),
        'max_win': r_vals.max(),
        'max_loss': r_vals.min(),
        'avg_mfe': tdf['mfe'].mean(),
        'avg_mae': tdf['mae'].mean(),
        'avg_dur': tdf['duration_bars'].mean(),
    }

    sig_b = "✓ SIGNIFICANT" if binom_res.pvalue < 0.05 else "✗ NOT SIG"
    sig_t = "✓ SIGNIFICANT" if t_pval          < 0.05 else "✗ NOT SIG"

    print(f"\n  {_hline()}")
    print(f"  {label}")
    print(f"  {_hline()}")
    print(f"  Trades          : {n:6d}    Wins: {wins}   Losses: {losses}")
    print(f"  Win Rate        : {wr:7.2%}   95% CI [{ci_lo:.2%} – {ci_hi:.2%}]")
    print(f"  Avg Win  (R)    : +{avg_win:6.3f}R")
    print(f"  Avg Loss (R)    :  {avg_loss:6.3f}R")
    print(f"  Payoff Ratio    :  {payoff:6.3f}x   (avg_win / |avg_loss|)")
    print(f"  Expected Value  : {ev:+7.4f}R per trade")
    print(f"  Sharpe (R)      :  {sharpe:6.3f}")
    print(f"  Median R        : {d['med_r']:+7.4f}R")
    print(f"  Std R           :  {d['std_r']:6.3f}R")
    print(f"  Max Win / Loss  : +{d['max_win']:.3f}R / {d['max_loss']:.3f}R")
    print(f"  Avg MFE / MAE   : +{d['avg_mfe']:.2f}R / -{d['avg_mae']:.2f}R")
    print(f"  Avg Duration    :  {d['avg_dur']:.1f} bars")
    print(f"  Binomial test   :  p = {binom_res.pvalue:.5f}  {sig_b}")
    print(f"  T-test (mean=0) :  p = {t_pval:.5f}  {sig_t}")
    return d


# ──────────────────────────────────────────────────────────────────────────────
#  5. BREAKDOWN ANALYSES
# ──────────────────────────────────────────────────────────────────────────────

def _binom_pval(sub: pd.DataFrame) -> float:
    wins = int((sub['outcome'] == 'win').sum())
    try:
        return binomtest(wins, len(sub), 0.5).pvalue
    except Exception:
        return 1.0


def directional_breakdown(tdf: pd.DataFrame, baseline: dict):
    print_header("DIRECTIONAL BREAKDOWN")
    for d in ('long', 'short'):
        sub = tdf[tdf['direction'] == d]
        if len(sub) >= MIN_TRADES:
            core_stats(sub, label=f"{d.upper()} TRADES")
        else:
            print(f"\n  Insufficient {d} trades: {len(sub)} (need {MIN_TRADES})")


def session_breakdown(tdf: pd.DataFrame, baseline_wr: float):
    print_header("HOUR-OF-DAY BREAKDOWN")
    print(f"\n  {'Hour':>4}  {'N':>5}  {'WR':>7}  {'EV(R)':>8}  {'p-val':>8}  {'sig':>4}  {'vs base':>8}")
    print(f"  {'─'*4}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*8}")
    rows = []
    for h in range(24):
        sub = tdf[tdf['entry_hour'] == h]
        if len(sub) < ALPHA_MIN_N:
            continue
        wr  = (sub['outcome'] == 'win').mean()
        ev  = sub['r_multiple'].mean()
        p   = _binom_pval(sub)
        sig = ' ** ' if p < 0.01 else (' *  ' if p < 0.05 else '    ')
        print(f"  {h:4d}h  {len(sub):5d}  {wr:7.2%}  {ev:+8.4f}  {p:8.4f}  {sig}  {wr - baseline_wr:+7.2%}")
        rows.append({'hour': h, 'n': len(sub), 'wr': wr, 'ev': ev, 'pval': p})
    return pd.DataFrame(rows)


def volatility_breakdown(tdf: pd.DataFrame, baseline_wr: float):
    print_header("VOLATILITY BREAKDOWN  (ATR quartile at entry)")
    print(f"\n  {'ATR_Q':>10}  {'N':>5}  {'WR':>7}  {'EV(R)':>8}  {'p-val':>8}  {'sig':>4}")
    print(f"  {'─'*10}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*4}")
    rows = []
    cats = tdf['atr_q'].unique() if hasattr(tdf['atr_q'], 'cat') else tdf['atr_q'].unique()
    for q in sorted([str(c) for c in cats]):
        sub = tdf[tdf['atr_q'].astype(str) == q]
        if len(sub) < ALPHA_MIN_N:
            continue
        wr  = (sub['outcome'] == 'win').mean()
        ev  = sub['r_multiple'].mean()
        p   = _binom_pval(sub)
        sig = ' ** ' if p < 0.01 else (' *  ' if p < 0.05 else '    ')
        print(f"  {q:>10}  {len(sub):5d}  {wr:7.2%}  {ev:+8.4f}  {p:8.4f}  {sig}")
        rows.append({'atr_q': q, 'n': len(sub), 'wr': wr, 'ev': ev, 'pval': p})
    return pd.DataFrame(rows)


def prior_trend_breakdown(tdf: pd.DataFrame):
    print_header("PRIOR TREND DURATION BREAKDOWN")
    print("  (Was the trend that just flipped brief or long-lived?)")
    print(f"\n  {'Prior_Q':>8}  {'N':>5}  {'WR':>7}  {'EV(R)':>8}  {'p-val':>8}  Avg_prior_bars")
    print(f"  {'─'*8}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*14}")
    rows = []
    cats = (tdf['prior_q'].cat.categories.tolist()
            if hasattr(tdf['prior_q'], 'cat')
            else sorted(tdf['prior_q'].unique()))
    for q in cats:
        sub = tdf[tdf['prior_q'] == q]
        if len(sub) < ALPHA_MIN_N:
            continue
        wr   = (sub['outcome'] == 'win').mean()
        ev   = sub['r_multiple'].mean()
        p    = _binom_pval(sub)
        avgp = sub['prior_bars'].mean()
        sig  = ' ** ' if p < 0.01 else (' *  ' if p < 0.05 else '    ')
        print(f"  {str(q):>8}  {len(sub):5d}  {wr:7.2%}  {ev:+8.4f}  {p:8.4f}{sig}   {avgp:.1f}")
        rows.append({'prior_q': q, 'n': len(sub), 'wr': wr, 'ev': ev,
                     'pval': p, 'avg_prior': avgp})
    return pd.DataFrame(rows)


def dow_breakdown(tdf: pd.DataFrame):
    print_header("DAY-OF-WEEK BREAKDOWN")
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"\n  {'Day':>4}  {'N':>5}  {'WR':>7}  {'EV(R)':>8}  {'p-val':>8}")
    print(f"  {'─'*4}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}")
    for d in range(7):
        sub = tdf[tdf['entry_dow'] == d]
        if len(sub) < ALPHA_MIN_N:
            continue
        wr  = (sub['outcome'] == 'win').mean()
        ev  = sub['r_multiple'].mean()
        p   = _binom_pval(sub)
        sig = ' ** ' if p < 0.01 else (' *  ' if p < 0.05 else '    ')
        print(f"  {dow_names[d]:>4}  {len(sub):5d}  {wr:7.2%}  {ev:+8.4f}  {p:8.4f}{sig}")


# ──────────────────────────────────────────────────────────────────────────────
#  6. HMM REGIME ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def run_hmm(df: pd.DataFrame, tdf: pd.DataFrame, n_states: int = 3):
    """
    Fit Gaussian HMM on (log_return, log_ATR, HL_range) per bar.
    Label each trade entry with the HMM state and test if state predicts outcome.

    Interpretation guide:
        BULL — positive mean return, moderate vol
        BEAR — negative mean return, moderate vol
        CHOP — near-zero return, HIGH vol (whipsaw regime — worst for ST)
    """
    if not HMM_AVAILABLE:
        print_header("HMM REGIME ANALYSIS")
        print("\n  [SKIPPED] Install hmmlearn:  pip install hmmlearn")
        return None, tdf

    print_header("HMM REGIME ANALYSIS")

    # Feature matrix: log return, log ATR (vol proxy), HL spread
    log_ret  = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
    log_atr  = np.log(df['atr'].clip(lower=1e-8)).values
    hl_range = ((df['high'] - df['low']) / df['close'].clip(lower=1e-8)).values

    X = np.column_stack([log_ret, log_atr, hl_range])

    # Manual z-score normalisation (no sklearn required)
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-12
    X_norm = (X - X_mean) / X_std

    model  = GaussianHMM(n_components=n_states, covariance_type='full',
                         n_iter=200, random_state=42)
    model.fit(X_norm)
    states = model.predict(X_norm)

    df        = df.copy()
    df['hmm'] = states

    # Label states by their mean log-return (lowest = BEAR, highest = BULL)
    mean_ret = {s: log_ret[states == s].mean() for s in range(n_states)}
    sorted_s = sorted(mean_ret.items(), key=lambda x: x[1])
    if n_states == 2:
        lbl_map = {sorted_s[0][0]: 'BEAR', sorted_s[1][0]: 'BULL'}
    else:
        lbl_map = {sorted_s[0][0]: 'BEAR',
                   sorted_s[1][0]: 'CHOP',
                   sorted_s[2][0]: 'BULL'}
    df['hmm_label'] = df['hmm'].map(lbl_map)

    # Tag each trade with the HMM state at its entry bar
    tdf = tdf.copy()
    tdf['hmm_state'] = tdf['entry_idx'].map(
        lambda i: df.loc[i, 'hmm_label'] if i in df.index else 'UNKNOWN'
    )

    # Transition matrix printout
    print(f"\n  HMM  |  {n_states} states  |  features: log_return, log_ATR, HL_range")
    lbl_order = [lbl_map[s] for s in range(n_states)]
    header_str = "             " + "".join(f"{l:>8}" for l in lbl_order)
    print(f"\n  Transition Matrix:")
    print(f"  {header_str}")
    tm = model.transmat_
    for i in range(n_states):
        row_vals = "".join(f"{tm[i, j]:8.4f}" for j in range(n_states))
        print(f"  {lbl_order[i]:>10}   {row_vals}")

    # Win rates per regime
    print(f"\n  {'State':>6}  {'Bars':>7}  {'Trades':>7}  {'WR':>7}  {'EV(R)':>8}  {'p-val':>8}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")
    for lbl in sorted(set(lbl_map.values())):
        n_bars = (df['hmm_label'] == lbl).sum()
        sub    = tdf[tdf['hmm_state'] == lbl]
        if len(sub) < ALPHA_MIN_N:
            print(f"  {lbl:>6}  {n_bars:7d}  {len(sub):7d}  {'—':>7}  {'—':>8}  {'—':>8}")
            continue
        wr  = (sub['outcome'] == 'win').mean()
        ev  = sub['r_multiple'].mean()
        p   = _binom_pval(sub)
        sig = ' ** ' if p < 0.01 else (' *  ' if p < 0.05 else '    ')
        print(f"  {lbl:>6}  {n_bars:7d}  {len(sub):7d}  {wr:7.2%}  {ev:+8.4f}  {p:8.4f}{sig}")

    # Chi-squared: is HMM state independent of outcome?
    ct = pd.crosstab(tdf['hmm_state'], tdf['outcome'])
    if ct.shape[0] >= 2 and ct.shape[1] == 2:
        chi2, chi_p, dof, _ = chi2_contingency(ct)
        sig_c = "✓ SIGNIFICANT" if chi_p < 0.05 else "✗ NOT SIGNIFICANT"
        print(f"\n  Chi² test (outcome ⊥ HMM state):")
        print(f"  χ² = {chi2:.3f}   dof = {dof}   p = {chi_p:.5f}")
        print(f"  → {sig_c}")
        print(f"  (Significant = HMM regime IS predictive of whether ST trade wins or loses)")

    return df, tdf


# ──────────────────────────────────────────────────────────────────────────────
#  7. ALPHA DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────

def find_alpha(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    Exhaustive single + two-way filter search.
    Filters: direction, ATR quartile, session bucket, prior trend quartile.
    Applies Bonferroni correction for multiple comparisons.
    Returns ranked DataFrame of all cells with their EV lift vs baseline.
    """
    print_header("ALPHA DISCOVERY — FILTER COMBINATIONS")

    base_wr = (tdf['outcome'] == 'win').mean()
    base_ev = tdf['r_multiple'].mean()
    print(f"\n  Baseline  WR: {base_wr:.2%}   EV: {base_ev:+.5f}R")
    print(f"  Minimum trades per cell: {ALPHA_MIN_N}")
    print(f"  Searching single + two-way filter combinations...\n")

    # Session bucket
    tdf = tdf.copy()
    tdf['session'] = pd.cut(
        tdf['entry_hour'],
        bins=[-1, 6, 11, 16, 23],
        labels=['Asian', 'London', 'NewYork', 'OffHours']
    )

    filter_cols = {
        'direction': tdf['direction'].unique().tolist(),
        'atr_q'    : sorted(tdf['atr_q'].astype(str).unique().tolist()),
        'session'  : tdf['session'].cat.categories.tolist(),
        'prior_q'  : (tdf['prior_q'].cat.categories.tolist()
                      if hasattr(tdf['prior_q'], 'cat')
                      else sorted(tdf['prior_q'].unique().tolist())),
    }

    results = []

    def _record(label, sub):
        if len(sub) < ALPHA_MIN_N:
            return
        wr  = (sub['outcome'] == 'win').mean()
        ev  = sub['r_multiple'].mean()
        p   = _binom_pval(sub)
        results.append({
            'filter'  : label,
            'n'       : len(sub),
            'win_rate': wr,
            'ev'      : ev,
            'pval'    : p,
            'wr_lift' : wr - base_wr,
            'ev_lift' : ev - base_ev,
        })

    # Single filters
    for col, vals in filter_cols.items():
        for v in vals:
            _record(f"{col}={v}", tdf[tdf[col].astype(str) == str(v)])

    # Two-way combinations
    cols = list(filter_cols.keys())
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            for v1, v2 in iproduct(filter_cols[c1], filter_cols[c2]):
                mask = (tdf[c1].astype(str) == str(v1)) & \
                       (tdf[c2].astype(str) == str(v2))
                _record(f"{c1}={v1} & {c2}={v2}", tdf[mask])

    if not results:
        print("  No filter cells with sufficient trades found.")
        return pd.DataFrame()

    adf = pd.DataFrame(results)
    # Bonferroni multiple-comparison correction
    adf['pval_adj'] = (adf['pval'] * len(adf)).clip(upper=1.0)
    adf = adf.sort_values('ev_lift', ascending=False).reset_index(drop=True)

    # Top positive alpha
    print(f"  {'Filter':<48}  {'N':>5}  {'WR':>7}  {'EV(R)':>8}  {'EV_lift':>8}  {'p_adj':>8}")
    print(f"  {'─'*48}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
    for _, row in adf[adf['ev_lift'] > 0].head(15).iterrows():
        sig = ' **' if row['pval_adj'] < 0.01 else (' * ' if row['pval_adj'] < 0.05 else '   ')
        print(f"  {row['filter']:<48}  {row['n']:5d}  {row['win_rate']:7.2%}  "
              f"{row['ev']:+8.4f}  {row['ev_lift']:+8.4f}  {row['pval_adj']:8.5f}{sig}")

    print(f"\n  Negative alpha — AVOID these conditions:")
    print(f"  {'─'*48}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
    for _, row in adf[adf['ev_lift'] < 0].tail(10).iterrows():
        sig = ' **' if row['pval_adj'] < 0.01 else (' * ' if row['pval_adj'] < 0.05 else '   ')
        print(f"  {row['filter']:<48}  {row['n']:5d}  {row['win_rate']:7.2%}  "
              f"{row['ev']:+8.4f}  {row['ev_lift']:+8.4f}  {row['pval_adj']:8.5f}{sig}")

    return adf


# ──────────────────────────────────────────────────────────────────────────────
#  8. VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────

WIN_C  = '#3fb950'
LOSS_C = '#f85149'
BLUE_C = '#388bfd'
TEXT_C = '#e6edf3'
GRID_C = '#21262d'
BG_AX  = '#161b22'
BG_FIG = '#0d1117'


def _style(ax, title=''):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_C)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.grid(True, color=GRID_C, alpha=0.5, linewidth=0.4)
    if title:
        ax.set_title(title, color=TEXT_C, fontsize=9, fontweight='bold', pad=5)


def make_charts(df: pd.DataFrame,
                tdf: pd.DataFrame,
                alpha_df: pd.DataFrame,
                out_path: str = OUTPUT_PNG):

    fig = plt.figure(figsize=(24, 30), facecolor=BG_FIG)
    fig.suptitle(
        f'SuperTrend Hypothesis Test — XAUUSD\n'
        f'ATR={ST_ATR_LEN}  Factor={ST_FACTOR}  |  '
        f'Trades={len(tdf)}  WR={(tdf["outcome"]=="win").mean():.1%}  '
        f'EV={tdf["r_multiple"].mean():+.4f}R',
        color=TEXT_C, fontsize=14, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.50, wspace=0.35,
        top=0.96, bottom=0.04, left=0.06, right=0.97
    )

    # ── Row 0: R distribution, bootstrap WR, equity curve ──────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    _style(ax0, 'R-Multiple Distribution')
    w_r = tdf.loc[tdf['outcome'] == 'win',  'r_multiple']
    l_r = tdf.loc[tdf['outcome'] == 'loss', 'r_multiple']
    ax0.hist(l_r, bins=40, color=LOSS_C, alpha=0.75, density=True, label='Loss')
    ax0.hist(w_r, bins=40, color=WIN_C,  alpha=0.75, density=True, label='Win')
    ax0.axvline(0, color='white', lw=1, ls='--')
    ax0.axvline(tdf['r_multiple'].mean(), color='yellow', lw=1.5,
                label=f"Mean={tdf['r_multiple'].mean():+.3f}R")
    ax0.legend(fontsize=7, facecolor=BG_AX, labelcolor=TEXT_C)
    ax0.set_xlabel('R-Multiple')

    ax1 = fig.add_subplot(gs[0, 1])
    _style(ax1, 'Win Rate — Bootstrap Distribution (10k)')
    wr_overall = (tdf['outcome'] == 'win').mean()
    rng   = np.random.default_rng(42)
    boot  = np.array([(rng.choice(tdf['outcome'].values,
                                  size=len(tdf), replace=True) == 'win').mean()
                      for _ in range(10_000)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    ax1.hist(boot, bins=60, color=BLUE_C, alpha=0.8, density=True)
    ax1.axvline(wr_overall, color='yellow', lw=2, label=f"WR={wr_overall:.2%}")
    ax1.axvline(0.50, color='white', lw=1, ls='--', label='50% ref')
    ax1.axvline(ci_lo, color=LOSS_C, lw=1, ls=':')
    ax1.axvline(ci_hi, color=WIN_C,  lw=1, ls=':',
                label=f"CI [{ci_lo:.2%}, {ci_hi:.2%}]")
    ax1.legend(fontsize=7, facecolor=BG_AX, labelcolor=TEXT_C)
    ax1.set_xlabel('Win Rate (bootstrap)')

    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2, 'Cumulative R — Equity Curve')
    cum = tdf['r_multiple'].cumsum().values
    ax2.plot(cum, color=BLUE_C, lw=1.2)
    ax2.fill_between(range(len(cum)), cum, alpha=0.12, color=BLUE_C)
    ax2.axhline(0, color='white', lw=0.8, ls='--')
    # Draw drawdown shading
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    ax2.fill_between(range(len(dd)), dd + running_max, running_max,
                     alpha=0.25, color=LOSS_C, label='Drawdown')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('Cumulative R')
    ax2.legend(fontsize=7, facecolor=BG_AX, labelcolor=TEXT_C)

    # ── Row 1: MFE/MAE, hour heatmap, direction comparison ─────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3, 'MFE vs MAE per Trade (R)')
    c_arr = [WIN_C if o == 'win' else LOSS_C for o in tdf['outcome']]
    ax3.scatter(tdf['mae'], tdf['mfe'], c=c_arr, alpha=0.35, s=12)
    ax3.set_xlabel('MAE (R) — adverse')
    ax3.set_ylabel('MFE (R) — favorable')
    ax3.legend(handles=[
        mpatches.Patch(color=WIN_C,  label='Win'),
        mpatches.Patch(color=LOSS_C, label='Loss')
    ], fontsize=7, facecolor=BG_AX, labelcolor=TEXT_C)

    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4, 'Win Rate by Hour of Day')
    h_wr  = (tdf.groupby('entry_hour')
               .apply(lambda x: (x['outcome'] == 'win').mean()
                      if len(x) >= ALPHA_MIN_N else np.nan)
               .dropna())
    bclrs = [WIN_C if v >= wr_overall else LOSS_C for v in h_wr.values]
    ax4.bar(h_wr.index, h_wr.values, color=bclrs, alpha=0.8)
    ax4.axhline(wr_overall, color='yellow', lw=1.2, ls='--',
                label=f'Baseline {wr_overall:.1%}')
    ax4.axhline(0.5, color='white', lw=0.7, ls=':', alpha=0.5)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Hour (server time)')
    ax4.set_ylabel('Win Rate')
    ax4.legend(fontsize=7, facecolor=BG_AX, labelcolor=TEXT_C)

    ax5 = fig.add_subplot(gs[1, 2])
    _style(ax5, 'Long vs Short — EV & Win Rate')
    dirs_data = []
    for d in ('long', 'short'):
        sub = tdf[tdf['direction'] == d]
        if len(sub) < MIN_TRADES:
            continue
        dirs_data.append({
            'label': d.capitalize(),
            'wr': (sub['outcome'] == 'win').mean(),
            'ev': sub['r_multiple'].mean(),
            'n' : len(sub)
        })
    if dirs_data:
        lbls = [x['label'] for x in dirs_data]
        wrs  = [x['wr']    for x in dirs_data]
        evs  = [x['ev']    for x in dirs_data]
        x    = np.arange(len(lbls))
        ax5b = ax5.twinx()
        ax5b.set_facecolor(BG_AX)
        ax5b.tick_params(colors=TEXT_C, labelsize=8)
        ax5.bar(x - 0.2, wrs, 0.35, color=BLUE_C,  alpha=0.8, label='Win Rate')
        ax5b.bar(x + 0.2, evs, 0.35, color='#d29922', alpha=0.8, label='EV(R)')
        ax5.axhline(0.5, color='white', lw=0.7, ls=':')
        ax5.set_xticks(x); ax5.set_xticklabels(lbls, color=TEXT_C)
        ax5.set_ylabel('Win Rate', color=TEXT_C)
        ax5b.set_ylabel('EV (R)', color=TEXT_C)
        ax5.set_ylim(0, 1)
        ax5.legend(fontsize=7, facecolor=BG_AX, labelcolor=TEXT_C, loc='upper left')
        ax5b.legend(fontsize=7, facecolor=BG_AX, labelcolor=TEXT_C, loc='upper right')

    # ── Row 2: ATR quartile, prior trend, alpha lift ────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    _style(ax6, 'Win Rate by Volatility (ATR Quartile)')
    atr_sub = (tdf.groupby('atr_q', observed=True)
                  .apply(lambda x: (x['outcome'] == 'win').mean()))
    bclrs6 = [WIN_C if v >= wr_overall else LOSS_C for v in atr_sub.values]
    ax6.bar(range(len(atr_sub)), atr_sub.values, color=bclrs6, alpha=0.8)
    ax6.set_xticks(range(len(atr_sub)))
    ax6.set_xticklabels(atr_sub.index.astype(str), rotation=25, fontsize=7)
    ax6.axhline(wr_overall, color='yellow', lw=1, ls='--')
    ax6.axhline(0.5, color='white', lw=0.7, ls=':', alpha=0.5)
    ax6.set_ylim(0, 1); ax6.set_ylabel('Win Rate')

    ax7 = fig.add_subplot(gs[2, 1])
    _style(ax7, 'Win Rate by Prior Trend Duration')
    cats = (tdf['prior_q'].cat.categories.tolist()
            if hasattr(tdf['prior_q'], 'cat')
            else sorted(tdf['prior_q'].unique().tolist()))
    pwr_vals, plbls = [], []
    for q in cats:
        sub = tdf[tdf['prior_q'] == q]
        if len(sub) < ALPHA_MIN_N: continue
        pwr_vals.append((sub['outcome'] == 'win').mean())
        plbls.append(str(q))
    bclrs7 = [WIN_C if v >= wr_overall else LOSS_C for v in pwr_vals]
    ax7.bar(range(len(plbls)), pwr_vals, color=bclrs7, alpha=0.8)
    ax7.set_xticks(range(len(plbls)))
    ax7.set_xticklabels(plbls, rotation=20, fontsize=7)
    ax7.axhline(wr_overall, color='yellow', lw=1, ls='--')
    ax7.axhline(0.5, color='white', lw=0.7, ls=':', alpha=0.5)
    ax7.set_ylim(0, 1); ax7.set_ylabel('Win Rate')

    ax8 = fig.add_subplot(gs[2, 2])
    _style(ax8, 'Top Alpha Filters — EV Lift vs Baseline')
    if not alpha_df.empty:
        top_pos = alpha_df[alpha_df['ev_lift'] > 0].head(10)
        top_neg = alpha_df[alpha_df['ev_lift'] < 0].tail(5)
        combined = pd.concat([top_neg, top_pos]).sort_values('ev_lift')
        clrs = [WIN_C if v > 0 else LOSS_C for v in combined['ev_lift']]
        ax8.barh(range(len(combined)), combined['ev_lift'], color=clrs, alpha=0.8)
        ax8.set_yticks(range(len(combined)))
        ax8.set_yticklabels(combined['filter'].str[:32], fontsize=6)
        ax8.axvline(0, color='white', lw=1)
        ax8.set_xlabel('EV Lift (R)')

    # ── Row 3: ST chart (last 300 bars) ─────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, :])
    _style(ax9, 'SuperTrend Chart — Last 300 Bars  (▲ Long entry  ▼ Short entry | Green=Win  Red=Loss)')

    tail_df = df.tail(300).reset_index(drop=True)
    plot_start_orig = df.tail(300).index[0]

    # Candlestick bars (wick + body)
    for i, row in tail_df.iterrows():
        body_c = WIN_C if row['close'] >= row['open'] else LOSS_C
        ax9.vlines(i, row['low'], row['high'],     color=body_c, lw=0.6, alpha=0.45)
        ax9.vlines(i, min(row['open'], row['close']),
                      max(row['open'], row['close']), color=body_c, lw=2.8, alpha=0.75)

    # SuperTrend dots
    bull_m = tail_df['st_direction'] == -1
    bear_m = tail_df['st_direction'] ==  1
    ax9.scatter(tail_df.index[bull_m], tail_df.loc[bull_m, 'st_value'],
                color=WIN_C,  s=5, zorder=3, label='ST Bull')
    ax9.scatter(tail_df.index[bear_m], tail_df.loc[bear_m, 'st_value'],
                color=LOSS_C, s=5, zorder=3, label='ST Bear')

    # Trade entry markers
    recent = tdf[tdf['entry_idx'] >= plot_start_orig]
    for _, t in recent.iterrows():
        pos = int(t['entry_idx'] - plot_start_orig)
        if 0 <= pos < len(tail_df):
            mkr  = '^' if t['direction'] == 'long' else 'v'
            clr  = WIN_C if t['outcome'] == 'win' else LOSS_C
            ax9.scatter(pos, t['entry_price'], marker=mkr, color=clr,
                        s=90, zorder=5, edgecolors='white', linewidths=0.6)

    ax9.set_xlabel('Bars (most recent 300)')
    ax9.set_ylabel('Price')
    ax9.legend(fontsize=7, facecolor=BG_AX, labelcolor=TEXT_C)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG_FIG)
    plt.close()
    print(f"\n  Chart saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
#  9. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  SUPERTREND HYPOTHESIS TESTER")
    print("  Probability of Riding Trend vs Hitting SL")
    print("=" * 68)

    # 1. Load
    print_header("DATA LOADING")
    df = load_csv(CSV_PATH)

    # 2. SuperTrend
    print_header("COMPUTING SUPERTREND  (RMA ATR — exact EA match)")
    df    = calc_supertrend(df, ST_ATR_LEN, ST_FACTOR)
    n_flips = df['st_flipped'].sum()
    bull_bars = (df['st_direction'] == -1).sum()
    bear_bars = (df['st_direction'] ==  1).sum()
    print(f"  Direction flips  : {n_flips}")
    print(f"  Bullish bars     : {bull_bars} ({bull_bars/len(df):.1%})")
    print(f"  Bearish bars     : {bear_bars} ({bear_bars/len(df):.1%})")

    # 3. Simulate
    print_header("SIMULATING TRADES")
    tdf = simulate_trades(df)
    print(f"  Completed trades : {len(tdf)}")
    print(f"  Long / Short     : {(tdf['direction']=='long').sum()} / "
          f"{(tdf['direction']=='short').sum()}")

    if len(tdf) < MIN_TRADES:
        print(f"\n  ERROR: Only {len(tdf)} trades — need {MIN_TRADES} minimum.")
        sys.exit(1)

    # 4. Core stats
    print_header("CORE STATISTICAL ANALYSIS")
    stats_all = core_stats(tdf, "ALL TRADES")

    # 5. Breakdowns
    directional_breakdown(tdf, stats_all)
    session_breakdown(tdf, stats_all['win_rate'])
    volatility_breakdown(tdf, stats_all['win_rate'])
    prior_trend_breakdown(tdf)
    dow_breakdown(tdf)

    # 6. HMM
    df_hmm, tdf = run_hmm(df, tdf, HMM_STATES)

    # 7. Alpha
    alpha_df = find_alpha(tdf)

    # 8. Charts
    print_header("GENERATING VISUAL REPORT")
    chart_df = df_hmm if df_hmm is not None else df
    make_charts(chart_df, tdf, alpha_df, OUTPUT_PNG)

    # 9. Final verdict
    print_header("FINAL VERDICT")
    wr  = stats_all['win_rate']
    ev  = stats_all['ev_r']
    pay = stats_all['payoff']
    bp  = stats_all['binom_pval']

    print(f"""
  Config  :  ST ATR={ST_ATR_LEN}  Factor={ST_FACTOR}  |  {len(tdf)} trades
  ────────────────────────────────────────────────────────────────────
  Win Rate       :  {wr:.2%}   95% CI [{stats_all['ci_lo']:.2%} – {stats_all['ci_hi']:.2%}]
  Payoff Ratio   :  {pay:.3f}x
  Expected Value : {ev:+.5f}R per trade
  Sharpe (R)     :  {stats_all['sharpe_r']:.3f}
  ────────────────────────────────────────────────────────────────────
  HYPOTHESIS VERDICT:""")

    if bp < 0.05 and ev > 0:
        print(f"""
  ✓ EDGE DETECTED — Positive and statistically significant
    Win rate {wr:.2%} beats 50/50 coin flip (p = {bp:.5f})
    Expected value {ev:+.5f}R means the strategy has a positive carry.
    SuperTrend flip entries RIDE the trend more often than they hit SL.
    → The base signal has alpha. Now use the filter table above to amplify it.""")
    elif ev > 0 and bp >= 0.05:
        print(f"""
  ~ MARGINAL EDGE — Positive EV but not yet statistically significant
    Win rate {wr:.2%} (p = {bp:.5f}) — need more data or tighter filters.
    → Check alpha discovery table for specific conditions that ARE significant.""")
    elif ev <= 0:
        print(f"""
  ✗ NO RAW EDGE — Negative or zero expected value in unfiltered signal
    Win rate {wr:.2%}, EV = {ev:+.5f}R
    → The raw ST flip is not enough alone. Study alpha filters above —
      specific session/volatility/direction combos may still show edge.""")

    if not alpha_df.empty:
        best = alpha_df.loc[alpha_df['ev_lift'].idxmax()]
        worst = alpha_df.loc[alpha_df['ev_lift'].idxmin()]
        print(f"""
  BEST ALPHA FILTER  (highest EV lift):
    Filter   :  {best['filter']}
    N trades :  {best['n']}
    Win Rate :  {best['win_rate']:.2%}
    EV       : {best['ev']:+.5f}R   (lift: {best['ev_lift']:+.5f}R vs baseline)
    p_adj    :  {best['pval_adj']:.5f}  {'✓ sig' if best['pval_adj'] < 0.05 else '✗ not sig'}

  WORST FILTER  (avoid in EA logic):
    Filter   :  {worst['filter']}
    N trades :  {worst['n']}
    Win Rate :  {worst['win_rate']:.2%}
    EV       : {worst['ev']:+.5f}R   (lift: {worst['ev_lift']:+.5f}R vs baseline)""")

    print(f"\n  Output PNG → {OUTPUT_PNG}")
    print("=" * 68)


if __name__ == "__main__":
    main()
