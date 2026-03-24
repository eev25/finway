# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "plotly",
#   "numpy",
#   "scipy",
# ]
# ///

import glob
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objs as go
from scipy import stats

# Known recurring vendors and their expected payment interval in days.
# WSSC is quarterly (~every 3 months); all others are monthly.
RECURRING_VENDORS: dict[str, int] = {
    "WASHINGTON GAS":       30,
    "PEPCO PAYMENTUS":      30,
    "PENNYMAC":             30,
    "VERIZON*RECURRING PAY": 30,
    "WSSC CONSUMER":        90,
}

# (date, description, amount | None, running_balance)
Transaction = tuple[date, str, "float | None", float]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_date(raw: str) -> date:
    return date(int(raw[6:]), int(raw[:2]), int(raw[3:5]))


def parse_statements(
    data_dir: str = "data",
) -> tuple[list[tuple[date, float]], list[Transaction]]:
    """
    Parse all statement .txt files.

    Returns
    -------
    records      : sorted (date, running_balance) — last balance per day
    transactions : every individual transaction row, sorted by date
    """
    files = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}/")

    # Rows that have BOTH an amount column AND a running-balance column
    two_num = re.compile(
        r"^(\d{2}/\d{2}/\d{4})\s+"
        r"(.+?)\s+"
        r"(-?[\d,]+\.\d{2})\s+"
        r"([\d,]+\.\d{2})\s*$"
    )
    # "Beginning balance" rows — only one number (the running balance)
    one_num = re.compile(
        r"^(\d{2}/\d{2}/\d{4})\s+"
        r"(.+?)\s+"
        r"([\d,]+\.\d{2})\s*$"
    )

    records: dict[date, float] = {}
    all_transactions: list[Transaction] = []
    seen: set[tuple] = set()

    for filepath in files:
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        in_transactions = False

        for line in lines:
            line = line.rstrip()
            if re.match(r"^Date\s+Description", line):
                in_transactions = True
                continue
            if not in_transactions or not line.strip():
                continue

            m2 = two_num.match(line)
            if m2:
                tx_date    = _parse_date(m2.group(1))
                desc       = m2.group(2).strip()
                amount     = float(m2.group(3).replace(",", ""))
                running    = float(m2.group(4).replace(",", ""))
                key = (tx_date, desc)
                if key not in seen:
                    seen.add(key)
                    all_transactions.append((tx_date, desc, amount, running))
                    records[tx_date] = running
                continue

            m1 = one_num.match(line)
            if m1:
                tx_date = _parse_date(m1.group(1))
                desc    = m1.group(2).strip()
                running = float(m1.group(3).replace(",", ""))
                key = (tx_date, desc)
                if key not in seen:
                    seen.add(key)
                    all_transactions.append((tx_date, desc, None, running))
                    records.setdefault(tx_date, running)

    if not records:
        raise ValueError("No transactions parsed from statement files.")

    sorted_records = sorted(records.items())
    sorted_txns    = sorted(all_transactions, key=lambda t: t[0])
    return sorted_records, sorted_txns


# ---------------------------------------------------------------------------
# Recurring payment analysis
# ---------------------------------------------------------------------------

def _matches_vendor(desc: str, vendor_key: str) -> bool:
    return vendor_key in desc.upper()


def extract_recurring(transactions: list[Transaction]) -> dict[str, dict]:
    """
    For each known recurring vendor, collect historical occurrences.

    Returns a dict keyed by vendor_key with:
        occurrences   : [(date, amount)]
        avg_amount    : mean debit (negative float)
        std_amount    : std-dev of debit amounts
        last_date     : date of most-recent occurrence
        interval_days : expected days between payments
    """
    result: dict[str, dict] = {}
    for vendor_key, interval in RECURRING_VENDORS.items():
        occurrences = [
            (tx_date, amount)
            for tx_date, desc, amount, _ in transactions
            if amount is not None and _matches_vendor(desc, vendor_key)
        ]

        if not occurrences:
            print(f"  [recurring] No history for: {vendor_key}  (will be skipped)")
            continue

        amounts = [amt for _, amt in occurrences]
        result[vendor_key] = {
            "occurrences":   occurrences,
            "avg_amount":    float(np.mean(amounts)),
            "std_amount":    float(np.std(amounts)) if len(amounts) > 1 else 0.0,
            "last_date":     max(d for d, _ in occurrences),
            "interval_days": interval,
        }

    return result


# ---------------------------------------------------------------------------
# Forward projection
# ---------------------------------------------------------------------------

def _monthly_recurring_cost(recurring_info: dict[str, dict]) -> float:
    """Annualised monthly cost of all tracked recurring bills (negative number)."""
    return sum(
        info["avg_amount"] * (30.44 / info["interval_days"])
        for info in recurring_info.values()
    )


def compute_forward_projection(
    records:        list[tuple[date, float]],
    transactions:   list[Transaction],
    recurring_info: dict[str, dict],
    max_days_ahead: int = 730,
) -> dict:
    """
    Project balance forward from the last known date using:
      - A 'non-recurring daily drift' estimated from historical data
        (total change minus the effect of known recurring payments)
      - Explicitly scheduled future recurring debits at their expected dates

    Uncertainty (error bands) is estimated from the standard deviation of
    historical residuals around the expected trajectory, propagated forward
    as σ · √(days_ahead).
    """
    dates, balances = zip(*records)
    first_date, last_date = dates[0], dates[-1]
    first_balance, last_balance = balances[0], balances[-1]
    total_days = max((last_date - first_date).days, 1)

    # ---- Non-recurring daily drift ----------------------------------------
    recurring_historical_total = sum(
        amount
        for _, desc, amount, _ in transactions
        if amount is not None
        for vk in RECURRING_VENDORS
        if _matches_vendor(desc, vk)
        # count each tx only once even if description matches multiple keys
    )
    # De-duplicate: a single transaction may match at most one vendor key
    counted: set[int] = set()
    recurring_historical_total = 0.0
    for i, (_, desc, amount, _) in enumerate(transactions):
        if amount is None:
            continue
        for vk in RECURRING_VENDORS:
            if _matches_vendor(desc, vk):
                if i not in counted:
                    counted.add(i)
                    recurring_historical_total += amount
                break

    total_change   = last_balance - first_balance
    non_rec_total  = total_change - recurring_historical_total
    daily_drift    = non_rec_total / total_days

    # ---- Historical residuals for uncertainty estimation ------------------
    # Expected balance at each record = anchor + drift*days + all recurring up to that date
    residuals: list[float] = []
    for rec_date, rec_bal in records:
        days_elapsed = (rec_date - first_date).days
        expected = first_balance + daily_drift * days_elapsed
        for i, (tx_date, desc, amount, _) in enumerate(transactions):
            if amount is None or tx_date > rec_date or i not in counted:
                continue
            # Only add recurring debits that have occurred by this date
            for vk in RECURRING_VENDORS:
                if _matches_vendor(desc, vk):
                    expected += amount
                    break
        residuals.append(rec_bal - expected)

    n_res = len(residuals)
    sigma = float(np.std(residuals)) if n_res > 2 else 0.0
    t_crit = stats.t.ppf(0.975, df=max(n_res - 2, 1))

    # ---- Schedule future payments ------------------------------------------
    end_date = last_date + timedelta(days=max_days_ahead)
    payment_by_date: dict[date, float] = {}
    future_payment_list: list[tuple[date, str, float]] = []

    for vk, info in recurring_info.items():
        nxt = info["last_date"] + timedelta(days=info["interval_days"])
        while nxt <= end_date:
            if nxt > last_date:
                payment_by_date[nxt] = payment_by_date.get(nxt, 0.0) + info["avg_amount"]
                future_payment_list.append((nxt, vk, info["avg_amount"]))
            nxt += timedelta(days=info["interval_days"])

    # ---- Day-by-day simulation --------------------------------------------
    proj_dates: list[date]  = [last_date]
    proj_bals:  list[float] = [last_balance]
    zero_date: date | None  = None
    horizon = max_days_ahead

    current = last_balance
    for offset in range(1, horizon + 1):
        sim_date = last_date + timedelta(days=offset)
        current += daily_drift
        if sim_date in payment_by_date:
            current += payment_by_date[sim_date]

        proj_dates.append(sim_date)
        proj_bals.append(current)

        if current <= 0 and zero_date is None:
            zero_date = sim_date
            horizon = min(offset + 45, max_days_ahead)  # extend slightly past zero

    y_hat  = np.array(proj_bals)
    t_days = np.array([(d - last_date).days for d in proj_dates], dtype=float)
    spread = t_crit * sigma * np.sqrt(t_days)

    return {
        "plot_dates":    proj_dates,
        "y_hat":         y_hat,
        "y_upper":       y_hat + spread,
        "y_lower":       y_hat - spread,
        "zero_date":     zero_date,
        "daily_drift":   daily_drift,
        "monthly_recurring_cost": _monthly_recurring_cost(recurring_info),
        "future_payments": sorted(future_payment_list),
        "sigma":         sigma,
    }


# ---------------------------------------------------------------------------
# Naive linear regression (kept for comparison)
# ---------------------------------------------------------------------------

def compute_regression(
    records:        list[tuple[date, float]],
    max_days_ahead: int = 730,
) -> dict:
    """Simple OLS regression on all balance observations."""
    dates, balances = zip(*records)
    origin = dates[0]

    x = np.array([(d - origin).days for d in dates], dtype=float)
    y = np.array(balances, dtype=float)

    slope, intercept, r_value, _, stderr = stats.linregress(x, y)

    zero_day  = (-intercept / slope) if slope < 0 else None
    zero_date = (origin + timedelta(days=int(zero_day))) if zero_day else None

    end_day = min(int(zero_day * 1.05) if zero_day else max_days_ahead, max_days_ahead)
    x_plot  = np.linspace(0, end_day, 300)
    y_hat   = intercept + slope * x_plot

    n      = len(x)
    x_mean = x.mean()
    sxx    = np.sum((x - x_mean) ** 2)
    t_crit = stats.t.ppf(0.975, df=n - 2)
    se     = stderr * np.sqrt(1 + 1 / n + (x_plot - x_mean) ** 2 / sxx)

    return {
        "plot_dates": [origin + timedelta(days=float(d)) for d in x_plot],
        "y_hat":      y_hat,
        "y_upper":    y_hat + t_crit * se,
        "y_lower":    y_hat - t_crit * se,
        "slope":      slope,
        "r_squared":  r_value ** 2,
        "zero_date":  zero_date,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(
    records:    list[tuple[date, float]],
    fwd:        dict,
    reg:        dict,
) -> go.Figure:
    actual_dates, actual_balances = zip(*records)
    traces = []

    # --- Naive regression band (faint, background) -------------------------
    rd = reg["plot_dates"]
    traces.append(go.Scatter(
        x=rd + rd[::-1],
        y=list(reg["y_upper"]) + list(reg["y_lower"][::-1]),
        fill="toself",
        fillcolor="rgba(150,150,150,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    traces.append(go.Scatter(
        x=reg["plot_dates"],
        y=reg["y_hat"],
        mode="lines",
        line=dict(color="rgba(160,160,160,0.7)", width=1.5, dash="dot"),
        name=f"Naïve linear trend (R²={reg['r_squared']:.2f})",
    ))

    # --- Recurring-adjusted projection band --------------------------------
    pd = fwd["plot_dates"]
    traces.append(go.Scatter(
        x=pd + pd[::-1],
        y=list(fwd["y_upper"]) + list(fwd["y_lower"][::-1]),
        fill="toself",
        fillcolor="rgba(255, 127, 14, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Prediction Interval (recurring-adjusted)",
    ))
    traces.append(go.Scatter(
        x=fwd["plot_dates"],
        y=fwd["y_hat"],
        mode="lines",
        line=dict(color="rgba(255, 127, 14, 0.9)", width=2, dash="dash"),
        name="Recurring-adjusted projection",
    ))

    # --- Actual balance ----------------------------------------------------
    traces.append(go.Scatter(
        x=list(actual_dates),
        y=list(actual_balances),
        mode="lines+markers",
        line=dict(color="rgb(31, 119, 180)", width=2),
        marker=dict(size=7),
        name="Running Balance",
    ))

    # --- Zero reference line ----------------------------------------------
    all_dates = list(actual_dates) + fwd["plot_dates"]
    traces.append(go.Scatter(
        x=[min(all_dates), max(all_dates)],
        y=[0, 0],
        mode="lines",
        line=dict(color="rgba(200,0,0,0.45)", width=1, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # --- Annotations -------------------------------------------------------
    annotations = []
    if fwd["zero_date"]:
        annotations.append(dict(
            x=fwd["zero_date"], y=0,
            xref="x", yref="y",
            text=f"Projected $0: {fwd['zero_date'].strftime('%b %d, %Y')}",
            showarrow=True, arrowhead=2, ax=0, ay=-44,
            font=dict(color="rgb(200,0,0)", size=13),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(200,0,0,0.4)",
        ))

    # Monthly recurring cost annotation
    monthly_rec = fwd["monthly_recurring_cost"]
    annotations.append(dict(
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        text=(
            f"Tracked recurring bills: <b>${abs(monthly_rec):,.0f}/mo</b>  |  "
            f"Non-recurring drift: <b>${fwd['daily_drift'] * 30.44:+,.0f}/mo</b>"
        ),
        showarrow=False,
        font=dict(size=11, color="rgba(80,80,80,1)"),
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(180,180,180,0.5)",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text="Bank Account Burn Rate", font=dict(size=22)),
        xaxis_title="Date",
        yaxis_title="Balance (USD)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        hovermode="x unified",
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    records, transactions = parse_statements("data")
    first_date, last_date = records[0][0], records[-1][0]
    print(f"Parsed {len(records)} balance snapshots  ({first_date} – {last_date})")
    print(f"Parsed {len(transactions)} transactions total\n")

    print("Recurring vendor history:")
    recurring_info = extract_recurring(transactions)
    for vk, info in recurring_info.items():
        n = len(info["occurrences"])
        print(
            f"  {vk:<28}  n={n}  "
            f"avg=${info['avg_amount']:+,.2f}  "
            f"last={info['last_date']}  "
            f"interval={info['interval_days']}d"
        )

    print()
    fwd = compute_forward_projection(records, transactions, recurring_info)
    reg = compute_regression(records)

    monthly_rec  = fwd["monthly_recurring_cost"]
    monthly_dft  = fwd["daily_drift"] * 30.44
    monthly_net  = monthly_dft + monthly_rec

    print(f"Non-recurring drift:   ${monthly_dft:+,.0f}/mo  (${fwd['daily_drift']:+.2f}/day)")
    print(f"Recurring bills:       ${abs(monthly_rec):,.0f}/mo")
    print(f"Net projected burn:    ${monthly_net:+,.0f}/mo")
    print(f"Uncertainty (σ):       ${fwd['sigma']:,.0f} per √day")
    if fwd["zero_date"]:
        print(f"\nProjected $0:          {fwd['zero_date'].strftime('%B %d, %Y')}")
    else:
        print("\nProjected $0:          not reached within projection window")

    fig = build_figure(records, fwd, reg)
    fig.write_html("burn_rate.html")
    print("\nSaved: burn_rate.html")
    fig.show()


if __name__ == "__main__":
    main()
