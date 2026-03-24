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
import statistics
from datetime import date, timedelta
from math import sqrt
from pathlib import Path

import numpy as np
import plotly.graph_objs as go
from scipy import stats

RECURRING_PAYEES = [
    "WASHINGTON GAS",
    "PEPCO PAYMENTUS",
    "PENNYMAC",
    "VERIZON*RECURRING PAY",
]

# Matches rows with both an Amount AND a Running Balance column
_TX_FULL = re.compile(
    r"^(\d{2}/\d{2}/\d{4})\s+"      # date
    r"(.+?)\s+"                       # description (non-greedy)
    r"(-?[\d,]+\.\d{2})\s+"          # amount (may be negative)
    r"([\d,]+\.\d{2})\s*$"           # running balance
)

# Matches rows with only a Running Balance (e.g. "Beginning balance" rows)
_TX_BAL_ONLY = re.compile(
    r"^(\d{2}/\d{2}/\d{4})\s+"
    r"(.+?)\s+"
    r"([\d,]+\.\d{2})\s*$"
)


def _parse_date(raw: str) -> date:
    return date(int(raw[6:]), int(raw[:2]), int(raw[3:5]))


def _parse_amount(raw: str) -> float:
    return float(raw.replace(",", ""))


def parse_statements(data_dir: str = "data") -> list[tuple[date, float]]:
    """Return sorted (date, running_balance) pairs — one per day (last tx wins)."""
    files = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}/")

    records: dict[date, float] = {}

    for filepath in files:
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        in_tx = False
        for line in text.splitlines():
            line = line.rstrip()
            if re.match(r"^Date\s+Description", line):
                in_tx = True
                continue
            if not in_tx or not line.strip():
                continue

            m_full = _TX_FULL.match(line)
            m_bal = _TX_BAL_ONLY.match(line)

            if m_full:
                tx_date = _parse_date(m_full.group(1))
                balance = _parse_amount(m_full.group(4))
                records[tx_date] = balance
            elif m_bal:
                tx_date = _parse_date(m_bal.group(1))
                balance = _parse_amount(m_bal.group(3))
                records.setdefault(tx_date, balance)  # only if not already set

    if not records:
        raise ValueError("No transactions parsed from statement files.")

    return sorted(records.items())


def parse_transactions(data_dir: str = "data") -> list[tuple[date, str, float, float]]:
    """Return all transactions that have an explicit Amount column.

    Returns list of (date, description, amount, running_balance).
    Amount is signed (negative for debits).
    """
    files = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    seen: set[tuple[date, str, float]] = set()
    results: list[tuple[date, str, float, float]] = []

    for filepath in files:
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        in_tx = False
        for line in text.splitlines():
            line = line.rstrip()
            if re.match(r"^Date\s+Description", line):
                in_tx = True
                continue
            if not in_tx or not line.strip():
                continue

            m = _TX_FULL.match(line)
            if not m:
                continue

            tx_date = _parse_date(m.group(1))
            description = m.group(2).strip()
            amount = _parse_amount(m.group(3))
            balance = _parse_amount(m.group(4))

            key = (tx_date, description[:30], amount)
            if key in seen:
                continue
            seen.add(key)
            results.append((tx_date, description, amount, balance))

    results.sort(key=lambda r: r[0])
    return results


def extract_recurring_debits(
    transactions: list[tuple[date, str, float, float]],
) -> dict[str, list[float]]:
    """Group observed amounts (as positive values) for each known recurring payee."""
    grouped: dict[str, list[float]] = {p: [] for p in RECURRING_PAYEES}

    for tx_date, description, amount, _balance in transactions:
        desc_upper = description.upper()
        for payee in RECURRING_PAYEES:
            if payee in desc_upper and amount < 0:
                grouped[payee].append(abs(amount))

    return {p: amounts for p, amounts in grouped.items() if amounts}


def compute_burn_projection(
    records: list[tuple[date, float]],
    recurring: dict[str, list[float]],
    max_days_ahead: int = 730,
) -> dict:
    """Project balance decline driven purely by guaranteed recurring debits.

    The slope is always negative. Uncertainty bands widen proportionally to
    sqrt(months elapsed), reflecting compounding variance across monthly payments.
    """
    last_date, last_balance = records[-1]

    # Per-payee statistics
    means = {p: statistics.mean(amounts) for p, amounts in recurring.items()}
    variances = {
        p: statistics.variance(amounts) if len(amounts) > 1 else 0.0
        for p, amounts in recurring.items()
    }

    monthly_burn = sum(means.values())          # total guaranteed monthly outflow ($)
    monthly_var = sum(variances.values())        # total monthly variance
    monthly_std = sqrt(monthly_var)

    daily_burn = monthly_burn / 30.44            # convert to per-day rate

    # Days until balance hits zero on the central projection
    zero_day = last_balance / daily_burn
    zero_date = last_date + timedelta(days=int(zero_day))

    end_day = min(int(zero_day * 1.1), max_days_ahead)
    x_days = np.linspace(0, end_day, 400)       # days from last_date

    # Central projection
    y_hat = last_balance - daily_burn * x_days

    # Widening uncertainty: after t days = t/30.44 months,
    # accumulated std = sqrt(months) * monthly_std
    months_elapsed = x_days / 30.44
    std_t = np.sqrt(months_elapsed) * monthly_std
    y_upper = y_hat + 1.96 * std_t
    y_lower = y_hat - 1.96 * std_t

    plot_dates = [last_date + timedelta(days=float(d)) for d in x_days]

    return {
        "slope": -daily_burn,
        "monthly_burn": monthly_burn,
        "monthly_std": monthly_std,
        "zero_date": zero_date,
        "plot_dates": plot_dates,
        "y_hat": y_hat,
        "y_upper": y_upper,
        "y_lower": y_lower,
        "recurring_breakdown": means,
    }


def build_figure(
    records: list[tuple[date, float]],
    proj: dict,
) -> go.Figure:
    actual_dates, actual_balances = zip(*records)
    pd_list = proj["plot_dates"]

    traces = []

    # Error band — drawn first (behind everything else)
    traces.append(
        go.Scatter(
            x=pd_list + pd_list[::-1],
            y=list(proj["y_upper"]) + list(proj["y_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Uncertainty Band",
        )
    )

    # Projection line
    monthly_burn = proj["monthly_burn"]
    traces.append(
        go.Scatter(
            x=pd_list,
            y=proj["y_hat"],
            mode="lines",
            line=dict(color="rgba(255, 127, 14, 0.9)", width=2, dash="dash"),
            name=f"Projected Burn (−${monthly_burn:,.0f}/mo guaranteed)",
        )
    )

    # Actual historical balance
    traces.append(
        go.Scatter(
            x=list(actual_dates),
            y=list(actual_balances),
            mode="lines+markers",
            line=dict(color="rgb(31, 119, 180)", width=2),
            marker=dict(size=7),
            name="Running Balance",
        )
    )

    # Zero reference line
    all_dates = list(actual_dates) + pd_list
    traces.append(
        go.Scatter(
            x=[min(all_dates), max(all_dates)],
            y=[0, 0],
            mode="lines",
            line=dict(color="rgba(200, 0, 0, 0.5)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    annotations = []
    zero_date = proj.get("zero_date")
    if zero_date:
        annotations.append(
            dict(
                x=zero_date,
                y=0,
                xref="x",
                yref="y",
                text=f"Projected zero: {zero_date.strftime('%b %d, %Y')}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                font=dict(color="rgb(200, 0, 0)", size=13),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(200,0,0,0.4)",
            )
        )

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


def main():
    records = parse_statements("data")
    print(f"Parsed {len(records)} balance snapshots: {records[0][0]} – {records[-1][0]}")
    print(f"Last known balance: ${records[-1][1]:,.2f} on {records[-1][0]}\n")

    transactions = parse_transactions("data")
    recurring = extract_recurring_debits(transactions)

    print("Recurring debits found:")
    for payee in RECURRING_PAYEES:
        if payee in recurring:
            amounts = recurring[payee]
            avg = statistics.mean(amounts)
            n = len(amounts)
            obs = ", ".join(f"${a:.2f}" for a in amounts)
            print(f"  {payee:<26}  avg ${avg:>8,.2f}/mo  ({n} obs: {obs})")
        else:
            print(f"  {payee:<26}  (no observations found)")

    if not recurring:
        print("ERROR: No recurring debits found — cannot project burn rate.")
        return

    proj = compute_burn_projection(records, recurring)
    print(f"\nGuaranteed monthly burn:  ${proj['monthly_burn']:,.2f}")
    print(f"Monthly std dev:          ±${proj['monthly_std']:,.2f}")
    if proj["zero_date"]:
        print(f"Projected $0:             {proj['zero_date'].strftime('%B %d, %Y')}")

    fig = build_figure(records, proj)
    fig.write_html("burn_rate.html")
    print("\nSaved: burn_rate.html")
    fig.show()


if __name__ == "__main__":
    main()
