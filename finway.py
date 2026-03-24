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
import subprocess
from datetime import date, timedelta
from math import sqrt
from pathlib import Path

import numpy as np
import plotly.graph_objs as go
from scipy import stats

# Strips ACH metadata fields (DES:, ID:, INDN:, CO ID:, SEC code)
_ACH_SUFFIX = re.compile(r'\s+DES:.*$', re.IGNORECASE)
# Strips inline MM/DD date patterns in non-ACH descriptions (e.g. "01/07")
_INLINE_DATE = re.compile(r'\s+\d{2}/\d{2}\b')

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


def normalize_payee(description: str) -> str:
    """Return a stable, uppercase payee key from a raw transaction description.

    ACH transactions (contain ' DES:'): take everything before that token.
    Non-ACH transactions: strip inline MM/DD date patterns.
    """
    desc = description.strip()
    if re.search(r'\bDES:', desc, re.IGNORECASE):
        desc = _ACH_SUFFIX.sub('', desc)
    else:
        desc = _INLINE_DATE.sub('', desc)
    return desc.upper().strip()


def detect_recurring_payees(
    transactions: list[tuple[date, str, float, float]],
    min_months: int = 2,
) -> dict[str, list[float]]:
    """Auto-detect recurring debit payees from transaction history.

    A payee is considered recurring if it appears as a debit in at least
    `min_months` distinct calendar months. Returns a dict mapping normalized
    payee key -> list of observed amounts (as positive values).
    """
    month_occurrences: dict[str, set[tuple[int, int]]] = {}
    all_amounts: dict[str, list[float]] = {}

    for tx_date, description, amount, _balance in transactions:
        if amount >= 0:
            continue
        key = normalize_payee(description)
        month_occurrences.setdefault(key, set()).add((tx_date.year, tx_date.month))
        all_amounts.setdefault(key, []).append(abs(amount))

    return {
        key: amounts
        for key, amounts in all_amounts.items()
        if len(month_occurrences[key]) >= min_months
    }


def detect_recurring_credits(
    transactions: list[tuple[date, str, float, float]],
    min_months: int = 2,
) -> dict[str, list[float]]:
    """Auto-detect recurring credit sources from transaction history.

    A source is considered recurring if it appears as a credit in at least
    `min_months` distinct calendar months. Returns a dict mapping normalized
    source key -> list of observed amounts.
    """
    month_occurrences: dict[str, set[tuple[int, int]]] = {}
    all_amounts: dict[str, list[float]] = {}

    for tx_date, description, amount, _balance in transactions:
        if amount <= 0:
            continue
        key = normalize_payee(description)
        month_occurrences.setdefault(key, set()).add((tx_date.year, tx_date.month))
        all_amounts.setdefault(key, []).append(amount)

    return {
        key: amounts
        for key, amounts in all_amounts.items()
        if len(month_occurrences[key]) >= min_months
    }


def compute_burn_projection(
    records: list[tuple[date, float]],
    recurring: dict[str, list[float]],
    credits: dict[str, list[float]] | None = None,
    max_days_ahead: int = 730,
) -> dict:
    """Project balance change driven by net recurring cash flows.

    Incorporates both recurring debits and (optionally) recurring credits to
    compute a net burn rate. Uncertainty bands widen proportionally to
    sqrt(months elapsed), reflecting compounding variance across monthly payments.
    If recurring income exceeds debits, the balance never hits zero.
    """
    last_date, last_balance = records[-1]

    # Per-payee debit statistics
    means = {p: statistics.mean(amounts) for p, amounts in recurring.items()}
    variances = {
        p: statistics.variance(amounts) if len(amounts) > 1 else 0.0
        for p, amounts in recurring.items()
    }

    monthly_burn = sum(means.values())          # total guaranteed monthly outflow ($)
    monthly_var = sum(variances.values())        # debit variance

    # Recurring income offsets outflows
    monthly_income = sum(
        statistics.mean(amts) for amts in (credits or {}).values()
    )
    credit_var = sum(
        statistics.variance(amts) if len(amts) > 1 else 0.0
        for amts in (credits or {}).values()
    )

    net_monthly_burn = monthly_burn - monthly_income
    monthly_var += credit_var
    monthly_std = sqrt(monthly_var)

    daily_burn = net_monthly_burn / 30.44       # net per-day rate

    if net_monthly_burn > 0:
        zero_day = last_balance / daily_burn
        zero_date = last_date + timedelta(days=int(zero_day))
        end_day = min(int(zero_day * 1.1), max_days_ahead)
    else:
        zero_date = None
        end_day = max_days_ahead

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
        "monthly_income": monthly_income,
        "monthly_std": monthly_std,
        "zero_date": zero_date,
        "plot_dates": plot_dates,
        "y_hat": y_hat,
        "y_upper": y_upper,
        "y_lower": y_lower,
        "recurring_breakdown": means,
    }


def _build_recurring_table_html(
    data: dict[str, list[float]],
    title: str,
    col_label: str,
    amount_color: str,
    amount_prefix: str,
) -> str:
    rows = []
    for name, amounts in data.items():
        avg = statistics.mean(amounts)
        rows.append(
            f"<tr style='border-bottom:1px solid #f0f0f0;'>"
            f"<td style='padding:8px 16px;'>{name.title()}</td>"
            f"<td style='padding:8px 16px;text-align:right;font-family:monospace;"
            f"color:{amount_color};'>{amount_prefix}${avg:,.2f}</td>"
            f"<td style='padding:8px 16px;color:#888;'>Monthly</td>"
            f"</tr>"
        )
    return (
        "<div style='max-width:900px;margin:24px auto;font-family:sans-serif;'>"
        f"<h3 style='margin-bottom:8px;font-size:15px;color:#444;'>{title}</h3>"
        "<table style='border-collapse:collapse;width:100%;font-size:14px;'>"
        "<thead><tr style='border-bottom:2px solid #ddd;color:#666;"
        "text-transform:uppercase;font-size:11px;letter-spacing:.05em;'>"
        f"<th style='padding:8px 16px;text-align:left;font-weight:600;'>{col_label}</th>"
        "<th style='padding:8px 16px;text-align:right;font-weight:600;'>Avg / Month</th>"
        "<th style='padding:8px 16px;text-align:left;font-weight:600;'>Interval</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table></div>"
    )


def build_payee_table_html(recurring: dict[str, list[float]]) -> str:
    return _build_recurring_table_html(recurring, "Detected Recurring Debits", "Payee", "#dc143c", "")


def build_credits_table_html(credits: dict[str, list[float]]) -> str:
    return _build_recurring_table_html(credits, "Detected Recurring Credits", "Source", "#2a7a3b", "+")


def build_page_html(
    chart_fragment: str,
    recurring: dict[str, list[float]],
    credits: dict[str, list[float]],
) -> str:
    """Assemble a complete HTML page with chart on the left, tables stacked on the right."""
    debits_html = build_payee_table_html(recurring)
    right_cells = f"<div class='table-col top'>{debits_html}</div>"
    if credits:
        right_cells += f"<div class='table-col bottom'>{build_credits_table_html(credits)}</div>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Bank Account Burn Rate</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 0; background: white; }}
  .layout {{
    display: grid;
    grid-template-columns: 2fr 1fr;
    grid-template-rows: 1fr 1fr;
    height: 100vh;
  }}
  .chart-col {{
    grid-row: span 2;
    overflow: hidden;
    padding: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .chart-col > * {{
    width: 100%;
    height: auto;
    aspect-ratio: 16 / 9;
  }}
  .chart-col .plotly-graph-div {{ height: 100% !important; }}
  .table-col {{
    overflow-y: auto;
    padding: 12px 16px;
    border-left: 1px solid #e8e8e8;
  }}
  .table-col.top {{ border-bottom: 1px solid #e8e8e8; }}
</style>
</head>
<body>
<div class="layout">
  <div class="chart-col">{chart_fragment}</div>
  {right_cells}
</div>
</body>
</html>"""


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
            name=f"Projected Burn",
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
    recurring = detect_recurring_payees(transactions)
    credits = detect_recurring_credits(transactions)

    print("Recurring debits detected:")
    for payee, amounts in recurring.items():
        avg = statistics.mean(amounts)
        n = len(amounts)
        obs = ", ".join(f"${a:.2f}" for a in amounts)
        print(f"  {payee:<40}  avg ${avg:>8,.2f}/mo  ({n} obs: {obs})")

    print("\nRecurring credits detected:")
    for source, amounts in credits.items():
        avg = statistics.mean(amounts)
        n = len(amounts)
        obs = ", ".join(f"${a:.2f}" for a in amounts)
        print(f"  {source:<40}  avg ${avg:>8,.2f}/mo  ({n} obs: {obs})")

    if not recurring:
        print("ERROR: No recurring debits found — cannot project burn rate.")
        return

    proj = compute_burn_projection(records, recurring, credits)
    print(f"\nGross monthly burn:       ${proj['monthly_burn']:,.2f}")
    print(f"Recurring income:        +${proj['monthly_income']:,.2f}")
    print(f"Net monthly burn:         ${proj['monthly_burn'] - proj['monthly_income']:,.2f}")
    print(f"Monthly std dev:          ±${proj['monthly_std']:,.2f}")
    if proj["zero_date"]:
        print(f"Projected $0:             {proj['zero_date'].strftime('%B %d, %Y')}")

    fig = build_figure(records, proj)
    chart_html = Path("burn_rate.html")
    chart_fragment = fig.to_html(full_html=False, include_plotlyjs=True)
    chart_html.write_text(build_page_html(chart_fragment, recurring, credits), encoding="utf-8")
    print("\nSaved: burn_rate.html")
    subprocess.run(["open", chart_html])


if __name__ == "__main__":
    main()
