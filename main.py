import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import math
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="FX Arb — Direct Rate Controls (Bellman-Ford)", page_icon="💱", layout="wide")

# -----------------------------
# Core configuration
# -----------------------------

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD']
USD_LEGS = ['EURUSD=X','GBPUSD=X','USDJPY=X','AUDUSD=X']

FALLBACK_USD = {
    'EURUSD=X': 1.0800,
    'GBPUSD=X': 1.2600,
    'USDJPY=X': 150.0000,
    'AUDUSD=X': 0.6500,
}

ANGLE = np.linspace(0, 2*np.pi, len(CURRENCIES), endpoint=False)
POS = {ccy: (float(np.cos(a)), float(np.sin(a))) for ccy, a in zip(CURRENCIES, ANGLE)}

# -----------------------------
# Data helpers
# -----------------------------

@st.cache_data(ttl=60)
def fetch_usd_legs_cached():
    rates = {}
    try:
        t = yf.Tickers(' '.join(USD_LEGS))
        for p in USD_LEGS:
            try:
                h = t.tickers[p].history(period='1d', interval='1m')
                if not h.empty and 'Close' in h:
                    rates[p] = float(h['Close'].dropna().iloc[-1])
            except Exception:
                pass
    except Exception:
        pass
    
    for p in USD_LEGS:
        if p not in rates:
            rates[p] = FALLBACK_USD[p]
    return rates

def matrix_from_usd_legs(usd_legs):
    eu = usd_legs['EURUSD=X']  # USD per 1 EUR
    gu = usd_legs['GBPUSD=X']  # USD per 1 GBP
    au = usd_legs['AUDUSD=X']  # USD per 1 AUD
    uj = usd_legs['USDJPY=X']  # JPY per 1 USD
    
    base = {}
    base['EURUSD=X'] = eu
    base['GBPUSD=X'] = gu
    base['AUDUSD=X'] = au
    base['USDJPY=X'] = uj
    base['EURGBP=X'] = eu / gu
    base['EURAUD=X'] = eu / au
    base['EURJPY=X'] = eu * uj
    base['GBPAUD=X'] = gu / au
    base['GBPJPY=X'] = gu * uj
    base['AUDJPY=X'] = au * uj
    
    inv = {}
    for k, v in base.items():
        s = k.replace('=X','')
        a, b = s[:3], s[3:]
        inv[f'{b}{a}=X'] = 1.0 / v
    
    base.update(inv)
    return base

# -----------------------------
# Graph & arbitrage (Bellman-Ford ONLY)
# -----------------------------

def build_graph_from_rates(rates):
    graph = {c1: {c2: (0.0 if c1==c2 else float('inf')) for c2 in CURRENCIES} for c1 in CURRENCIES}
    
    for pair, r in rates.items():
        # Extract currency codes from pair (e.g., EURUSD=X -> EUR, USD)
        pair_clean = pair.replace('=X', '')
        if len(pair_clean) == 6:  # Should be exactly 6 characters for currency pairs
            a, b = pair_clean[:3], pair_clean[3:]
            if a in CURRENCIES and b in CURRENCIES and r > 0:
                graph[a][b] = -math.log(r)
    return graph

def bellman_ford_cycles(graph, min_cycle_len=3):
    currencies = list(CURRENCIES)
    cycles = []
    
    # Try each currency as a starting point to find all possible cycles
    for start_currency in currencies:
        # Initialize distances and predecessors
        dist = {c: float('inf') for c in currencies}
        pred = {c: None for c in currencies}
        dist[start_currency] = 0.0
        
        edges = []
        for u in currencies:
            for v in currencies:
                w = graph[u][v]
                if u != v and w != float('inf'):
                    edges.append((u, v, w))
        
        # Relax edges V-1 times
        for iteration in range(len(currencies) - 1):
            for u, v, w in edges:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u
        
        # Check for negative cycles and extract them
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                # Found negative cycle, extract it
                cycle_node = v
                visited = set()
                
                # Follow predecessors to get into the cycle
                while cycle_node not in visited:
                    visited.add(cycle_node)
                    cycle_node = pred[cycle_node]
                    if cycle_node is None:
                        break
                
                if cycle_node is not None:
                    # Extract the actual cycle
                    cycle = []
                    current = cycle_node
                    while True:
                        cycle.append(current)
                        current = pred[current]
                        if current == cycle_node or current is None:
                            break
                    
                    if len(cycle) >= min_cycle_len:
                        # Normalize cycle
                        min_idx = cycle.index(min(cycle))
                        normalized_cycle = cycle[min_idx:] + cycle[:min_idx]
                        cycles.append(normalized_cycle)
    
    # Remove duplicates
    unique_cycles = []
    seen = set()
    for cycle in cycles:
        # Create both forward and reverse representations
        forward = tuple(cycle)
        reverse = tuple([cycle[0]] + list(reversed(cycle[1:])))
        if forward not in seen and reverse not in seen:
            seen.add(forward)
            seen.add(reverse)
            unique_cycles.append(cycle)
    
    return unique_cycles

def cycle_profit(cycle, rates):
    prod = 1.0
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i+1) % len(cycle)]
        key, rkey = f"{a}{b}=X", f"{b}{a}=X"
        if key in rates and rates[key] > 0:
            r = rates[key]
        elif rkey in rates and rates[rkey] > 0:
            r = 1.0 / rates[rkey]
        else:
            return 0.0
        prod *= r
    return max(0.0, prod - 1.0)

def render_graph(rates, highlight_cycle=None):
    fig = go.Figure()
    
    xs = [POS[c][0] for c in CURRENCIES]
    ys = [POS[c][1] for c in CURRENCIES]
    
    # Nodes
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=CURRENCIES,
        textposition="top center",
        marker=dict(size=24),
        hoverinfo="text",
        showlegend=False
    ))
    
    # All faint edges (no labels)
    for a in CURRENCIES:
        for b in CURRENCIES:
            if a == b: continue
            x0, y0 = POS[a]; x1, y1 = POS[b]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.25
            ))
    
    # Highlighted cycle with numbers on each segment
    if highlight_cycle:
        cyc = highlight_cycle + [highlight_cycle[0]]
        for i in range(len(cyc)-1):
            a, b = cyc[i], cyc[i+1]
            x0, y0 = POS[a]; x1, y1 = POS[b]
            
            # thick edge
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(width=4),
                hoverinfo="skip",
                showlegend=False
            ))
            
            # arrow head
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2
            )
            
            # step number at the midpoint
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_annotation(
                x=mx, y=my,
                text=str(i+1),
                showarrow=False,
                font=dict(size=14, color="white"),
                bgcolor="black",
                opacity=0.85
            )
    
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=520)
    return fig

# -----------------------------
# Session init & sidebar
# -----------------------------

st.sidebar.header("⚙️ Data source")
LIVE_YF = st.sidebar.toggle("Pull live USD legs from Yahoo", value=False)
st.sidebar.caption("If off, you can change currency manually (avoids SSL issues).")

if "usd_legs" not in st.session_state:
    st.session_state.usd_legs = fetch_usd_legs_cached() if LIVE_YF else FALLBACK_USD.copy()

if "pair_overrides" not in st.session_state:
    st.session_state.pair_overrides = {}  # {'EURJPY=X': 163.2, ...}

# Auto-reset when switching to live data
if "prev_live_yf" not in st.session_state:
    st.session_state.prev_live_yf = LIVE_YF

if st.session_state.prev_live_yf != LIVE_YF and LIVE_YF:
    # User just switched to live data - reset to clean state
    st.session_state.pair_overrides.clear()
    st.session_state.usd_legs = fetch_usd_legs_cached()
    st.sidebar.success("✅ Switched to live data - overrides cleared!")

st.session_state.prev_live_yf = LIVE_YF

# --- ONE-CLICK DEMO (SINGLE SECTION) ---
st.sidebar.markdown("---")
st.sidebar.header("🎯 One-click demo arbitrage")

# Show active overrides
if st.session_state.pair_overrides:
    st.sidebar.warning(f"⚠️ {len(st.session_state.pair_overrides)} override(s) active:")
    for pair, val in st.session_state.pair_overrides.items():
        st.sidebar.write(f"• {pair}: {val}")

# Reset button
if st.sidebar.button("🔄 Reset to clean state", type="secondary"):
    st.session_state.pair_overrides.clear()
    # 🔁 reload the USD legs based on the toggle
    st.session_state.usd_legs = fetch_usd_legs_cached() if LIVE_YF else FALLBACK_USD.copy()
    st.sidebar.success("✅ Baselines reloaded & overrides cleared!")
    st.rerun()

st.sidebar.markdown("---")

# Function to adjust currency strength - ONLY adjust USD pairs to create imbalances
def adjust_currency_strength(currency, factor):
    """
    Adjust the strength of a currency by modifying only its USD pair.
    This creates imbalances in cross rates, leading to arbitrage opportunities.
    factor > 1.0 makes currency stronger, factor < 1.0 makes it weaker
    """
    # Only adjust the major USD pair for this currency
    usd_pair = None
    if currency == 'EUR':
        usd_pair = 'EURUSD=X'
    elif currency == 'GBP':
        usd_pair = 'GBPUSD=X'
    elif currency == 'JPY':
        usd_pair = 'USDJPY=X'
    elif currency == 'AUD':
        usd_pair = 'AUDUSD=X'
    
    if usd_pair:
        # Get current rate (either override or base)
        if usd_pair in st.session_state.pair_overrides:
            current_rate = st.session_state.pair_overrides[usd_pair]
        else:
            current_rate = st.session_state.usd_legs[usd_pair]
        
        pair_clean = usd_pair.replace('=X', '')
        first_ccy = pair_clean[:3]
        
        if first_ccy == currency:
            # Currency is base (EUR, GBP, AUD), strengthen = higher rate
            new_rate = current_rate * factor
        else:
            # Currency is quote (JPY), strengthen = lower rate
            new_rate = current_rate / factor
        
        st.session_state.pair_overrides[usd_pair] = round(new_rate, 6)

# Currency strength adjustment buttons
st.sidebar.subheader("💪 Currency Strength")
currencies_to_adjust = ['EUR', 'GBP', 'JPY', 'AUD']

for currency in currencies_to_adjust:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button(f"💪 {currency} stronger", key=f"strong_{currency}"):
            adjust_currency_strength(currency, 1.01)  # 1% stronger
            affected_pairs = len([p for p in st.session_state.pair_overrides.keys() if currency in p])
            st.sidebar.success(f"✅ {currency} strengthened! ({affected_pairs} pairs affected)")
    with col2:
        if st.button(f"📉 {currency} weaker", key=f"weak_{currency}"):
            adjust_currency_strength(currency, 0.99)  # 1% weaker
            affected_pairs = len([p for p in st.session_state.pair_overrides.keys() if currency in p])
            st.sidebar.success(f"✅ {currency} weakened! ({affected_pairs} pairs affected)")

# -----------------------------
# Build rates (apply overrides) - FIXED to preserve imbalances
# -----------------------------

base_rates = matrix_from_usd_legs(st.session_state.usd_legs)
adj_rates = base_rates.copy()

# Apply ONLY explicit pair overrides - don't recalculate cross-rates
for pair, val in st.session_state.pair_overrides.items():
    if val > 0:
        pair_clean = pair.replace('=X', '')
        if len(pair_clean) == 6:
            a, b = pair_clean[:3], pair_clean[3:]
            inv_pair = f"{b}{a}=X"
            adj_rates[pair] = float(val)
            adj_rates[inv_pair] = 1.0 / float(val)

# DON'T recalculate cross-rates - this is the key fix!
# The cross-rates will remain at their original calculated values
# while the USD pairs get adjusted, creating arbitrage opportunities

# -----------------------------
# UI layout
# -----------------------------

st.title("💱 FX Arbitrage")
st.caption("Bellman-Ford algorithm. Cycles must have length ≥ 3. Override any cross to create arbitrage deterministically.")

left, right = st.columns([3.0, 1.25])

graph_slot = left.empty()

with right:
    st.subheader("🔍 Scan")
    min_profit = st.number_input("Min Profit (%)", 0.0000, 10.0, 0.0010, 0.0001, format="%.4f")
    scan = st.button("🚀 Scan for Arbitrage", type="primary")
    


# -----------------------------
# Scan (Bellman-Ford only)
# -----------------------------

highlight_cycle = None
opps = []

if scan:
    with st.spinner("Scanning for arbitrage opportunities..."):
        graph = build_graph_from_rates(adj_rates)
        
        # Debug: Show graph construction
        st.write("**Debug: Graph edges (showing -log rates):**")
        debug_edges = []
        for a in CURRENCIES:
            for b in CURRENCIES:
                if a != b and graph[a][b] != float('inf'):
                    debug_edges.append(f"{a}→{b}: {graph[a][b]:.6f}")
        st.write(", ".join(debug_edges[:10]) + "..." if len(debug_edges) > 10 else ", ".join(debug_edges))
        
        cycles = bellman_ford_cycles(graph, min_cycle_len=3)
        
        # Check ALL possible triangles (comprehensive)
        st.write("**Debug: ALL possible triangles:**")
        all_triangles = []
        for i, c1 in enumerate(CURRENCIES):
            for j, c2 in enumerate(CURRENCIES):
                if i >= j: continue
                for k, c3 in enumerate(CURRENCIES):
                    if k <= j: continue
                    # Create triangle in both directions
                    triangle1 = [c1, c2, c3]
                    triangle2 = [c1, c3, c2]
                    all_triangles.extend([triangle1, triangle2])
        
        profitable_triangles = []
        for triangle in all_triangles:
            profit = cycle_profit(triangle, adj_rates) * 100.0
            if abs(profit) > 0.01:  # Show any triangle with >0.01% profit/loss
                profitable_triangles.append((triangle, profit))
        
        # Sort by absolute profit
        profitable_triangles.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Show top 10
        for triangle, profit in profitable_triangles[:10]:
            path = " → ".join(triangle + [triangle[0]])
            st.write(f"• **{path}**: {profit:+.4f}%")
            if abs(profit) >= min_profit and triangle not in cycles:
                cycles.append(triangle)
        
        for cyc in cycles:
            p = cycle_profit(cyc, adj_rates) * 100.0
            if p >= min_profit:
                opps.append({
                    "cycle": cyc,
                    "path": " → ".join(cyc + [cyc[0]]),
                    "profit_pct": p,
                    "length": len(cyc)
                })
        
        opps.sort(key=lambda x: x["profit_pct"], reverse=True)
        
        if opps:
            highlight_cycle = opps[0]["cycle"]

# -----------------------------
# Render & results
# -----------------------------

fig = render_graph(adj_rates, highlight_cycle=highlight_cycle)
graph_slot.plotly_chart(fig, use_container_width=True)

if scan:
    if opps:
        right.success(f"🎉 Found {len(opps)} opportunity(s)!")
        
        with right.expander("Best Opportunity Details", expanded=True):
            best = opps[0]
            st.write(f"**Profit:** {best['profit_pct']:.4f}%")
            st.write(f"**Path:** {best['path']}")
            
            # Show detailed calculation
            cyc = best['cycle'] + [best['cycle'][0]]
            prod = 1.0
            for j in range(len(cyc)-1):
                a, b = cyc[j], cyc[j+1]
                key, rkey = f"{a}{b}=X", f"{b}{a}=X"
                r = adj_rates.get(key) or (1.0 / adj_rates[rkey] if adj_rates.get(rkey, 0) > 0 else None)
                st.write(f"{j+1}. {a} → {b}" + (f" @ {r:.6f}" if r else " (no rate)"))
                if r: prod *= r
            st.write(f"**Final Product:** {prod:.8f}")
        
        # Show all opportunities if more than one
        if len(opps) > 1:
            with right.expander(f"All {len(opps)} Opportunities"):
                for i, opp in enumerate(opps):
                    st.write(f"{i+1}. {opp['path']} → {opp['profit_pct']:.4f}%")
    else:
        right.warning("No arbitrage opportunities found")
        right.info("💡 Try using the one-click demo buttons or add manual overrides to create arbitrage.")

st.markdown("---")

st.subheader("📊 Current vs baseline rates")

pairs_to_show = [
    'EURUSD=X','GBPUSD=X','AUDUSD=X','USDJPY=X',
    'EURGBP=X','EURAUD=X','EURJPY=X','GBPAUD=X','GBPJPY=X','AUDJPY=X'
]

rows = []
for k in pairs_to_show:
    base = base_rates[k]
    adj = adj_rates[k]
    dev_pct = (adj / base - 1.0) * 100.0 if base != 0 else 0.0
    s = k.replace('=X','')
    is_override = k in st.session_state.pair_overrides
    
    rows.append({
        "Pair": f"{s[:3]}/{s[3:]}",
        "Baseline": f"{base:.6f}",
        "Current": f"{adj:.6f}",
        "Δ%": f"{dev_pct:+.4f}%",
        "Override": "✅" if is_override else ""
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)