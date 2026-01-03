import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Retirement Monte Carlo Simulator", layout="wide")

st.title("🏖️ Retirement Portfolio Monte Carlo Simulator")
st.markdown("Simulate thousands of retirement scenarios to estimate portfolio success probability")

# Sidebar for inputs
st.sidebar.header("Simulation Settings")

# Basic Parameters
st.sidebar.subheader("📊 Basic Parameters")
years = st.sidebar.number_input("Retirement Duration (years)", min_value=1, max_value=60, value=47)
simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=50000, value=10000, step=1000)
initial_portfolio = st.sidebar.number_input("Initial Portfolio ($)", min_value=0, max_value=20000000, value=3900000, step=100000)

# Asset Allocation
st.sidebar.subheader("📈 Asset Allocation")
equity_pct = st.sidebar.slider("Equities (%)", min_value=0, max_value=100, value=60)
bond_pct = st.sidebar.slider("Bonds (%)", min_value=0, max_value=100, value=35)
cash_pct = st.sidebar.slider("Cash (%)", min_value=0, max_value=100, value=5)

total_allocation = equity_pct + bond_pct + cash_pct
if total_allocation != 100:
    st.sidebar.error(f"⚠️ Allocation must sum to 100% (currently {total_allocation}%)")

asset_mix = {
    "equities": equity_pct / 100,
    "bonds": bond_pct / 100,
    "cash": cash_pct / 100
}

# Returns
st.sidebar.subheader("💰 Expected Returns (Annual)")
equity_return = st.sidebar.slider("Equity Return", min_value=0.0, max_value=0.20, value=0.065, step=0.005, format="%.3f")
equity_vol = st.sidebar.slider("Equity Volatility", min_value=0.0, max_value=0.30, value=0.16, step=0.01, format="%.2f")
bond_return = st.sidebar.slider("Bond Return", min_value=0.0, max_value=0.15, value=0.03, step=0.005, format="%.3f")
bond_vol = st.sidebar.slider("Bond Volatility", min_value=0.0, max_value=0.15, value=0.06, step=0.01, format="%.2f")
cash_return = st.sidebar.slider("Cash Return", min_value=0.0, max_value=0.10, value=0.01, step=0.005, format="%.3f")
inflation_rate = st.sidebar.slider("Inflation Rate", min_value=0.0, max_value=0.10, value=0.02, step=0.005, format="%.3f")

# Spending Glidepath
st.sidebar.subheader("💳 Spending Glidepath (Monthly, Today's $)")
go_go_spend = st.sidebar.number_input("Go-Go Years (Monthly)", min_value=0, max_value=50000, value=11100, step=100)
go_go_years = st.sidebar.number_input("Go-Go Duration (years)", min_value=0, max_value=50, value=10)

slow_go_spend = st.sidebar.number_input("Slow-Go Years (Monthly)", min_value=0, max_value=50000, value=9100, step=100)
slow_go_years = st.sidebar.number_input("Slow-Go Duration (years)", min_value=0, max_value=50, value=15)

no_go_spend = st.sidebar.number_input("No-Go Years (Monthly)", min_value=0, max_value=50000, value=8350, step=100)

# Mortgage
st.sidebar.subheader("🏠 Mortgage")
has_mortgage = st.sidebar.checkbox("Include Mortgage", value=True)
if has_mortgage:
    mortgage_balance = st.sidebar.number_input("Mortgage Balance ($)", min_value=0, max_value=5000000, value=363400, step=10000)
    mortgage_rate = st.sidebar.slider("Mortgage Rate", min_value=0.0, max_value=0.15, value=0.06, step=0.0025, format="%.4f")
    mortgage_term_years = st.sidebar.number_input("Mortgage Term (years)", min_value=1, max_value=40, value=30)
    
    if mortgage_balance > 0:
        mortgage_payment = mortgage_balance * (mortgage_rate / 12) / (
            1 - (1 + mortgage_rate / 12) ** (-mortgage_term_years * 12)
        )
        annual_mortgage = mortgage_payment * 12
        st.sidebar.info(f"Monthly Payment: ${mortgage_payment:,.0f}")
    else:
        annual_mortgage = 0
else:
    mortgage_balance = 0
    annual_mortgage = 0
    mortgage_term_years = 0

# Social Security
st.sidebar.subheader("👴 Social Security")
has_ss = st.sidebar.checkbox("Include Social Security", value=True)
if has_ss:
    ss_monthly = st.sidebar.number_input("SS Monthly Benefit (Today's $)", min_value=0, max_value=10000, value=5088, step=100)
    ss_start_year = st.sidebar.number_input("SS Start Year", min_value=1, max_value=60, value=17)
else:
    ss_monthly = 0
    ss_start_year = 999  # Never starts

# Temporary Income
st.sidebar.subheader("💼 Temporary Earned Income")
has_temp_income = st.sidebar.checkbox("Include Temporary Income", value=True)
if has_temp_income:
    temp_income_annual = st.sidebar.number_input("Annual Income ($, nominal)", min_value=0, max_value=1000000, value=120000, step=10000)
    temp_income_years = st.sidebar.number_input("Income Duration (years)", min_value=0, max_value=20, value=3)
else:
    temp_income_annual = 0
    temp_income_years = 0

# Market Drawdown Rule
st.sidebar.subheader("📉 Rebalancing Rules")
drawdown_trigger = st.sidebar.slider("Drawdown Trigger (no rebalance if exceeded)", min_value=0.0, max_value=0.50, value=0.15, step=0.05, format="%.2f")

# Run button
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Main content area
if run_simulation and total_allocation == 100:
    
    with st.spinner(f"Running {simulations:,} simulations... This may take a moment..."):
        
        np.random.seed(42)
        
        final_portfolios = []
        success_flags = []
        failure_years_list = []
        
        years_go = []
        years_slow = []
        years_no = []
        
        for sim in range(simulations):
            
            # Initialize portfolio buckets
            equities = initial_portfolio * asset_mix["equities"]
            bonds = initial_portfolio * asset_mix["bonds"]
            cash = initial_portfolio * asset_mix["cash"]
            
            market_index = 1.0
            market_high = 1.0
            
            y_go = y_slow = y_no = 0
            success = True
            
            for year in range(years):
                
                # Returns
                equity_r = np.random.normal(equity_return, equity_vol)
                bond_r = np.random.normal(bond_return, bond_vol)
                cash_r = cash_return
                
                equities *= (1 + equity_r)
                bonds *= (1 + bond_r)
                cash *= (1 + cash_r)
                
                market_index *= (1 + equity_r)
                market_high = max(market_high, market_index)
                
                # Spending Phase
                if year < go_go_years:
                    monthly_spend = go_go_spend
                    y_go += 1
                elif year < (go_go_years + slow_go_years):
                    monthly_spend = slow_go_spend
                    y_slow += 1
                else:
                    monthly_spend = no_go_spend
                    y_no += 1
                
                annual_spend = monthly_spend * 12 * ((1 + inflation_rate) ** year)
                
                # Mortgage
                if year < mortgage_term_years:
                    annual_spend += annual_mortgage
                
                # Social Security
                if year >= ss_start_year:
                    ss_annual = ss_monthly * 12 * ((1 + inflation_rate) ** (year - ss_start_year))
                else:
                    ss_annual = 0
                
                net_spend = max(annual_spend - ss_annual, 0)
                
                # Temporary earned income
                if year < temp_income_years:
                    net_spend = max(net_spend - temp_income_annual, 0)
                
                # Withdrawal: proportional across all buckets
                total = equities + bonds + cash
                if net_spend >= total:
                    success = False
                    failure_years_list.append(year + 1)
                    equities = bonds = cash = 0
                    break
                
                prop_eq = equities / total
                prop_bd = bonds / total
                prop_cash = cash / total
                
                equities -= net_spend * prop_eq
                bonds -= net_spend * prop_bd
                cash -= net_spend * prop_cash
                
                # Annual rebalance if market is above drawdown threshold
                drawdown = (market_high - market_index) / market_high
                if drawdown < drawdown_trigger:
                    total = equities + bonds + cash
                    equities = total * asset_mix['equities']
                    bonds = total * asset_mix['bonds']
                    cash = total * asset_mix['cash']
                
                if equities + bonds + cash <= 0:
                    success = False
                    failure_years_list.append(year + 1)
                    break
            
            final_portfolios.append(equities + bonds + cash)
            success_flags.append(success)
            years_go.append(y_go)
            years_slow.append(y_slow)
            years_no.append(y_no)
    
    # Calculate results
    final_portfolios = np.array(final_portfolios)
    success_rate = np.mean(success_flags)
    
    # Display Results
    st.header("📊 Results")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Success Rate", f"{success_rate*100:.1f}%")
    with col2:
        st.metric("Failed Simulations", f"{simulations - sum(success_flags):,}")
    with col3:
        median_final = np.median(final_portfolios)
        st.metric("Median Final Portfolio", f"${median_final:,.0f}")
    with col4:
        if failure_years_list:
            avg_failure_year = np.mean(failure_years_list)
            st.metric("Avg Failure Year", f"{avg_failure_year:.1f}")
        else:
            st.metric("Avg Failure Year", "N/A")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio Distribution", "⏱️ Survival Curve", "📉 Failure Analysis", "📋 Detailed Stats"])
    
    with tab1:
        st.subheader("Final Portfolio Distribution")
        
        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=final_portfolios,
            nbinsx=50,
            name="Final Portfolio Value",
            marker_color='lightblue'
        ))
        fig.update_layout(
            xaxis_title="Final Portfolio Value ($)",
            yaxis_title="Number of Simulations",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Percentiles table
        st.subheader("Portfolio Value Percentiles")
        percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        percentile_values = np.percentile(final_portfolios, percentiles)
        
        df_percentiles = pd.DataFrame({
            'Percentile': [f"{p}th" for p in percentiles],
            'Portfolio Value': [f"${v:,.0f}" for v in percentile_values]
        })
        st.dataframe(df_percentiles, hide_index=True, use_container_width=True)
    
    with tab2:
        st.subheader("Probability of Success Over Time")
        
        if failure_years_list:
            failure_counts = np.zeros(years, dtype=int)
            for y in failure_years_list:
                failure_counts[y-1] += 1
            cumulative_failures = np.cumsum(failure_counts)
            
            survival_prob = 1 - cumulative_failures / simulations
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, years+1)),
                y=survival_prob * 100,
                mode='lines',
                name='Survival Probability',
                line=dict(color='green', width=2),
                fill='tozeroy'
            ))
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Success Probability (%)",
                yaxis_range=[0, 105],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 100% success rate across all years!")
    
    with tab3:
        if failure_years_list:
            st.subheader("Failure Timing Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Earliest Failure", f"Year {min(failure_years_list)}")
                st.metric("Latest Failure", f"Year {max(failure_years_list)}")
            
            with col2:
                st.metric("Average Failure Year", f"{np.mean(failure_years_list):.1f}")
                st.metric("Median Failure Year", f"{np.median(failure_years_list):.0f}")
            
            # Failure distribution histogram
            st.subheader("When Do Failures Occur?")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=failure_years_list,
                nbinsx=years,
                name="Failures by Year",
                marker_color='red'
            ))
            fig.update_layout(
                xaxis_title="Year of Failure",
                yaxis_title="Number of Failed Simulations",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No failures occurred in any simulation!")
    
    with tab4:
        st.subheader("Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Spending Phase Summary**")
            st.write(f"Average Go-Go years: {np.mean(years_go):.1f}")
            st.write(f"Average Slow-Go years: {np.mean(years_slow):.1f}")
            st.write(f"Average No-Go years: {np.mean(years_no):.1f}")
            
            if has_ss and ss_start_year < years:
                ss_year1_nominal = ss_monthly * 12 * ((1 + inflation_rate) ** (ss_start_year - 1))
                st.markdown(f"**Social Security in Year {ss_start_year}** (nominal $): ${ss_year1_nominal:,.0f}")
        
        with col2:
            st.markdown("**Portfolio Statistics**")
            st.write(f"Mean final portfolio: ${np.mean(final_portfolios):,.0f}")
            st.write(f"Std dev: ${np.std(final_portfolios):,.0f}")
            st.write(f"Min final portfolio: ${np.min(final_portfolios):,.0f}")
            st.write(f"Max final portfolio: ${np.max(final_portfolios):,.0f}")

elif run_simulation and total_allocation != 100:
    st.error("⚠️ Please ensure asset allocation sums to 100% before running simulation")
else:
    st.info("👈 Configure your parameters in the sidebar and click 'Run Simulation' to begin")
    
    # Show example parameters
    st.subheader("About This Simulator")
    st.markdown("""
    This Monte Carlo simulator helps you evaluate retirement portfolio sustainability by:
    
    - Running thousands of random market scenarios
    - Accounting for variable spending patterns over time
    - Including mortgages, Social Security, and temporary income
    - Modeling realistic market volatility and drawdowns
    - Showing success probability and failure timing
    
    **How to use:**
    1. Adjust parameters in the left sidebar
    2. Click "Run Simulation"
    3. Review results across multiple tabs
    4. Experiment with different scenarios to test sensitivity
    """)
