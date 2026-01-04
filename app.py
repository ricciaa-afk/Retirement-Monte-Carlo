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
st.sidebar.subheader("📈 Asset Allocation (Normal Mode)")
equity_pct = st.sidebar.slider("Equities (%)", min_value=0, max_value=100, value=80)
bond_pct = st.sidebar.slider("Bonds (%)", min_value=0, max_value=100, value=15)
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
equity_return = st.sidebar.slider("Equity Return", min_value=0.0, max_value=0.20, value=0.08, step=0.005, format="%.3f")
equity_vol = st.sidebar.slider("Equity Volatility", min_value=0.0, max_value=0.30, value=0.18, step=0.01, format="%.2f")
bond_return = st.sidebar.slider("Bond Return", min_value=0.0, max_value=0.15, value=0.03, step=0.005, format="%.3f")
bond_vol = st.sidebar.slider("Bond Volatility", min_value=0.0, max_value=0.15, value=0.06, step=0.01, format="%.2f")
cash_return = st.sidebar.slider("Cash Return", min_value=0.0, max_value=0.10, value=0.01, step=0.005, format="%.3f")
inflation_rate = st.sidebar.slider("Inflation Rate", min_value=0.0, max_value=0.10, value=0.02, step=0.005, format="%.3f")

# Tax Drag
st.sidebar.subheader("💸 Tax Drag by Period")
st.sidebar.markdown("*Percentage of withdrawals lost to taxes*")

go_go_tax_drag = st.sidebar.slider(
    "Go-Go Years Tax Drag (%)", 
    min_value=0, 
    max_value=30, 
    value=5, 
    step=1,
    help="Early years: pulling from taxable accounts, cap gains treatment"
) / 100

slow_go_tax_drag = st.sidebar.slider(
    "Slow-Go Years Tax Drag (%)", 
    min_value=0, 
    max_value=30, 
    value=12, 
    step=1,
    help="Middle years: RMDs start, mix of account types"
) / 100

no_go_tax_drag = st.sidebar.slider(
    "No-Go Years Tax Drag (%)", 
    min_value=0, 
    max_value=30, 
    value=15, 
    step=1,
    help="Later years: higher RMDs, more ordinary income"
) / 100

# Spending Glidepath
st.sidebar.subheader("💳 Spending Glidepath (Monthly, After-Tax $)")
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

# Bucket Strategy - Defensive Withdrawal Rules
st.sidebar.subheader("🛡️ Defensive Withdrawal Strategy")
st.sidebar.markdown("*Withdraw from cash/bonds first during market downturns*")
defensive_trigger = st.sidebar.slider("Defensive Trigger (market down %)", min_value=0, max_value=50, value=15, step=5, format="%d%%") / 100
recovery_threshold = st.sidebar.slider("Recovery Threshold (within % of peak)", min_value=0, max_value=20, value=5, step=1, format="%d%%") / 100

# Guardrails Strategy
st.sidebar.subheader("🚨 Dynamic Guardrails Strategy")
enable_guardrails = st.sidebar.checkbox("Enable Guardrails", value=False)

if enable_guardrails:
    st.sidebar.markdown("*Automatically adjust spending and allocation when portfolio drops*")
    
    guardrail_threshold = st.sidebar.number_input(
        "Guardrail Trigger ($)", 
        min_value=0, 
        max_value=10000000, 
        value=1500000, 
        step=100000,
        help="If portfolio drops below this, activate guardrails"
    )
    
    spending_reduction = st.sidebar.slider(
        "Reduced Spending (%)", 
        min_value=50, 
        max_value=100, 
        value=75, 
        step=5,
        help="Percentage of base spending when guardrails active"
    ) / 100
    
    st.sidebar.markdown("**Defensive Allocation (when triggered):**")
    
    defensive_equity_pct = st.sidebar.slider("Defensive Equities (%)", min_value=0, max_value=100, value=50, key="def_eq")
    defensive_bond_pct = st.sidebar.slider("Defensive Bonds (%)", min_value=0, max_value=100, value=40, key="def_bond")
    defensive_cash_pct = st.sidebar.slider("Defensive Cash (%)", min_value=0, max_value=100, value=10, key="def_cash")
    
    defensive_allocation_total = defensive_equity_pct + defensive_bond_pct + defensive_cash_pct
    if defensive_allocation_total != 100:
        st.sidebar.error(f"⚠️ Defensive allocation must sum to 100% (currently {defensive_allocation_total}%)")
    
    defensive_asset_mix = {
        "equities": defensive_equity_pct / 100,
        "bonds": defensive_bond_pct / 100,
        "cash": defensive_cash_pct / 100
    }
    
    recovery_buffer = st.sidebar.slider(
        "Recovery Buffer (%)", 
        min_value=0, 
        max_value=20, 
        value=5,
        help="Portfolio must exceed threshold by this % to exit guardrails"
    ) / 100
    
    st.sidebar.info(f"Guardrails activate if portfolio < ${guardrail_threshold:,.0f}. Resume normal at ${guardrail_threshold * (1 + recovery_buffer):,.0f}")

# Run button
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Main content area
if run_simulation and total_allocation == 100:
    if enable_guardrails and defensive_allocation_total != 100:
        st.error("⚠️ Please ensure defensive allocation sums to 100% before running simulation")
    else:
        with st.spinner(f"Running {simulations:,} simulations... This may take a moment..."):
            
            np.random.seed(42)
            
            final_portfolios = []
            success_flags = []
            failure_years_list = []
            
            years_go = []
            years_slow = []
            years_no = []
            
            defensive_years_count = []
            guardrail_years_count = []
            total_taxes_paid = []
            
            for sim in range(simulations):
                
                # Initialize portfolio buckets
                equities = initial_portfolio * asset_mix["equities"]
                bonds = initial_portfolio * asset_mix["bonds"]
                cash = initial_portfolio * asset_mix["cash"]
                
                market_index = 1.0
                market_high = 1.0
                
                y_go = y_slow = y_no = 0
                defensive_years = 0
                guardrail_years = 0
                taxes_paid = 0
                success = True
                
                guardrails_active = False
                
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
                    
                    # Check portfolio value for guardrails
                    current_portfolio = equities + bonds + cash
                    
                    # Guardrails logic
                    if enable_guardrails:
                        if not guardrails_active and current_portfolio < guardrail_threshold:
                            # Activate guardrails
                            guardrails_active = True
                        elif guardrails_active and current_portfolio > guardrail_threshold * (1 + recovery_buffer):
                            # Deactivate guardrails - recovered
                            guardrails_active = False
                        
                        if guardrails_active:
                            guardrail_years += 1
                    
                    # Calculate current drawdown
                    drawdown = (market_high - market_index) / market_high
                    
                    # Determine if we're in defensive withdrawal mode
                    in_defensive_mode = drawdown > defensive_trigger
                    
                    # Check if we've recovered enough to exit defensive mode
                    if in_defensive_mode and drawdown < recovery_threshold:
                        in_defensive_mode = False
                    
                    if in_defensive_mode:
                        defensive_years += 1
                    
                    # Spending Phase - apply guardrail reduction if active
                    if year < go_go_years:
                        monthly_spend = go_go_spend
                        y_go += 1
                        current_tax_drag = go_go_tax_drag
                    elif year < (go_go_years + slow_go_years):
                        monthly_spend = slow_go_spend
                        y_slow += 1
                        current_tax_drag = slow_go_tax_drag
                    else:
                        monthly_spend = no_go_spend
                        y_no += 1
                        current_tax_drag = no_go_tax_drag
                    
                    # Apply spending reduction if guardrails active
                    if enable_guardrails and guardrails_active:
                        monthly_spend = monthly_spend * spending_reduction
                    
                    # Calculate after-tax spending need
                    annual_spend_after_tax = monthly_spend * 12 * ((1 + inflation_rate) ** year)
                    
                    # Mortgage (after-tax - already paid from after-tax dollars)
                    if year < mortgage_term_years:
                        annual_spend_after_tax += annual_mortgage
                    
                    # Social Security (after-tax benefit received)
                    if year >= ss_start_year:
                        ss_annual = ss_monthly * 12 * ((1 + inflation_rate) ** (year - ss_start_year))
                    else:
                        ss_annual = 0
                    
                    net_spend_after_tax = max(annual_spend_after_tax - ss_annual, 0)
                    
                    # Temporary earned income (after-tax)
                    if year < temp_income_years:
                        net_spend_after_tax = max(net_spend_after_tax - temp_income_annual, 0)
                    
                    # Calculate gross withdrawal needed (before taxes)
                    if current_tax_drag > 0:
                        gross_withdrawal = net_spend_after_tax / (1 - current_tax_drag)
                        tax_amount = gross_withdrawal - net_spend_after_tax
                        taxes_paid += tax_amount
                    else:
                        gross_withdrawal = net_spend_after_tax
                        tax_amount = 0
                    
                    # Check if we have enough total
                    total = equities + bonds + cash
                    if gross_withdrawal >= total:
                        success = False
                        failure_years_list.append(year + 1)
                        equities = bonds = cash = 0
                        break
                    
                    # WITHDRAWAL STRATEGY - depends on market conditions
                    if in_defensive_mode:
                        # Defensive: withdraw from Cash → Bonds → Equities
                        remaining_need = gross_withdrawal
                        
                        # Take from cash first
                        cash_withdrawal = min(remaining_need, cash)
                        cash -= cash_withdrawal
                        remaining_need -= cash_withdrawal
                        
                        # Take from bonds next
                        if remaining_need > 0:
                            bond_withdrawal = min(remaining_need, bonds)
                            bonds -= bond_withdrawal
                            remaining_need -= bond_withdrawal
                        
                        # Take from equities last
                        if remaining_need > 0:
                            equities -= remaining_need
                    
                    else:
                        # Normal times: withdraw proportionally
                        prop_eq = equities / total
                        prop_bd = bonds / total
                        prop_cash = cash / total
                        
                        equities -= gross_withdrawal * prop_eq
                        bonds -= gross_withdrawal * prop_bd
                        cash -= gross_withdrawal * prop_cash
                    
                    # REBALANCING - only if NOT in defensive mode
                    if not in_defensive_mode:
                        total = equities + bonds + cash
                        
                        # Use guardrail allocation if active, otherwise use normal allocation
                        if enable_guardrails and guardrails_active:
                            target_mix = defensive_asset_mix
                        else:
                            target_mix = asset_mix
                        
                        equities = total * target_mix['equities']
                        bonds = total * target_mix['bonds']
                        cash = total * target_mix['cash']
                    
                    if equities + bonds + cash <= 0:
                        success = False
                        failure_years_list.append(year + 1)
                        break
                
                final_portfolios.append(equities + bonds + cash)
                success_flags.append(success)
                years_go.append(y_go)
                years_slow.append(y_slow)
                years_no.append(y_no)
                defensive_years_count.append(defensive_years)
                guardrail_years_count.append(guardrail_years)
                total_taxes_paid.append(taxes_paid)
        
        # Calculate results
        final_portfolios = np.array(final_portfolios)
        success_rate = np.mean(success_flags)
        
        # Display Results
        st.header("📊 Results")
        
        # Key metrics in columns
        if enable_guardrails:
            col1, col2, col3, col4, col5 = st.columns(5)
        else:
            col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Success Rate", f"{success_rate*100:.1f}%")
        with col2:
            st.metric("Failed Simulations", f"{simulations - sum(success_flags):,}")
        with col3:
            median_final = np.median(final_portfolios)
            st.metric("Median Final Portfolio", f"${median_final:,.0f}")
        with col4:
            avg_defensive_years = np.mean(defensive_years_count)
            st.metric("Avg Years in Defensive Mode", f"{avg_defensive_years:.1f}")
        
        if enable_guardrails:
            with col5:
                avg_guardrail_years = np.mean(guardrail_years_count)
                st.metric("Avg Years with Guardrails", f"{avg_guardrail_years:.1f}")
        
        # Tax metrics
        avg_taxes = np.mean(total_taxes_paid)
        median_taxes = np.median(total_taxes_paid)
        st.info(f"💸 **Lifetime Taxes Paid:** Average: ${avg_taxes:,.0f} | Median: ${median_taxes:,.0f}")
        
        # Tabs for different views
        if enable_guardrails:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Portfolio Distribution", "⏱️ Survival Curve", "📉 Failure Analysis", "🚨 Guardrails Analysis", "📋 Detailed Stats", "🤖 AI Analysis Export"])
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Portfolio Distribution", "⏱️ Survival Curve", "📉 Failure Analysis", "📋 Detailed Stats", "🤖 AI Analysis Export"])
        
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
        
        if enable_guardrails:
            with tab4:
                st.subheader("Guardrails Strategy Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Simulations Using Guardrails", f"{100 * np.sum(np.array(guardrail_years_count) > 0) / simulations:.1f}%")
                    st.metric("Average Years with Guardrails", f"{np.mean(guardrail_years_count):.1f}")
                    st.metric("Max Years with Guardrails", f"{np.max(guardrail_years_count)}")
                
                with col2:
                    if np.sum(np.array(guardrail_years_count) > 0) > 0:
                        guardrail_users = [g for g in guardrail_years_count if g > 0]
                        st.metric("Avg Years (when used)", f"{np.mean(guardrail_users):.1f}")
                        st.metric("Median Years (when used)", f"{np.median(guardrail_users):.1f}")
                
                # Histogram of guardrail usage
                st.subheader("Distribution of Guardrail Years")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=guardrail_years_count,
                    nbinsx=30,
                    name="Years in Guardrails",
                    marker_color='orange'
                ))
                fig.update_layout(
                    xaxis_title="Years Spent in Guardrails Mode",
                    yaxis_title="Number of Simulations",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            tab_idx_ai = tab5
            tab_idx = tab6
        else:
            tab_idx_ai = tab4
            tab_idx = tab5
        
        with tab_idx_ai:
            st.subheader("🤖 AI Analysis Export")
            st.markdown("Copy the text below and paste into an AI chat to analyze or compare multiple scenarios")
            
            # Generate comprehensive text summary
            percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
            percentile_values = np.percentile(final_portfolios, percentiles)
            
            analysis_text = f"""RETIREMENT MONTE CARLO SIMULATION RESULTS
{'='*60}

PARAMETERS:
- Simulations: {simulations:,}
- Time Horizon: {years} years
- Initial Portfolio: ${initial_portfolio:,.0f}
- Asset Allocation: {equity_pct}% Equities / {bond_pct}% Bonds / {cash_pct}% Cash

MARKET ASSUMPTIONS:
- Equity Return: {equity_return*100:.1f}% (Volatility: {equity_vol*100:.0f}%)
- Bond Return: {bond_return*100:.1f}% (Volatility: {bond_vol*100:.0f}%)
- Cash Return: {cash_return*100:.1f}%
- Inflation: {inflation_rate*100:.1f}%
- Real Equity Return: {(equity_return - inflation_rate)*100:.1f}%

TAX ASSUMPTIONS:
- Go-Go Years Tax Drag: {go_go_tax_drag*100:.0f}%
- Slow-Go Years Tax Drag: {slow_go_tax_drag*100:.0f}%
- No-Go Years Tax Drag: {no_go_tax_drag*100:.0f}%

SPENDING PLAN (After-Tax, Today's Dollars):
- Go-Go Years (1-{go_go_years}): ${go_go_spend:,.0f}/month
- Slow-Go Years ({go_go_years+1}-{go_go_years+slow_go_years}): ${slow_go_spend:,.0f}/month
- No-Go Years ({go_go_years+slow_go_years+1}+): ${no_go_spend:,.0f}/month

OTHER INCOME/EXPENSES:
- Mortgage: {"Yes" if has_mortgage else "No"}"""

            if has_mortgage:
                analysis_text += f" (${mortgage_balance:,.0f} @ {mortgage_rate*100:.2f}%, {mortgage_term_years} years, ${mortgage_payment:,.0f}/month)"
            
            analysis_text += f"""
- Social Security: {"Yes" if has_ss else "No"}"""
            
            if has_ss:
                analysis_text += f" (${ss_monthly:,.0f}/month starting year {ss_start_year})"
            
            analysis_text += f"""
- Temporary Income: {"Yes" if has_temp_income else "No"}"""
            
            if has_temp_income:
                analysis_text += f" (${temp_income_annual:,.0f}/year for {temp_income_years} years)"

            analysis_text += f"""

WITHDRAWAL STRATEGY:
- Defensive Trigger: Market down {defensive_trigger*100:.0f}%+
- Recovery Threshold: Within {recovery_threshold*100:.0f}% of peak
- Strategy: Withdraw Cash→Bonds→Equities when defensive

GUARDRAILS:
- Enabled: {"Yes" if enable_guardrails else "No"}"""

            if enable_guardrails:
                analysis_text += f"""
- Trigger Threshold: ${guardrail_threshold:,.0f}
- Spending Reduction: {spending_reduction*100:.0f}%
- Defensive Allocation: {defensive_equity_pct}% / {defensive_bond_pct}% / {defensive_cash_pct}%
- Recovery Buffer: {recovery_buffer*100:.0f}%"""

            analysis_text += f"""

{'='*60}
KEY RESULTS:
{'='*60}

SUCCESS RATE: {success_rate*100:.1f}%
Failed Simulations: {simulations - sum(success_flags):,} ({(1-success_rate)*100:.1f}%)

FINAL PORTFOLIO DISTRIBUTION:
"""
            for p, val in zip(percentiles, percentile_values):
                analysis_text += f"  {p:3d}th percentile: ${val:15,.0f}\n"

            analysis_text += f"""
PORTFOLIO STATISTICS:
- Mean: ${np.mean(final_portfolios):,.0f}
- Median: ${np.median(final_portfolios):,.0f}
- Std Dev: ${np.std(final_portfolios):,.0f}

TAX BURDEN:
- Average Lifetime Taxes: ${np.mean(total_taxes_paid):,.0f}
- Median Lifetime Taxes: ${np.median(total_taxes_paid):,.0f}

DEFENSIVE WITHDRAWALS:
- Average Years in Defensive Mode: {np.mean(defensive_years_count):.1f}
- Max Years in Defensive Mode: {np.max(defensive_years_count)}
- Simulations Using Defensive Mode: {100 * np.sum(np.array(defensive_years_count) > 0) / simulations:.1f}%
"""

            if enable_guardrails:
                analysis_text += f"""
GUARDRAILS USAGE:
- Simulations Triggering Guardrails: {100 * np.sum(np.array(guardrail_years_count) > 0) / simulations:.1f}%
- Average Years with Guardrails: {np.mean(guardrail_years_count):.1f}
- Max Years with Guardrails: {np.max(guardrail_years_count)}"""
                
                if np.sum(np.array(guardrail_years_count) > 0) > 0:
                    guardrail_users = [g for g in guardrail_years_count if g > 0]
                    analysis_text += f"""
- Average Years (when used): {np.mean(guardrail_users):.1f}
- Median Years (when used): {np.median(guardrail_users):.1f}"""

            if failure_years_list:
                analysis_text += f"""

FAILURE ANALYSIS:
- Earliest Failure: Year {min(failure_years_list)}
- Latest Failure: Year {max(failure_years_list)}
- Average Failure Year: {np.mean(failure_years_list):.1f}
- Median Failure Year: {np.median(failure_years_list):.0f}
"""
            else:
                analysis_text += f"""

FAILURE ANALYSIS:
- No failures occurred in any simulation
"""

            analysis_text += f"""
{'='*60}
END OF SIMULATION RESULTS
{'='*60}
"""
            
            # Display in a text area that can be copied
            st.text_area("Simulation Results (Copy All)", analysis_text, height=600)
            
            st.info("💡 **Tip:** Run multiple simulations with different parameters, copy each result, then paste them all into an AI chat with a prompt like: 'Analyze and compare these three retirement scenarios and recommend which is best.'")

        
        with tab_idx:
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
                
                st.markdown("**Defensive Withdrawal Summary**")
                st.write(f"Average years in defensive mode: {np.mean(defensive_years_count):.1f}")
                st.write(f"Max years in defensive mode: {np.max(defensive_years_count)}")
                st.write(f"% of sims that used defensive mode: {100 * np.sum(np.array(defensive_years_count) > 0) / simulations:.1f}%")
                
                if enable_guardrails:
                    st.markdown("**Guardrails Summary**")
                    st.write(f"Average years with guardrails: {np.mean(guardrail_years_count):.1f}")
                    st.write(f"Max years with guardrails: {np.max(guardrail_years_count)}")
                    st.write(f"% of sims that used guardrails: {100 * np.sum(np.array(guardrail_years_count) > 0) / simulations:.1f}%")
            
            with col2:
                st.markdown("**Portfolio Statistics**")
                st.write(f"Mean final portfolio: ${np.mean(final_portfolios):,.0f}")
                st.write(f"Std dev: ${np.std(final_portfolios):,.0f}")
                st.write(f"Min final portfolio: ${np.min(final_portfolios):,.0f}")
                st.write(f"Max final portfolio: ${np.max(final_portfolios):,.0f}")
                
                st.markdown("**Tax Statistics**")
                st.write(f"Mean lifetime taxes: ${np.mean(total_taxes_paid):,.0f}")
                st.write(f"Median lifetime taxes: ${np.median(total_taxes_paid):,.0f}")
                st.write(f"Min lifetime taxes: ${np.min(total_taxes_paid):,.0f}")
                st.write(f"Max lifetime taxes: ${np.max(total_taxes_paid):,.0f}")

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
    - **Modeling realistic tax drag** that varies by retirement phase
    - Including mortgages, Social Security, and temporary income
    - Modeling realistic market volatility and drawdowns
    - **Using a defensive bucket strategy during market downturns** to protect against sequence of returns risk
    - **Optional dynamic guardrails** to automatically adjust spending and allocation when portfolio drops
    
    **Tax Modeling:**
    Accounts for the fact that early years (taxable accounts, cap gains) are more tax-efficient than later years (RMDs, ordinary income):
    - Go-Go years: Lower tax drag (default 5%)
    - Slow-Go years: Medium tax drag as RMDs start (default 12%)
    - No-Go years: Higher tax drag from larger RMDs (default 15%)
    
    **Defensive Withdrawal Strategy:**
    When markets drop significantly, the simulator switches to defensive mode:
    - Withdrawals come from Cash → Bonds → Equities (preserving growth assets)
    - Rebalancing is paused until markets recover
    - Helps avoid "selling low" during crashes
    
    **Dynamic Guardrails Strategy (Optional):**
    Automatically respond to portfolio distress:
    - If portfolio drops below threshold, reduce spending and shift to defensive allocation
    - When portfolio recovers, resume normal spending and allocation
    - Maximizes flexibility while protecting against failure
    
    **How to use:**
    1. Adjust parameters in the left sidebar
    2. Set tax drag percentages for each phase
    3. Set your defensive trigger and recovery thresholds
    4. Optionally enable guardrails for dynamic adjustment
    5. Click "Run Simulation"
    6. Review results across multiple tabs
    7. Experiment with different scenarios to test sensitivity
    """)
