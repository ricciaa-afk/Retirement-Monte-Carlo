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
inflation_vol = st.sidebar.slider("Inflation Volatility", min_value=0.0, max_value=0.05, value=0.01, step=0.005, format="%.3f", help="Standard deviation of annual inflation")

# Regime-Based Inflation
st.sidebar.subheader("📊 Inflation Regimes (Optional)")
use_inflation_regimes = st.sidebar.checkbox("Use Regime-Based Inflation", value=False, help="Model economic cycles with persistent inflation periods")

if use_inflation_regimes:
    st.sidebar.markdown("*Inflation will shift between distinct economic regimes*")
    
    st.sidebar.markdown("**Low Inflation Regime:**")
    low_inflation_mean = st.sidebar.slider("Low - Mean", min_value=0.0, max_value=0.05, value=0.015, step=0.005, format="%.3f", key="low_mean")
    low_inflation_vol = st.sidebar.slider("Low - Volatility", min_value=0.0, max_value=0.03, value=0.0075, step=0.0025, format="%.4f", key="low_vol")
    low_inflation_duration = st.sidebar.slider("Low - Avg Duration (years)", min_value=3, max_value=20, value=8, step=1, key="low_dur")
    
    st.sidebar.markdown("**Normal Inflation Regime:**")
    normal_inflation_mean = st.sidebar.slider("Normal - Mean", min_value=0.0, max_value=0.08, value=0.025, step=0.005, format="%.3f", key="normal_mean")
    normal_inflation_vol = st.sidebar.slider("Normal - Volatility", min_value=0.0, max_value=0.03, value=0.01, step=0.0025, format="%.4f", key="normal_vol")
    normal_inflation_duration = st.sidebar.slider("Normal - Avg Duration (years)", min_value=3, max_value=20, value=12, step=1, key="normal_dur")
    
    st.sidebar.markdown("**High Inflation Regime:**")
    high_inflation_mean = st.sidebar.slider("High - Mean", min_value=0.0, max_value=0.15, value=0.05, step=0.005, format="%.3f", key="high_mean")
    high_inflation_vol = st.sidebar.slider("High - Volatility", min_value=0.0, max_value=0.05, value=0.015, step=0.0025, format="%.4f", key="high_vol")
    high_inflation_duration = st.sidebar.slider("High - Avg Duration (years)", min_value=3, max_value=20, value=7, step=1, key="high_dur")
    
    starting_regime = st.sidebar.selectbox("Starting Regime", ["Normal", "Low", "High"], index=0)
    
    # Store regime parameters
    inflation_regimes = {
        "Low": {"mean": low_inflation_mean, "vol": low_inflation_vol, "duration": low_inflation_duration},
        "Normal": {"mean": normal_inflation_mean, "vol": normal_inflation_vol, "duration": normal_inflation_duration},
        "High": {"mean": high_inflation_mean, "vol": high_inflation_vol, "duration": high_inflation_duration}
    }
else:
    starting_regime = None
    inflation_regimes = None

# Tax Drag
st.sidebar.subheader("💸 Tax Drag by Period")
st.sidebar.markdown("*Percentage of withdrawals lost to taxes*")

high_spend_tax_drag = st.sidebar.slider(
    "High Spend Years Tax Drag (%)", 
    min_value=0, 
    max_value=30, 
    value=5, 
    step=1,
    help="Early years: pulling from taxable accounts, cap gains treatment"
) / 100

med_spend_tax_drag = st.sidebar.slider(
    "Medium Spend Years Tax Drag (%)", 
    min_value=0, 
    max_value=30, 
    value=12, 
    step=1,
    help="Middle years: RMDs start, mix of account types"
) / 100

low_spend_tax_drag = st.sidebar.slider(
    "Low Spend Years Tax Drag (%)", 
    min_value=0, 
    max_value=30, 
    value=15, 
    step=1,
    help="Later years: higher RMDs, more ordinary income"
) / 100

# Spending Glidepath
st.sidebar.subheader("💳 Spending Glidepath (Monthly, After-Tax $)")
high_spend_monthly = st.sidebar.number_input("High Spend Years (Monthly)", min_value=0, max_value=50000, value=11100, step=100)
high_spend_years = st.sidebar.number_input("High Spend Duration (years)", min_value=0, max_value=50, value=10)

med_spend_monthly = st.sidebar.number_input("Medium Spend Years (Monthly)", min_value=0, max_value=50000, value=9100, step=100)
med_spend_years = st.sidebar.number_input("Medium Spend Duration (years)", min_value=0, max_value=50, value=15)

low_spend_monthly = st.sidebar.number_input("Low Spend Years (Monthly)", min_value=0, max_value=50000, value=8350, step=100)

# Spending Model Selection
st.sidebar.subheader("📊 Spending Model")
use_conditional_spend = st.sidebar.checkbox("Use Conditional Spending Model", value=False, 
    help="Dynamically adjust spending tier based on portfolio value vs present value of remaining needs")

if use_conditional_spend:
    st.sidebar.markdown("*Spending tier selected each year based on portfolio health*")
    discount_rate = st.sidebar.slider("Discount Rate for PV Calculation", 
        min_value=0.0, max_value=0.10, value=0.04, step=0.005, format="%.3f",
        help="Rate used to calculate present value of future spending needs. Higher rate = more aggressive spending.")
    st.sidebar.info(f"With {inflation_rate*100:.1f}% inflation, this implies {(discount_rate-inflation_rate)*100:.1f}% real discount rate")
else:
    discount_rate = None
    st.sidebar.markdown("*Spending follows planned year-based tiers*")

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

# Delayed Retirement Income
st.sidebar.subheader("💼 Delayed Retirement Income")
has_delayed_income = st.sidebar.checkbox("Include Delayed Retirement Income", value=True)
if has_delayed_income:
    delayed_income_annual = st.sidebar.number_input("Annual Income ($, nominal)", min_value=0, max_value=1000000, value=120000, step=10000)
    delayed_income_years = st.sidebar.number_input("Years Before Full Retirement", min_value=0, max_value=20, value=3, 
        help="Work this many years before fully retiring")
else:
    delayed_income_annual = 0
    delayed_income_years = 0

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
    
    st.sidebar.markdown("**Spending Reductions (% to CUT):**")
    
    high_spend_cut = st.sidebar.slider(
        "High Spend Years Cut (%)", 
        min_value=0, 
        max_value=50, 
        value=20, 
        step=5,
        help="Percentage reduction in high spend when guardrails active"
    ) / 100
    
    med_spend_cut = st.sidebar.slider(
        "Medium Spend Years Cut (%)", 
        min_value=0, 
        max_value=50, 
        value=10, 
        step=5,
        help="Percentage reduction in medium spend when guardrails active"
    ) / 100
    
    low_spend_cut = st.sidebar.slider(
        "Low Spend Years Cut (%)", 
        min_value=0, 
        max_value=50, 
        value=5, 
        step=5,
        help="Percentage reduction in low spend when guardrails active"
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
    
    # Contingency Income (Return to Work)
    st.sidebar.markdown("**Contingency Income (Return to Work):**")
    enable_contingency_income = st.sidebar.checkbox("Enable Contingency Income", value=False,
        help="Return to work if guardrails trigger")
    
    if enable_contingency_income:
        contingency_income_annual = st.sidebar.number_input(
            "Annual Contingency Income ($, today's $)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=5000,
            help="Income from returning to work, inflation-adjusted"
        )
        contingency_min_years = st.sidebar.number_input(
            "Minimum Years to Work",
            min_value=1,
            max_value=20,
            value=2,
            help="Minimum continuous years when contingency activated"
        )
        contingency_max_years = st.sidebar.number_input(
            "Maximum Years to Work",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum continuous years when contingency activated"
        )
        st.sidebar.info(f"Work {contingency_min_years}-{contingency_max_years} years if guardrails activate")
    else:
        contingency_income_annual = 0
        contingency_min_years = 0
        contingency_max_years = 0

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
            max_consecutive_guardrails = []  # Track longest continuous guardrail period
            guardrail_patterns = []  # Track year-by-year guardrail status for each sim
            
            # Conditional spending tracking
            if use_conditional_spend:
                tier_usage_high = []
                tier_usage_med = []
                tier_usage_low = []
                max_consecutive_low_tier = []
                tier_patterns = []  # Track which tier each year: 1=high, 2=med, 3=low
            
            # Lifestyle/spending tracking (actual spending including guardrail cuts)
            lifestyle_patterns = []  # Track actual spending level each year
            # Categories: 1=High, 2=High-cut, 3=Med, 4=Med-cut, 5=Low, 6=Low-cut
            
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
                consecutive_guardrails = 0
                max_consecutive = 0
                guardrail_pattern = []  # Track each year: 1=active, 0=inactive
                cumulative_inflation = 1.0  # Track cumulative inflation multiplier
                
                # Inflation regime tracking
                if use_inflation_regimes:
                    current_regime = starting_regime
                    years_in_regime = 0
                else:
                    current_regime = None
                    years_in_regime = 0
                
                # Conditional spending tracking
                if use_conditional_spend:
                    years_high_tier = 0
                    years_med_tier = 0
                    years_low_tier = 0
                    consecutive_low_tier = 0
                    max_consecutive_low = 0
                    tier_pattern = []
                
                # Contingency income tracking
                if enable_guardrails and enable_contingency_income:
                    contingency_active = False
                    contingency_years_worked = 0
                    contingency_start_year = 0  # Track when guardrails first triggered
                
                # Lifestyle tracking
                lifestyle_pattern = []  # Track actual spending level each year
                
                for year in range(years):
                    
                    # Generate variable inflation for this year
                    if use_inflation_regimes:
                        # Regime-based inflation with transitions
                        years_in_regime += 1
                        
                        # Check if regime should transition
                        regime_params = inflation_regimes[current_regime]
                        expected_duration = regime_params["duration"]
                        
                        # Probability of transitioning increases as we exceed expected duration
                        # Use exponential probability: low chance early, higher chance as time goes on
                        transition_prob = 1 - np.exp(-years_in_regime / expected_duration)
                        
                        if np.random.random() < transition_prob:
                            # Transition to new regime
                            # Weighted transitions: 50% Normal, 25% Low, 25% High
                            regime_weights = {"Normal": 0.5, "Low": 0.25, "High": 0.25}
                            # But can't transition to current regime
                            del regime_weights[current_regime]
                            # Renormalize
                            total = sum(regime_weights.values())
                            regime_weights = {k: v/total for k, v in regime_weights.items()}
                            
                            current_regime = np.random.choice(list(regime_weights.keys()), p=list(regime_weights.values()))
                            years_in_regime = 0
                        
                        # Generate inflation for this year from current regime
                        regime_params = inflation_regimes[current_regime]
                        year_inflation = max(0, np.random.normal(regime_params["mean"], regime_params["vol"]))
                        
                    elif inflation_vol > 0:
                        # Simple random inflation
                        year_inflation = max(0, np.random.normal(inflation_rate, inflation_vol))
                    else:
                        # Fixed inflation
                        year_inflation = inflation_rate
                    
                    # Update cumulative inflation
                    cumulative_inflation *= (1 + year_inflation)
                    
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
                            # Reset consecutive counter when exiting
                            max_consecutive = max(max_consecutive, consecutive_guardrails)
                            consecutive_guardrails = 0
                        
                        if guardrails_active:
                            guardrail_years += 1
                            consecutive_guardrails += 1
                            guardrail_pattern.append(1)
                        else:
                            guardrail_pattern.append(0)
                    
                    # Calculate current drawdown
                    drawdown = (market_high - market_index) / market_high
                    
                    # Determine if we're in defensive withdrawal mode
                    in_defensive_mode = drawdown > defensive_trigger
                    
                    # Check if we've recovered enough to exit defensive mode
                    if in_defensive_mode and drawdown < recovery_threshold:
                        in_defensive_mode = False
                    
                    if in_defensive_mode:
                        defensive_years += 1
                    
                    # Spending Phase - determine monthly spend based on model
                    if use_conditional_spend:
                        # CONDITIONAL SPENDING MODEL
                        # Calculate PV thresholds based on current portfolio needs
                        
                        years_remaining = years - year
                        
                        if years_remaining > 0:
                            # Calculate PV factor for annuity
                            pv_factor = (1 - (1 + discount_rate)**-years_remaining) / discount_rate
                            
                            # Calculate current year's nominal spending for each tier (already inflated)
                            current_inflation_factor = cumulative_inflation
                            
                            # High tier net withdrawal
                            high_gross = high_spend_monthly * 12 * current_inflation_factor
                            if year < mortgage_term_years:
                                high_gross += annual_mortgage
                            high_net = high_gross
                            if year >= ss_start_year:
                                ss_current = ss_monthly * 12 * ((1 + inflation_rate) ** (year - ss_start_year))
                                high_net -= ss_current
                            if year < delayed_income_years:
                                high_net -= delayed_income_annual
                            high_net = max(high_net, 0)
                            
                            # Med tier net withdrawal
                            med_gross = med_spend_monthly * 12 * current_inflation_factor
                            if year < mortgage_term_years:
                                med_gross += annual_mortgage
                            med_net = med_gross
                            if year >= ss_start_year:
                                med_net -= ss_current
                            if year < delayed_income_years:
                                med_net -= delayed_income_annual
                            med_net = max(med_net, 0)
                            
                            # Low tier net withdrawal  
                            low_gross = low_spend_monthly * 12 * current_inflation_factor
                            if year < mortgage_term_years:
                                low_gross += annual_mortgage
                            low_net = low_gross
                            if year >= ss_start_year:
                                low_net -= ss_current
                            if year < delayed_income_years:
                                low_net -= delayed_income_annual
                            low_net = max(low_net, 0)
                            
                            # Calculate PV thresholds
                            high_threshold = high_net * pv_factor
                            med_threshold = med_net * pv_factor
                            low_threshold = low_net * pv_factor
                            
                            # Compare current portfolio to thresholds
                            current_portfolio = equities + bonds + cash
                            
                            if current_portfolio >= high_threshold:
                                monthly_spend = high_spend_monthly
                                current_tier = 1
                                years_high_tier += 1
                                consecutive_low_tier = 0
                            elif current_portfolio >= med_threshold:
                                monthly_spend = med_spend_monthly
                                current_tier = 2
                                years_med_tier += 1
                                consecutive_low_tier = 0
                            else:
                                monthly_spend = low_spend_monthly
                                current_tier = 3
                                years_low_tier += 1
                                consecutive_low_tier += 1
                                max_consecutive_low = max(max_consecutive_low, consecutive_low_tier)
                            
                            tier_pattern.append(current_tier)
                        else:
                            # Last year - default to low tier
                            monthly_spend = low_spend_monthly
                            current_tier = 3
                            tier_pattern.append(current_tier)
                        
                        # Set tax drag based on tier selected
                        if current_tier == 1:
                            current_tax_drag = high_spend_tax_drag
                            current_spending_cut = high_spend_cut if enable_guardrails and guardrails_active else 0
                        elif current_tier == 2:
                            current_tax_drag = med_spend_tax_drag
                            current_spending_cut = med_spend_cut if enable_guardrails and guardrails_active else 0
                        else:
                            current_tax_drag = low_spend_tax_drag
                            current_spending_cut = low_spend_cut if enable_guardrails and guardrails_active else 0
                    
                    else:
                        # PLANNED SPENDING MODEL (original logic)
                        if year < high_spend_years:
                            monthly_spend = high_spend_monthly
                            y_go += 1
                            current_tax_drag = high_spend_tax_drag
                            current_spending_cut = high_spend_cut if enable_guardrails and guardrails_active else 0
                        elif year < (high_spend_years + med_spend_years):
                            monthly_spend = med_spend_monthly
                            y_slow += 1
                            current_tax_drag = med_spend_tax_drag
                            current_spending_cut = med_spend_cut if enable_guardrails and guardrails_active else 0
                        else:
                            monthly_spend = low_spend_monthly
                            y_no += 1
                            current_tax_drag = low_spend_tax_drag
                            current_spending_cut = low_spend_cut if enable_guardrails and guardrails_active else 0
                    
                    # Apply spending reduction (cut is percentage to reduce, not percentage to keep)
                    monthly_spend = monthly_spend * (1 - current_spending_cut)
                    
                    # Track lifestyle category: 1=High, 2=High-cut, 3=Med, 4=Med-cut, 5=Low, 6=Low-cut
                    if use_conditional_spend:
                        # For conditional, current_tier is already set (1/2/3)
                        if current_tier == 1:
                            lifestyle_cat = 2 if (enable_guardrails and guardrails_active) else 1
                        elif current_tier == 2:
                            lifestyle_cat = 4 if (enable_guardrails and guardrails_active) else 3
                        else:
                            lifestyle_cat = 6 if (enable_guardrails and guardrails_active) else 5
                    else:
                        # For planned, determine from year
                        if year < high_spend_years:
                            lifestyle_cat = 2 if (enable_guardrails and guardrails_active) else 1
                        elif year < (high_spend_years + med_spend_years):
                            lifestyle_cat = 4 if (enable_guardrails and guardrails_active) else 3
                        else:
                            lifestyle_cat = 6 if (enable_guardrails and guardrails_active) else 5
                    
                    lifestyle_pattern.append(lifestyle_cat)
                    
                    # Calculate after-tax spending need using cumulative inflation
                    annual_spend_after_tax = monthly_spend * 12 * cumulative_inflation
                    
                    # Mortgage (after-tax - already paid from after-tax dollars)
                    if year < mortgage_term_years:
                        annual_spend_after_tax += annual_mortgage
                    
                    # Social Security (after-tax benefit received) - use cumulative inflation from SS start
                    if year >= ss_start_year:
                        ss_inflation_years = year - ss_start_year
                        ss_cumulative = (1 + inflation_rate) ** ss_inflation_years
                        ss_annual = ss_monthly * 12 * ss_cumulative
                    else:
                        ss_annual = 0
                    
                    net_spend_after_tax = max(annual_spend_after_tax - ss_annual, 0)
                    
                    # Delayed retirement income (after-tax)
                    if year < delayed_income_years:
                        net_spend_after_tax = max(net_spend_after_tax - delayed_income_annual, 0)
                    
                    # Contingency income (return to work if guardrails active)
                    if enable_guardrails and enable_contingency_income:
                        # Activate contingency ONLY on first guardrail trigger (not if already working)
                        if guardrails_active and not contingency_active:
                            contingency_active = True
                            contingency_years_worked = 0
                            contingency_guardrail_initially_active = True
                        
                        # Once working, complete commitment regardless of subsequent guardrail changes
                        if contingency_active:
                            should_work = False
                            
                            # Rule 1: Always work at least min_years
                            if contingency_years_worked < contingency_min_years:
                                should_work = True
                            
                            # Rule 2: Never work more than max_years
                            elif contingency_years_worked >= contingency_max_years:
                                should_work = False
                            
                            # Rule 3: Between min and max, work as long as guardrails stay active
                            # (Check current guardrail status, but this is the ORIGINAL guardrail period)
                            else:
                                should_work = guardrails_active
                            
                            if should_work:
                                contingency_income_inflated = contingency_income_annual * cumulative_inflation
                                net_spend_after_tax = max(net_spend_after_tax - contingency_income_inflated, 0)
                                contingency_years_worked += 1
                            else:
                                # Done with work commitment
                                contingency_active = False
                                contingency_years_worked = 0
                    
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
                    
                    # REBALANCING - only if NOT in defensive mode AND equities have grown significantly
                    if not in_defensive_mode:
                        total = equities + bonds + cash
                        
                        # Calculate target equity amount based on allocation
                        if enable_guardrails and guardrails_active:
                            target_equity_allocation = defensive_asset_mix['equities']
                        else:
                            target_equity_allocation = asset_mix['equities']
                        
                        target_equity_amount = total * target_equity_allocation
                        
                        # Only rebalance if equities have grown 20% above target
                        # This ensures we only trim gains, not sell during early recovery
                        if equities > target_equity_amount * 1.20:
                            # Rebalance: trim equity gains to restore bonds/cash
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
                
                # Check max consecutive one final time at end
                if enable_guardrails:
                    max_consecutive = max(max_consecutive, consecutive_guardrails)
                
                if use_conditional_spend:
                    max_consecutive_low = max(max_consecutive_low, consecutive_low_tier)
                
                final_portfolios.append(equities + bonds + cash)
                success_flags.append(success)
                years_go.append(y_go)
                years_slow.append(y_slow)
                years_no.append(y_no)
                defensive_years_count.append(defensive_years)
                guardrail_years_count.append(guardrail_years)
                total_taxes_paid.append(taxes_paid)
                max_consecutive_guardrails.append(max_consecutive)
                guardrail_patterns.append(guardrail_pattern)
                
                if use_conditional_spend:
                    tier_usage_high.append(years_high_tier)
                    tier_usage_med.append(years_med_tier)
                    tier_usage_low.append(years_low_tier)
                    max_consecutive_low_tier.append(max_consecutive_low)
                    tier_patterns.append(tier_pattern)
                
                # Store lifestyle pattern
                lifestyle_patterns.append(lifestyle_pattern)
        
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
        if use_conditional_spend and enable_guardrails:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📈 Portfolio Distribution", "⏱️ Survival Curve", "📉 Failure Analysis", "🚨 Guardrails Analysis", "🎯 Conditional Spending", "💰 Lifestyle Analysis", "🤖 AI Analysis Export", "📋 Detailed Stats"])
        elif use_conditional_spend:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Portfolio Distribution", "⏱️ Survival Curve", "📉 Failure Analysis", "🎯 Conditional Spending", "💰 Lifestyle Analysis", "🤖 AI Analysis Export", "📋 Detailed Stats"])
        elif enable_guardrails:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Portfolio Distribution", "⏱️ Survival Curve", "📉 Failure Analysis", "🚨 Guardrails Analysis", "💰 Lifestyle Analysis", "🤖 AI Analysis Export", "📋 Detailed Stats"])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Portfolio Distribution", "⏱️ Survival Curve", "📉 Failure Analysis", "💰 Lifestyle Analysis", "🤖 AI Analysis Export", "📋 Detailed Stats"])
        
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
                
                # Year-by-year success probability
                st.subheader("Success Probability by Year")
                
                # Calculate survival probability at end of each year
                failure_counts = np.zeros(years, dtype=int)
                for fail_year in failure_years_list:
                    failure_counts[fail_year-1] += 1
                
                cumulative_failures = np.cumsum(failure_counts)
                survival_prob = 1 - cumulative_failures / simulations
                
                # Find first year with failure
                first_failure_year = min(failure_years_list)
                
                # Create table starting from first failure year
                year_range = list(range(first_failure_year, years + 1))
                survival_pcts = [survival_prob[y-1] * 100 for y in year_range]
                failure_counts_display = [failure_counts[y-1] for y in year_range]
                
                df_survival = pd.DataFrame({
                    'End of Year': year_range,
                    'Success Probability': [f"{p:.2f}%" for p in survival_pcts],
                    'Failures This Year': failure_counts_display
                })
                
                # Display in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Probability of Success:**")
                    # Display first half
                    mid_point = len(df_survival) // 2
                    st.dataframe(df_survival.iloc[:mid_point], hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("**&nbsp;**")  # Spacer for alignment
                    # Display second half
                    st.dataframe(df_survival.iloc[mid_point:], hide_index=True, use_container_width=True)
                
                # Summary stats
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    peak_failure_year = year_range[np.argmax(failure_counts_display)]
                    st.metric("Peak Failure Year", f"Year {peak_failure_year}")
                with col2:
                    st.metric("Failures in Peak Year", f"{max(failure_counts_display)}")
                with col3:
                    avg_annual_failures = np.mean([f for f in failure_counts_display if f > 0])
                    st.metric("Avg Failures/Year (when >0)", f"{avg_annual_failures:.1f}")
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
                        
                        # Add consecutive metrics
                        max_consec_all = [m for m in max_consecutive_guardrails if m > 0]
                        if max_consec_all:
                            st.metric("Avg Max Consecutive Years", f"{np.mean(max_consec_all):.1f}")
                            st.metric("Longest Consecutive Stretch", f"{np.max(max_consecutive_guardrails)}")
                
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
                
                # Guardrails Timeline Visualization
                st.subheader("Guardrails Engagement Timeline")
                st.markdown("*Showing when guardrails are active across a sample of simulations*")
                
                # Select a few representative simulations to visualize
                sample_sims = min(20, len(guardrail_patterns))
                
                # Create heatmap data - showing guardrail status by year and simulation
                heatmap_data = []
                sim_labels = []
                
                for i in range(sample_sims):
                    if len(guardrail_patterns[i]) > 0:
                        heatmap_data.append(guardrail_patterns[i])
                        total_gr_years = guardrail_years_count[i]
                        max_consec = max_consecutive_guardrails[i]
                        sim_labels.append(f"Sim {i+1} ({total_gr_years}yr, max{max_consec})")
                
                if heatmap_data:
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data,
                        x=list(range(1, years+1)),
                        y=sim_labels,
                        colorscale=[[0, 'lightgreen'], [1, 'red']],
                        showscale=True,
                        colorbar=dict(
                            title="Status",
                            tickvals=[0, 1],
                            ticktext=["Normal", "Guardrails"]
                        )
                    ))
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Simulation",
                        height=max(400, sample_sims * 25),
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("🟢 Green = Normal spending/allocation | 🔴 Red = Guardrails active (reduced spending + defensive allocation)")
                else:
                    st.info("No simulations triggered guardrails in this run")
        
        # Conditional Spending Analysis Tab
        if use_conditional_spend:
            # Determine which tab index based on whether guardrails enabled
            if enable_guardrails:
                tab_cond = tab5
                tab_idx_ai_temp = tab6
                tab_idx_temp = tab7
            else:
                tab_cond = tab4
                tab_idx_ai_temp = tab5
                tab_idx_temp = tab6
            
            with tab_cond:
                st.subheader("🎯 Conditional Spending Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Years High Tier", f"{np.mean(tier_usage_high):.1f}")
                    st.metric("% Sims Using High Tier", f"{100 * np.sum(np.array(tier_usage_high) > 0) / simulations:.1f}%")
                
                with col2:
                    st.metric("Avg Years Med Tier", f"{np.mean(tier_usage_med):.1f}")
                    st.metric("% Sims Using Med Tier", f"{100 * np.sum(np.array(tier_usage_med) > 0) / simulations:.1f}%")
                
                with col3:
                    st.metric("Avg Years Low Tier", f"{np.mean(tier_usage_low):.1f}")
                    st.metric("% Sims Dropping to Low", f"{100 * np.sum(np.array(tier_usage_low) > 0) / simulations:.1f}%")
                
                # Consecutive low tier metrics
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    low_tier_users = [m for m in max_consecutive_low_tier if m > 0]
                    if low_tier_users:
                        st.metric("Avg Max Consecutive Years in Low", f"{np.mean(low_tier_users):.1f}")
                        st.metric("Longest Stretch in Low Tier", f"{np.max(max_consecutive_low_tier)}")
                
                with col2:
                    if low_tier_users:
                        st.metric("Median Consecutive Low (when used)", f"{np.median(low_tier_users):.1f}")
                
                # Tier distribution chart
                st.subheader("Spending Tier Distribution")
                
                tier_data = pd.DataFrame({
                    'High Tier': tier_usage_high,
                    'Medium Tier': tier_usage_med,
                    'Low Tier': tier_usage_low
                })
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=tier_usage_high, name='High Tier', marker_color='green', opacity=0.7, nbinsx=30))
                fig.add_trace(go.Histogram(x=tier_usage_med, name='Med Tier', marker_color='yellow', opacity=0.7, nbinsx=30))
                fig.add_trace(go.Histogram(x=tier_usage_low, name='Low Tier', marker_color='red', opacity=0.7, nbinsx=30))
                
                fig.update_layout(
                    xaxis_title="Years Spent in Tier",
                    yaxis_title="Number of Simulations",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tier Timeline Visualization
                st.subheader("Spending Tier Timeline")
                st.markdown("*Showing spending tier by year across sample simulations*")
                
                sample_sims = min(20, len(tier_patterns))
                
                heatmap_data = []
                sim_labels = []
                
                for i in range(sample_sims):
                    if len(tier_patterns[i]) > 0:
                        heatmap_data.append(tier_patterns[i])
                        high_yrs = tier_usage_high[i]
                        med_yrs = tier_usage_med[i]
                        low_yrs = tier_usage_low[i]
                        max_low = max_consecutive_low_tier[i]
                        sim_labels.append(f"Sim {i+1} (H:{high_yrs} M:{med_yrs} L:{low_yrs}, max_low:{max_low})")
                
                if heatmap_data:
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data,
                        x=list(range(1, years+1)),
                        y=sim_labels,
                        colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
                        showscale=True,
                        colorbar=dict(
                            title="Tier",
                            tickvals=[1, 2, 3],
                            ticktext=["High", "Med", "Low"]
                        ),
                        zmin=1,
                        zmax=3
                    ))
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Simulation",
                        height=max(400, sample_sims * 25),
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("🟢 Green = High Spend | 🟡 Yellow = Medium Spend | 🔴 Red = Low Spend")
            
            # Set tab indices based on configuration
            if enable_guardrails and use_conditional_spend:
                tab_lifestyle = tab6
                tab_idx_ai = tab7
                tab_idx = tab8
            elif enable_guardrails:
                tab_lifestyle = tab5
                tab_idx_ai = tab6
                tab_idx = tab7
            elif use_conditional_spend:
                tab_lifestyle = tab5
                tab_idx_ai = tab6
                tab_idx = tab7
            else:
                tab_lifestyle = tab4
                tab_idx_ai = tab5
                tab_idx = tab6
        
        with tab_lifestyle:
            st.subheader("💰 Lifestyle Analysis")
            st.markdown("*What does your actual spending look like across all scenarios?*")
            
            # Calculate years in each lifestyle category
            lifestyle_counts = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
            
            for pattern in lifestyle_patterns:
                for cat in range(1, 7):
                    count = pattern.count(cat)
                    lifestyle_counts[cat].append(count)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**High Spend Tier:**")
                st.metric("Avg Years at High Spend", f"{np.mean(lifestyle_counts[1]):.1f}")
                if enable_guardrails:
                    st.metric("Avg Years at High Spend (w/ cuts)", f"{np.mean(lifestyle_counts[2]):.1f}")
                    cut_pct = high_spend_cut * 100 if enable_guardrails else 0
                    st.caption(f"*High with cuts = {100-cut_pct:.0f}% of normal High*")
            
            with col2:
                st.markdown("**Medium Spend Tier:**")
                st.metric("Avg Years at Med Spend", f"{np.mean(lifestyle_counts[3]):.1f}")
                if enable_guardrails:
                    st.metric("Avg Years at Med Spend (w/ cuts)", f"{np.mean(lifestyle_counts[4]):.1f}")
                    cut_pct = med_spend_cut * 100 if enable_guardrails else 0
                    st.caption(f"*Med with cuts = {100-cut_pct:.0f}% of normal Med*")
            
            with col3:
                st.markdown("**Low Spend Tier:**")
                st.metric("Avg Years at Low Spend", f"{np.mean(lifestyle_counts[5]):.1f}")
                if enable_guardrails:
                    st.metric("Avg Years at Low Spend (w/ cuts)", f"{np.mean(lifestyle_counts[6]):.1f}")
                    cut_pct = low_spend_cut * 100 if enable_guardrails else 0
                    st.caption(f"*Low with cuts = {100-cut_pct:.0f}% of normal Low*")
            
            # Distribution chart
            st.subheader("Lifestyle Distribution Across Simulations")
            
            fig = go.Figure()
            
            colors = {
                1: 'darkgreen',
                2: 'lightgreen',
                3: 'gold',
                4: 'yellow',
                5: 'orangered',
                6: 'lightcoral'
            }
            
            labels = {
                1: 'High Spend',
                2: 'High Spend (cut)',
                3: 'Med Spend',
                4: 'Med Spend (cut)',
                5: 'Low Spend',
                6: 'Low Spend (cut)'
            }
            
            for cat in range(1, 7):
                if np.sum(lifestyle_counts[cat]) > 0:  # Only show if used
                    fig.add_trace(go.Histogram(
                        x=lifestyle_counts[cat],
                        name=labels[cat],
                        marker_color=colors[cat],
                        opacity=0.7,
                        nbinsx=30
                    ))
            
            fig.update_layout(
                xaxis_title="Years in Lifestyle Category",
                yaxis_title="Number of Simulations",
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary percentages
            st.subheader("Average Lifestyle Mix")
            total_years = years
            
            lifestyle_pcts = {}
            for cat in range(1, 7):
                pct = (np.mean(lifestyle_counts[cat]) / total_years) * 100
                lifestyle_pcts[cat] = pct
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Percentage of Retirement:**")
                st.write(f"• High Spend: {lifestyle_pcts[1]:.1f}%")
                if enable_guardrails and lifestyle_pcts[2] > 0:
                    st.write(f"• High Spend (cut): {lifestyle_pcts[2]:.1f}%")
                st.write(f"• Med Spend: {lifestyle_pcts[3]:.1f}%")
                if enable_guardrails and lifestyle_pcts[4] > 0:
                    st.write(f"• Med Spend (cut): {lifestyle_pcts[4]:.1f}%")
                st.write(f"• Low Spend: {lifestyle_pcts[5]:.1f}%")
                if enable_guardrails and lifestyle_pcts[6] > 0:
                    st.write(f"• Low Spend (cut): {lifestyle_pcts[6]:.1f}%")
            
            with col2:
                # Overall quality of life metric
                st.markdown("**Overall Lifestyle Quality:**")
                
                # Weighted score: High=100, High-cut=80, Med=70, Med-cut=56, Low=60, Low-cut=48
                weights = {1: 100, 2: 80, 3: 70, 4: 56, 5: 60, 6: 48}
                
                weighted_scores = []
                for pattern in lifestyle_patterns:
                    score = sum(weights[cat] for cat in pattern) / len(pattern)
                    weighted_scores.append(score)
                
                avg_score = np.mean(weighted_scores)
                st.metric("Avg Quality Score", f"{avg_score:.1f}/100")
                st.caption("*Higher = better lifestyle (weighted by spending level)*")
        
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
- Inflation: {inflation_rate*100:.1f}% (Volatility: {inflation_vol*100:.1f}%)"""
            
            if use_inflation_regimes:
                analysis_text += f"""
- REGIME-BASED INFLATION ENABLED:
  - Starting Regime: {starting_regime}
  - Low Regime: {low_inflation_mean*100:.1f}% ± {low_inflation_vol*100:.2f}%, avg {low_inflation_duration} years
  - Normal Regime: {normal_inflation_mean*100:.1f}% ± {normal_inflation_vol*100:.2f}%, avg {normal_inflation_duration} years
  - High Regime: {high_inflation_mean*100:.1f}% ± {high_inflation_vol*100:.2f}%, avg {high_inflation_duration} years"""
            
            analysis_text += f"""
- Real Equity Return: {(equity_return - inflation_rate)*100:.1f}%

TAX ASSUMPTIONS:
- High Spend Years Tax Drag: {high_spend_tax_drag*100:.0f}%
- Medium Spend Years Tax Drag: {med_spend_tax_drag*100:.0f}%
- Low Spend Years Tax Drag: {low_spend_tax_drag*100:.0f}%

SPENDING PLAN (After-Tax, Today's Dollars):
"""
            
            if use_conditional_spend:
                analysis_text += f"""- Spending Model: CONDITIONAL (Dynamic tier selection based on portfolio value)
- Discount Rate: {discount_rate*100:.1f}%
- High Spend Tier: ${high_spend_monthly:,.0f}/month
- Medium Spend Tier: ${med_spend_monthly:,.0f}/month
- Low Spend Tier: ${low_spend_monthly:,.0f}/month
"""
            else:
                analysis_text += f"""- Spending Model: PLANNED (Fixed tiers by year)
- High Spend Years (1-{high_spend_years}): ${high_spend_monthly:,.0f}/month
- Medium Spend Years ({high_spend_years+1}-{high_spend_years+med_spend_years}): ${med_spend_monthly:,.0f}/month
- Low Spend Years ({high_spend_years+med_spend_years+1}+): ${low_spend_monthly:,.0f}/month
"""
            
            analysis_text += f"""OTHER INCOME/EXPENSES:
- Mortgage: {"Yes" if has_mortgage else "No"}"""

            if has_mortgage and mortgage_balance > 0:
                # Recalculate mortgage payment for display
                mort_pmt = mortgage_balance * (mortgage_rate / 12) / (
                    1 - (1 + mortgage_rate / 12) ** (-mortgage_term_years * 12)
                )
                analysis_text += f" (${mortgage_balance:,.0f} @ {mortgage_rate*100:.2f}%, {mortgage_term_years} years, ${mort_pmt:,.0f}/month)"
            
            analysis_text += f"""
- Social Security: {"Yes" if has_ss else "No"}"""
            
            if has_ss:
                analysis_text += f" (${ss_monthly:,.0f}/month starting year {ss_start_year})"
            
            analysis_text += f"""
- Delayed Retirement Income: {"Yes" if has_delayed_income else "No"}"""
            
            if has_delayed_income:
                analysis_text += f" (${delayed_income_annual:,.0f}/year for {delayed_income_years} years)"

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
- Spending Reductions: High {high_spend_cut*100:.0f}% / Med {med_spend_cut*100:.0f}% / Low {low_spend_cut*100:.0f}%
- Defensive Allocation: {defensive_equity_pct}% / {defensive_bond_pct}% / {defensive_cash_pct}%
- Recovery Buffer: {recovery_buffer*100:.0f}%"""
                
                if enable_contingency_income:
                    analysis_text += f"""
- Contingency Income: ${contingency_income_annual:,.0f}/year (work {contingency_min_years}-{contingency_max_years} years if triggered)"""

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
                    max_consec_all = [m for m in max_consecutive_guardrails if m > 0]
                    analysis_text += f"""
- Average Years (when used): {np.mean(guardrail_users):.1f}
- Median Years (when used): {np.median(guardrail_users):.1f}"""
                    if max_consec_all:
                        analysis_text += f"""
- Average Max Consecutive Years: {np.mean(max_consec_all):.1f}
- Longest Consecutive Stretch: {np.max(max_consecutive_guardrails)}"""
            
            if use_conditional_spend:
                analysis_text += f"""

CONDITIONAL SPENDING ANALYSIS:
- Spending Model: Dynamic (tier based on portfolio vs PV of needs)
- Discount Rate Used: {discount_rate*100:.1f}%
- Average Years High Tier: {np.mean(tier_usage_high):.1f}
- Average Years Med Tier: {np.mean(tier_usage_med):.1f}
- Average Years Low Tier: {np.mean(tier_usage_low):.1f}
- Simulations Dropping to Low Tier: {100 * np.sum(np.array(tier_usage_low) > 0) / simulations:.1f}%"""
                
                low_tier_users = [m for m in max_consecutive_low_tier if m > 0]
                if low_tier_users:
                    analysis_text += f"""
- Avg Max Consecutive Years in Low: {np.mean(low_tier_users):.1f}
- Longest Stretch in Low Tier: {np.max(max_consecutive_low_tier)}"""
            
            # Lifestyle Analysis
            lifestyle_counts = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
            for pattern in lifestyle_patterns:
                for cat in range(1, 7):
                    count = pattern.count(cat)
                    lifestyle_counts[cat].append(count)
            
            analysis_text += f"""

LIFESTYLE ANALYSIS:
- Average Years at High Spend: {np.mean(lifestyle_counts[1]):.1f}"""
            
            if enable_guardrails:
                analysis_text += f"""
- Average Years at High Spend (w/ cuts): {np.mean(lifestyle_counts[2]):.1f}"""
            
            analysis_text += f"""
- Average Years at Med Spend: {np.mean(lifestyle_counts[3]):.1f}"""
            
            if enable_guardrails:
                analysis_text += f"""
- Average Years at Med Spend (w/ cuts): {np.mean(lifestyle_counts[4]):.1f}"""
            
            analysis_text += f"""
- Average Years at Low Spend: {np.mean(lifestyle_counts[5]):.1f}"""
            
            if enable_guardrails:
                analysis_text += f"""
- Average Years at Low Spend (w/ cuts): {np.mean(lifestyle_counts[6]):.1f}"""
            
            # Lifestyle percentages
            total_years_calc = years
            lifestyle_pcts = {}
            for cat in range(1, 7):
                pct = (np.mean(lifestyle_counts[cat]) / total_years_calc) * 100
                lifestyle_pcts[cat] = pct
            
            analysis_text += f"""
- Percentage at High Spend: {lifestyle_pcts[1]:.1f}%"""
            if enable_guardrails and lifestyle_pcts[2] > 0.1:
                analysis_text += f"""
- Percentage at High Spend (cut): {lifestyle_pcts[2]:.1f}%"""
            analysis_text += f"""
- Percentage at Med Spend: {lifestyle_pcts[3]:.1f}%"""
            if enable_guardrails and lifestyle_pcts[4] > 0.1:
                analysis_text += f"""
- Percentage at Med Spend (cut): {lifestyle_pcts[4]:.1f}%"""
            analysis_text += f"""
- Percentage at Low Spend: {lifestyle_pcts[5]:.1f}%"""
            if enable_guardrails and lifestyle_pcts[6] > 0.1:
                analysis_text += f"""
- Percentage at Low Spend (cut): {lifestyle_pcts[6]:.1f}%"""
            
            # Quality score
            weights = {{1: 100, 2: 80, 3: 70, 4: 56, 5: 60, 6: 48}}
            weighted_scores = []
            for pattern in lifestyle_patterns:
                score = sum(weights[cat] for cat in pattern) / len(pattern)
                weighted_scores.append(score)
            avg_score = np.mean(weighted_scores)
            
            analysis_text += f"""
- Overall Lifestyle Quality Score: {avg_score:.1f}/100"""

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
                st.write(f"Average High Spend years: {np.mean(years_go):.1f}")
                st.write(f"Average Medium Spend years: {np.mean(years_slow):.1f}")
                st.write(f"Average Low Spend years: {np.mean(years_no):.1f}")
                
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
    - High Spend years: Lower tax drag (default 5%)
    - Medium Spend years: Medium tax drag as RMDs start (default 12%)
    - Low Spend years: Higher tax drag from larger RMDs (default 15%)
    
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
