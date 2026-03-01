import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load the trained model
model = joblib.load("medicaid_spending_model.pkl")

# Hardcoded aggregates from notebook (top drugs, popular, state totals)
top_exp_drugs = pd.DataFrame({
    "Product Name": ["biktarvy", "jardiance", "trulicity", "invega sus", "humira(cf)",
                     "humira pen", "ozempic", "dupixent s", "eliquis", "zepbound",
                     "dupixent p", "ozempic 0.", "abilify ma", "stelara 90", "vraylar (c"],
    "Total Amount Reimbursed": [1.487752e9, 1.224935e9, 9.942157e8, 8.841717e8, 8.621136e8,
                                7.755576e8, 6.612600e8, 6.432030e8, 6.114578e8, 5.080595e8,
                                4.911089e8, 4.176484e8, 4.130071e8, 3.995239e8, 3.830521e8]
})

popular_drugs = pd.DataFrame({
    "Product Name": ["amoxicilli", "albuterol", "ibuprofen", "fluticason", "atorvastat",
                     "gabapentin", "ondansetro", "cetirizine", "metformin", "sertraline",
                     "hydroxyzin", "omeprazole", "amlodipine", "lisinopril", "trazodone"],
    "No of prescriptions": [8243538, 7264826, 6543001, 6042960, 5987791,
                            5969243, 5741383, 5275403, 4655398, 4481159,
                            4438048, 4397245, 4231409, 4016472, 3957128]
})

state_spending = pd.DataFrame({
    "State Full Name": ["California", "New York", "Pennsylvania", "Ohio", "North Carolina",
                        "Michigan", "Illinois", "Florida", "Indiana", "Virginia",
                        "Kentucky", "Texas", "Louisiana", "Massachusetts", "Wisconsin"],
    "Total Amount Reimbursed": [6.730810e9, 5.143920e9, 2.345845e9, 2.009043e9, 2.001101e9,
                                1.755784e9, 1.442779e9, 1.387497e9, 1.267096e9, 1.240000e9,
                                1.181363e9, 1.163783e9, 1.108245e9, 9.837195e8, 9.464532e8]
})

state_usage = pd.DataFrame({
    "State Full Name": ["California", "New York", "Ohio", "Pennsylvania", "Texas",
                        "North Carolina", "Michigan", "Florida", "Kentucky", "Illinois",
                        "Indiana", "New Jersey", "Virginia", "Missouri", "Massachusetts"],
    "Units Reimbursed": [2.037401e9, 1.488311e9, 7.663352e8, 7.039589e8, 6.490379e8,
                         5.959167e8, 5.167147e8, 4.880454e8, 4.733878e8, 4.474310e8,
                         3.938806e8, 3.557094e8, 3.473175e8, 3.401117e8, 3.317646e8]
})

# State abbreviation lookup for map
state_full_to_abbr = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
    "Wyoming": "WY", "District of Columbia": "DC", "Puerto Rico": "PR"
}

# Recommendations text (from notebook page 30)
recommendations = """
1. Prioritize high-volume prescription management  
   Number of Prescriptions is the strongest driver → focus on prior authorization, step therapy, generic substitution.

2. Benchmark high-impact states  
   Compare high-spending states (CA, NY, PA…) against lower-cost peers to identify best practices.

3. Focus on drug-specific cost optimization  
   Negotiate rebates for high-impact drugs and promote therapeutic alternatives.

4. Develop state-specific cost containment strategies  
   Tailor interventions to enrollment size, disease burden, and local patterns.

5. Enhance predictive budget planning  
   Use forecasting tools to anticipate pressures and adjust policies early.
"""

# Page config & Medicaid styling
st.set_page_config(page_title="MedicaidRx Estimator", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f5f7fa; }
    h1, h2, h3 { color: #003366; }
    .stButton>button { 
        background-color: #003366; 
        color: white; 
        border: none; 
        border-radius: 6px; 
        padding: 0.6em 1.4em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e6f0ff;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: #003366;
    }
    .stTabs [aria-selected="true"] {
        background-color: #003366 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💊 MedicaidRx: Spending Estimator & Insights")
st.caption("Predict reimbursement and explore Medicaid drug spending patterns")

# Tabs
tab_calc, tab_insights, tab_about = st.tabs(["Calculator", "Insights", "About & Recommendations"])

# ───── Calculator Tab ─────
with tab_calc:
    st.subheader("Estimate Reimbursement Spending")

    col1, col2 = st.columns(2)
    with col1:
        state_full = st.selectbox("State", sorted(state_spending["State Full Name"]))
    with col2:
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])

    product = st.text_input("Drug Name", placeholder="e.g. ozempic, biktarvy")

    col3, col4 = st.columns(2)
    with col3:
        units = st.number_input("Units Reimbursed", min_value=0.0, step=100.0, format="%.0f")
    with col4:
        prescriptions = st.number_input("Number of Prescriptions", min_value=0.0, step=100.0, format="%.0f")

    utilization_intensity = units * prescriptions

    if st.button("Calculate Estimate"):
        if not product.strip():
            st.error("Please enter a drug name")
        elif units == 0 and prescriptions == 0:
            st.warning("Zero units and prescriptions → estimate may be unreliable")
        else:
            input_df = pd.DataFrame({
                "State Full Name": [state_full],
                "Product Name": [product.lower().strip()],
                "Quarter": [quarter],
                "Units Reimbursed": [units],
                "Number of Prescriptions": [prescriptions],
                "utilization_intensity": [utilization_intensity]
            })

            log_pred = model.predict(input_df)[0]
            spending = np.expm1(log_pred)

            st.success(f"**Estimated spending in {state_full}: ${spending:,.0f}**")

            # Download option
            export_df = input_df.copy()
            export_df["Predicted Spending"] = spending
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download result (CSV)",
                csv,
                "medicaid_estimate.csv",
                "text/csv"
            )

# ───── Insights Tab ─────
with tab_insights:
    st.subheader("Spending & Utilization Patterns")

    # Map section
    st.markdown("#### Geographic Overview")
    map_view = st.radio("Show", ["Spending ($ Billion)", "Units Reimbursed (Billion)"], horizontal=True)

    if map_view == "Spending ($ Billion)":
        df_map = state_spending.copy()
        df_map["Value"] = df_map["Total Amount Reimbursed"] / 1e9
        colorbar_title = "$ Billion"
        colors = "YlOrRd"
    else:
        df_map = state_usage.copy()
        df_map["Value"] = df_map["Units Reimbursed"] / 1e9
        colorbar_title = "Billion units"
        colors = "Blues"

    df_map["State Code"] = df_map["State Full Name"].map(state_full_to_abbr)

    fig_map = px.choropleth(
        df_map,
        locations="State Code",
        locationmode="USA-states",
        color="Value",
        hover_name="State Full Name",
        hover_data={"Value": ":.2f"},
        color_continuous_scale=colors,
        scope="usa",
        labels={"Value": colorbar_title}
    )

    fig_map.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0},
        height=500,
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Bar chart section
    st.markdown("#### Top Items")
    view = st.selectbox("View", ["Top Spending Drugs", "Most Prescribed Drugs", "Top Spending States", "Top Usage States"])
    n_top = st.slider("Show top", 5, 15, 10)

    if view == "Top Spending Drugs":
        df_plot = top_exp_drugs.head(n_top)
        x, y = "Total Amount Reimbursed", "Product Name"
        title = "Top High-Cost Drugs"
    elif view == "Most Prescribed Drugs":
        df_plot = popular_drugs.head(n_top)
        x, y = "No of prescriptions", "Product Name"
        title = "Most Frequently Prescribed Drugs"
    elif view == "Top Spending States":
        df_plot = state_spending.head(n_top)
        x, y = "Total Amount Reimbursed", "State Full Name"
        title = "Top Spending States"
    else:
        df_plot = state_usage.head(n_top)
        x, y = "Units Reimbursed", "State Full Name"
        title = "Top States by Units Reimbursed"

    fig_bar, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=df_plot, x=x, y=y, ax=ax, palette="Blues_d")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    st.pyplot(fig_bar)

# ───── About Tab ─────
with tab_about:
    st.subheader("Project Overview")
    st.markdown("""
    This tool analyzes **Medicaid State Drug Utilization Data (2025)** to  
    - identify high-cost and high-volume drugs  
    - show spending & usage variation across states  
    - predict reimbursement amounts based on utilization inputs  

    The XGBoost model achieves R² > 0.80 on cross-validation and test sets.
    """)

    st.markdown("#### Key Recommendations")
    st.markdown(recommendations)