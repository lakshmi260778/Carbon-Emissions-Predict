import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CO₂ Emissions Predictor", page_icon="🌍", layout="wide")

# 1. Load trained model and encoders
@st.cache_resource
def load_model():
    with open("co2_emissions_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
model = model_data["model"]
le_country = model_data["le_country"]
le_region = model_data["le_region"]

# (Optional) load data once to get valid choices for country and region
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Carbon_(CO2)_Emissions_by_Country.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    # Create country-region mapping for validation
    country_region_map = df.groupby("Country")["Region"].first().to_dict()
    regions = sorted(df["Region"].unique().tolist())
else:
    # fallback if CSV not present in production
    country_region_map = {}
    regions = []

# Sidebar for inputs and predict button
#st.sidebar.title("CO₂ Emissions Predictor")
#st.sidebar.markdown("<small>Predict kilotons of CO₂ based on country, region, and year.</small>", unsafe_allow_html=True)
#st.sidebar.markdown("---")

# Input widgets in sidebar
st.sidebar.subheader("Input Parameters")

if regions:
    region = st.sidebar.selectbox("Region", regions)
    # Filter countries based on selected region
    countries_in_region = sorted([c for c, r in country_region_map.items() if r == region])
    country = st.sidebar.selectbox("Country", countries_in_region)
else:
    region = st.sidebar.text_input("Region (type name)")
    country = st.sidebar.text_input("Country (type name)")

year = st.sidebar.number_input("Year", min_value=1900, max_value=2100, value=2030, step=1)

st.sidebar.markdown("---")

# Predict button in sidebar (prominent)
#prediction_result = None
#if st.sidebar.button("🚀 Predict CO₂ Emissions", type="primary", use_container_width=True):
#    try:
#        # Encode the user inputs using the same encoders from training
#        country_encoded = le_country.transform([country])[0]
#        region_encoded = le_region.transform([region])[0]

        # Construct one-row DataFrame with encoded values
#        X_user = pd.DataFrame([{
#            "Country": country_encoded,
#            "Region": region_encoded,
#            "Year": year
#        }])

#        pred = model.predict(X_user)[0]
#        prediction_result = pred
#        st.sidebar.success(f"**Predicted: {pred:,.0f} kilotons**")
#    except Exception as e:
#        st.sidebar.error(f"Error: {e}")

# Main page content
st.title("CO₂ Emissions Predictor")
#st.markdown("<medium>Predict kilotons of CO₂ emissions based on country, region, and year.</medium>", unsafe_allow_html=True)   
#st.info("Use the sidebar on the left to select parameters and click 'Predict CO₂ Emissions'.")
st.markdown("<medium>Be prudent. Act wisely. Protect our resources.</medium>", unsafe_allow_html=True)  
st.info("This application helps you explore, predict, and visualize CO₂ emissions for different countries and regions, showing how emission levels evolve over time.")
 

prediction_result = None
#if st.button("🚀 Predict CO₂ Emissions", type="primary", use_container_width=True):
if st.button("🚀 Predict CO₂ Emissions", type="primary"):
    try:
        # Encode the user inputs using the same encoders from training
        country_encoded = le_country.transform([country])[0]
        region_encoded = le_region.transform([region])[0]

        # Construct one-row DataFrame with encoded values
        X_user = pd.DataFrame([{
            "Country": country_encoded,
            "Region": region_encoded,
            "Year": year
        }])

        pred = model.predict(X_user)[0]
        prediction_result = pred
        st.success(f"**Predicted: {pred:,.0f} kilotons**")
    except Exception as e:
        st.error(f"Error: {e}")

# Visualization section in sidebar
#st.sidebar.markdown("---")
st.sidebar.subheader("Visualizations")

show_top_emitters = st.sidebar.button("📊 View vs Top Emitters", use_container_width=True)
show_regional_avg = st.sidebar.button("📈 View vs Regional Average", use_container_width=True)
show_latest_comparison = st.sidebar.button("📉 View Latest Year Comparison", use_container_width=True)

# 4. Visualizations Section - Only show when button is clicked
if df is not None and country:
    # Prepare data for visualization
    df["Year"] = pd.to_datetime(df["Date"], format="%d-%m-%Y").dt.year
    
    # Plot 1: Selected Country vs Top Emitters Over Time
    if show_top_emitters:
        with st.expander(f"📊 {country} vs Top Emitting Countries", expanded=True):
            # Get top 5 countries by total emissions for comparison
            top_countries = df.groupby("Country")["Kilotons of Co2"].sum().nlargest(5).index.tolist()
            
            # Ensure selected country is included
            if country not in top_countries:
                comparison_countries = top_countries[:4] + [country]
            else:
                comparison_countries = top_countries
            
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            for c in comparison_countries:
                country_data = df[df["Country"] == c].sort_values("Year")
                if c == country:
                    ax1.plot(country_data["Year"], country_data["Kilotons of Co2"], 
                            marker="o", linewidth=3, label=f"{c} (Selected)", color="red")
                else:
                    ax1.plot(country_data["Year"], country_data["Kilotons of Co2"], 
                            marker="o", alpha=0.7, label=c)
            
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Kilotons of CO2")
            ax1.set_title(f"CO2 Emissions Over Time: {country} vs Top Emitters")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
    
    # Plot 2: Selected Country vs Regional Average
    if show_regional_avg:
        with st.expander(f"📈 {country} vs {region} Regional Average", expanded=True):
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            # Selected country data
            country_data = df[df["Country"] == country].sort_values("Year")
            ax2.plot(country_data["Year"], country_data["Kilotons of Co2"], 
                    marker="o", linewidth=3, label=f"{country}", color="red")
            
            # Regional average
            region_data = df[df["Region"] == region].groupby("Year")["Kilotons of Co2"].mean().reset_index()
            ax2.plot(region_data["Year"], region_data["Kilotons of Co2"], 
                    marker="s", linewidth=2, label=f"{region} Average", color="blue", linestyle="--")
            
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Kilotons of CO2")
            ax2.set_title(f"{country} vs {region} Regional Average Over Time")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
    
    # Plot 3: Bar Chart Comparison - Latest Year
    if show_latest_comparison:
        with st.expander("📉 Latest Year Comparison", expanded=True):
            latest_year = df["Year"].max()
            latest_data = df[df["Year"] == latest_year].groupby("Country")["Kilotons of Co2"].sum().nlargest(10)
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            colors = ["red" if c == country else "steelblue" for c in latest_data.index]
            sns.barplot(x=latest_data.values, y=latest_data.index, palette=colors, ax=ax3)
            ax3.set_xlabel("Kilotons of CO2")
            ax3.set_title(f"Top 10 Countries by CO2 Emissions ({latest_year})")
            st.pyplot(fig3)
