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


# Main page content
st.title("CO₂ Emissions Predictor")
st.markdown("<medium>Be prudent. Act wise. Protect our resources.</medium>", unsafe_allow_html=True)  
st.info("This application helps you explore, predict, and visualize CO₂ emissions for different countries over time.\n\n- Choose the input parameters in the sidebar and click 'Predict CO₂ Emissions' to see the result.")
 

prediction_result = None
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
    
    # Helper function to predict future values
    def predict_future(c_name, r_name, years):
        c_encoded = le_country.transform([c_name])[0]
        r_encoded = le_region.transform([r_name])[0]
        predictions = []
        for y in years:
            X_pred = pd.DataFrame([{
                "Country": c_encoded,
                "Region": r_encoded,
                "Year": y
            }])
            pred = model.predict(X_pred)[0]
            predictions.append(pred)
        return predictions
    
    # Get year range for predictions
    last_historical_year = df["Year"].max()
    future_years = list(range(last_historical_year + 1, year + 1)) if year > last_historical_year else []
    
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
                # Historical data
                country_data = df[df["Country"] == c].sort_values("Year")
                c_region = country_region_map[c]
                
                if c == country:
                    # Plot historical data for selected country
                    ax1.plot(country_data["Year"], country_data["Kilotons of Co2"], 
                            marker="o", linewidth=3, label=f"{c} (Historical)", color="red")
                    
                    # Plot future predictions for selected country
                    if future_years:
                        future_preds = predict_future(c, c_region, future_years)
                        ax1.plot(future_years, future_preds, 
                                marker="x", linewidth=3, linestyle="--", 
                                label=f"{c} (Predicted)", color="darkred")
                else:
                    # Plot historical data for other countries
                    ax1.plot(country_data["Year"], country_data["Kilotons of Co2"], 
                            marker="o", alpha=0.7, label=f"{c} (Historical)")
                    
                    # Plot future predictions for other countries
                    if future_years:
                        future_preds = predict_future(c, c_region, future_years)
                        ax1.plot(future_years, future_preds, 
                                marker="x", alpha=0.7, linestyle="--", label=f"{c} (Predicted)")
            
            # Add vertical line to separate historical and predicted
            if future_years:
                ax1.axvline(x=last_historical_year + 0.5, color="gray", linestyle=":", alpha=0.7, label="Prediction Start")
            
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Kilotons of CO2")
            ax1.set_title(f"CO2 Emissions Over Time: {country} vs Top Emitters (with Predictions till {year})")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
    
    # Plot 2: Selected Country vs Regional Average
    if show_regional_avg:
        with st.expander(f"📈 {country} vs {region} Regional Average", expanded=True):
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            # Selected country - Historical data
            country_data = df[df["Country"] == country].sort_values("Year")
            ax2.plot(country_data["Year"], country_data["Kilotons of Co2"], 
                    marker="o", linewidth=3, label=f"{country} (Historical)", color="red")
            
            # Selected country - Future predictions
            if future_years:
                future_preds = predict_future(country, region, future_years)
                ax2.plot(future_years, future_preds, 
                        marker="x", linewidth=3, linestyle="--", 
                        label=f"{country} (Predicted)", color="darkred")
            
            # Regional average - Historical
            region_data = df[df["Region"] == region].groupby("Year")["Kilotons of Co2"].mean().reset_index()
            ax2.plot(region_data["Year"], region_data["Kilotons of Co2"], 
                    marker="s", linewidth=2, label=f"{region} Average (Historical)", color="blue", linestyle="-")
            
            # Regional average - Future predictions (predict for all countries in region and average)
            if future_years:
                region_countries = [c for c, r in country_region_map.items() if r == region]
                region_future_preds = []
                for y in future_years:
                    year_preds = []
                    for c in region_countries:
                        c_region = country_region_map[c]
                        pred = predict_future(c, c_region, [y])[0]
                        year_preds.append(pred)
                    region_future_preds.append(sum(year_preds) / len(year_preds))
                
                ax2.plot(future_years, region_future_preds, 
                        marker="x", linewidth=2, linestyle="--", 
                        label=f"{region} Average (Predicted)", color="darkblue")
            
            # Add vertical line to separate historical and predicted
            if future_years:
                ax2.axvline(x=last_historical_year + 0.5, color="gray", linestyle=":", alpha=0.7, label="Prediction Start")
            
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Kilotons of CO2")
            ax2.set_title(f"{country} vs {region} Regional Average Over Time (with Predictions till {year})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
    
    # Plot 3: Bar Chart Comparison - Selected Year
    if show_latest_comparison:
        with st.expander(f"📉 Top 10 Countries Comparison ({year})", expanded=True):
            # Get historical data for latest year
            latest_historical_year = df["Year"].max()
            
            if year <= latest_historical_year:
                # Use historical data
                year_data = df[df["Year"] == year].groupby("Country")["Kilotons of Co2"].sum().nlargest(10)
            else:
                # Use predictions for all countries
                all_countries = df["Country"].unique()
                year_preds = {}
                for c in all_countries:
                    c_region = country_region_map[c]
                    pred = predict_future(c, c_region, [year])[0]
                    year_preds[c] = pred
                year_data = pd.Series(year_preds).nlargest(10)
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            colors = ["red" if c == country else "steelblue" for c in year_data.index]
            sns.barplot(x=year_data.values, y=year_data.index, palette=colors, ax=ax3)
            ax3.set_xlabel("Kilotons of CO2")
            title_type = "Historical" if year <= latest_historical_year else "Predicted"
            ax3.set_title(f"Top 10 Countries by CO2 Emissions ({year}) - {title_type}")
            st.pyplot(fig3)
