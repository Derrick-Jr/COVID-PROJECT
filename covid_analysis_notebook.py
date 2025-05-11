# COVID-19 Data Analysis Project
# ===========================
# This script performs data loading, cleaning, exploratory data analysis (EDA), and visualization
# on COVID-19 data from Our World in Data (OWID).
# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# 2 Data Loading & Exploration
# ==============================
print("Loading COVID-19 data...")

df = pd.read_csv('owid-covid-data.csv')

print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# Check column names
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Display first few rows to understand data structure
print("\nPreview of the dataset:")
print(df.head())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values by column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

# Display basic statistics for numeric columns
print("\nSummary statistics:")
pd.set_option('display.max_columns', None)
print(df.describe())

# 3️⃣ Data Cleaning
# =================
print("\nCleaning the data...")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
print("Date column converted to datetime.")

# Select countries of interest (Kenya, USA, India)
countries_of_interest = ['Kenya', 'United States', 'India']
df_selected = df[df['location'].isin(countries_of_interest)]
print(f"Selected data for {countries_of_interest}.")


print("\nNumber of rows for each selected country:")
print(df_selected['location'].value_counts())


critical_columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']

for country in countries_of_interest:
    country_mask = df_selected['location'] == country
    for col in critical_columns:
        if col.startswith('total_'):
            # For cumulative metrics, use forward fill
            df_selected.loc[country_mask, col] = df_selected.loc[country_mask, col].ffill()
        elif col.startswith('new_'):
            # For daily metrics, fill NAs with 0 (assuming no report = 0)
            df_selected.loc[country_mask, col] = df_selected.loc[country_mask, col].fillna(0)

print("Handled missing values in critical columns.")

# Check remaining missing values in critical columns
print("\nRemaining missing values in critical columns:")
print(df_selected[critical_columns].isnull().sum())

# 4️⃣ Exploratory Data Analysis (EDA)
# ==================================
print("\nPerforming exploratory data analysis...")

# Create a figure for total cases over time
plt.figure(figsize=(14, 8))

for country in countries_of_interest:
    country_data = df_selected[df_selected['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country)

plt.title('Total COVID-19 Cases Over Time by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Cases', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('total_cases_by_country.png')
plt.show()

# Create a figure for total deaths over time
plt.figure(figsize=(14, 8))

for country in countries_of_interest:
    country_data = df_selected[df_selected['location'] == country]
    plt.plot(country_data['date'], country_data['total_deaths'], label=country)

plt.title('Total COVID-19 Deaths Over Time by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Deaths', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('total_deaths_by_country.png')
plt.show()

# Create a figure for new cases over time
plt.figure(figsize=(14, 8))

for country in countries_of_interest:
    country_data = df_selected[df_selected['location'] == country]
    # Calculate 7-day moving average
    country_data['new_cases_7day_avg'] = country_data['new_cases'].rolling(window=7).mean()
    plt.plot(country_data['date'], country_data['new_cases_7day_avg'], label=country)

plt.title('New COVID-19 Cases (7-day Average) by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('New Cases (7-day Average)', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('new_cases_by_country.png')
plt.show()

# Calculate and plot death rates (total_deaths / total_cases)
plt.figure(figsize=(14, 8))

for country in countries_of_interest:
    country_data = df_selected[df_selected['location'] == country]
    # Calculate death rate
    country_data['death_rate'] = (country_data['total_deaths'] / country_data['total_cases']) * 100
    plt.plot(country_data['date'], country_data['death_rate'], label=country)

plt.title('COVID-19 Death Rate (Deaths/Cases) by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Death Rate (%)', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('death_rate_by_country.png')
plt.show()

# Compare cases per million for a more normalized comparison
plt.figure(figsize=(14, 8))

for country in countries_of_interest:
    country_data = df_selected[df_selected['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases_per_million'], label=country)

plt.title('COVID-19 Cases per Million by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cases per Million', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cases_per_million_by_country.png')
plt.show()

# 5️⃣ Visualizing Vaccination Progress
# ==================================
print("\nAnalyzing vaccination progress...")

# Create a figure for total vaccinations over time
plt.figure(figsize=(14, 8))

for country in countries_of_interest:
    country_data = df_selected[df_selected['location'] == country]
   
    if 'total_vaccinations' in country_data.columns and not country_data['total_vaccinations'].isna().all():
        plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)

plt.title('Total COVID-19 Vaccinations Over Time by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Vaccinations', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('total_vaccinations_by_country.png')
plt.show()


plt.figure(figsize=(14, 8))

for country in countries_of_interest:
    country_data = df_selected[df_selected['location'] == country]
    
    if 'people_vaccinated_per_hundred' in country_data.columns and not country_data['people_vaccinated_per_hundred'].isna().all():
        plt.plot(country_data['date'], country_data['people_vaccinated_per_hundred'], label=country)

plt.title('Percentage of Population Vaccinated by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('People Vaccinated (%)', fontsize=12)
plt.legend()
plt.grid(True)
plt.axhline(y=70, color='r', linestyle='--', label='70% Target')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('vaccination_percentage_by_country.png')
plt.show()


latest_data = df_selected.groupby('location').apply(lambda x: x.nlargest(1, 'date')).reset_index(drop=True)


plt.figure(figsize=(12, 8))

for i, country in enumerate(countries_of_interest):
    plt.subplot(1, len(countries_of_interest), i+1)
    
    country_latest = latest_data[latest_data['location'] == country]
    
    
    if 'people_fully_vaccinated_per_hundred' in country_latest.columns and not country_latest['people_fully_vaccinated_per_hundred'].isna().all():
        fully_vacc = country_latest['people_fully_vaccinated_per_hundred'].values[0]
        if pd.isna(fully_vacc):
            fully_vacc = 0
        not_fully_vacc = 100 - fully_vacc
        
        plt.pie([fully_vacc, not_fully_vacc], 
                labels=['Fully Vaccinated', 'Not Fully Vaccinated'],
                autopct='%1.1f%%', 
                colors=['green', 'lightgray'],
                startangle=90)
        plt.title(f'{country} - Vaccination Status')

plt.tight_layout()
plt.savefig('vaccination_status_pie_charts.png')
plt.show()

# 6️⃣ Build a Choropleth Map
# ===================================
print("\nCreating a choropleth map of COVID-19 data...")

latest_date = df['date'].max()
latest_global_data = df[df['date'] == latest_date]

# Create a choropleth map of total cases per million
try:
    fig = px.choropleth(
        latest_global_data,
        locations="iso_code",
        color="total_cases_per_million",
        hover_name="location",
        color_continuous_scale=px.colors.sequential.Plasma,
        title=f"COVID-19 Cases per Million by Country (as of {latest_date.strftime('%Y-%m-%d')})"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Cases per Million")
    )
    fig.write_html("covid_cases_choropleth.html")
    
    print("Choropleth map created and saved as 'covid_cases_choropleth.html'")
except Exception as e:
    print(f"Could not create choropleth map: {e}")


# 7️⃣ Insights & Reporting
# =======================
print("\nGenerating insights from the data...")

# Calculate key metrics for insights
insights_data = []

for country in countries_of_interest:
    country_data = df_selected[df_selected['location'] == country]
    
    # Get the latest data
    latest = country_data.iloc[-1]
    
    # Calculate peak daily cases and when it occurred
    peak_cases_idx = country_data['new_cases'].idxmax()
    peak_cases_date = country_data.loc[peak_cases_idx, 'date']
    peak_cases = country_data.loc[peak_cases_idx, 'new_cases']
    
    # Calculate current death rate
    current_death_rate = (latest['total_deaths'] / latest['total_cases']) * 100 if latest['total_cases'] > 0 else 0
    
    # Check vaccination progress if data is available
    vacc_progress = "Unknown"
    if 'people_vaccinated_per_hundred' in latest and not pd.isna(latest['people_vaccinated_per_hundred']):
        vacc_progress = f"{latest['people_vaccinated_per_hundred']:.1f}%"
    
    insights_data.append({
        "Country": country,
        "Total Cases": f"{latest['total_cases']:,.0f}",
        "Total Deaths": f"{latest['total_deaths']:,.0f}",
        "Death Rate": f"{current_death_rate:.2f}%",
        "Peak Daily Cases": f"{peak_cases:,.0f} on {peak_cases_date.strftime('%Y-%m-%d')}",
        "Vaccination Progress": vacc_progress
    })

# Display insights table
insights_df = pd.DataFrame(insights_data)
print("\nKey COVID-19 Metrics by Country:")
print(insights_df)

# INSIGHTS NARRATIVE
# -----------------
print("\n===== COVID-19 DATA ANALYSIS INSIGHTS =====")
print("""
Based on the analysis of COVID-19 data for Kenya, United States, and India, the following key insights emerge:

1. CASE PROGRESSION PATTERNS:
   - The United States showed the highest absolute number of cases, but when normalized by population (cases per million), we can see different patterns of spread intensity across countries.
   - India experienced rapid case growth during its major waves, while Kenya maintained relatively lower case counts throughout the pandemic.

2. DEATH RATES AND MORTALITY:
   - Death rates (deaths/cases) showed significant variation between countries, which could be attributed to differences in healthcare capacity, reporting methods, testing availability, and demographic factors.
   - The death rates generally declined over time as treatment protocols improved and vulnerable populations became vaccinated.

3. VACCINATION ROLLOUT:
   - The United States achieved faster initial vaccination coverage compared to India and Kenya.
   - Vaccination inequity is evident in the data, with significant differences in vaccination timelines and coverage between these countries.

4. WAVE PATTERNS:
   - Each country experienced distinct COVID-19 waves at different times, showing how the pandemic progressed differently across geographical regions.
   - The timing of peaks differed significantly, demonstrating the global yet asynchronous nature of the pandemic.

5. REPORTING CONSISTENCY:
   - Data quality and reporting consistency varied between countries, with some showing gaps or inconsistencies in reporting that might affect interpretation.

These insights highlight the complex and varied impact of COVID-19 across different countries, influenced by factors including healthcare infrastructure, government policies, population demographics, and vaccine availability.
""")

print("\nAnalysis complete! All visualizations have been saved to the current directory.")
