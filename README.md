# 🦠 COVID-19 Data Analysis Project

## 📋 Overview
This project analyzes COVID-19 data across multiple countries, focusing on case trends, death rates, and vaccination progress. The analysis provides insights into how different countries (Kenya, USA, and India) experienced the pandemic, with visualizations and comparative metrics.

## 🔍 Key Features
- Comprehensive data exploration and cleaning
- Multi-country comparative analysis
- Time-series visualizations of key COVID metrics
- Vaccination progress tracking
- Interactive choropleth world map
- Detailed insights and interpretations

## 📊 Visualizations Generated
- Total cases over time by country
- Total deaths over time by country
- 7-day rolling average of new cases
- Death rate trends by country
- Cases per million (population-normalized)
- Vaccination progress and coverage
- Vaccination status pie charts
- Global choropleth map of COVID impacts

## 🛠️ Requirements
- Python 3.7+
- Required libraries:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  plotly
  ```

## 📥 Installation & Setup
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/covid-analysis.git
   cd covid-analysis
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - The project uses the "Our World in Data" COVID-19 dataset
   - Download from: https://github.com/owid/covid-19-data/tree/master/public/data
   - Save as `owid-covid-data.csv` in the project directory

## 🚀 Usage
1. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open the `COVID-19_Data_Analysis_Project.ipynb` notebook

3. Run cells sequentially to reproduce the analysis

## 📊 Analysis Components

### 1️⃣ Data Exploration
- Dataset overview and structure analysis
- Missing value identification
- Data type verification

### 2️⃣ Data Cleaning
- Date format standardization
- Country filtering
- Missing value handling with appropriate techniques

### 3️⃣ Exploratory Data Analysis
- Time-series visualizations
- Country-by-country comparisons
- Death rate calculations

### 4️⃣ Vaccination Analysis
- Vaccination rollout tracking
- Population coverage comparisons
- Vaccination status visualization

### 5️⃣ Geospatial Visualization
- Interactive choropleth world map
- Country-level COVID metric mapping

### 6️⃣ Insights & Interpretation
- Data-driven findings
- Cross-country comparative analysis
- Pandemic pattern identification

## 🔑 Key Insights
The analysis reveals:
- Distinct wave patterns across different countries
- Variations in death rates and their potential causes
- Vaccination inequity between countries
- Impact of population density and healthcare infrastructure
- Data quality considerations affecting interpretation

