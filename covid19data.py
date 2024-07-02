import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('C:/Users/faran/Downloads/covid_19_data.csv')

data = load_data()

# Display the first few rows of the dataset
st.title("COVID-19 Data Analysis and Model Deployment")
st.subheader("Dataset")
st.dataframe(data.head())

# Display basic statistics of the dataset
st.subheader("Basic Statistics")
st.write(data.describe())

# Exploratory Data Analysis (EDA)
st.subheader("Exploratory Data Analysis (EDA)")

# Select only numeric columns for the correlation matrix
numeric_cols = ['Confirmed', 'Deaths', 'Recovered']

st.write("Heatmap of Correlations")
plt.figure(figsize=(10, 8))
sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)

# Handle infinite values by replacing them with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Time series plot of confirmed cases
st.write("Time Series Plot of Confirmed Cases")
plt.figure(figsize=(12, 6))
data_grouped_date = data.groupby('ObservationDate').sum().reset_index()

# Handle any remaining NaN values by filling them with 0
data_grouped_date.fillna(0, inplace=True)

sns.lineplot(x='ObservationDate', y='Confirmed', data=data_grouped_date)
plt.title('Time Series Plot of Confirmed Cases')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.xticks(rotation=45)
st.pyplot(plt)

# Distribution plots for each feature
for col in numeric_cols:
    st.write(f"Distribution of {col}")
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col].fillna(0), bins=30, kde=True)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Data Preprocessing
st.subheader("Data Preprocessing and Model Training")

# Select features for the model
features = ['Deaths', 'Recovered']
target = 'Confirmed'

# Handle missing values by filling with 0 (or you can use other imputation methods)
data[features] = data[features].fillna(0)

# Split the data into training and testing sets
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Machine Learning Model: Linear Regression
st.write("Training Linear Regression Model")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'Linear Regression - Mean Squared Error: {mse}')
st.write(f'Linear Regression - R^2 Score: {r2}')

# Visualize the Predictions vs Actual for Linear Regression
st.write("Linear Regression: Actual vs Predicted Confirmed Cases")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')
plt.xlabel('Actual Confirmed Cases')
plt.ylabel('Predicted Confirmed Cases')
st.pyplot(plt)

# Machine Learning Model: Random Forest Regression
st.write("Training Random Forest Regression Model")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

st.write(f'Random Forest - Mean Squared Error: {mse_rf}')
st.write(f'Random Forest - R^2 Score: {r2_rf}')

# Visualize the Predictions vs Actual for Random Forest
st.write("Random Forest: Actual vs Predicted Confirmed Cases")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')
plt.xlabel('Actual Confirmed Cases')
plt.ylabel('Predicted Confirmed Cases')
st.pyplot(plt)

# Comparison of the models
if mse_rf < mse:
    st.write("Random Forest Regression is more accurate than Linear Regression.")
else:
    st.write("Linear Regression is more accurate than Random Forest Regression.")