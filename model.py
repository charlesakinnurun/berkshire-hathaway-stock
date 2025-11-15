# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

# %% [markdown]
# Data Acquisition and Preprocesing

# %%
#  Define the stock ticker for Berkshire Hathway Class B shares (more liquid)
TICKER = "BRK-B"
# Define the data range for data download (e.g, last 5 years)
START_DATE = "2020-01-01"
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')
# Define the number of days into the future we want to predict
FORECAST_OUT = 1 # Predict the next day's closing price

# %%
print(f"Downloading historical data for {TICKER}.........")

# Download the historical stock data using yfinance
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# %%
df.to_csv("berkshire_hathway_data.csv",index=False)

# %%
# Check if the dataframe is empty (e.g, due to a data error or a bad ticker)
if df.empty:
    print("Error: DataFrame is empty. Check ticker or date range")

# %%
#df.reset_index(inplace=True)

# %%
# Display the first few rows of the data
print("----- Initial Head -----")
df.head()

# %%
# Display the data information 
print("----- Data Info -----")
print(df.info())

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("----- Missing Values -----")
print(df_missing)

# %%
# Check for dupliacted rows
df_duplicated = df.duplicated().sum()
print("----- Duplicated Rows -----")
print(df_duplicated)

# %% [markdown]
# Feature Engineering: Creating predictiv variables

# %%
# 1. Target Variable (y): The price we want to predict
# Create a new future column "Future_Close" by shiftinh the "Close" price up by FORECAST_OUT days
# This makes the current row's features predict the closing price 1 day later
df["Future_Close"] = df["Close"].shift(-FORECAST_OUT)

# %%
# 2. Lag Feature: Yesterday's Close price
df["Prev_Close"] = df["Close"].shift(1)

# %%
# 3. Volatility Feature: Difference between High and Low
df["High_Low_Diff"] = df["High"] - df["Low"]

# %%
# 4. Moving Average (Technical Indicator): Simple 50-day rolling mean of the closing price
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# %%
# Drop the first few rows that now contain NaN values due to shifting the rolling calculations
# This ensures we only use complete data points for training
df.dropna(inplace=True)

# %%
#  Select the features matrix (X) and the target variable (y)
# Features selected: Open, High, Low, Volume, Prev_Close, High_Low_Diff, SMA_50
X = df[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'High_Low_Diff', 'SMA_50']]
# Target variable: The actual closing price one day in the future
y = df["Future_Close"]

# %% [markdown]
# Visualization before training

# %%
plt.Figure(figsize=(12,6))
# Plot the actual closing price over time
plt.plot(df.index,df["Close"],label="Actual Close Price")
# Plot the 50-day Simple Moving Average (SMA) as a context feature
plt.plot(df.index,df["SMA_50"],label="50-Day Moving Average",linestyle="--")
plt.title("Berkshire Hathway Historical Price and Technical Indicator (Feature)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show() # Display the show visualization

# %% [markdown]
# Data Splitting

# %%
# Split the data into training and testing sets
# We use a time-series split (shuffling is bad for financial data where order matters)
# To maintain the time order, we use the standard array slicing
test_size=0.2 # 20% of the data will be used for testing 
test_split_index = int(len(X) - (1 - test_size))

# X_train and y_train are the data used to teach the model (earlier period)
X_train = X[:test_split_index]
y_train = y [:test_split_index]

#  X_test and y_test are the data used to evaluate the model (later period)
X_test = X[test_split_index:]
y_test = y[test_split_index:]

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# %% [markdown]
# Data Scaling

# %%
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler only on the training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
# Apply the *same* transformation to the test data
X_test_scaled = scaler.transform(X_test)

# Convert the scaled arrays back to DataFrames for easily handling (optional but helpful)
X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns,index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns,index=X_test.index)

# %% [markdown]
# Model Definition, Hyperparameter Tuning and Training

# %%
# Dictionary to hold all models (Linear Regression has no hyperparameters to tune here)
models = {
    "Linear Regression" : LinearRegression()
}

# Hyperparameter Tuning using GridSearchCV

# Ridge Regression (L2 Regularization)
#  Define the grid of alpha values to search (alpha controls the strength of regularization)
ridge_params = {
    "alpha": [0.1,1.0,10.0,100.0]
}
# Initialize GridSearchCV for Ridge
ridge_gs = GridSearchCV(Ridge(),ridge_params,cv=5,scoring="neg_mean_squared_error")
print("----- Tuning Ridge Regression -----")
# Fit the GridSearchCV object to the scaled training data
ridge_gs.fit(X_train_scaled,y_train)
# Add the best estimator (the tuned model) to our models accuracy
models["Ridge"] = ridge_gs.best_estimator_
print(f"Best Ridge Alpha: {ridge_gs.best_params_["alpha"]}")

# Lasso Regression (L1 Regularization)
# Define the grid of alpha values to search
lasso_params = {
    "alpha":[0.001,0.01,0.1,1.0]
}
# Initialize the GridSearchCV for Lasso
lasso_gs = GridSearchCV(Lasso(max_iter=5000),lasso_params,cv=5,scoring="neg_mean_squared_error")
print("----- Tuning Lasso Regression -----")
# Fit the GridSearchCV object
lasso_gs.fit(X_train_scaled,y_train)
# Add the best estimator to our models dictionary
models["Lasso"] = lasso_gs.best_estimator_
print(f"Best Lasso Alpha: {lasso_gs.best_params_["alpha"]}")

# Support the Vector Regression (SVR)
# Define the grid of parameters to search (C for regularization, gamma for kernel influence)
svr_params = {
    "C":[0.1,1,10],
    "gamma":["scale","auto"]
}
# Initialize the GridSearchCV for SVR (using a Radial Basis Fucntion kernel by default)
svr_gs = GridSearchCV(SVR(kernel="rbf"),svr_params,cv=3,scoring="neg_mean_squared_error")
print("----- Tuning SVR -----")
# Fit the GridSearch object
svr_gs.fit(X_train_scaled,y_train)
# Add the best estimator to our models dictionary
models["SVR"] = svr_gs.best_estimator_
print(f"Best SVR Params: {svr_gs.best_params_}")

# %%
# Training and Initial Prediction (Linear Regression Only)
# Note: Ridge,Lasso and SVR are already trained via GridSearchCV

# Train the standard Linear Regression model (which wasn't tuned)
print("'----- Training Linear Regression -----")
models["Linear Regression"].fit(X_train_scaled,y_train)

# Dictionary to store performance metrics
performance = {}

# %% [markdown]
# Model Evaluation and Comparison

# %%
# Iterate through all trained models to evaluate their performance on the test set
for name, model in models.items():
    # Make predictions on the scaled test data
    y_pred = model.predict(X_test_scaled)

    # Calculate the evaluation metrics
    # Root Mean Squared Error (RMSE): Square root of the average squared differences. Penalty for large errors
    rmse = np.sqrt(mean_absolute_error(y_test,y_pred))
    # Mean Absolute Error (MAE): Average absolute difference. Easier to interpret in dollars
    mae = mean_absolute_error(y_test,y_pred)
    # R-squared (R2): Proportion of the variance in the dependent variable explained by the model
    r2 = r2_score(y_test,y_pred)

    # Store the metrics
    performance[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
  

    # Print the results for comparison
    print(f"{name} Results")
    print(f"R-squared  (R2): {r2:.4f}")
    print(f"Mean Absolute Error(MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")


# Convert the peformance dictionary into a DataFrame for easy viewing and sorting
performance_df = pd.DataFrame(performance).T

# Sort the DataFrame to find the best model based on R2 (higher is better)
best_model_name =performance_df["R2"].max()
best_model = models[best_model_name]



print("=" * 50)
print(f"The Best Performing Model is: {best_model_name}")
print(f"With R2: {performance_df["R2"].max():.4f}")
print("="*50)
print("Comparative Performance Table (Sortedby R2):")
print(performance_df.sort_values(by="R2",ascending=False))

# %% [markdown]
# Visualization After Training

# %%
# Create a DataFrame to hold actual and predicted values for plotting
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': best_model_predictions}, index=X_test.index)

plt.figure(figsize=(14, 7))
# Plot the actual closing prices from the test set
plt.plot(predictions_df.index, predictions_df['Actual'], label='Actual Closing Price (Test Set)', color='blue')
# Plot the predicted closing prices from the best model
plt.plot(predictions_df.index, predictions_df['Predicted'], label=f'{best_model_name} Predictions', color='red', linestyle='--')

plt.title(f'Actual vs. Predicted Stock Price ({TICKER}) - Best Model: {best_model_name}')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show() # Display the post-training visualization


