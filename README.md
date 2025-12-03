# Berkshire-Hathway Stock
![Berkshire-Hathway](/brk-b.jpg)

## Procedures
- Import Libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Data Acquisition
    - Stock data acquired from the yahoo finance api
- Data Loading

| Date       | Price      | Close      | High       | Low        | Open       | Volume   |
|------------|------------|------------|------------|------------|------------|----------|
| 2020-01-02 | 228.389999 | 228.389999 | 226.710007 | 227.509995 | 227.509995 | 3,764,000 |
| 2020-01-03 | 226.179993 | 227.429993 | 225.479996 | 225.690002 | 225.690002 | 3,023,900 |
| 2020-01-06 | 226.990005 | 227.130005 | 224.699997 | 224.990005 | 224.990005 | 4,263,000 |
| 2020-01-07 | 225.919998 | 227.259995 | 225.440002 | 227.259995 | 227.259995 | 3,517,000 |
| 2020-01-08 | 225.990005 | 227.839996 | 225.869995 | 226.029999 | 226.029999 | 3,780,300 |
| ...        | ...        | ...        | ...        | ...        | ...        | ...      |
| 2025-11-25 | 508.570007 | 512.359985 | 506.859985 | 508.929993 | 508.929993 | 4,064,000 |
| 2025-11-26 | 511.230011 | 512.700012 | 507.649994 | 509.250000 | 509.250000 | 3,686,700 |
| 2025-11-28 | 513.809998 | 516.849976 | 511.559998 | 511.890015 | 511.890015 | 2,312,200 |
| 2025-12-01 | 508.549988 | 514.489990 | 508.179993 | 513.650024 | 513.650024 | 3,995,200 |
| 2025-12-02 | 506.649994 | 508.109985 | 503.105011 | 507.589996 | 507.589996 | 3,377,984 |


- Data Preprocessing
    - Check for missing values
    - Check for duplicated rows
- Feature Engineering
    - Features: Open, High, Low, Volume, Prev_Close, High_Low_Diff, SMA_50
- Pre-Training Visualization

![pre-training-visualization.png](/output.png)
- Data Splitting
    - Split the data into training and testing sets
    - We use a time-series split (shuffling is bad for financial data where order matters)
- Data Scaling
    - Initialize the StandardScaler
    - Fit the scaler only on the training data to prevent data leakage
- Model Definition and Training
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Support Vector Regression
- Hyperparameter Tuning
    - GridSearchCV
- Model Comparison
- Model Evaluation
    - Linear Regression Results
        - R-squared (R2): nan 
        - Mean Absolute Error(MAE): $3.22 
        - Root Mean Squared Error (RMSE): $1.79
    - Ridge Results 
        - R-squared (R2): nan 
        - Mean Absolute Error(MAE): $2.51 
        - Root Mean Squared Error (RMSE): $1.58 
    - Lasso Results 
        - R-squared (R2): nan 
        - Mean Absolute Error(MAE): $3.01 
        - Root Mean Squared Error (RMSE): $1.74 
    - SVR Results
        - R-squared (R2): nan 
        - Mean Absolute Error(MAE): $10.49 
        - Root Mean Squared Error (RMSE): $3.24
- Post-Training Visualization


## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
    - Anaconda
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```

## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/berkshire-hathaway-stock.git
cd berkshire-hathaway-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
berkshire-hathaway-stock/
│
├── model.ipynb  
|── model.py    
├── requirements.txt 
├── brk-b.jpg   
|── berkshire_hathaway_data.csv
|── output.png
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── README.md          

```

