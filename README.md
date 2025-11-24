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
- Data Acquisition and Loading
    - Stock data acquired from the yahoo finance api
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

