# Forecasting-with-Python

import pandas as pd
import matplotlib.pyplot as plt

# Read Excel file
df = pd.read_excel(r'C:\Users\vaish\Downloads\Amazon_Sales.xlsx', engine='openpyxl')
print(df.head())
 


import numpy as np
# We need to see if we have any missing data
# Let's loop through the data and see if there is anything missing

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
print(df.dtypes)
print("Duplicated Rows")
print(df[df.duplicated])
category_sales=df.groupby('Category')['Sales'].sum().reset_index()
sns.barplot(data = category_sales,x='Category',y='Sales')
plt.title("Total Sales By category")
plt.xlabel("Category")
plt.xticks(rotation=45)
plt.ylabel("Sales")
plt.show()

# Convert Order Date to datetime if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Set the 'Order Date' as the index for time series analysis
df.set_index('Order Date', inplace=True)

# Resample by month and calculate total profit per month
monthly_profit = df['Profit'].resample('M').sum()

# Plot the monthly profit
import matplotlib.pyplot as plt

monthly_profit.plot(title='Monthly Profit Analysis')
plt.ylabel('Total Profit')
plt.show()

# Convert 'Order Date' to datetime if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Set the 'Order Date' as the index for time series analysis
df.set_index('Order Date', inplace=True)

# Now, resample by day and calculate total sales per day
sts = df['Sales'].resample('D').sum()

# Split data into train and test sets (last 30 days for testing)
train = sts[:-30]
test = sts[-30:]

# Fit the ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # p=5, d=1, q=0
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=30)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(test.index, predictions, label='Forecasted Sales', color='red')
plt.title('Sales Forecasting using ARIMA')
plt.legend()
plt.show()
plt.show()
