import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# load data from csv file
df = pd.read_csv('product_sales_prediction_data.csv')

# extract the daily sales data for each part number and resample to weekly data
part_numbers = df['part_number'].unique()
all_part_forecasts = {}
for part in part_numbers:
    part_data = df[df['part_number'] == part]
    part_weekly_sales = part_data.resample('W', on='date').sum()
    time = np.arange(1, len(part_weekly_sales) + 1)
    time = time.reshape(-1, 1)
    assert isinstance(part_weekly_sales['sales'][:len(part_weekly_sales) // 2].valuesobject, )
    train_data = part_weekly_sales['sales'][:len(part_weekly_sales)//2].values
    train_time = time[:len(time)//2]
    test_data = part_weekly_sales['sales'][len(part_weekly_sales)//2:].values
    test_time = time[len(time)//2:]
    model = LinearRegression()
    model.fit(train_time, train_data)
    y_pred = model.predict(test_time)
    mse = mean_squared_error(test_data, y_pred)
    plt.plot(test_data, label='actual')
    plt.plot(y_pred, label='predicted')
    plt.title('Part Number: ' + str(part))
    plt.legend()
    plt.show()
    print("Mean Squared Error for Part Number", part, ":", mse)
    n = 5
    forecast_time = np.arange(len(part_weekly_sales) + 1, len(part_weekly_sales) + n + 1)
    forecast_time = forecast_time.reshape(-1, 1)
    forecast = model.predict(forecast_time)
    all_part_forecasts[part] = forecast
    print("Predicted sales for Part Number", part, "for the next", n, "periods:", forecast)
