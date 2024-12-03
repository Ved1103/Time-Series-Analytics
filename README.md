# Time-Series-Analytics

## Overview
This project involves analyzing and forecasting monthly sales data from January 2008 to September 2013. The analysis utilizes advanced time series modeling techniques, such as ETS (Error-Trend-Seasonality) models and Seasonal ARIMA, to explore trends, seasonality, and error components in the dataset. The goal is to develop accurate models to forecast future sales and evaluate their performance using various error metrics.

## Dataset
The dataset contains Monthly Sales records spanning:
- **Time Period**: January 2008 - September 2013
 - **Frequency**: Monthly
   
**Key Preprocessing Steps:**

- Time Indexing: The Month column is converted to a datetime index for analysis.
- Data Splitting:
    - Training set: January 2008 - May 2013
    -  Testing set: June 2013 - September 2013

## Methodology
The project begins with a thorough exploration of the dataset, including visualizing the time series to identify trends, seasonality, and irregularities. Decomposition techniques are applied to separate the series into its components: trend, seasonality, and error, revealing an overall upward trajectory with recurring patterns of varying magnitudes. For model building, an ETS (Error-Trend-Seasonality) model with multiplicative error, additive trend, and multiplicative seasonality was employed, leveraging its ability to handle fluctuating seasonal effects. The model's accuracy was assessed using metrics such as RMSE, MAE, MAPE, and MASE to ensure robust performance. Additionally, a Seasonal ARIMA model was developed after achieving stationarity through seasonal differencing, verified using the Augmented Dickey-Fuller test. ACF and PACF plots guided the selection of optimal parameters for the ARIMA model, ensuring it effectively captured the temporal dynamics of the data.

## Evaluation
``` python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate RMSE
rmse = mean_squared_error(actual_values, predicted_values, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate MAE
mae = mean_absolute_error(actual_values, predicted_values)
print("Mean Absolute Error (MAE):", mae)

# Calculate MAPE
mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
print("Mean Absolute Percentage Error (MAPE):", mape)

# Calculate MASE
def calculate_mase(actual, predicted, naive_forecast):
    naive_mae = np.mean(np.abs(np.diff(naive_forecast)))
    mase = np.mean(np.abs(actual - predicted)) / naive_mae
    return mase

mase = calculate_mase(actual_values, predicted_values, naive_forecast=actual_values[:-1])
print("Mean Absolute Scaled Error (MASE):", mase)
```

## Conclusion

In conclusion, this project successfully implements a robust model that leverages advanced techniques in AI for data analysis and prediction. By using rigorous preprocessing and hyperparameter tuning, the model ensures high accuracy and performance. The evaluation metrics, such as RMSE, MAE, MAPE, and MASE, confirm the model's reliability and effectiveness in real-world scenarios. Through continuous improvements and iterative refinements, this project lays the foundation for future enhancements, ensuring that it can adapt to various use cases and datasets. The results demonstrate that the model is capable of delivering meaningful insights, contributing to more informed decision-making processes.

