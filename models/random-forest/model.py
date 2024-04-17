import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#import matplotlib.pyplot as plt

target_days = 31
occupancy_source = '../../output/cut-data.csv'
output_file = '../../output/latest-random-forest.csv'



# Load CSV, set date as index
data= pd.read_csv(occupancy_source)
data['dates'] = pd.to_datetime(data['dates'])
data.set_index('dates', inplace=True)

# Create features and target variable
data['target'] = data['occupancy'].shift(-target_days)
data.dropna(inplace=True)  # Drop rows with NaN target (due to shifting)

# Extract datetime features
data['day_of_year'] = data.index.dayofyear
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data['year'] = data.index.year

x = data[['day_of_year', 'day_of_week', 'month', 'year']]
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)

latest_date = data.index.max()
future_dates = [latest_date + pd.DateOffset(days=i) for i in range(1, target_days + 1)]
future_features = pd.DataFrame(index=future_dates, columns=['day_of_year', 'day_of_week', 'month', 'year'])

future_features['day_of_year'] = [date.dayofyear for date in future_dates]
future_features['day_of_week'] = [date.dayofweek for date in future_dates]
future_features['month'] = [date.month for date in future_dates]
future_features['year'] = [date.year for date in future_dates]

future_predictions = rf_model.predict(future_features)

for count,index in enumerate(future_features.index):
    print(index, ';', future_predictions[count])


print("The End")

