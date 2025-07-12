import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

url = "https://zenodo.org/records/4770937/files/weather_prediction_dataset.csv"
df = pd.read_csv(url)

df = df.dropna()
# on-hot encoding 
df = pd.get_dummies(df, drop_first=True)

# feature selection

X = df.drop('BASEL_temp_mean', axis = 1)
y = df['BASEL_temp_mean']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

# train the model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# calculate mean squarred error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# calculate r-squared score 
r2 = r2_score(y_test, y_pred)
print(f'R-Squared Score {r2}')

# visualise the results 
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Weather Conditions')
plt.show()