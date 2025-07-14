import pandas as pd
import numpy as np

class MyLinearRegression:
    def __init__(self):
        self.beta_matrix = []
    def SteepestDescent(self, m_now, b_now, points, learning_rate):
        m_gradient = 0
        b_gradient = 0
        n = len(points)
        for i in range(len(points)):
            x = points.iloc[i].arg
            y = points.iloc[i].val
            m_gradient += -(2/n) * (y - m_now * x - b_now) * x
            b_gradient += -(2/n) * (y - m_now * x - b_now) 
        m_new = m_now - m_gradient * learning_rate
        b_new = m_now - b_gradient * learning_rate
        self.m_new, self.b_new = m_new, b_new

    def NormalEquations(self, independent_data, dependant_data):
        X = np.c_[(np.ones(len(independent_data)), independent_data.to_numpy())]
        y_raw = dependant_data.to_numpy()
        y = y_raw[np.newaxis].T

        beta = (np.linalg.inv(X.T @ X)) @ X.T @ y
        self.beta_matrix = beta
        # print(self.beta_matrix)
    
    def Fit(self, independent_data, dependent_data):
        self.NormalEquations(independent_data, dependent_data)

    def Predict(self, test_features):
        X_hat = np.c_[np.ones(len(test_features)), test_features.to_numpy()]
        print(X_hat)
        return np.dot(X_hat, self.beta_matrix)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    import time
    my_model = MyLinearRegression()
    df = pd.read_csv("../data/MultipleLR_data.csv", sep=";")
    X = df.drop("Satisfaction", axis=1)    
    y = df.Satisfaction
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    
    start_time = time.time()
    my_model.Fit(X_train, y_train)
    y_pred = my_model.Predict(X_test)
    end_time = time.time()
    my_mse = mean_squared_error(y_test, y_pred)
    my_r2 = r2_score(y_test, y_pred)

    print(f"My model: mse = {my_mse}, r2 = {my_r2}, time = {end_time - start_time}")
    plt.scatter(range(0, len(y_test)), y_pred, color = "blue")

    start_time = time.time()
    model = LinearRegression()
    # train the model
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)
    end_time = time.time()
    # calculate mean squarred error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Sklearn model: mse = {mse}, r2 = {r2}, time = {end_time - start_time}')

    plt.scatter(range(0, len(y_test)), y_test, color = "red")
    plt.show()
    # calculate r-squared score 

