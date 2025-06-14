import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

def create_ts_data(df, window_size=10):
    window_size = 5
    count = 1

    while count < window_size:
        df[f'co2_{count}'] = df['co2'].shift(-count)
        count += 1
    # 'target': predict column
    df['target'] = df['co2'].shift(-count)
    df = df.dropna(axis=0)
    return df

def build_model():
    reg_model = Pipeline(steps=[
        ('preprocessor',StandardScaler()),
        # ('model', RandomForestRegressor())
        ('model', LinearRegression())

    ])
    return reg_model

def split_data(data, train_ratio=0.8):
    # Split data
    X = data.drop(['time', 'target'], axis=1)
    y = data['target']

    # Split into training & testing
    train_ratio = 0.8
    num_of_samples = X.shape[0]

    training_size = int(num_of_samples * train_ratio)

    # X_train = X.iloc[:X_train_size,:]
    X_train = X[:training_size]
    X_test = X[training_size:]
    y_train = y[:training_size]
    y_test = y[training_size:]
    return X_train, X_test, y_train, y_test, training_size

def evaluate(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE: ",mse)
    print("MSE: ",mae)
    print("MSE: ",r2)


def main():
    df = pd.read_csv("../datasets/Time-series-datasets/co2.csv",sep=',')
    # Handle data type and missing values
    df['time'] = pd.to_datetime(df['time'])
    df['co2'] = df['co2'].interpolate()

    # Create time-series data
    window_size = 5
    df = create_ts_data(df, window_size)

    # Split data
    X_train, X_test, y_train, y_test, training_size = split_data(df)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    # Build the model
    model = build_model()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    evaluate(y_test, y_pred)

    # Visualize
    fg, ax = plt.subplots()

    ax.set_title("Co2 trending prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Co2")
    ax.plot(df['time'][:training_size], y_train, label='train')
    ax.plot(df['time'][training_size:], y_test, label='test')
    ax.plot(df['time'][training_size:], y_pred,label='prediction')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()