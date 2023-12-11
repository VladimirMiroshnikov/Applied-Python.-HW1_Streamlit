from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pickle import dump, load
import pandas as pd

DATA = 'data/df_client_agg.csv'


def open_data(path=DATA):
    df = pd.read_csv(path).iloc[:, 2:]

    return df

def split_data(df: pd.DataFrame):
    y = df['TARGET']
    X = df.drop('TARGET', axis = 1)

    return X, y

def fit_and_save_model(X, y, path="data/model.pkl"):
    model = RandomForestClassifier()
    model.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)

    test_prediction = model.predict(X_test)
    accuracy = accuracy_score(test_prediction, y_test)
    print(f"Model accuracy is {accuracy}")

    with open(path, 'wb') as f:
        dump(model, f)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/model.pkl"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]

    prediction_proba = model.predict_proba(df)[0]

    encode_prediction_proba = {
        0: "Вы не склонны откликнуться на предложение банка с вероятностью",
        1: "Вы склонны откликнуться на предложение банка с вероятностью"
    }

    encode_prediction = {
        0: "Вы не откликнитесь на предложение банка",
        1: "Вы не склонны откликнуться на предложение банка"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = split_data(df)
    fit_and_save_model(X_df, y_df)