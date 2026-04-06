import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df["Heart Disease"] = df["Heart Disease"].map({"Presence": 1, "Absence": 0})
    
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    return X, y
