from sklearn.model_selection import train_test_split
import lightgbm as lgb
from preprocess import load_data, preprocess_data

def train():
    df = load_data("data/train.csv")
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test
