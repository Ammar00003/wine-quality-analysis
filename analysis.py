import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns

# Load datasets
red = pd.read_csv("wine quality data/winequality-red.csv", sep=';')
white = pd.read_csv("wine quality data/winequality-white.csv", sep=';')

def evaluate_model(X, y, model, name="Model"):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{name} Results:")
    print(f"R² Score: {r2:.3f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Plot: True vs Predicted
    plt.figure(figsize=(6, 4))
    
    # Histplot with labels for the key
    sns.histplot(x=y_test, y=y_pred, label=f'{name} Predicted vs True', color='green')
    
    # Labels, title, and legend
    plt.xlabel('True Quality')
    plt.ylabel('Predicted Quality')
    plt.title(f'{name} - True vs Predicted')
    plt.tight_layout()
    plt.savefig(f"graphs/{name} quality comparison.png")

    return r2, mae, rmse

# Prepare features and targets
X_red = red.drop('quality', axis=1)
y_red = red['quality']

X_white = white.drop('quality', axis=1)
y_white = white['quality']

# --- Linear Regression ---
linreg = LinearRegression()
evaluate_model(X_red, y_red, linreg, "Linear Regression - Red Wine")
evaluate_model(X_white, y_white, linreg, "Linear Regression - White Wine")

# --- Random Forest Regressor ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model(X_red, y_red, rf, "Random Forest - Red Wine")
evaluate_model(X_white, y_white, rf, "Random Forest - White Wine")

# --- Gradient Boosting Regressor ---
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.3, max_depth=3, random_state=42)
evaluate_model(X_red, y_red, gbr, "Gradient Boosting - Red Wine")
evaluate_model(X_white, y_white, gbr, "Gradient Boosting - White Wine")

# --- SVR Regressor ---
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, epsilon=0.2))
evaluate_model(X_red, y_red, svr, "SVR - Red Wine")
evaluate_model(X_white, y_white, svr, "SVR - White Wine")

# --- KNN Regressor ---
knn = KNeighborsRegressor(n_neighbors=3)
evaluate_model(X_red, y_red, knn, "KNN - Red Wine")
evaluate_model(X_white, y_white, knn, "KNN - White Wine")