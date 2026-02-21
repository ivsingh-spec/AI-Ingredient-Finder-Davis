"""
model_training.py

This file is responsible for training and evaluating the ML model


- Loads the prepared dataset 
- Defines the prediction target (race outcome for a single race)
- Splits the data using time-aware validation
- Trains a baseline model
- Trains an improved model
- Evaluates model performance using appropriate metrics
- Saves the trained model for future predictions

The goal of this file is to produce a trained model
that can generalize to unseen races.
"""
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from merge_tables import merge_tables


def load_data():
    """Load and merge raw F1 data to create the dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "F1Data")
    
    print("Loading raw F1 data...")
    dataframes = {
        "results": pd.read_csv(os.path.join(data_dir, "results.csv")),
        "races": pd.read_csv(os.path.join(data_dir, "races.csv")),
        "drivers": pd.read_csv(os.path.join(data_dir, "drivers.csv")),
        "constructors": pd.read_csv(os.path.join(data_dir, "constructors.csv")),
        "qualifying": pd.read_csv(os.path.join(data_dir, "qualifying.csv")),
        "status": pd.read_csv(os.path.join(data_dir, "status.csv")),
    }
    
    print("Merging tables...")
    df = merge_tables(dataframes)
    return df


def engineer_features(df):
    """
    Engineer features from historical performance.
    Must be done BEFORE splitting.
    """
    df = df.copy()
    
    # Previous race finish position
    df["prev_finish"] = df.groupby("driverid")["finish_position"].shift(1)
    
    # Rolling average finish position (last 5 races)
    df["rolling_finish_5"] = (
        df.groupby("driverid")["finish_position"]
          .rolling(5)
          .mean()
          .reset_index(0, drop=True)
    )
    
    # Constructor rolling average finish position
    df["constructor_avg_finish"] = (
        df.groupby("constructorid")["finish_position"]
          .rolling(5)
          .mean()
          .reset_index(0, drop=True)
    )
    
    # Constructor rolling average points
    df["constructor_avg_points"] = (
        df.groupby("constructorid")["points"]
          .rolling(5)
          .mean()
          .reset_index(0, drop=True)
    )
    
    # Driver experience (number of races)
    df["driver_races"] = df.groupby("driverid").cumcount()
    
    # Driver rolling average points (captures driver skill)
    df["driver_avg_points"] = (
        df.groupby("driverid")["points"]
          .rolling(5)
          .mean()
          .reset_index(0, drop=True)
    )
    
    # Grid minus qualifying (shows qualifying performance)
    df["grid_quali_diff"] = df["grid"] - df["quali_position"]
    
    # Drop rows with NaN values from feature engineering
    df = df.dropna()
    
    return df


def prepare_features(df):
    """
    Select features and target.
    Features must already be engineered.
    """

    # Target variable
    y = df["finish_position"]

    # Pre-race features
    feature_cols = [
        "grid",
        "quali_position",
        "prev_finish",
        "rolling_finish_5",
        "constructor_avg_finish",
        "constructor_avg_points",
        "driver_races",
        "driver_avg_points",
        "grid_quali_diff"
    ]

    X = df[feature_cols]

    return X, y


def time_based_split(df, X, y, split_year=2019):
    """
    Split dataset based on year to prevent data leakage.
    """
    train_df = df[df["year"] < split_year]
    test_df = df[df["year"] >= split_year]

    X_train = train_df[X.columns]
    X_test = test_df[X.columns]
    y_train = train_df["finish_position"]
    y_test = test_df["finish_position"]

    return X_train, X_test, y_train, y_test


def train_improved_model(X_train, y_train):
    """
    Train Random Forest model with optimized hyperparameters.
    """
    model = RandomForestRegressor(
        n_estimators=300,      # Increase from 200
        max_depth=15,          # Increase from 10
        min_samples_split=5,   # NEW: Prevent overfitting
        min_samples_leaf=2,    # NEW: Prevent overfitting
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """
    # Get predictions (keep full precision internally)
    y_pred = model.predict(X_test)
    
    # Round for presentation (clamp to valid F1 range 1-20)
    y_pred_rounded = [max(1, min(20, int(round(pred)))) for pred in y_pred]
    
    # Calculate metrics on raw predictions (for statistical accuracy)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    # Calculate metrics on rounded predictions (for practical accuracy)
    mae_rounded = mean_absolute_error(y_test, y_pred_rounded)
    
    print(f"Mean Absolute Error (RAW): {mae:.4f}")
    print(f"Mean Absolute Error (ROUNDED): {mae_rounded:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return y_pred_rounded  # or y_pred if you prefer raw predictions


def assign_unique_positions(predictions):
    """
    Convert regression predictions to unique F1 positions (1-20).
    Each position can only be assigned once.
    """
    # Create list of (index, raw_prediction) tuples
    pred_with_idx = [(i, pred) for i, pred in enumerate(predictions)]
    
    # Sort by prediction value (descending - best drivers first)
    pred_with_idx.sort(key=lambda x: x[1], reverse=True)
    
    # Assign unique positions
    final_positions = [0] * len(predictions)
    assigned_positions = set()
    
    for idx, (original_idx, pred) in enumerate(pred_with_idx):
        # Find nearest available position (lower is better in F1)
        position = max(1, min(20, int(round(pred))))
        
        # If position already taken, find nearest available
        while position in assigned_positions and position < 20:
            position += 1
        
        if position not in assigned_positions and position <= 20:
            final_positions[original_idx] = position
            assigned_positions.add(position)
    
    return final_positions


def main():

    # Load data
    df = load_data()

    # Engineer features FIRST
    df = engineer_features(df)

    # Then prepare features
    X, y = prepare_features(df)

    # Then split
    X_train, X_test, y_train, y_test = time_based_split(df, X, y, split_year=2020)

    print("Training size:", X_train.shape)
    print("Test size:", X_test.shape)

    # ---- Improved Model ----
    print("\n\nTraining Improved Model...")
    improved_model = train_improved_model(X_train, y_train)
    print("Improved Model Results:")
    evaluate_model(improved_model, X_test, y_test)

    # ---- Save Model ----
    print("\n\nSaving improved model...")
    joblib.dump(improved_model, "f1_trained_model.pkl")
    print("Model saved as 'f1_trained_model.pkl'")


if __name__ == "__main__":
    main()
