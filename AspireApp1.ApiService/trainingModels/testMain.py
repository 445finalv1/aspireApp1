import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings
import concurrent.futures
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import shap
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

# Configure root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Constants
COMPANY_LAGS = 5
ECON_LAGS = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Load the dataset
def load_data(file_path):
    logging.info("Loading data...")
    df = pd.read_csv(file_path)
    return df


# Preprocess the data
def preprocess_data(df, company_name, company_columns, lags=COMPANY_LAGS):
    logging.info(f"Preprocessing data for {company_name}...")

    # Create a copy to avoid modifying the original DataFrame in parallel processes
    df = df.copy()

    # Infer better dtypes for object columns
    df.infer_objects(copy=False)

    # Convert all columns to numeric, coercing errors to NaN
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Handle missing values
    df[numeric_columns] = df[numeric_columns].interpolate(method='linear', inplace=False)

    # Initialize a dictionary to hold new features
    new_features = {}

    # Create lag features for the target company
    for lag in range(1, lags + 1):
        new_features[f'{company_name}_lag_{lag}'] = df[company_name].shift(lag)

    # Create lag features for economic indicators
    economic_indicators = df.columns[-4:]
    for econ in economic_indicators:
        for lag in range(1, ECON_LAGS + 1):
            new_features[f'{econ}_lag_{lag}'] = df[econ].shift(lag)

    # Add technical indicators (e.g., moving averages)
    new_features[f'{company_name}_MA_5'] = df[company_name].rolling(window=5).mean()
    new_features[f'{company_name}_MA_10'] = df[company_name].rolling(window=10).mean()

    # Add lagged features of related companies (e.g., the next 5 companies)
    related_companies = [comp for comp in company_columns if comp != company_name][:5]
    for comp in related_companies:
        if comp in df.columns:
            new_features[f'{comp}_lag_1'] = df[comp].shift(1)
        else:
            logging.warning(f"Related company '{comp}' not found in DataFrame columns.")

    # Convert the dictionary to a DataFrame
    new_features_df = pd.DataFrame(new_features)

    # Concatenate all new features to the original DataFrame in one operation
    df = pd.concat([df, new_features_df], axis=1)

    # Combine features and target
    target = df[company_name]
    data = pd.concat([new_features_df, target], axis=1).dropna()
    features = data.drop(company_name, axis=1)
    target = data[company_name]

    return features, target


# Feature selection
def feature_selection(X_train, y_train):
    logging.info("Selecting important features...")
    selector = RFE(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), n_features_to_select=10)
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.support_]
    return selected_features


# Time series cross-validation
def time_series_cv(model, X, y):
    logging.info("Performing time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)
        mse_cv = mean_squared_error(y_test_cv, y_pred_cv)
        scores.append(mse_cv)
    average_score = np.mean(scores)
    logging.info(f"Average CV Mean Squared Error: {average_score}")
    return average_score


# Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    logging.info("Tuning hyperparameters...")
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20]
    }
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=RANDOM_STATE))
    ])
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


# Build and train LSTM model
def build_lstm_model(X_train, y_train, X_test):
    logging.info("Building LSTM model...")
    # Reshape data for LSTM
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(X_test_lstm)
    return y_pred, model


# Model evaluation and visualization
def evaluate_model(y_test, y_pred, model, X_test, company_name):
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Model Mean Squared Error for {company_name}: {mse}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'Stock Price Prediction for {company_name}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f"{company_name}_prediction_plot.png")  # Save plot instead of showing
    plt.close()

    # Feature importance (for tree-based models)
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        plt.title(f"Feature Importances for {company_name}")
        plt.bar(range(len(importances)), importances[indices], color="r", align="center")
        plt.xticks(range(len(importances)), [X_test.columns[i] for i in indices], rotation=90)
        plt.xlim([-1, len(importances)])
        plt.tight_layout()
        plt.savefig(f"{company_name}_feature_importances.png")  # Save plot instead of showing
        plt.close()

    # SHAP values
    logging.info("Calculating SHAP values...")
    try:
        explainer = shap.Explainer(model.named_steps['model'], X_test)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(f"{company_name}_shap_summary.png")  # Save plot instead of showing
        plt.close()
    except Exception as e:
        logging.error(f"SHAP analysis failed for {company_name}: {e}")


# Export model to ONNX
def export_model_onnx(pipeline, X_train, model_name):
    logging.info(f"Exporting model to ONNX format as {model_name}...")
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = skl2onnx.convert_sklearn(pipeline, initial_types=initial_type)
    with open(model_name, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logging.info(f"Model exported to {model_name}")


def process_company(company_name, df, company_columns, economic_indicators):
    # Configure logging for each process
    logger = logging.getLogger(company_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{company_name}.log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        logger.info(f"Processing company: {company_name}")

        # Preprocess data
        features, target = preprocess_data(df, company_name, company_columns)

        # Skip if insufficient data after preprocessing
        if len(features) < 10:
            logger.warning(f"Not enough data for {company_name}, skipping...")
            return (company_name, [])  # Return empty list

        # Split data
        split_point = int(len(features) * (1 - TEST_SIZE))
        X_train, X_test = features.iloc[:split_point], features.iloc[split_point:]
        y_train, y_test = target.iloc[:split_point], target.iloc[split_point:]

        # Feature selection
        selected_features = feature_selection(X_train, y_train)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        # Save selected features to individual CSV
        selected_features_df = pd.DataFrame(selected_features, columns=['Selected_Features'])
        selected_features_df.to_csv(f"{company_name}_selected_features.csv", index=False)
        logger.info(f"Selected features saved to '{company_name}_selected_features.csv'")

        # Build pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection',
             RFE(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), n_features_to_select=10)),
            ('model', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))
        ])

        # Time series cross-validation
        time_series_cv(pipeline, X_train, y_train)

        # Hyperparameter tuning
        best_pipeline = hyperparameter_tuning(X_train, y_train)

        # Train final model
        logger.info(f"Training final model for {company_name}...")
        best_pipeline.fit(X_train, y_train)
        y_pred_rf = best_pipeline.predict(X_test)

        # Evaluate model
        evaluate_model(y_test, y_pred_rf, best_pipeline, X_test, company_name)

        # Export model to ONNX
        onnx_model_name = f"{company_name}_model.onnx"
        export_model_onnx(best_pipeline, X_train, onnx_model_name)

        # Build and evaluate LSTM model
        y_pred_lstm, lstm_model = build_lstm_model(X_train, y_train, X_test)
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        logger.info(f"LSTM Model Mean Squared Error for {company_name}: {mse_lstm}")

        # Save LSTM model
        lstm_model_name = f"{company_name}_lstm_model.keras"
        lstm_model.save(lstm_model_name)
        logger.info(f"LSTM model saved as '{lstm_model_name}'")

        # Optionally close plots if any remain
        plt.close('all')

    except Exception as e:
        logger.error(f"Error processing {company_name}: {e}")

    finally:
        # Remove handlers to prevent duplicate logs in subsequent runs
        logger.removeHandler(handler)
        handler.close()

    return (company_name, selected_features.tolist())  # Convert Index to list


def main():
    # Load data
    df = load_data('Final_Combined.csv')

    # Exclude 'Date' and select company columns
    company_columns = df.columns[1:-4]  # Exclude 'Date' and last four columns
    economic_indicators = df.columns[-4:]

    # Define the number of workers; adjust based on your system
    max_workers = min(4, len(company_columns))  # Example: 4 workers

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_company, company, df, company_columns, economic_indicators)
            for company in company_columns
        ]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                company_name, selected_features = future.result()
                if selected_features:
                    logging.info(f"Company '{company_name}' selected features saved.")
                else:
                    logging.info(f"Company '{company_name}' has no selected features.")
            except Exception as exc:
                logging.error(f'Generated an exception: {exc}')

    # Removed the export_selected_features function and related logic
    # since each company's selected features are now saved individually.


if __name__ == '__main__':
    main()