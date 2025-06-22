import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import DataPreprocessor from the sibling module
from data_preprocessing_module import DataPreprocessor

# --- Module 3: Cyclic Multidimensional Gray Model (NSGM(1,N)) Module ---

class NSGM1NModel:
    def __init__(self):
        self.model_params = None

    def _ago(self, data):
        # Accumulated Generating Operation (AGO)
        return np.cumsum(data, axis=0)

    def _iago(self, data_ago):
        # Inverse Accumulated Generating Operation (IAGO)
        # The first element remains the same, subsequent elements are differences
        data = np.zeros_like(data_ago)
        data[0] = data_ago[0]
        data[1:] = data_ago[1:] - data_ago[:-1]
        return data

    def train(self, X_train, y_train):
        # X_train: (n_samples, n_features) - input features (related series)
        # y_train: (n_samples,) - target variable (primary series)

        # Combine y_train and X_train to form the multivariate series
        # The first column is the primary series (y_train), rest are related series (X_train)
        y_train_reshaped = y_train.reshape(-1, 1)
        data_combined = np.hstack((y_train_reshaped, X_train))

        n_samples, n_vars = data_combined.shape

        # Step 1: Perform AGO on all series
        data_ago = self._ago(data_combined)

        # Step 2: Construct the B matrix and Y vector for parameter estimation
        # Primary series (y1_ago) is the first column of data_ago
        y1_ago = data_ago[:, 0]

        # Z1 for the 'a' coefficient (average of adjacent AGO values of primary series)
        Z1 = 0.5 * (y1_ago[1:] + y1_ago[:-1])

        # Y vector: -X1(0)(k) (negative of the original differenced primary series)
        # The equation is X1(0)(k) + a*Z1(k) = sum(bi*Xi(1)(k))
        # So, Y = X1(0)(k) and B = [-Z1(k), Xi(1)(k)]
        # Or, for least squares: Y = X1(0)(k) and B = [-Z1(k), X2(1)(k), ..., XN(1)(k)]
        # Let's use the form: Y = X1(0)(k) and B = [-Z1(k), X2(1)(k), ..., XN(1)(k)]

        Y = data_combined[1:, 0].reshape(-1, 1) # Original differenced primary series

        # B matrix: [-Z1(k), X2(1)(k), X3(1)(k), ...]
        B = np.zeros((n_samples - 1, n_vars))
        B[:, 0] = -Z1 # Coefficient for 'a'
        B[:, 1:] = data_ago[1:, 1:] # Coefficients for 'b's (related series AGO values)

        # Step 3: Estimate parameters (a, b2, ..., bN) using least squares
        # beta = (B.T * B)^-1 * B.T * Y
        try:
            # Add a small regularization term to B.T @ B to prevent singularity
            lambda_reg = 1e-6
            beta = np.linalg.inv(B.T @ B + lambda_reg * np.eye(B.shape[1])) @ B.T @ Y
        except np.linalg.LinAlgError:
            print("Singular matrix encountered in NSGM(1,N) parameter estimation. Returning None.")
            self.model_params = None
            return

        self.model_params = beta.flatten()
        print("NSGM(1,N) model training complete.")

    def predict(self, X_test_sequence):
        # X_test_sequence: (timesteps, n_features) - last `timesteps` of data, including primary series.
        # The primary series is assumed to be the first column (index 0).
        # This method predicts the next value of the primary series.

        if self.model_params is None:
            raise ValueError("Model not trained. Call train() first.")

        a = self.model_params[0]
        b_coeffs = self.model_params[1:] # Coefficients for related series

        # Perform AGO on the entire input sequence to get the AGO values up to the current point.
        # This allows the model to be self-contained for prediction given a historical window.
        ago_sequence = self._ago(X_test_sequence)

        # Get the last AGO values from the sequence for prediction
        primary_series_ago_current = ago_sequence[-1, 0]
        related_series_ago_current = ago_sequence[-1, 1:]

        # Calculate the sum(bi*Xi(1)(k)) for related series
        # Ensure b_coeffs and related_series_ago_current have compatible shapes
        if len(b_coeffs) != len(related_series_ago_current):
            raise ValueError("Number of b coefficients does not match number of related series.")

        sum_b_xi = np.sum(b_coeffs * related_series_ago_current)

        # Predict the next AGO value of the primary series (X1(1)(k+1))
        # X1(1)(k+1) = (X1(1)(k) - (sum(bi*Xi(1)(k)) / a)) * exp(-a) + (sum(bi*Xi(1)(k)) / a)
        # Note: if 'a' is very close to 0, this formula can be unstable. Add a small epsilon.
        epsilon = 1e-9
        if abs(a) < epsilon:
            # Handle case where 'a' is close to zero (linear approximation)
            predicted_ago_primary = primary_series_ago_current + sum_b_xi
        else:
            predicted_ago_primary = (primary_series_ago_current - (sum_b_xi / a)) * np.exp(-a) + (sum_b_xi / a)

        # Inverse AGO to get the predicted actual value (X1(0)(k+1))
        # X1(0)(k+1) = X1(1)(k+1) - X1(1)(k)
        # This requires the actual AGO value at time k, which is `primary_series_ago_current`.
        predicted_actual_primary = predicted_ago_primary - primary_series_ago_current

        return predicted_actual_primary

# Example Usage (for testing the module)
if __name__ == '__main__':
    print("\n--- NSGM1NModel Module Test ---")

    # 1. Get preprocessed data
    # Using AAPL for 2 years of data.
    data_preprocessor = DataPreprocessor(stock_ticker='AAPL', years_of_data=2, random_seed=42)
    processed_df, data_scaler = data_preprocessor.preprocess()

    if processed_df is None or processed_df.empty:
        print("Failed to preprocess data. Aborting NSGM1NModel test.")
    else:
        print(f"Shape of preprocessed_df: {processed_df.shape}")

        # NSGM(1,N) expects the primary series (target) as the first column for its internal logic.
        # The DataPreprocessor currently puts the target ('Close_ticker') as the last column.
        # We need to reorder processed_df for NSGM.
        target_column_name = processed_df.columns[-1] # e.g., 'Close_AAPL'

        # Create a new DataFrame with target as the first column
        cols = [target_column_name] + [col for col in processed_df.columns if col != target_column_name]
        nsgm_input_df = processed_df[cols]
        print(f"Reordered columns for NSGM input (target '{target_column_name}' is first): {nsgm_input_df.columns.tolist()}")

        # Split data: 80% for training, 20% for testing NSGM predictions step-by-step
        # NSGM is typically trained on a contiguous block of historical data.
        train_size = int(len(nsgm_input_df) * 0.8)
        train_df_nsgm = nsgm_input_df.iloc[:train_size]
        test_df_nsgm = nsgm_input_df.iloc[train_size:]

        print(f"train_df_nsgm shape: {train_df_nsgm.shape}")
        print(f"test_df_nsgm shape: {test_df_nsgm.shape}")

        if train_df_nsgm.shape[0] < 2 or test_df_nsgm.shape[0] == 0 : # NSGM needs at least 2 samples for training.
            print("Not enough data for NSGM training/testing after split. Aborting.")
        else:
            # Prepare data for NSGM training:
            # y_train is the primary series (first column of train_df_nsgm)
            # X_train are the related series (other columns of train_df_nsgm)
            nsgm_y_train = train_df_nsgm.iloc[:, 0].values
            nsgm_X_train = train_df_nsgm.iloc[:, 1:].values

            nsgm_model = NSGM1NModel()
            print("\nTraining NSGM(1,N) model...")
            nsgm_model.train(nsgm_X_train, nsgm_y_train)

            if nsgm_model.model_params is None:
                print("NSGM model training failed (model_params is None). Aborting further tests.")
            else:
                print(f"NSGM model trained. Parameters: {nsgm_model.model_params}")

                # Test prediction step-by-step on the test_df_nsgm
                # For NSGM, prediction is typically one step ahead using a rolling window of historical data.
                # The `predict` method expects a sequence of `look_back` length.
                # For this test, we'll use a fixed look_back similar to LSTM, e.g., 60.
                # However, NSGM's `predict` method itself takes the *current* sequence leading up to prediction point.
                # The training data for NSGM parameters is train_df_nsgm.
                # To predict for test_df_nsgm, we need to provide sequences from nsgm_input_df.

                look_back_nsgm = 60 # A window size for prediction inputs
                nsgm_predictions_scaled = []
                nsgm_actuals_scaled = []

                print(f"\nMaking step-by-step predictions on test data (look_back={look_back_nsgm})...")
                # We need to ensure that for each prediction point in test_df_nsgm,
                # we can form a sequence of `look_back_nsgm` points ending just before it.
                # The input to nsgm_model.predict should be from nsgm_input_df (which has target as first col).

                # Start predicting from the first point in test_df_nsgm.
                # The sequence for the first test point comes from data ending at train_size -1.
                for i in range(len(test_df_nsgm)):
                    # Index in the original nsgm_input_df for the end of the current sequence
                    current_sequence_end_idx = train_size + i -1
                    # Index for the start of the sequence
                    current_sequence_start_idx = current_sequence_end_idx - look_back_nsgm + 1

                    if current_sequence_start_idx < 0:
                        # print(f"Skipping prediction for test point {i} due to insufficient history for look_back.")
                        continue

                    test_sequence = nsgm_input_df.iloc[current_sequence_start_idx : current_sequence_end_idx + 1].values

                    if test_sequence.shape[0] < look_back_nsgm:
                        # This case should ideally be caught by current_sequence_start_idx < 0,
                        # but as a safeguard if data has gaps.
                        # print(f"Skipping prediction for test point {i}, sequence length {test_sequence.shape[0]} is less than look_back {look_back_nsgm}.")
                        continue

                    predicted_val = nsgm_model.predict(test_sequence)
                    actual_val = test_df_nsgm.iloc[i, 0] # Actual value is the first column (target) of current test point

                    nsgm_predictions_scaled.append(predicted_val)
                    nsgm_actuals_scaled.append(actual_val)

                if not nsgm_predictions_scaled:
                    print("No predictions were made by NSGM model. Check data length and look_back setting.")
                else:
                    nsgm_predictions_scaled = np.array(nsgm_predictions_scaled)
                    nsgm_actuals_scaled = np.array(nsgm_actuals_scaled)

                    mse_nsgm_scaled = mean_squared_error(nsgm_actuals_scaled, nsgm_predictions_scaled)
                    mae_nsgm_scaled = mean_absolute_error(nsgm_actuals_scaled, nsgm_predictions_scaled)
                    rmse_nsgm_scaled = np.sqrt(mse_nsgm_scaled)

                    print("\n--- NSGM(1,N) Model Performance (Scaled Data on Test Set) ---")
                    print(f"Test MSE (scaled): {mse_nsgm_scaled:.6f}")
                    print(f"Test MAE (scaled): {mae_nsgm_scaled:.6f}")
                    print(f"Test RMSE (scaled): {rmse_nsgm_scaled:.6f}")
                    print(f"Number of test predictions made: {len(nsgm_predictions_scaled)}")

                    # Visualization
                    plt.figure(figsize=(12, 6))
                    sample_size_nsgm = min(100, len(nsgm_actuals_scaled)) # Plot up to 100 points
                    plt.plot(nsgm_actuals_scaled[:sample_size_nsgm], label='Actual Values (Scaled)', marker='.')
                    plt.plot(nsgm_predictions_scaled[:sample_size_nsgm], label='NSGM Predicted Values (Scaled)', linestyle='--')
                    plt.title(f'NSGM(1,N) Predictions vs Actuals (First {sample_size_nsgm} Test Samples - Scaled)')
                    plt.xlabel('Sample Index in Test Set')
                    plt.ylabel('Value (Scaled)')
                    plt.legend()
                    plt.grid(True)
                    nsgm_plot_filename = "nsgm_module_preds_vs_actuals.png"
                    plt.savefig(nsgm_plot_filename)
                    print(f"NSGM predictions vs actuals plot saved as {nsgm_plot_filename} in {os.path.abspath('.')}")
                    plt.close()

    print("\n--- End of NSGM1NModel Module Test ---")


