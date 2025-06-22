import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# --- Module 4: Ensemble (Weighted Fusion) Module ---

class EnsembleModel:
    def __init__(self, optimization_method='mse_optimization', random_seed=42):
        self.optimization_method = optimization_method
        self.weights = None
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def _objective_function(self, weights, att_lstm_preds, nsgm_preds, actual_values):
        # Objective function to minimize (e.g., Mean Squared Error)
        # Ensure weights sum to 1 and are non-negative
        weights = np.array(weights)
        weights = np.maximum(0, weights) # Ensure non-negative
        weights = weights / np.sum(weights) # Normalize to sum to 1

        combined_predictions = (weights[0] * att_lstm_preds) + (weights[1] * nsgm_preds)
        return mean_squared_error(actual_values, combined_predictions)

    def train_weights(self, att_lstm_val_preds, nsgm_val_preds, actual_val_targets):
        print(f"Training ensemble weights using {self.optimization_method}...")

        if self.optimization_method == 'mse_optimization':
            # Initial guess for weights (e.g., equal weighting)
            initial_weights = np.array([0.5, 0.5])

            # Bounds for weights (0 to 1)
            bounds = ((0.0, 1.0), (0.0, 1.0))

            # Constraints: weights must sum to 1
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

            # Minimize the objective function
            result = minimize(
                self._objective_function,
                initial_weights,
                args=(att_lstm_val_preds, nsgm_val_preds, actual_val_targets),
                method='SLSQP', # Sequential Least Squares Programming
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                self.weights = np.maximum(0, result.x) # Ensure non-negative after optimization
                self.weights = self.weights / np.sum(self.weights) # Normalize again
                print(f"Ensemble weights optimized: {self.weights}")
            else:
                print(f"Weight optimization failed: {result.message}. Using equal weights.")
                self.weights = np.array([0.5, 0.5])
        elif self.optimization_method == 'fixed':
            self.weights = np.array([0.5, 0.5]) # Default to equal weights
            print(f"Using fixed equal weights: {self.weights}")
        else:
            raise ValueError("Unsupported optimization method.")

        print("Ensemble weight training complete.")

    def predict(self, att_lstm_preds, nsgm_preds):
        if self.weights is None:
            raise ValueError("Ensemble weights not trained. Call train_weights() first.")

        # Ensure predictions are numpy arrays
        att_lstm_preds = np.asarray(att_lstm_preds).flatten()
        nsgm_preds = np.asarray(nsgm_preds).flatten()

        if len(att_lstm_preds) != len(nsgm_preds):
            raise ValueError("ATT-LSTM and NSGM predictions must have the same length.")

        combined_predictions = (self.weights[0] * att_lstm_preds) + (self.weights[1] * nsgm_preds)
        return combined_predictions

# Example Usage (for testing the module)
if __name__ == '__main__':
    print("\n--- EnsembleModel Module Test ---")
    # Simulate validation predictions from ATT-LSTM and NSGM
    n_val_samples = 100
    np.random.seed(42) # Ensure reproducibility for simulated data
    att_lstm_val_preds = np.random.rand(n_val_samples) * 100
    nsgm_val_preds = np.random.rand(n_val_samples) * 90 # Slightly different scale for NSGM
    actual_val_targets = (0.6 * att_lstm_val_preds + 0.4 * nsgm_val_preds) + np.random.normal(0, 5, n_val_samples) # Actuals related to inputs + noise

    # --- Test with 'mse_optimization' ---
    print("\n1. Testing with 'mse_optimization':")
    ensemble_model_optimized = EnsembleModel(optimization_method='mse_optimization', random_seed=42)
    ensemble_model_optimized.train_weights(att_lstm_val_preds, nsgm_val_preds, actual_val_targets)

    print(f"   Optimized Weights (ATT-LSTM, NSGM): {ensemble_model_optimized.weights}")

    # Visualize optimized weights
    if ensemble_model_optimized.weights is not None:
        plt.figure(figsize=(6, 4))
        model_names = ['ATT-LSTM', 'NSGM']
        plt.bar(model_names, ensemble_model_optimized.weights, color=['skyblue', 'lightgreen'])
        plt.ylabel("Weight Value")
        plt.title("Optimized Ensemble Weights (mse_optimization)")
        plt.ylim(0, 1)
        for i, weight in enumerate(ensemble_model_optimized.weights):
            plt.text(i, weight + 0.02, f"{weight:.3f}", ha='center')

        plot_filename = "ensemble_module_optimized_weights.png"
        plt.savefig(plot_filename)
        print(f"   Optimized weights plot saved as {plot_filename} in {os.path.abspath('.')}")
        plt.close()

    # Simulate test predictions from ATT-LSTM and NSGM
    n_test_samples = 50
    att_lstm_test_preds = np.random.rand(n_test_samples) * 100
    nsgm_test_preds = np.random.rand(n_test_samples) * 90

    # Make combined predictions with optimized model
    if ensemble_model_optimized.weights is not None:
        combined_predictions_optimized = ensemble_model_optimized.predict(att_lstm_test_preds, nsgm_test_preds)
        print(f"   Sample combined predictions (optimized weights, first 5): {combined_predictions_optimized[:5]}")
    else:
        print("   Skipping prediction with optimized weights as weights were not trained.")

    # --- Test with 'fixed' weights ---
    print("\n2. Testing with 'fixed' weights:")
    ensemble_model_fixed = EnsembleModel(optimization_method='fixed', random_seed=42)
    # actual_val_targets are ignored for 'fixed' method but passed for consistency
    ensemble_model_fixed.train_weights(att_lstm_val_preds, nsgm_val_preds, actual_val_targets)
    print(f"   Fixed Weights (ATT-LSTM, NSGM): {ensemble_model_fixed.weights}")

    # Make combined predictions with fixed weights
    combined_predictions_fixed = ensemble_model_fixed.predict(att_lstm_test_preds, nsgm_test_preds)
    print(f"   Sample combined predictions (fixed weights, first 5): {combined_predictions_fixed[:5]}")

    print("\n--- End of EnsembleModel Module Test ---")


