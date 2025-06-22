import keras_tuner as kt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from data_preprocessing_module import DataPreprocessor
from att_lstm_module import ATTLSTMModel

def build_hypermodel(hp):
    """
    Builds a hypermodel for KerasTuner.
    This function is called by the tuner to create a model with a given set of hyperparameters.
    """
    # --- Data Preprocessing ---
    # For tuning, we might fix look_back or make it a hyperparameter.
    # If look_back is a hyperparameter, data preprocessing needs to be efficient per trial.
    # For this first pass, let's assume a fixed look_back for data generation.
    # The input_shape required by ATTLSTMModel will be determined by the data preprocessed
    # in main_tuner() using a fixed look_back.

    # This function is effectively replaced by the closure build_hypermodel_with_shape.
    # The original build_hypermodel(hp) is not called by the tuner.
    pass


def main_tuner(
    stock_ticker="^AEX",
    years_of_data=5, # Increased years of data for more robust tuning
    project_name_prefix="stock_att_lstm_hyperband_aggressive",
    look_back_period=60,
    max_epochs_hyperband=150, # Increased max_epochs for Hyperband
    hyperband_iterations=3,  # Increased Hyperband iterations
    early_stopping_patience=20, # Increased patience for early stopping
    use_differencing_for_tuning=False # Added parameter to control differencing during tuning
):
    """
    Main function to run KerasTuner with Hyperband for a specific look_back_period.
    Includes tuning for batch_size.
    """
    project_name = f"{project_name_prefix}_{years_of_data}yr_{look_back_period}d_lookback"
    print(f"Starting hyperparameter tuning for {stock_ticker} using {years_of_data} years of data.")
    print(f"Using look_back_period = {look_back_period} for data preparation and model input shape.")
    print(f"Hyperband: max_epochs={max_epochs_hyperband}, iterations={hyperband_iterations}, early_stopping_patience={early_stopping_patience}")

    # --- 1. Data Preparation ---
    # Use a default lasso_alpha for tuning runs; it can be experimented with separately.
    # Pass use_differencing_for_tuning to DataPreprocessor
    data_preprocessor = DataPreprocessor(
        stock_ticker=stock_ticker,
        years_of_data=years_of_data,
        random_seed=42,
        lasso_alpha=0.005, # Default LASSO alpha for tuning
        use_differencing=use_differencing_for_tuning # Control differencing
    )

    # Ensure this matches the return signature of DataPreprocessor.preprocess()
    # It now returns 5 values: scaled_df, target_scaler, selected_features_names, df_with_all_indicators_cleaned, first_price_before_diff
    processed_df, _, selected_features, _, first_price_val_if_diff = data_preprocessor.preprocess()

    if processed_df.empty:
        print("Error: Preprocessed data is empty. Aborting tuning.")
        return
    print(f"Data preprocessed. Number of selected features by LASSO: {len(selected_features)}")
    print(f"Processed data shape: {processed_df.shape}")


    target_column_name = processed_df.columns[-1]

    # Create sequences function (adapted from FullStockPredictionModel)
    # Ensure 'look_back' used in range is the current_look_back being tuned.
    def create_sequences_for_tuning(data, current_look_back, target_col_name):
        target_col_idx = data.columns.get_loc(target_col_name)
        X, y = [], []
        # Corrected: use current_look_back in the loop range
        for i in range(len(data) - current_look_back):
            X.append(data.iloc[i:(i + current_look_back), :].values)
            y.append(data.iloc[i + current_look_back, target_col_idx])
        return np.array(X), np.array(y)

    # Use the passed look_back_period for sequence creation
    X_seq, y_seq = create_sequences_for_tuning(processed_df, look_back_period, target_column_name)

    if len(X_seq) == 0:
        print(f"Error: No sequences created with look_back = {look_back_period}. Check data length. Aborting tuning for this look_back.")
        return

    # Split data: 70% train, 15% validation (for tuner), 15% test (final holdout, not used by tuner)
    # Temporal split is important.
    train_size = int(len(X_seq) * 0.70)
    val_size = int(len(X_seq) * 0.15)

    X_train_seq, X_temp_seq, y_train_seq, y_temp_seq = train_test_split(
        X_seq, y_seq, train_size=train_size, shuffle=False
    )
    X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(
        X_temp_seq, y_temp_seq, train_size=val_size, shuffle=False # val_size from remaining
    )

    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        print("Error: Not enough data for train/validation split after sequencing. Aborting tuning.")
        return

    print(f"X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
    print(f"X_val_seq shape: {X_val_seq.shape}, y_val_seq shape: {y_val_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")

    # Update build_hypermodel to remove look_back tuning and use fixed_look_back_data_prep for input_shape
    # This is a workaround for the complexity of tuning look_back directly with KerasTuner's default flow.
    # The ATTLSTMModel's internal look_back is less critical if input_shape is correctly set.
    # The input_shape for the model will be derived from X_train_seq, which is created using look_back_period.

    current_input_shape = (X_train_seq.shape[1], X_train_seq.shape[2]) # (timesteps, features)
                                                                    # timesteps here is look_back_period

    def build_hypermodel_with_shape(hp): # Reverted name
        # This inner function captures current_input_shape and look_back_period
        model_instance = ATTLSTMModel(
            input_shape=current_input_shape,
            look_back=look_back_period,
            random_seed=42
        )
        # ATTLSTMModel.build_model uses hp to get units, lr, dropout etc.
        # We can add hp.Choice for batch_size here if the tuner itself supports it for model.fit()
        # For now, batch_size is handled outside this function for tuner.search()
        model = model_instance.build_model(hp)
        return model

    # --- 2. KerasTuner Setup ---
    # Using Hyperband tuner
    tuner = kt.Hyperband(
        hypermodel=build_hypermodel_with_shape, # Reverted name
        objective='val_loss',
        max_epochs=max_epochs_hyperband,
        factor=3,
        hyperband_iterations=hyperband_iterations,
        directory='keras_tuner_dir',
        project_name=project_name,
        overwrite=True
    )

    tuner.search_space_summary()

    # --- 3. Run Search ---
    print("Starting KerasTuner Hyperband search...")
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True
    )

    # Define batch_size as a hyperparameter to be tuned by KerasTuner
    # This hp will be used by the tuner to pass to model.fit()
    hp_batch_size = hp.Choice('batch_size', values=[16, 32, 64])

    # Note: For KerasTuner to properly use hp.Choice for batch_size in model.fit(),
    # the hyperparameter must be defined within the hypermodel's build method.
    # The tuner then typically handles passing this to fit.
    # If tuner.search() is called with a batch_size argument, that fixed value is used.
    # To tune batch_size, it must be part of the hp space passed to the hypermodel.
    # Then, during the search, the tuner will manage fitting with different batch_sizes.

    # The following line `current_batch_size = 32` makes batch_size fixed.
    # To tune it, we would need to ensure the `hp` object within `build_hypermodel_with_shape`
    # defines `hp.Choice('batch_size', ...)` and that the tuner uses this.
    # KerasTuner's default behavior is that `model.fit` (called internally by `tuner.search`)
    # will receive the `batch_size` from the hyperparameters if it's defined.

    # Let's ensure 'batch_size' is defined in the hypermodel scope for the tuner to pick it up.
    # We need to modify `build_hypermodel_with_shape` to include `hp.Choice('batch_size', ...)`
    # even if it's not directly used in model architecture. KerasTuner will then use it for `fit`.

    # Re-defining build_hypermodel_with_shape to include batch_size in its HPs
    def build_hypermodel_with_shape(hp):
        model_instance = ATTLSTMModel(
            input_shape=current_input_shape,
            look_back=look_back_period,
            random_seed=42
        )
        # Define batch_size as a hyperparameter here so KerasTuner is aware of it
        _ = hp.Choice('batch_size', values=[16, 32, 64, 128]) # Tuner will use this for model.fit()

        model = model_instance.build_model(hp) # hp here is for architecture (layers, units, lr, etc.)
        return model

    # Re-initialize tuner with the corrected hypermodel function
    tuner = kt.Hyperband(
        hypermodel=build_hypermodel_with_shape,
        objective='val_loss',
        max_epochs=max_epochs_hyperband,
        factor=3,
        hyperband_iterations=hyperband_iterations,
        directory='keras_tuner_dir',
        project_name=project_name,
        overwrite=True
    )
    tuner.search_space_summary() # Re-print summary with batch_size potentially included by tuner

    tuner.search(
        X_train_seq, y_train_seq,
        epochs=max_epochs_hyperband,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[early_stopping_cb]
        # batch_size argument is NOT explicitly passed here; KerasTuner will use the 'batch_size' from hp if defined.
    )

    # --- 4. Results ---
    print("\nTuning complete.")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nBest Hyperparameters Found:")
    for param, value in best_hps.values.items():
        print(f"- {param}: {value}")

    # Build the best model with the best hyperparameters
    best_model = tuner.hypermodel.build(best_hps) # or tuner.get_best_models(num_models=1)[0]

    # Optional: Train the best model on combined training and validation data for a few more epochs
    # Or evaluate it directly on the test set
    print("\nEvaluating the best model found by KerasTuner on the test set...")
    loss = best_model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print(f"Best model test loss (MSE): {loss:.4f}")
    # To get other metrics like MAE, RMSE, you'd predict and calculate manually.

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Example of how one might loop through different look_back periods:
    # This part would typically be in a separate script that calls main_tuner.
    # For demonstration, it's included here.
    # --- Configuration for the tuning runs ---
    # For a quick test of the script:
    # test_look_back_values = [30, 60] # Test with a couple of look_back values
    # test_years = 3                   # Use 3 years of data for faster runs
    # test_max_epochs = 27             # Max epochs for Hyperband's last bracket
    # test_hyperband_iterations = 1    # Number of Hyperband iterations
    # test_early_stopping_patience = 5

    # For a more comprehensive tuning run (can be time-consuming):
    # AGGRESSIVE TUNING CONFIGURATION
    aggressive_look_back_values = [20, 30, 40, 50, 60, 75, 90, 120, 150] # Expanded look_back
    aggressive_years = 7 # Increased years of data, e.g., 5 to 7 for more data if available and relevant
    aggressive_max_epochs = 200  # Significantly increased max_epochs for Hyperband
    aggressive_hyperband_iterations = 4 # Increased Hyperband iterations
    aggressive_early_stopping_patience = 30 # Increased patience

    # --- Select which configuration to run ---
    # Set to aggressive values
    current_look_back_values = aggressive_look_back_values
    current_years = aggressive_years
    current_max_epochs = aggressive_max_epochs
    current_hyperband_iterations = aggressive_hyperband_iterations
    current_early_stopping_patience = aggressive_early_stopping_patience
    # Ensure project_prefix reflects the aggressive nature and key params like years.
    project_prefix = f"stock_att_lstm_aggressive_{current_years}yr"

    # Allow choosing differencing strategy for the entire tuning campaign
    tune_with_differencing = False # Set to True to run all tuning with differencing enabled

    print(f"--- Starting KerasTuner Optimization Script (Aggressive Tuning) ---")
    print(f"Running with: Look_backs={current_look_back_values}, Years={current_years}, MaxEpochs={current_max_epochs}, HB_Iterations={current_hyperband_iterations}, Differencing for Tuning: {tune_with_differencing}")

    for lb_period in current_look_back_values:
        print(f"\n--- Running Tuner for Look-Back Period: {lb_period} ---")

        # Construct a specific project name for this run to include differencing status
        current_project_name_prefix = f"{project_prefix}_diff_{tune_with_differencing}"

        main_tuner(
            stock_ticker="^AEX",
            years_of_data=current_years,
            project_name_prefix=current_project_name_prefix, # Pass the modified prefix
            look_back_period=lb_period,
            max_epochs_hyperband=current_max_epochs,
            hyperband_iterations=current_hyperband_iterations,
            early_stopping_patience=current_early_stopping_patience,
            use_differencing_for_tuning=tune_with_differencing # Pass differencing choice
        )
        print(f"--- Tuner Run for Look-Back Period: {lb_period} Finished ---")

    print("\n--- KerasTuner Script for Look-Back Optimization (Aggressive) Finished ---")

# Note on look_back tuning:
# The current setup runs the entire KerasTuner search for each look_back period.
# After these runs, one would compare the best model performance (e.g., test loss from tuner.evaluate)
# and potentially training times from each tuner run (project_name directory) to select an optimal look_back
# and its corresponding hyperparameters.
# The `input_shape` (derived from look_back_period here) is the primary driver for the model layers.
