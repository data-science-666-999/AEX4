import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
import matplotlib.pyplot as plt
import os

# Import DataPreprocessor from the sibling module
from data_preprocessing_module import DataPreprocessor


# --- Module 2: Attention-Enhanced LSTM (ATT-LSTM) Module ---

class ATTLSTMModel:
    def __init__(self, input_shape, look_back=60, random_seed=42, model_params=None):
        self.input_shape = input_shape  # (timesteps, features)
        self.look_back = look_back
        self.model = None
        self.random_seed = random_seed
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Store model parameters if provided, otherwise use defaults in build_model
        self.model_params = model_params if model_params else {}

        # Set default values for parameters that might be passed in model_params
        # These will be overridden by model_params if keys exist,
        # and used by build_model if hp object is None.
        self.lstm_units_1 = self.model_params.get('lstm_units_1', 64)
        self.lstm_units_2 = self.model_params.get('lstm_units_2', 64) # Default for potential second layer
        self.num_lstm_layers = self.model_params.get('num_lstm_layers', 1)

        self.dense_units_1 = self.model_params.get('dense_units_1', 32)
        self.dense_units_2 = self.model_params.get('dense_units_2', 32) # Default for potential second layer
        self.num_dense_layers = self.model_params.get('num_dense_layers', 1)

        self.learning_rate = self.model_params.get('learning_rate', 0.001)
        self.dropout_rate_lstm = self.model_params.get('dropout_rate_lstm', 0.2)
        self.dropout_rate_dense = self.model_params.get('dropout_rate_dense', 0.2)
        self.activation_dense = self.model_params.get('activation_dense', 'relu')


    def _create_sequences(self, data, target_column_index):
        # This method might be better placed in DataPreprocessor or main script
        # if look_back is tuned, as ATTLSTMModel instance might not know the tuned look_back
        # For now, assuming look_back is fixed for a given ATTLSTMModel instance.
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:(i + self.look_back), :])
            y.append(data[i + self.look_back, target_column_index])
        return np.array(X), np.array(y)

    def build_model(self, hp=None): # Accept hp object
        if hp: # Use hyperparameters from tuner if provided
            num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=2, step=1)
            lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=512, step=32)
            # Only define lstm_units_2 if num_lstm_layers is 2
            if num_lstm_layers > 1:
                lstm_units_2 = hp.Int('lstm_units_2', min_value=32, max_value=256, step=32)

            num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=2, step=1)
            dense_units_1 = hp.Int('dense_units_1', min_value=32, max_value=256, step=32)
            if num_dense_layers > 1:
                dense_units_2 = hp.Int('dense_units_2', min_value=16, max_value=128, step=16)

            learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
            dropout_rate_lstm = hp.Float('dropout_rate_lstm', min_value=0.1, max_value=0.5, step=0.1)
            dropout_rate_dense = hp.Float('dropout_rate_dense', min_value=0.1, max_value=0.5, step=0.1)
            activation_dense = hp.Choice('activation_dense', values=['relu', 'tanh', 'elu'])

        else: # Use instance attributes (from self.model_params or their defaults)
            num_lstm_layers = self.num_lstm_layers
            lstm_units_1 = self.lstm_units_1
            lstm_units_2 = self.lstm_units_2 # Will be used if num_lstm_layers > 1

            num_dense_layers = self.num_dense_layers
            dense_units_1 = self.dense_units_1
            dense_units_2 = self.dense_units_2 # Will be used if num_dense_layers > 1

            learning_rate = self.learning_rate
            dropout_rate_lstm = self.dropout_rate_lstm
            dropout_rate_dense = self.dropout_rate_dense
            activation_dense = self.activation_dense

        inputs = Input(shape=self.input_shape)
        x = inputs

        # LSTM layers
        # First LSTM layer
        x = LSTM(units=lstm_units_1,
                 return_sequences=True, # Crucial: True if followed by another LSTM or Attention that sees all sequences
                 dropout=dropout_rate_lstm if hp else 0.0,
                 recurrent_dropout=dropout_rate_lstm if hp else 0.0
                )(x)

        # Second LSTM layer (optional)
        if num_lstm_layers > 1:
            # The second (now last) LSTM layer also needs return_sequences=True for the attention mechanism
            x = LSTM(units=lstm_units_2, # lstm_units_2 is defined if num_lstm_layers > 1
                     return_sequences=True,
                     dropout=dropout_rate_lstm if hp else 0.0,
                     recurrent_dropout=dropout_rate_lstm if hp else 0.0
                    )(x)

        # --- Attention Mechanism (applied to the output of the last LSTM layer) ---
        # Query: last hidden state of the final LSTM layer
        query = x[:, -1, :]
        # Value and Key: all hidden states of the final LSTM layer
        value = x
        key = x

        query_reshaped = keras.ops.expand_dims(query, axis=1)
        key_transposed = keras.ops.transpose(key, axes=(0, 2, 1))
        scores = keras.ops.matmul(query_reshaped, key_transposed)
        scores = keras.ops.squeeze(scores, axis=1)
        attention_weights = keras.ops.softmax(scores, axis=-1)
        attention_weights_reshaped = keras.ops.expand_dims(attention_weights, axis=-1)
        context_vector = keras.ops.multiply(value, attention_weights_reshaped)
        context_vector = keras.ops.sum(context_vector, axis=1)

        merged_output = Concatenate()([query, context_vector])
        x = merged_output

        # Dense layers for prediction
        # First Dense layer
        x = Dense(units=dense_units_1, activation=activation_dense if hp else 'relu')(x)
        x = Dropout(dropout_rate_dense if hp else 0.2)(x)

        # Second Dense layer (optional)
        if num_dense_layers > 1:
            x = Dense(units=dense_units_2, activation=activation_dense if hp else 'relu')(x) # dense_units_2 is defined if num_dense_layers > 1
            x = Dropout(dropout_rate_dense if hp else 0.2)(x)

        outputs = Dense(1)(x) # Output a single value for stock price prediction

        # Create the model object
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model with the potentially tuned learning rate
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

        # If build_model is called directly (not by KerasTuner),
        # it should still assign the created model to self.model
        if not hp:
            self.model = model

        return model # Crucial: return the compiled model for KerasTuner

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, early_stopping_patience=10, reduce_lr_patience=5, reduce_lr_factor=0.2):
        # If self.model is not set (e.g. if only build_model(hp) was called by tuner),
        # this method might not work as expected unless it receives a model or self.model is set by caller after tuning.
        # For direct training of an ATTLSTMModel instance, ensure build_model() (no hp) is called if model is None.
        if self.model is None:
            print("Warning: self.model is None in train(). Calling build_model() without hp to initialize.")
            self.build_model() # This will use instance attributes and assign to self.model

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            verbose=1,
            restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            verbose=1,
            min_lr=1e-6 # Do not reduce LR below this value
        )

        print("Training ATT-LSTM model with Early Stopping and ReduceLROnPlateau...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
            shuffle=False # Important for time series data
        )
        print("ATT-LSTM model training complete.")
        return history

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not built or trained. Call build_model() and train() first.")
        return self.model.predict(X_test)

# Example Usage (for testing the module)
if __name__ == '__main__':
    print("\n--- ATTLSTMModel Module Test ---")
    # 1. Get preprocessed data
    # Using AAPL for 2 years to get a bit more data for LSTM training than 1 year.
    # look_back is 60, so we need at least 60+ data points for one sequence.
    # train/val/test split will further reduce this.
    # (0.8 * 0.75) * (N - 60) for X_train. If N=250 (1 year), (250-60)*0.6 = 114 samples for training.
    # If N=500 (2 years), (500-60)*0.6 = 264 samples for training. This is better.
    data_preprocessor = DataPreprocessor(stock_ticker='AAPL', years_of_data=2, random_seed=42)
    processed_df, data_scaler = data_preprocessor.preprocess()

    if processed_df is None or processed_df.empty:
        print("Failed to preprocess data. Aborting ATTLSTMModel test.")
    else:
        print(f"Shape of preprocessed_df: {processed_df.shape}")
        # Assuming target column is the last one, as per DataPreprocessor output
        target_column_name = processed_df.columns[-1]
        target_column_index = processed_df.columns.get_loc(target_column_name)
        print(f"Target column for LSTM: {target_column_name} at index {target_column_index}")

        look_back = 60 # Standard look_back

        # Create sequences (using the model's own _create_sequences if it's suitable, or a local one)
        # The model's _create_sequences is simple, let's use a local version for clarity here.
        def create_sequences_for_test(data_df, target_idx, lb):
            X_data, y_data = [], []
            raw_data = data_df.values # Convert to numpy array for faster iloc-like access
            for i in range(len(raw_data) - lb):
                X_data.append(raw_data[i:(i + lb), :]) # All features
                y_data.append(raw_data[i + lb, target_idx]) # Target feature
            return np.array(X_data), np.array(y_data)

        X, y = create_sequences_for_test(processed_df, target_column_index, look_back)

        if X.shape[0] == 0:
            print("Not enough data to create sequences. Aborting.")
        else:
            # Split data: 80% train, 20% temp -> then 20% of temp becomes 25% of train for val
            # Temporal split is important: shuffle=False
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False) # Split temp into 50% val, 50% test

            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
                print("Not enough data after splitting for training/validation/testing. Aborting.")
            else:
                input_shape = (X_train.shape[1], X_train.shape[2]) # (timesteps, features)

                # Initialize and build model
                # Using default units from the class for this test
                att_lstm_test_model = ATTLSTMModel(
                    input_shape=input_shape,
                    lstm_units=50, # Smaller units for faster test
                    dense_units=25,
                    look_back=look_back,
                    random_seed=42
                )
                att_lstm_test_model.build_model() # Build with default params
                print("\nATTLSTM Model Summary (for test):")
                att_lstm_test_model.model.summary()

                # Train model
                print("\nTraining ATTLSTM model for test...")
                # Train for fewer epochs for a module test, relying on EarlyStopping
                history = att_lstm_test_model.train(
                    X_train, y_train, X_val, y_val,
                    epochs=10, # Reduced epochs for test
                    batch_size=16, # Smaller batch size for smaller dataset
                    early_stopping_patience=3,
                    reduce_lr_patience=2
                )

                # Evaluate model
                print("\nEvaluating ATTLSTM model on test set...")
                predictions_scaled = att_lstm_test_model.predict(X_test).flatten()

                # Since y_test is already scaled (it came from processed_df),
                # we can calculate metrics directly on scaled data for this module test.
                from sklearn.metrics import mean_squared_error, mean_absolute_error

                mse_scaled = mean_squared_error(y_test, predictions_scaled)
                mae_scaled = mean_absolute_error(y_test, predictions_scaled)
                rmse_scaled = np.sqrt(mse_scaled)

                print("\n--- ATTLSTM Model Performance (Scaled Data) ---")
                print(f"Test MSE (scaled): {mse_scaled:.6f}")
                print(f"Test MAE (scaled): {mae_scaled:.6f}")
                print(f"Test RMSE (scaled): {rmse_scaled:.6f}")

                # Visualizations
                print("\nGenerating visualizations for ATTLSTM module test...")

                # 1. Plot training & validation loss
                plt.figure(figsize=(10, 6))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('ATTLSTM Model Training and Validation Loss (Module Test)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.legend()
                plt.grid(True)
                loss_plot_filename = "att_lstm_module_loss_curve.png"
                plt.savefig(loss_plot_filename)
                print(f"Loss curve plot saved as {loss_plot_filename} in {os.path.abspath('.')}")
                plt.close()

                # 2. Plot predictions vs. actuals for a sample from the test set
                sample_size = min(50, len(y_test)) # Plot up to 50 points
                plt.figure(figsize=(12, 6))
                plt.plot(y_test[:sample_size], label='Actual Values (Scaled)', marker='.')
                plt.plot(predictions_scaled[:sample_size], label='Predicted Values (Scaled)', linestyle='--')
                plt.title(f'ATTLSTM Predictions vs Actuals (First {sample_size} Test Samples - Scaled)')
                plt.xlabel('Sample Index')
                plt.ylabel('Value (Scaled)')
                plt.legend()
                plt.grid(True)
                preds_plot_filename = "att_lstm_module_preds_vs_actuals.png"
                plt.savefig(preds_plot_filename)
                print(f"Predictions vs actuals plot saved as {preds_plot_filename} in {os.path.abspath('.')}")
                plt.close()

    print("\n--- End of ATTLSTMModel Module Test ---")


