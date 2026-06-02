import os
import joblib
import datetime
import numpy as np
from config import Config
from utils.firebase_db import db_client

class PredictionModule:
    @staticmethod
    def load_model_package(model_filename):
        """Loads the serialized model package dictionary from the file."""
        model_path = os.path.join(Config.SAVED_MODELS_FOLDER, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
        return joblib.load(model_path)

    @staticmethod
    def get_features_schema(model_package):
        """
        Analyzes the preprocessor and returns a list of features with their types and categories.
        Used by the frontend to dynamically build form inputs.
        """
        prep = model_package['preprocessor']
        schema = []
        
        for col in prep.original_columns:
            if col in prep.categorical_cols:
                # Retrieve category options by cleaning dummy prefixes
                options = []
                dummy_cols = prep.one_hot_columns.get(col, [])
                prefix = f"{col}_"
                for d_col in dummy_cols:
                    if d_col.startswith(prefix):
                        options.append(d_col[len(prefix):])
                
                schema.append({
                    'name': col,
                    'type': 'categorical',
                    'options': options
                })
            else:
                # Numerical input
                schema.append({
                    'name': col,
                    'type': 'numerical',
                    # Try to infer clean bounds
                    'mean': float(prep.impute_values.get(col, 0.0))
                })
        
        return schema

    @staticmethod
    def predict(model_package, input_dict, username=None):
        """
        Executes preprocessing and model inference for a single input record.
        Logs prediction to database and returns predictions details.
        """
        model = model_package['model']
        prep = model_package['preprocessor']
        
        # 1. Transform single input dictionary to match scaling and encoding shape
        X_single = prep.transform_single(input_dict)
        
        # 2. Predict
        prediction_val = model.predict(X_single)[0]
        
        # 3. Calculate Confidence Score (if classification)
        confidence = None
        prediction_label = prediction_val
        
        if prep.task_type == 'classification':
            try:
                probs = model.predict_proba(X_single)[0]
                max_prob_idx = np.argmax(probs)
                confidence = float(probs[max_prob_idx])
            except Exception:
                confidence = 1.0  # Fallback if model doesn't support probability
                
            # If target was encoded, map label back to original class name
            if prep.target_col in prep.label_encoders:
                le = prep.label_encoders[prep.target_col]
                prediction_label = str(le.inverse_transform([prediction_val])[0])
        else:
            # For regression, round numerical output for cleaner display
            prediction_label = float(prediction_val)
            
        # 4. Save to Prediction History database
        history_record = {
            'username': username or 'guest',
            'model_name': model_package['algorithm_name'],
            'dataset_name': model_package['dataset_name'],
            'inputs': input_dict,
            'prediction': str(prediction_label),
            'confidence': confidence,
            'created_at': datetime.datetime.now().isoformat()
        }
        
        db_client.add_document('predictions', history_record)
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'history_record': history_record
        }

    @staticmethod
    def get_prediction_history(username):
        """Retrieves prediction logs for a specific user."""
        records = db_client.query_documents('predictions', 'username', username)
        # Sort by creation date descending
        records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return records
