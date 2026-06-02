import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.logs = []
        self.impute_values = {}
        self.scalers = {}
        self.label_encoders = {}
        self.one_hot_columns = {}
        self.original_columns = []
        self.feature_cols = []
        self.target_col = None
        self.task_type = None  # 'regression' or 'classification'
        self.categorical_cols = []
        self.numerical_cols = []

    def fit_transform(self, df, target_column, task_type='regression', missing_strategy='mean', scaling_strategy='standard', test_size=0.2, random_state=42):
        """
        Fits the preprocessing pipeline on the dataframe and transforms it.
        Returns:
            X_train, X_test, y_train, y_test, feature_names, logs
        """
        self.logs = []
        self.target_col = target_column
        self.task_type = task_type
        
        self.logs.append(f"Starting preprocessing pipeline for dataset. Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        self.logs.append(f"Target column designated: '{target_column}' ({task_type})")
        
        # Make a copy to avoid setting with copy warnings
        data = df.copy()
        
        # 1. Identify columns
        self.original_columns = [col for col in data.columns if col != target_column]
        
        # Separate features and target
        X = data[self.original_columns].copy()
        y = data[target_column].copy()
        
        # Identify variable types
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logs.append(f"Features list - Numerical: {self.numerical_cols}, Categorical: {self.categorical_cols}")
        
        # 2. Missing Value Imputation
        for col in self.numerical_cols:
            if X[col].isnull().sum() > 0:
                missing_count = X[col].isnull().sum()
                if missing_strategy == 'mean':
                    fill_val = X[col].mean()
                elif missing_strategy == 'median':
                    fill_val = X[col].median()
                else:  # mode
                    fill_val = X[col].mode().iloc[0] if not X[col].mode().empty else 0
                
                self.impute_values[col] = fill_val
                X[col] = X[col].fillna(fill_val)
                self.logs.append(f"Imputed {missing_count} missing values in numerical column '{col}' using {missing_strategy} ({fill_val:.3f})")
            else:
                self.impute_values[col] = X[col].mean() if not X[col].empty else 0
                
        for col in self.categorical_cols:
            if X[col].isnull().sum() > 0:
                missing_count = X[col].isnull().sum()
                fill_val = X[col].mode().iloc[0] if not X[col].mode().empty else 'Missing'
                self.impute_values[col] = fill_val
                X[col] = X[col].fillna(fill_val)
                self.logs.append(f"Imputed {missing_count} missing values in categorical column '{col}' using mode ('{fill_val}')")
            else:
                self.impute_values[col] = X[col].mode().iloc[0] if not X[col].mode().empty else 'Missing'

        # Check target missing values
        if y.isnull().sum() > 0:
            missing_count = y.isnull().sum()
            if task_type == 'regression':
                fill_val = y.mean()
            else:
                fill_val = y.mode().iloc[0]
            y = y.fillna(fill_val)
            self.logs.append(f"WARNING: Imputed {missing_count} missing values in target column '{target_column}' with '{fill_val}'")

        # 3. Encoding target (if classification and categorical)
        y_encoded = y.copy()
        if task_type == 'classification' and (y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype)):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))
            self.label_encoders[target_column] = le
            self.logs.append(f"Encoded target column '{target_column}' using LabelEncoder. Classes: {list(le.classes_)}")

        # 4. Encoding categorical features (One-Hot Encoding preferred)
        X_encoded = X.copy()
        if self.categorical_cols:
            self.logs.append(f"Encoding categorical features: {self.categorical_cols}")
            # Use pandas get_dummies for multi-column one-hot encoding
            # We record dummy column names to enforce them later on test predictions
            dummy_df = pd.get_dummies(X[self.categorical_cols], drop_first=False)
            self.one_hot_columns = {col: dummy_df.columns[dummy_df.columns.str.startswith(col)].tolist() for col in self.categorical_cols}
            
            # Remove categorical columns, append dummy columns
            X_encoded = X_encoded.drop(columns=self.categorical_cols)
            X_encoded = pd.concat([X_encoded, dummy_df], axis=1)
            self.logs.append(f"One-Hot encoded categorical features. Expanded feature count from {len(self.original_columns)} to {X_encoded.shape[1]}")

        # Keep track of final feature columns structure
        self.feature_cols = X_encoded.columns.tolist()

        # 5. Scaling Numerical Features
        if scaling_strategy in ['standard', 'minmax'] and self.numerical_cols:
            self.logs.append(f"Scaling numerical features using {scaling_strategy} scaler")
            for col in self.numerical_cols:
                if scaling_strategy == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                
                # Fit and transform
                X_encoded[col] = scaler.fit_transform(X_encoded[[col]])
                self.scalers[col] = scaler
            self.logs.append(f"Scaling complete for columns: {self.numerical_cols}")
        else:
            self.logs.append("Skipped feature scaling as per user request.")

        # Ensure all columns are numeric type (e.g. converting boolean dummy columns to int)
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'bool':
                X_encoded[col] = X_encoded[col].astype(int)

        # 6. Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded.values, 
            y_encoded.values if hasattr(y_encoded, 'values') else y_encoded, 
            test_size=test_size, 
            random_state=random_state
        )
        self.logs.append(f"Split data into train and test sets. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]} (Split: {int((1-test_size)*100)}/{int(test_size*100)})")
        
        return X_train, X_test, y_train, y_test, self.feature_cols, self.logs

    def transform_single(self, input_dict):
        """
        Transforms a single user input dictionary for real-time predictions.
        Must receive columns and values matching the original features.
        """
        # Create a single row DataFrame
        row = pd.DataFrame([input_dict])
        
        # 1. Fill missing values with saved values
        for col in self.original_columns:
            if col not in row.columns or pd.isna(row.loc[0, col]):
                row[col] = self.impute_values.get(col, 0)
        
        # 2. Impute any categoricals
        for col in self.categorical_cols:
            if row[col].iloc[0] is None:
                row[col] = self.impute_values.get(col, 'Missing')

        # 3. Handle One-Hot encoding of categoricals manually to match shape
        # Create a df with numerical values
        X_numeric = row[self.numerical_cols].copy()
        
        # Scale numerical features
        for col in self.numerical_cols:
            if col in self.scalers:
                X_numeric[col] = self.scalers[col].transform(X_numeric[[col]])
        
        # Recreate dummy columns
        dummies = {}
        for col in self.categorical_cols:
            val = str(row[col].iloc[0])
            expected_dummies = self.one_hot_columns.get(col, [])
            active_dummy = f"{col}_{val}"
            
            for dummy_col in expected_dummies:
                dummies[dummy_col] = 1 if dummy_col == active_dummy else 0
        
        X_dummies = pd.DataFrame([dummies])
        
        # Concatenate and align columns exactly to self.feature_cols
        X_final = pd.concat([X_numeric, X_dummies], axis=1)
        
        # Add missing columns just in case
        for col in self.feature_cols:
            if col not in X_final.columns:
                X_final[col] = 0
                
        # Arrange columns in original training order
        X_final = X_final[self.feature_cols]
        
        return X_final.values
