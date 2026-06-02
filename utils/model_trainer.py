import time
import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics as skmetrics
from config import Config

# Helper to import XGBoost if available, fallback to HistGradientBoosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class ModelTrainer:
    @staticmethod
    def get_algorithm_name(algo_id):
        names = {
            'linear_regression': 'Linear Regression',
            'multiple_linear_regression': 'Multiple Linear Regression',
            'polynomial_regression': 'Polynomial Regression',
            'ridge_regression': 'Ridge Regression',
            'lasso_regression': 'Lasso Regression',
            'decision_tree_regression': 'Decision Tree Regressor',
            'random_forest_regression': 'Random Forest Regressor',
            'logistic_regression': 'Logistic Regression',
            'decision_tree_classifier': 'Decision Tree Classifier',
            'random_forest_classifier': 'Random Forest Classifier',
            'knn': 'K-Nearest Neighbors (KNN)',
            'naive_bayes': 'Naive Bayes Classifier',
            'svm': 'Support Vector Machine (SVM)',
            'gradient_boosting': 'Gradient Boosting Classifier',
            'xgboost': 'XGBoost Classifier (HistGradientBoosting)'
        }
        return names.get(algo_id, algo_id.replace('_', ' ').title())

    @staticmethod
    def build_model(algo_id, params):
        """Builds a model instance based on algorithm ID and parameters."""
        if algo_id == 'linear_regression' or algo_id == 'multiple_linear_regression':
            return LinearRegression()
            
        elif algo_id == 'polynomial_regression':
            degree = int(params.get('degree', 2))
            return make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            
        elif algo_id == 'ridge_regression':
            alpha = float(params.get('alpha', 1.0))
            return Ridge(alpha=alpha)
            
        elif algo_id == 'lasso_regression':
            alpha = float(params.get('alpha', 1.0))
            return Lasso(alpha=alpha)
            
        elif algo_id == 'decision_tree_regression':
            max_depth = params.get('max_depth')
            max_depth = int(max_depth) if max_depth and max_depth != 'None' else None
            min_samples_split = int(params.get('min_samples_split', 2))
            return DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            
        elif algo_id == 'random_forest_regression':
            n_estimators = int(params.get('n_estimators', 100))
            max_depth = params.get('max_depth')
            max_depth = int(max_depth) if max_depth and max_depth != 'None' else None
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            
        elif algo_id == 'logistic_regression':
            C = float(params.get('C', 1.0))
            penalty = params.get('penalty', 'l2')
            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
            return LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42)
            
        elif algo_id == 'decision_tree_classifier':
            max_depth = params.get('max_depth')
            max_depth = int(max_depth) if max_depth and max_depth != 'None' else None
            criterion = params.get('criterion', 'gini')
            return DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
            
        elif algo_id == 'random_forest_classifier':
            n_estimators = int(params.get('n_estimators', 100))
            max_depth = params.get('max_depth')
            max_depth = int(max_depth) if max_depth and max_depth != 'None' else None
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            
        elif algo_id == 'knn':
            n_neighbors = int(params.get('n_neighbors', 5))
            weights = params.get('weights', 'uniform')
            return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            
        elif algo_id == 'naive_bayes':
            return GaussianNB()
            
        elif algo_id == 'svm':
            C = float(params.get('C', 1.0))
            kernel = params.get('kernel', 'rbf')
            return SVC(C=C, kernel=kernel, probability=True, random_state=42)
            
        elif algo_id == 'gradient_boosting':
            n_estimators = int(params.get('n_estimators', 100))
            learning_rate = float(params.get('learning_rate', 0.1))
            return GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            
        elif algo_id == 'xgboost':
            learning_rate = float(params.get('learning_rate', 0.1))
            max_iter = int(params.get('max_iter', 100))
            if XGBOOST_AVAILABLE:
                # Use real XGBoost
                return xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=max_iter, random_state=42, eval_metric='logloss')
            else:
                # Use Scikit-learn HistGradientBoostingClassifier as transparent fallback
                return HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter, random_state=42)
        
        raise ValueError(f"Unknown algorithm ID: {algo_id}")

    @staticmethod
    def train_and_evaluate(algo_id, params, X_train, X_test, y_train, y_test, preprocessor, target_column, dataset_name):
        """Trains a model, calculates evaluation metrics, and saves the pipeline."""
        start_time = time.time()
        
        # 1. Build and Fit Model
        model = ModelTrainer.build_model(algo_id, params)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # 2. Make Predictions
        y_pred = model.predict(X_test)
        
        # 3. Calculate metrics
        task_type = preprocessor.task_type
        metrics = {}
        
        if task_type == 'regression':
            mae = skmetrics.mean_absolute_error(y_test, y_pred)
            mse = skmetrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = skmetrics.r2_score(y_test, y_pred)
            
            # Adjusted R2
            n = len(y_test)
            p = X_test.shape[1]
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r2
            
            metrics = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'adjusted_r2': float(adj_r2)
            }
        else:
            # Classification
            accuracy = skmetrics.accuracy_score(y_test, y_pred)
            
            # Use 'weighted' average to handle multiclass safely
            precision = skmetrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = skmetrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = skmetrics.f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = skmetrics.confusion_matrix(y_test, y_pred)
            
            # ROC AUC (handling multiclass and binary safely)
            try:
                y_prob = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    # Binary
                    roc_auc = skmetrics.roc_auc_score(y_test, y_prob[:, 1])
                else:
                    # Multiclass
                    roc_auc = skmetrics.roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            except Exception:
                roc_auc = 0.0  # Fallback if model doesn't support probability
                
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'confusion_matrix': cm.tolist()
            }
            
        # 4. Save entire pipeline
        model_filename = f"{algo_id}_{dataset_name.split('.')[0]}_{int(time.time())}.pkl"
        model_path = os.path.join(Config.SAVED_MODELS_FOLDER, model_filename)
        
        saved_package = {
            'model': model,
            'preprocessor': preprocessor,
            'algorithm_id': algo_id,
            'algorithm_name': ModelTrainer.get_algorithm_name(algo_id),
            'dataset_name': dataset_name,
            'task_type': task_type,
            'metrics': metrics,
            'features': preprocessor.feature_cols,
            'target_column': target_column,
            'train_time': training_time,
            'hyperparameters': params,
            'created_at': time.time()
        }
        
        joblib.dump(saved_package, model_path)
        
        return saved_package, model_filename
