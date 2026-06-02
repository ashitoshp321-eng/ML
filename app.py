import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from utils.firebase_db import db_client
from utils.preprocessing import DataPreprocessor
from utils.model_trainer import ModelTrainer
from utils.prediction import PredictionModule
from utils.visualization import Visualizer
from utils.report_generator import ReportGenerator

app = Flask(__name__)
app.config.from_object(Config)

# Ensure required folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['REPORTS_FOLDER'], app.config['SAVED_MODELS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Helper function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper to verify login
def is_logged_in():
    return 'username' in session

# Helper to get sample datasets list
def get_sample_datasets():
    try:
        files = os.listdir(app.config['DATASETS_FOLDER'])
        return [f for f in files if f.endswith('.csv')]
    except Exception:
        return []

# Context processor to inject active page and login state into templates
@app.context_processor
def inject_global_vars():
    return {
        'samples': get_sample_datasets(),
        'is_logged_in': is_logged_in()
    }

# Custom Jinja2 filter to format Unix timestamps to dates
@app.template_filter('format_timestamp')
def format_timestamp(timestamp):
    """Convert Unix timestamp to formatted date string."""
    try:
        if isinstance(timestamp, str):
            return timestamp
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError, OSError):
        return str(timestamp)

# ----------------------------------------------------
# AUTHENTICATION ROUTES
# ----------------------------------------------------
@app.route('/auth/register', methods=['GET', 'POST'])
def register():
    if is_logged_in():
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check if username/email already exists
        existing_user_uname = db_client.query_documents('users', 'username', username)
        existing_user_email = db_client.query_documents('users', 'email', email)
        
        if existing_user_uname or existing_user_email:
            flash("Username or Email already registered. Please choose another or login.", "error")
            return redirect(url_for('register'))
            
        hashed_password = generate_password_hash(password)
        user_data = {
            'fullname': fullname,
            'email': email,
            'username': username,
            'password_hash': hashed_password,
            'created_at': time.time()
        }
        
        db_client.add_document('users', user_data)
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for('login'))
        
    return render_template('auth/register.html', active_page='register')

@app.route('/auth/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user_records = db_client.query_documents('users', 'username', username)
        if not user_records:
            flash("Invalid username or password.", "error")
            return redirect(url_for('login'))
            
        user = user_records[0]
        if check_password_hash(user['password_hash'], password):
            session['username'] = user['username']
            session['fullname'] = user['fullname']
            session['email'] = user['email']
            flash(f"Welcome back, {user['fullname']}!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.", "error")
            return redirect(url_for('login'))
            
    return render_template('auth/login.html', active_page='login')

@app.route('/auth/logout')
def logout():
    session.clear()
    flash("Successfully logged out.", "success")
    return redirect(url_for('home'))

# ----------------------------------------------------
# MAIN WEBSITE PAGES
# ----------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html', active_page='home')

@app.route('/models')
def models_list():
    return render_template('models/index.html', active_page='models')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/contact')
def contact():
    return render_template('contact.html', active_page='contact')

@app.route('/contact/submit', methods=['POST'])
def contact_submit():
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')
    
    msg_data = {
        'name': name,
        'email': email,
        'subject': subject,
        'message': message,
        'created_at': time.time()
    }
    
    db_client.add_document('contact_messages', msg_data)
    flash("Thank you for your message! Our team will get back to you shortly.", "success")
    return redirect(url_for('contact'))

# ----------------------------------------------------
# DATASET ACTIONS (UPLOAD & PREVIEW)
# ----------------------------------------------------
@app.route('/upload/<dashboard_type>', methods=['POST'])
def upload_dataset(dashboard_type):
    # Determine source (sample vs uploaded file)
    sample_name = request.form.get('sample_name')
    filepath = None
    dataset_name = None
    
    if sample_name:
        filepath = os.path.join(app.config['DATASETS_FOLDER'], sample_name)
        dataset_name = sample_name
    elif 'file' in request.files:
        file = request.files['file']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            dataset_name = filename
            
    if not filepath or not os.path.exists(filepath):
        flash("No valid dataset selected or uploaded.", "error")
        return redirect(url_for(f"supervised_dashboard" if dashboard_type in ['regression', 'classification'] else f"unsupervised_dashboard", task_type=dashboard_type))

    # Read dataset preview details
    try:
        df = pd.read_csv(filepath)
        # Store metadata in session
        session['active_dataset_path'] = filepath
        session['active_dataset_name'] = dataset_name
        
        # Clear any old model results from session when a new dataset is uploaded
        session.pop('last_model_results', None)
        session.pop('last_model_filename', None)
        session.pop('last_unsupervised_results', None)
        
        flash(f"Successfully loaded dataset: {dataset_name}", "success")
    except Exception as e:
        flash(f"Error reading CSV: {e}", "error")
        
    if dashboard_type in ['regression', 'classification']:
        return redirect(url_for('supervised_dashboard', task_type=dashboard_type))
    else:
        return redirect(url_for('unsupervised_dashboard', task_type=dashboard_type))

# ----------------------------------------------------
# SUPERVISED DASHBOARD ROUTE
# ----------------------------------------------------
@app.route('/dashboard/supervised/<task_type>')
@app.route('/dashboard/supervised/<task_type>/<algo_id>')
def supervised_dashboard(task_type, algo_id=None):
    if not is_logged_in():
        flash("Please login to access the interactive dashboards.", "warning")
        return redirect(url_for('login'))
        
    df_exists = False
    shape = (0, 0)
    columns = []
    preview_data = []
    summary = {}
    missing_sum = 0
    
    dataset_path = session.get('active_dataset_path')
    dataset_name = session.get('active_dataset_name', '')
    
    if dataset_path and os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            df_exists = True
            shape = df.shape
            columns = df.columns.tolist()
            preview_data = df.head(10).to_dict(orient='records')
            missing_sum = int(df.isnull().sum().sum())
            
            # Simple numeric properties summary
            numeric_df = df.select_dtypes(include=[np.number])
            summary = numeric_df.describe().to_dict()
        except Exception as e:
            flash(f"Error loading preview: {e}", "error")
            
    # Model results if trained in this session
    model_results = session.get('last_model_results')
    model_filename = session.get('last_model_filename', '')
    
    actual_vs_predicted_json = None
    residual_json = None
    confusion_matrix_json = None
    roc_json = None
    feature_importance_json = None
    correlation_plot_json = None
    features_schema = []
    prediction_history = []
    
    if df_exists and os.path.exists(dataset_path):
        # Generate correlation plot for data analysis
        try:
            df_data = pd.read_csv(dataset_path)
            correlation_plot_json = Visualizer.get_correlation_heatmap(df_data)
        except Exception as e:
            print(f"Error generating correlation plot: {e}")
    
    if model_results and model_results.get('task_type') == task_type:
        try:
            # Re-read metrics and load plots
            model_package = PredictionModule.load_model_package(model_filename)
            features_schema = PredictionModule.get_features_schema(model_package)
            prediction_history = PredictionModule.get_prediction_history(session['username'])
            
            # Recreate plotting json strings
            if task_type == 'regression':
                # Since we don't store X_test/y_test/y_pred in the session, we regenerate it briefly to plot
                preprocessor = model_package['preprocessor']
                df_data = pd.read_csv(dataset_path)
                X_train, X_test, y_train, y_test, _, _ = preprocessor.fit_transform(
                    df_data, 
                    model_package['target_column'], 
                    task_type='regression',
                    missing_strategy=model_package['hyperparameters'].get('missing_strategy', 'mean'),
                    scaling_strategy=model_package['hyperparameters'].get('scaling_strategy', 'standard'),
                    test_size=model_package['hyperparameters'].get('test_size', 0.2)
                )
                y_pred = model_package['model'].predict(X_test)
                
                actual_vs_predicted_json = Visualizer.get_actual_vs_predicted(y_test, y_pred)
                residual_json = Visualizer.get_residual_plot(y_test, y_pred)
            else:
                # Classification
                preprocessor = model_package['preprocessor']
                df_data = pd.read_csv(dataset_path)
                X_train, X_test, y_train, y_test, _, _ = preprocessor.fit_transform(
                    df_data, 
                    model_package['target_column'], 
                    task_type='classification',
                    missing_strategy=model_package['hyperparameters'].get('missing_strategy', 'median'),
                    scaling_strategy=model_package['hyperparameters'].get('scaling_strategy', 'standard'),
                    test_size=model_package['hyperparameters'].get('test_size', 0.2)
                )
                y_pred = model_package['model'].predict(X_test)
                cm = np.array(model_package['metrics']['confusion_matrix'])
                classes = np.unique(y_test)
                
                confusion_matrix_json = Visualizer.get_confusion_matrix(cm, classes)
                
                # Check for ROC AUC
                try:
                    y_prob = model_package['model'].predict_proba(X_test)
                    if len(classes) == 2:
                        roc_json = Visualizer.get_roc_curve(y_test, y_prob[:, 1], model_package['algorithm_name'])
                except Exception:
                    pass
            
            # Check for feature importances
            model_obj = model_package['model']
            if hasattr(model_obj, 'feature_importances_'):
                feature_importance_json = Visualizer.get_feature_importance(model_obj.feature_importances_, model_package['features'])
        except Exception as e:
            print(f"Error rebuilding visualizations: {e}")
            
    template_name = 'dashboard/regression.html' if task_type == 'regression' else 'dashboard/classification.html'
    
    return render_template(
        template_name,
        active_page=task_type,
        df_exists=df_exists,
        active_dataset=dataset_name,
        shape=shape,
        columns=columns,
        preview_data=preview_data,
        summary=summary,
        missing_sum=missing_sum,
        model_results=model_results,
        model_filename=model_filename,
        actual_vs_predicted_json=actual_vs_predicted_json,
        residual_json=residual_json,
        confusion_matrix_json=confusion_matrix_json,
        roc_json=roc_json,
        feature_importance_json=feature_importance_json,
        correlation_plot_json=correlation_plot_json,
        features_schema=features_schema,
        prediction_history=prediction_history
    )

# ----------------------------------------------------
# MODEL TRAINING ROUTE
# ----------------------------------------------------
@app.route('/train/supervised/<task_type>', methods=['POST'])
def train_supervised(task_type):
    if not is_logged_in():
        flash("Please log in to train models.", "warning")
        return redirect(url_for('login'))
        
    dataset_path = session.get('active_dataset_path')
    dataset_name = session.get('active_dataset_name')
    
    if not dataset_path or not os.path.exists(dataset_path):
        flash("Dataset path not found. Please upload dataset first.", "error")
        return redirect(url_for('supervised_dashboard', task_type=task_type))

    target_column = request.form.get('target_column')
    missing_strategy = request.form.get('missing_strategy', 'mean')
    scaling_strategy = request.form.get('scaling_strategy', 'standard')
    test_size = float(request.form.get('test_size', 20)) / 100.0
    algorithm = request.form.get('algorithm')
    
    # Gather hyperparameters
    params = {
        'missing_strategy': missing_strategy,
        'scaling_strategy': scaling_strategy,
        'test_size': test_size
    }
    
    if algorithm == 'polynomial_regression':
        params['degree'] = request.form.get('poly_degree', 2)
    elif algorithm in ['ridge_regression', 'lasso_regression']:
        params['alpha'] = request.form.get('alpha', 1.0)
    elif algorithm == 'decision_tree_regression':
        params['max_depth'] = request.form.get('tree_max_depth', '')
        params['min_samples_split'] = request.form.get('tree_min_samples', 2)
    elif algorithm == 'random_forest_regression':
        params['n_estimators'] = request.form.get('forest_n_estimators', 100)
        params['max_depth'] = request.form.get('forest_max_depth', '')
    elif algorithm == 'logistic_regression':
        params['C'] = request.form.get('logistic_C', 1.0)
        params['penalty'] = request.form.get('logistic_penalty', 'l2')
    elif algorithm == 'decision_tree_classifier':
        params['max_depth'] = request.form.get('tree_max_depth', '')
        params['criterion'] = request.form.get('tree_criterion', 'gini')
    elif algorithm == 'random_forest_classifier':
        params['n_estimators'] = request.form.get('forest_n_estimators', 100)
        params['max_depth'] = request.form.get('forest_max_depth', '')
    elif algorithm == 'knn':
        params['n_neighbors'] = request.form.get('knn_n_neighbors', 5)
        params['weights'] = request.form.get('knn_weights', 'uniform')
    elif algorithm == 'svm':
        params['C'] = request.form.get('svm_C', 1.0)
        params['kernel'] = request.form.get('svm_kernel', 'rbf')
    elif algorithm in ['gradient_boosting', 'xgboost']:
        params['n_estimators'] = request.form.get('boosting_n_estimators', 100)
        params['max_iter'] = request.form.get('boosting_n_estimators', 100)
        params['learning_rate'] = request.form.get('boosting_learning_rate', 0.1)

    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Fit Preprocessor
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, feature_names, logs = preprocessor.fit_transform(
            df,
            target_column=target_column,
            task_type=task_type,
            missing_strategy=missing_strategy,
            scaling_strategy=scaling_strategy,
            test_size=test_size
        )
        
        # Train and Evaluate Model
        model_package, model_filename = ModelTrainer.train_and_evaluate(
            algo_id=algorithm,
            params=params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            target_column=target_column,
            dataset_name=dataset_name
        )
        
        # Save results metadata to db collection 'saved_models'
        model_record = {
            'username': session['username'],
            'model_name': model_filename,
            'algorithm_name': model_package['algorithm_name'],
            'dataset_name': dataset_name,
            'task_type': task_type,
            'metrics_json': jsonify(model_package['metrics']).json,
            'filepath': os.path.join(app.config['SAVED_MODELS_FOLDER'], model_filename),
            'features_json': feature_names,
            'target_column': target_column,
            'created_at': time.time()
        }
        db_client.add_document('saved_models', model_record)
        
        # Insert performance metric to Leaderboard
        primary_metric_name = 'r2' if task_type == 'regression' else 'accuracy'
        primary_metric_value = model_package['metrics'][primary_metric_name]
        
        leaderboard_entry = {
            'username': session['username'],
            'algorithm_name': model_package['algorithm_name'],
            'dataset_name': dataset_name,
            'metric_name': primary_metric_name.upper(),
            'metric_value': primary_metric_value,
            'created_at': time.time()
        }
        db_client.add_document('leaderboard', leaderboard_entry)
        
        # Store metadata in session
        session['last_model_results'] = {
            'algorithm_name': model_package['algorithm_name'],
            'task_type': task_type,
            'metrics': model_package['metrics']
        }
        session['last_model_filename'] = model_filename
        
        flash(f"Successfully trained {model_package['algorithm_name']}!", "success")
    except Exception as e:
        flash(f"Training failed: {e}", "error")
        
    return redirect(url_for('supervised_dashboard', task_type=task_type))

# ----------------------------------------------------
# LIVE PREDICTIONS ENDPOINT
# ----------------------------------------------------
@app.route('/predict/supervised', methods=['POST'])
def predict_supervised():
    if not is_logged_in():
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
    data = request.get_json()
    if not data or 'model_filename' not in data:
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400
        
    model_filename = data.pop('model_filename')
    
    try:
        model_package = PredictionModule.load_model_package(model_filename)
        # Execute single row inference
        res = PredictionModule.predict(model_package, data, username=session['username'])
        
        return jsonify({
            'success': True,
            'prediction': res['prediction'],
            'confidence': res['confidence'],
            'history_record': res['history_record']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ----------------------------------------------------
# UNSUPERVISED ROUTE
# ----------------------------------------------------
@app.route('/dashboard/unsupervised/<task_type>')
def unsupervised_dashboard(task_type):
    if not is_logged_in():
        flash("Please login to access the unsupervised dashboard.", "warning")
        return redirect(url_for('login'))
        
    df_exists = False
    shape = (0, 0)
    columns = []
    numeric_columns = []
    preview_data = []
    
    dataset_path = session.get('active_dataset_path')
    dataset_name = session.get('active_dataset_name', '')
    
    if dataset_path and os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            df_exists = True
            shape = df.shape
            columns = df.columns.tolist()
            preview_data = df.head(10).to_dict(orient='records')
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        except Exception as e:
            flash(f"Error reading CSV: {e}", "error")
            
    results = session.get('last_unsupervised_results')
    
    cluster_plot_json = None
    elbow_json = None
    plot2d_json = None
    plot3d_json = None
    variance_json = None
    dim_reduced = False
    
    if results and results.get('task_type') == task_type:
        try:
            # Recreate plotting strings from session results
            if task_type == 'clustering':
                df_data = pd.read_csv(dataset_path)
                # Fill nulls
                features = results['features']
                df_sub = df_data[features].fillna(df_data[features].median())
                labels = results['labels']
                
                # Check dimensions for plotting
                if len(features) > 2:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(df_sub)
                    df_plot = pd.DataFrame(coords, columns=['Component 1', 'Component 2'])
                    cluster_plot_json = Visualizer.get_clustering_plot(df_plot, labels, 'Component 1', 'Component 2')
                    dim_reduced = True
                else:
                    cluster_plot_json = Visualizer.get_clustering_plot(df_sub, labels, features[0], features[1])
                    
                if 'elbow_inertias' in results:
                    elbow_fig = go.Figure()
                    elbow_fig.add_trace(go.Scatter(
                        x=list(range(2, len(results['elbow_inertias'])+2)),
                        y=results['elbow_inertias'],
                        mode='lines+markers',
                        line=dict(color='#7C3AED', width=2)
                    ))
                    elbow_fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='gray',
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    elbow_json = json.dumps(elbow_fig, cls=PlotlyJSONEncoder)
            else:
                # Dimensionality Reduction
                coords = np.array(results['coords'])
                labels = results.get('labels')
                
                plot2d_json = Visualizer.get_pca_2d_plot(coords[:, :2], labels)
                if coords.shape[1] == 3:
                    plot3d_json = Visualizer.get_pca_3d_plot(coords, labels)
                if 'explained_variance' in results:
                    variance_json = Visualizer.get_pca_variance_plot(results['explained_variance'])
        except Exception as e:
            print(f"Error rebuilding unsupervised plots: {e}")

    template_name = 'dashboard/clustering.html' if task_type == 'clustering' else 'dashboard/dimensionality.html'
    
    return render_template(
        template_name,
        active_page=task_type,
        df_exists=df_exists,
        active_dataset=dataset_name,
        shape=shape,
        columns=columns,
        numeric_columns=numeric_columns,
        preview_data=preview_data,
        results=results,
        cluster_plot_json=cluster_plot_json,
        elbow_json=elbow_json,
        plot2d_json=plot2d_json,
        plot3d_json=plot3d_json,
        variance_json=variance_json,
        dim_reduced=dim_reduced
    )

@app.route('/train/clustering', methods=['POST'])
def train_clustering():
    if not is_logged_in():
        flash("Login to access clustering.", "warning")
        return redirect(url_for('login'))
        
    dataset_path = session.get('active_dataset_path')
    dataset_name = session.get('active_dataset_name')
    
    if not dataset_path or not os.path.exists(dataset_path):
        flash("Dataset not loaded.", "error")
        return redirect(url_for('unsupervised_dashboard', task_type='clustering'))

    algorithm = request.form.get('algorithm')
    features = request.form.getlist('cluster_features')
    impute_strategy = request.form.get('impute_strategy', 'median')

    if len(features) < 2:
        flash("Select at least 2 numerical features for clustering.", "error")
        return redirect(url_for('unsupervised_dashboard', task_type='clustering'))

    try:
        df = pd.read_csv(dataset_path)
        # Basic imputation
        df_sub = df[features].copy()
        for col in features:
            fill_val = df_sub[col].median() if impute_strategy == 'median' else df_sub[col].mean()
            df_sub[col] = df_sub[col].fillna(fill_val)

        labels = []
        n_clusters = 0
        silhouette = None
        elbow_inertias = None

        if algorithm == 'kmeans':
            from sklearn.cluster import KMeans
            k = int(request.form.get('kmeans_k', 5))
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(df_sub).tolist()
            n_clusters = k
            
            # Silhouette
            from sklearn.metrics import silhouette_score
            silhouette = float(silhouette_score(df_sub, labels))
            
            # Elbow method
            inertias = []
            for i in range(2, 11):
                km = KMeans(n_clusters=i, random_state=42)
                km.fit(df_sub)
                inertias.append(float(km.inertia_))
            elbow_inertias = inertias
            
        elif algorithm == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            k = int(request.form.get('hierarchical_k', 5))
            linkage = request.form.get('linkage', 'ward')
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = model.fit_predict(df_sub).tolist()
            n_clusters = k
            
            from sklearn.metrics import silhouette_score
            silhouette = float(silhouette_score(df_sub, labels))
            
        elif algorithm == 'dbscan':
            from sklearn.cluster import DBSCAN
            eps = float(request.form.get('dbscan_eps', 3.0))
            min_samples = int(request.form.get('dbscan_min_samples', 5))
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(df_sub).tolist()
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Silhouette (only if clusters > 1)
            if n_clusters > 1:
                from sklearn.metrics import silhouette_score
                # Exclude noise for silhouette
                valid_mask = np.array(labels) != -1
                if sum(valid_mask) > 1:
                    silhouette = float(silhouette_score(df_sub.values[valid_mask], np.array(labels)[valid_mask]))
            
        noise_ratio = float(labels.count(-1) / len(labels)) if -1 in labels else 0.0
        
        session['last_unsupervised_results'] = {
            'task_type': 'clustering',
            'algorithm': algorithm,
            'features': features,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'noise_ratio': noise_ratio,
            'elbow_inertias': elbow_inertias
        }
        flash("Clustering execution complete!", "success")
    except Exception as e:
        flash(f"Clustering failed: {e}", "error")
        
    return redirect(url_for('unsupervised_dashboard', task_type='clustering'))

@app.route('/train/dimensionality', methods=['POST'])
def train_dimensionality():
    if not is_logged_in():
        flash("Please login.", "warning")
        return redirect(url_for('login'))
        
    dataset_path = session.get('active_dataset_path')
    
    if not dataset_path or not os.path.exists(dataset_path):
        flash("Dataset not loaded.", "error")
        return redirect(url_for('unsupervised_dashboard', task_type='dimensionality'))

    algorithm = request.form.get('algorithm')
    features = request.form.getlist('reduction_features')
    color_column = request.form.get('color_column')

    if len(features) < 2:
        flash("Select at least 2 features for reduction.", "error")
        return redirect(url_for('unsupervised_dashboard', task_type='dimensionality'))

    try:
        df = pd.read_csv(dataset_path)
        # basic imputation
        df_sub = df[features].copy()
        for col in features:
            df_sub[col] = df_sub[col].fillna(df_sub[col].median())
            
        # Scale inputs (recommended for PCA/t-SNE)
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(df_sub)
        
        labels = None
        if color_column:
            labels = df[color_column].tolist()
            
        coords = None
        explained_variance = None

        if algorithm == 'pca':
            from sklearn.decomposition import PCA
            n_components = int(request.form.get('pca_components', 3))
            model = PCA(n_components=n_components)
            coords = model.fit_transform(X_scaled).tolist()
            explained_variance = model.explained_variance_ratio_.tolist()
            
        elif algorithm == 'tsne':
            from sklearn.manifold import TSNE
            perplexity = float(request.form.get('tsne_perplexity', 30))
            lr = float(request.form.get('tsne_lr', 200))
            
            # Perplexity must be less than number of samples
            if perplexity >= len(X_scaled):
                perplexity = max(5.0, float(len(X_scaled) - 1))
                
            model = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, random_state=42)
            coords = model.fit_transform(X_scaled).tolist()

        session['last_unsupervised_results'] = {
            'task_type': 'dimensionality',
            'algorithm': algorithm,
            'features': features,
            'coords': coords,
            'labels': labels,
            'explained_variance': explained_variance
        }
        flash("Dimensionality projection complete!", "success")
    except Exception as e:
        flash(f"Projection failed: {e}", "error")
        
    return redirect(url_for('unsupervised_dashboard', task_type='dimensionality'))

# ----------------------------------------------------
# EXTENDED SYSTEM SERVICES (COMPARISON, AUTOML, LEADERBOARD)
# ----------------------------------------------------
@app.route('/dashboard/comparison', methods=['GET', 'POST'])
def model_comparison():
    if not is_logged_in():
        flash("Login to access model comparisons.", "warning")
        return redirect(url_for('login'))
        
    # Fetch all saved models for the user
    saved_records = db_client.query_documents('saved_models', 'username', session['username'])
    
    regression_models = [r for r in saved_records if r['task_type'] == 'regression']
    classification_models = [r for r in saved_records if r['task_type'] == 'classification']
    
    # Parse metrics JSON in models
    for m in regression_models + classification_models:
        try:
            m['metrics_dict'] = m['metrics_json'] if isinstance(m['metrics_json'], dict) else json.loads(m['metrics_json'])
        except Exception:
            m['metrics_dict'] = {}

    selected_ids = []
    comparison_results = []
    task_type = None

    if request.method == 'POST':
        selected_ids = request.form.getlist('selected_models')
        
        # Load details of selected models
        for m_id in selected_ids:
            doc = db_client.get_document('saved_models', m_id)
            if doc:
                try:
                    doc['metrics_dict'] = doc['metrics_json'] if isinstance(doc['metrics_json'], dict) else json.loads(doc['metrics_json'])
                except Exception:
                    doc['metrics_dict'] = {}
                comparison_results.append(doc)
                
        if comparison_results:
            # Check matching task types
            task_types = set([m['task_type'] for m in comparison_results])
            if len(task_types) > 1:
                flash("Cannot compare regression models with classification models together. Select matching types.", "error")
                comparison_results = []
            else:
                task_type = list(task_types)[0]
                
                # Sort rankings based on primary metrics
                if task_type == 'regression':
                    comparison_results.sort(key=lambda x: x['metrics_dict'].get('r2', 0.0), reverse=True)
                else:
                    comparison_results.sort(key=lambda x: x['metrics_dict'].get('accuracy', 0.0), reverse=True)

    return render_template(
        'dashboard/comparison.html',
        active_page='comparison',
        regression_models=regression_models,
        classification_models=classification_models,
        selected_ids=selected_ids,
        comparison_results=comparison_results,
        task_type=task_type
    )

@app.route('/dashboard/automl', methods=['GET', 'POST'])
def automl_recommend():
    if not is_logged_in():
        flash("Login to access the AutoML Advisor.", "warning")
        return redirect(url_for('login'))
        
    df_exists = False
    shape = (0, 0)
    columns = []
    missing_ratio = 0
    recommendation = None
    
    dataset_path = session.get('active_dataset_path')
    dataset_name = session.get('active_dataset_name', '')
    
    if dataset_path and os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            df_exists = True
            shape = df.shape
            columns = df.columns.tolist()
            missing_ratio = float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) if df.size > 0 else 0
            
            # Predict task type based on target column datatype (last column)
            target = columns[-1]
            y = df[target]
            
            # Automatic heuristic check
            is_classification = y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype) or len(y.unique()) < 10
            task_type = 'classification' if is_classification else 'regression'
            
            # Setup rule-based recommendations
            n_rows = shape[0]
            n_cols = shape[1]
            
            recommendation = {
                'task_type': task_type,
                'rationales': [
                    f"Dataset target type identified as: {task_type.upper()}",
                    f"Dataset volume: {n_rows} rows (N)",
                    f"Feature dimension complexity: {n_cols} features (P)",
                    f"Clean cells completeness: {(1 - missing_ratio)*100:.1f}% data cells filled"
                ]
            }
            
            if task_type == 'regression':
                if n_rows < 500:
                    recommendation['id'] = 'ridge_regression'
                    recommendation['name'] = 'Ridge Regression'
                    recommendation['description'] = 'Ideal for smaller regression datasets. Regularization limits weights overfitting on test splits.'
                    recommendation['rationales'].append("N < 500 rows triggers regularized linear regression to block overfitting.")
                    
                    recommendation['alternatives'] = [
                        {'name': 'Linear Regression', 'reason': 'Baseline model', 'desc': 'Simple OLS fit, highly interpretable.'},
                        {'name': 'Decision Tree Regression', 'reason': 'Non-linear splits', 'desc': 'Good alternative if parameters boundaries are strict.'}
                    ]
                else:
                    recommendation['id'] = 'random_forest_regression'
                    recommendation['name'] = 'Random Forest Regressor'
                    recommendation['description'] = 'Robust ensemble bagging tree model. Effectively captures multi-feature non-linear interactions.'
                    recommendation['rationales'].append("Robust datasets sizing (N >= 500) supports deep tree ensembles learning.")
                    
                    recommendation['alternatives'] = [
                        {'name': 'Decision Tree Regressor', 'reason': 'Single tree model', 'desc': 'Faster training time, easy node visualization.'},
                        {'name': 'Ridge Regression', 'reason': 'Regularization fallback', 'desc': 'Preferred if relationships are strictly linear.'}
                    ]
            else:
                # Classification
                if n_rows < 300:
                    recommendation['id'] = 'naive_bayes'
                    recommendation['name'] = 'Gaussian Naive Bayes'
                    recommendation['description'] = 'Fast, simple baseline classifier. Extremely effective on tiny datasets with conditional assumptions.'
                    recommendation['rationales'].append("N < 300 rows triggers probabilistic Naive Bayes to avoid model bias.")
                    
                    recommendation['alternatives'] = [
                        {'name': 'SVM Classifier', 'reason': 'High dimension boundary', 'desc': 'Effective kernel boundary separation.'},
                        {'name': 'Logistic Regression', 'reason': 'Linear probability', 'desc': 'Stable baseline classifier.'}
                    ]
                elif n_cols > 25:
                    recommendation['id'] = 'svm'
                    recommendation['name'] = 'Support Vector Machine (SVM)'
                    recommendation['description'] = 'SVM works extremely well in high-dimensional feature spaces by finding maximum boundary margins.'
                    recommendation['rationales'].append("High features complexity (P > 25) triggers kernel SVM vector maximization.")
                    
                    recommendation['alternatives'] = [
                        {'name': 'Random Forest Classifier', 'reason': 'Forest ensemble', 'desc': 'Less sensitive to scaling, checks importances.'},
                        {'name': 'Logistic Regression', 'reason': 'Ridge penalized', 'desc': 'Fast baseline.'}
                    ]
                else:
                    recommendation['id'] = 'random_forest_classifier'
                    recommendation['name'] = 'Random Forest Classifier'
                    recommendation['description'] = 'Highly reliable bagging ensemble. Minimizes variance mistakes and requires little parameter tuning.'
                    recommendation['rationales'].append("Mid-to-large dataset size fits forest decision trees bagging.")
                    
                    recommendation['alternatives'] = [
                        {'name': 'XGBoost Classifier', 'reason': 'Boosting accuracy', 'desc': 'High accuracy boosting, slightly longer training.'},
                        {'name': 'KNN Classifier', 'reason': 'Instance lookup', 'desc': 'Neighborhood distances vote classifier.'}
                    ]
        except Exception as e:
            flash(f"Analysis failed: {e}", "error")
            
    return render_template(
        'dashboard/automl.html',
        active_page='automl',
        df_exists=df_exists,
        active_dataset=dataset_name,
        shape=shape,
        missing_ratio=missing_ratio,
        recommendation=recommendation
    )

@app.route('/dashboard/leaderboard')
def leaderboard_view():
    # Retrieve all leaderboard submissions
    records = db_client.get_documents('leaderboard')
    
    classification_leaderboard = [r for r in records if r.get('metric_name') == 'ACCURACY']
    regression_leaderboard = [r for r in records if r.get('metric_name') == 'R2']
    
    # Sort descending
    classification_leaderboard.sort(key=lambda x: x.get('metric_value', 0.0), reverse=True)
    regression_leaderboard.sort(key=lambda x: x.get('metric_value', 0.0), reverse=True)
    
    return render_template(
        'dashboard/leaderboard.html',
        active_page='leaderboard',
        classification_leaderboard=classification_leaderboard,
        regression_leaderboard=regression_leaderboard
    )

# ----------------------------------------------------
# FILE DOWNLOAD SERVICE ENDPOINTS
# ----------------------------------------------------
@app.route('/download/model/<filename>')
def download_model(filename):
    if not is_logged_in():
        flash("Access denied.", "error")
        return redirect(url_for('login'))
        
    return send_from_directory(app.config['SAVED_MODELS_FOLDER'], filename, as_attachment=True)

@app.route('/download/report/<file_format>/<filename>')
def download_report(file_format, filename):
    if not is_logged_in():
        flash("Access denied.", "error")
        return redirect(url_for('login'))
        
    try:
        model_package = PredictionModule.load_model_package(filename)
        
        # Load dataset to generate evaluations charts
        dataset_path = session.get('active_dataset_path')
        df_data = pd.read_csv(dataset_path)
        
        preprocessor = model_package['preprocessor']
        X_train, X_test, y_train, y_test, _, _ = preprocessor.fit_transform(
            df_data, 
            model_package['target_column'], 
            task_type=model_package['task_type'],
            missing_strategy=model_package['hyperparameters'].get('missing_strategy', 'mean'),
            scaling_strategy=model_package['hyperparameters'].get('scaling_strategy', 'standard'),
            test_size=model_package['hyperparameters'].get('test_size', 0.2)
        )
        y_pred = model_package['model'].predict(X_test)
        
        if file_format == 'csv':
            report_path, out_name = ReportGenerator.generate_csv_report(model_package)
        else:
            # PDF
            report_path, out_name = ReportGenerator.generate_pdf_report(model_package, X_test, y_test, y_pred)
            
        return send_from_directory(app.config['REPORTS_FOLDER'], out_name, as_attachment=True)
    except Exception as e:
        flash(f"Report download failed: {e}", "error")
        return redirect(url_for('supervised_dashboard', task_type=model_package.get('task_type', 'regression')))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
