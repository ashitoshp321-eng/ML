import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless mode for web server safety
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from config import Config

class Visualizer:
    @staticmethod
    def _get_unique_plot_path(filename):
        """Generates a unique path for static plots."""
        filename_clean = f"{int(time.time())}_{filename}"
        path = os.path.join(Config.BASE_DIR, 'static', 'plots', filename_clean)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path, f"/static/plots/{filename_clean}"

    # ----------------------------------------------------
    # DATASET ANALYSIS VISUALIZATIONS
    # ----------------------------------------------------
    @staticmethod
    def get_correlation_heatmap(df):
        """Generates a Plotly correlation heatmap JSON."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return None
        
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Matrix',
            aspect='auto'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    @staticmethod
    def get_distribution_plot(df, column):
        """Generates a Plotly distribution plot (histogram + density estimation)."""
        if df[column].dtype == 'object' or isinstance(df[column].dtype, pd.CategoricalDtype):
            # Categorical bar count
            counts = df[column].value_counts().reset_index()
            counts.columns = [column, 'Count']
            fig = px.bar(counts, x=column, y='Count', color=column, title=f"Value Distribution: {column}")
        else:
            # Numerical histogram
            fig = px.histogram(df, x=column, marginal="box", title=f"Value Distribution: {column}", color_discrete_sequence=['#4F46E5'])
            
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    # ----------------------------------------------------
    # REGRESSION DIAGNOSTIC VISUALIZATIONS
    # ----------------------------------------------------
    @staticmethod
    def get_actual_vs_predicted(y_test, y_pred):
        """Generates actual vs predicted scatter plot with unity line."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode='markers',
            marker=dict(color='#7C3AED', opacity=0.6, size=8),
            name='Predictions'
        ))
        
        # Unity line (y = x)
        mn = min(min(y_test), min(y_pred))
        mx = max(max(y_test), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode='lines',
            line=dict(color='#06B6D4', width=2, dash='dash'),
            name='Ideal Fit'
        ))
        
        fig.update_layout(
            title='Actual vs. Predicted Values',
            xaxis_title='Actual values',
            yaxis_title='Predicted values',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    @staticmethod
    def get_residual_plot(y_test, y_pred):
        """Generates residual plot (Residuals vs. Predicted)."""
        residuals = y_test - y_pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            marker=dict(color='#ef4444', opacity=0.6, size=8),
            name='Residuals'
        ))
        # Zero line
        fig.add_trace(go.Scatter(
            x=[min(y_pred), max(y_pred)], y=[0, 0],
            mode='lines',
            line=dict(color='#4F46E5', width=2),
            name='Zero Line'
        ))
        fig.update_layout(
            title='Residuals vs. Predicted Values',
            xaxis_title='Predicted values',
            yaxis_title='Residuals (Error)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    # ----------------------------------------------------
    # CLASSIFICATION DIAGNOSTIC VISUALIZATIONS
    # ----------------------------------------------------
    @staticmethod
    def get_confusion_matrix(cm, classes):
        """Confusion Matrix heatmap plot."""
        fig = px.imshow(
            cm,
            x=[str(c) for c in classes],
            y=[str(c) for c in classes],
            text_auto=True,
            color_continuous_scale='Purples',
            title='Confusion Matrix'
        )
        fig.update_layout(
            xaxis_title='Predicted Class',
            yaxis_title='Actual Class',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    @staticmethod
    def get_roc_curve(y_test, y_prob_pos, algo_name=""):
        """Plotly ROC Curve for classification."""
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            line=dict(color='#4F46E5', width=3),
            name=f'ROC Curve (AUC = {roc_auc:.3f})'
        ))
        # Random guess line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='gray', width=1.5, dash='dash'),
            name='Random Guess'
        ))
        
        fig.update_layout(
            title=f'Receiver Operating Characteristic (ROC) - {algo_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    # ----------------------------------------------------
    # FEATURE IMPORTANCE
    # ----------------------------------------------------
    @staticmethod
    def get_feature_importance(importances, feature_names):
        """Generates feature importance horizontal bar chart."""
        indices = np.argsort(importances)
        sorted_features = [feature_names[i] for i in indices][-15:]  # Top 15 features
        sorted_importances = [importances[i] for i in indices][-15:]
        
        fig = px.bar(
            x=sorted_importances,
            y=sorted_features,
            orientation='h',
            labels={'x': 'Relative Importance', 'y': 'Feature'},
            title='Feature Importance Analysis',
            color_discrete_sequence=['#06B6D4']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    # ----------------------------------------------------
    # STATIC MATPLOTLIB GENERATORS FOR REPORTLAB PDF
    # ----------------------------------------------------
    @staticmethod
    def save_static_confusion_matrix(cm, classes):
        """Saves a static Confusion Matrix image and returns its local web path."""
        path, web_path = Visualizer._get_unique_plot_path("confusion_matrix.png")
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes, cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    @staticmethod
    def save_static_actual_vs_predicted(y_test, y_pred):
        """Saves a static scatter plot of actual vs predicted."""
        path, web_path = Visualizer._get_unique_plot_path("actual_vs_pred.png")
        
        plt.figure(figsize=(5, 4))
        plt.scatter(y_test, y_pred, color='#7C3AED', alpha=0.5, edgecolors='none')
        mn = min(min(y_test), min(y_pred))
        mx = max(max(y_test), max(y_pred))
        plt.plot([mn, mx], [mn, mx], color='#06B6D4', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    # ----------------------------------------------------
    # UNSUPERVISED & DIMENSIONALITY REDUCTION VISUALS
    # ----------------------------------------------------
    @staticmethod
    def get_clustering_plot(df, labels, x_col, y_col):
        """Generates 2D scatter of clustering results."""
        df_plot = df.copy()
        df_plot['Cluster'] = [f"Cluster {l}" if l != -1 else 'Noise' for l in labels]
        
        fig = px.scatter(
            df_plot, x=x_col, y=y_col,
            color='Cluster',
            title=f"Clustering Cluster Space ({x_col} vs {y_col})",
            template='plotly_white'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    @staticmethod
    def get_pca_variance_plot(explained_variance):
        """Generates Explained Variance bar/line chart."""
        variance_ratio = explained_variance
        cumulative_variance = np.cumsum(variance_ratio)
        
        fig = go.Figure()
        # Individual variance bars
        fig.add_trace(go.Bar(
            x=[f"PC {i+1}" for i in range(len(variance_ratio))],
            y=variance_ratio,
            name='Individual Variance Explained',
            marker_color='#4F46E5'
        ))
        # Cumulative variance line
        fig.add_trace(go.Scatter(
            x=[f"PC {i+1}" for i in range(len(variance_ratio))],
            y=cumulative_variance,
            name='Cumulative Variance Explained',
            line=dict(color='#06B6D4', width=3)
        ))
        
        fig.update_layout(
            title='Explained Variance by Principal Components',
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance Ratio',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    @staticmethod
    def get_pca_2d_plot(coords, labels=None):
        """Generates 2D PCA representation scatter plot."""
        df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
        if labels is not None:
            df['Target'] = [str(l) for l in labels]
            fig = px.scatter(df, x='PC1', y='PC2', color='Target', title='2D Projection representation')
        else:
            fig = px.scatter(df, x='PC1', y='PC2', title='2D Projection representation', color_discrete_sequence=['#7C3AED'])
            
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    @staticmethod
    def get_pca_3d_plot(coords, labels=None):
        """Generates interactive 3D PCA projection."""
        df = pd.DataFrame(coords, columns=['PC1', 'PC2', 'PC3'])
        if labels is not None:
            df['Target'] = [str(l) for l in labels]
            fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Target', title='3D Projection representation')
        else:
            fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', title='3D Projection representation', color_discrete_sequence=['#7C3AED'])
            
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='gray',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)
