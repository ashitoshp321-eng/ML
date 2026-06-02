import os
import csv
import time
import datetime
import numpy as np
from config import Config
from utils.visualization import Visualizer

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class ReportGenerator:
    @staticmethod
    def generate_csv_report(model_package, filename=None):
        """Generates a CSV report summarizing the model performance metrics."""
        if not filename:
            filename = f"report_{model_package['algorithm_id']}_{int(time.time())}.csv"
        
        path = os.path.join(Config.REPORTS_FOLDER, filename)
        
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Title block
            writer.writerow(["Machine Learning Testing Platform - Model Report"])
            writer.writerow(["Generated At", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            
            # Model Info
            writer.writerow(["Model Metadata"])
            writer.writerow(["Algorithm", model_package['algorithm_name']])
            writer.writerow(["Dataset", model_package['dataset_name']])
            writer.writerow(["Task Type", model_package['task_type']])
            writer.writerow(["Target Column", model_package['target_column']])
            writer.writerow(["Training Time (s)", f"{model_package['train_time']:.4f}"])
            writer.writerow([])
            
            # Hyperparameters
            writer.writerow(["Model Hyperparameters"])
            for k, v in model_package['hyperparameters'].items():
                writer.writerow([k, v])
            writer.writerow([])
            
            # Metrics
            writer.writerow(["Evaluation Metrics"])
            for k, v in model_package['metrics'].items():
                if k != 'confusion_matrix':
                    writer.writerow([k.upper(), f"{v:.5f}" if isinstance(v, float) else v])
                    
        return path, filename

    @staticmethod
    def generate_pdf_report(model_package, X_test, y_test, y_pred, filename=None):
        """Generates a high-quality PDF summary report including evaluations and charts."""
        if not filename:
            filename = f"report_{model_package['algorithm_id']}_{int(time.time())}.pdf"
            
        path = os.path.join(Config.REPORTS_FOLDER, filename)
        
        # Build ReportLab Document template
        doc = SimpleDocTemplate(
            path,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=40, bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        
        # Define brand custom styles
        primary_color = colors.HexColor("#4F46E5")
        secondary_color = colors.HexColor("#7C3AED")
        accent_color = colors.HexColor("#06B6D4")
        dark_text = colors.HexColor("#0f172a")
        
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=primary_color,
            spaceAfter=15
        )
        
        section_style = ParagraphStyle(
            'ReportSection',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=secondary_color,
            spaceBefore=15,
            spaceAfter=10
        )
        
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=dark_text,
            leading=14,
            spaceAfter=8
        )
        
        header_table_style = TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), primary_color),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 16),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ])
        
        metrics_table_style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ])

        story = []
        
        # 1. Header Banner
        header_data = [[f"MODEL EVALUATION REPORT: {model_package['algorithm_name'].upper()}"]]
        header_table = Table(header_data, colWidths=[530])
        header_table.setStyle(header_table_style)
        story.append(header_table)
        story.append(Spacer(1, 15))
        
        # 2. Metadata Section
        story.append(Paragraph("1. Metadata Summary", section_style))
        meta_intro = f"This diagnostic report evaluates the performance of the <b>{model_package['algorithm_name']}</b> algorithm trained on the dataset <b>{model_package['dataset_name']}</b>, target variable <b>'{model_package['target_column']}'</b>."
        story.append(Paragraph(meta_intro, body_style))
        
        metadata_records = [
            [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style)],
            ["Algorithm ID", model_package['algorithm_id']],
            ["Task Type", model_package['task_type'].capitalize()],
            ["Total Target Classes", str(len(np.unique(y_test))) if model_package['task_type'] == 'classification' else 'N/A (Continuous)'],
            ["Features Count", str(len(model_package['features']))],
            ["Training Time (seconds)", f"{model_package['train_time']:.4f} seconds"],
            ["Evaluation Timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        meta_table = Table(metadata_records, colWidths=[200, 330])
        meta_table.setStyle(metrics_table_style)
        story.append(meta_table)
        story.append(Spacer(1, 15))
        
        # 3. Model Hyperparameters
        story.append(Paragraph("2. Training Parameters", section_style))
        param_records = [[Paragraph("<b>Hyperparameter Key</b>", body_style), Paragraph("<b>Configured Value</b>", body_style)]]
        for k, v in model_package['hyperparameters'].items():
            param_records.append([str(k), str(v)])
            
        param_table = Table(param_records, colWidths=[200, 330])
        param_table.setStyle(metrics_table_style)
        story.append(param_table)
        story.append(Spacer(1, 15))
        
        # 4. Evaluation Metrics
        story.append(Paragraph("3. Evaluation Results", section_style))
        metric_records = [[Paragraph("<b>Metric Name</b>", body_style), Paragraph("<b>Performance Score</b>", body_style)]]
        for k, v in model_package['metrics'].items():
            if k != 'confusion_matrix':
                metric_records.append([k.upper().replace('_', ' '), f"{v:.5f}" if isinstance(v, float) else str(v)])
                
        metric_table = Table(metric_records, colWidths=[200, 330])
        metric_table.setStyle(metrics_table_style)
        story.append(metric_table)
        story.append(Spacer(1, 20))
        
        # Page break before charts to make layout premium
        story.append(PageBreak())
        
        # 5. Visualizations Section
        story.append(Paragraph("4. Diagnostic Visualizations", section_style))
        
        chart_elements = []
        if model_package['task_type'] == 'classification':
            cm = np.array(model_package['metrics']['confusion_matrix'])
            classes = np.unique(y_test)
            # Save static confusion matrix
            plot_path = Visualizer.save_static_confusion_matrix(cm, classes)
            img = Image(plot_path, width=320, height=256)
            chart_elements.append(Paragraph("<b>Confusion Matrix Heatmap</b>", body_style))
            chart_elements.append(img)
        else:
            # Regression Actual vs Predicted plot
            plot_path = Visualizer.save_static_actual_vs_predicted(y_test, y_pred)
            img = Image(plot_path, width=320, height=256)
            chart_elements.append(Paragraph("<b>Actual vs. Predicted Scatter Fit</b>", body_style))
            chart_elements.append(img)
            
        story.append(KeepTogether(chart_elements))
        story.append(Spacer(1, 15))
        
        # Build Document
        doc.build(story)
        
        return path, filename
