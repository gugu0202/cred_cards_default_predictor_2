"""
Мониторинг дрифта данных и концептуального дрифта с использованием Evidently AI
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Evidently imports
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
    ColumnSummaryMetric,
    RegressionQualityMetric,
    ClassificationQualityMetric,
    TargetDriftMetric,
    DataQualityTable,
    DatasetSummaryMetric
)
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.workspace import Workspace, Project
import mlflow

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftMonitor:
    """Класс для мониторинга дрифта данных и моделей"""
    
    def __init__(self, project_name: str = "credit-scoring"):
        self.project_name = project_name
        self.reference_data = None
        self.current_data = None
        self.drift_results = {}
        
        # Создаем директории для отчетов
        self.report_dir = Path("../monitoring/reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка Evidently workspace
        self.workspace_path = Path("../monitoring/evidently_workspace")
        self.workspace_path.mkdir(parents=True, exist_ok=True)
    
    def load_reference_data(self, data_path: str) -> pd.DataFrame:
        """Загрузка референсных данных (тренировочных)"""
        logger.info(f"Загрузка референсных данных из {data_path}")
        
        if data_path.endswith('.csv'):
            self.reference_data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            self.reference_data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Не поддерживаемый формат файла: {data_path}")
        
        # Преобразование типов для категориальных признаков
        categorical_cols = ['employment_status', 'home_ownership', 'loan_purpose', 'marital_status']
        for col in categorical_cols:
            if col in self.reference_data.columns:
                self.reference_data[col] = self.reference_data[col].astype('category')
        
        logger.info(f"Референсные данные загружены: {self.reference_data.shape}")
        return self.reference_data
    
    def load_current_data(self, data_path: str, days_back: int = 7) -> pd.DataFrame:
        """Загрузка текущих данных (например, за последнюю неделю)"""
        logger.info(f"Загрузка текущих данных из {data_path}")
        
        if data_path.endswith('.csv'):
            self.current_data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            self.current_data = pd.read_parquet(data_path)
        else:
            # Симуляция данных для демонстрации
            self.current_data = self._simulate_current_data(days_back)
        
        # Преобразование типов для категориальных признаков
        categorical_cols = ['employment_status', 'home_ownership', 'loan_purpose', 'marital_status']
        for col in categorical_cols:
            if col in self.current_data.columns:
                self.current_data[col] = self.current_data[col].astype('category')
        
        logger.info(f"Текущие данные загружены: {self.current_data.shape}")
        return self.current_data
    
    def _simulate_current_data(self, days_back: int) -> pd.DataFrame:
        """Симуляция текущих данных с дрифтом"""
        np.random.seed(42)
        n_samples = 1000
        
        # Базовые данные
        data = {
            'age': np.random.normal(45, 15, n_samples).astype(int),
            'income': np.random.lognormal(10.5, 0.8, n_samples),
            'loan_amount': np.random.uniform(10000, 100000, n_samples),
            'credit_history_length': np.random.exponential(10, n_samples).astype(int),
            'debt_to_income_ratio': np.random.beta(2, 5, n_samples) * 100,
            'employment_status': np.random.choice(['employed', 'self-employed', 'unemployed'], n_samples),
            'home_ownership': np.random.choice(['mortgage', 'own', 'rent'], n_samples),
            'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'business'], n_samples),
            'marital_status': np.random.choice(['married', 'single', 'divorced'], n_samples),
            'target': np.random.binomial(1, 0.3, n_samples)  # Симулированные метки
        }
        
        # Добавляем дрифт
        data['age'] = data['age'] + np.random.normal(5, 2, n_samples)  # Сдвиг распределения
        data['income'] = data['income'] * 1.1  # Увеличение дохода
        data['loan_amount'] = data['loan_amount'] * 1.2  # Увеличение суммы кредита
        
        return pd.DataFrame(data)
    
    def detect_data_drift(self) -> Dict:
        """Детектирование дрифта данных"""
        logger.info("Детектирование дрифта данных...")
        
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Сначала загрузите референсные и текущие данные")
        
        # Создаем отчет о дрифте данных
        data_drift_report = Report(metrics=[
            DataDriftTable(),
            DatasetDriftMetric(),
            DatasetSummaryMetric(),
            DataQualityTable()
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )
        
        # Сохраняем отчет
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.report_dir / f"data_drift_report_{timestamp}.html"
        data_drift_report.save_html(str(html_path))
        
        # Получаем результаты
        report_result = data_drift_report.as_dict()
        
        # Извлекаем ключевые метрики
        drift_metrics = {
            'dataset_drift_detected': report_result['metrics'][1]['result']['drift_detected'],
            'dataset_drift_score': report_result['metrics'][1]['result']['drift_score'],
            'number_of_drifted_columns': report_result['metrics'][0]['result']['number_of_drifted_columns'],
            'share_of_drifted_columns': report_result['metrics'][0]['result']['share_of_drifted_columns'],
            'timestamp': timestamp,
            'report_path': str(html_path)
        }
        
        # Детальный анализ по колонкам
        column_drifts = []
        for column in report_result['metrics'][0]['result']['drift_by_columns'].keys():
            col_result = report_result['metrics'][0]['result']['drift_by_columns'][column]
            column_drifts.append({
                'column': column,
                'drift_detected': col_result['drift_detected'],
                'drift_score': col_result['drift_score'],
                'current_distribution': col_result.get('current_distribution', {}),
                'reference_distribution': col_result.get('reference_distribution', {})
            })
        
        drift_metrics['column_drifts'] = column_drifts
        
        # Логирование в MLflow
        with mlflow.start_run(run_name=f"data_drift_{timestamp}"):
            mlflow.log_metric("dataset_drift_score", drift_metrics['dataset_drift_score'])
            mlflow.log_metric("drifted_columns", drift_metrics['number_of_drifted_columns'])
            mlflow.log_artifact(str(html_path), "drift_reports")
        
        logger.info(f"Дрифт данных обнаружен: {drift_metrics['dataset_drift_detected']}")
        logger.info(f"Счет дрифта: {drift_metrics['dataset_drift_score']:.3f}")
        
        self.drift_results['data_drift'] = drift_metrics
        return drift_metrics
    
    def detect_concept_drift(self, y_true: pd.Series, y_pred: pd.Series) -> Dict:
        """Детектирование концептуального дрифта"""
        logger.info("Детектирование концептуального дрифта...")
        
        # Создаем датафреймы с предсказаниями
        reference_predictions = pd.DataFrame({
            'prediction': y_pred[:len(self.reference_data)],
            'target': y_true[:len(self.reference_data)]
        })
        
        current_predictions = pd.DataFrame({
            'prediction': y_pred[-len(self.current_data):],
            'target': y_true[-len(self.current_data):]
        })
        
        # Анализ концептуального дрифта
        concept_drift_report = Report(metrics=[
            TargetDriftMetric(),
            ClassificationQualityMetric(),
            ColumnDriftMetric(column_name='prediction'),
            ColumnDriftMetric(column_name='target')
        ])
        
        concept_drift_report.run(
            reference_data=reference_predictions,
            current_data=current_predictions
        )
        
        # Сохраняем отчет
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.report_dir / f"concept_drift_report_{timestamp}.html"
        concept_drift_report.save_html(str(html_path))
        
        # Получаем результаты
        report_result = concept_drift_report.as_dict()
        
        concept_metrics = {
            'target_drift_detected': report_result['metrics'][0]['result']['drift_detected'],
            'target_drift_score': report_result['metrics'][0]['result']['drift_score'],
            'prediction_drift_detected': report_result['metrics'][2]['result']['drift_detected'],
            'prediction_drift_score': report_result['metrics'][2]['result']['drift_score'],
            'accuracy_reference': report_result['metrics'][1]['result']['reference']['accuracy'],
            'accuracy_current': report_result['metrics'][1]['result']['current']['accuracy'],
            'accuracy_difference': report_result['metrics'][1]['result']['current']['accuracy'] - 
                                 report_result['metrics'][1]['result']['reference']['accuracy'],
            'timestamp': timestamp,
            'report_path': str(html_path)
        }
        
        # Логирование в MLflow
        with mlflow.start_run(run_name=f"concept_drift_{timestamp}"):
            mlflow.log_metric("target_drift_score", concept_metrics['target_drift_score'])
            mlflow.log_metric("prediction_drift_score", concept_metrics['prediction_drift_score'])
            mlflow.log_metric("accuracy_difference", concept_metrics['accuracy_difference'])
            mlflow.log_artifact(str(html_path), "concept_drift_reports")
        
        logger.info(f"Концептуальный дрифт обнаружен: {concept_metrics['target_drift_detected']}")
        logger.info(f"Разница в точности: {concept_metrics['accuracy_difference']:.3f}")
        
        self.drift_results['concept_drift'] = concept_metrics
        return concept_metrics
    
    def detect_model_performance_decay(self, y_true: pd.Series, y_pred: pd.Series, 
                                      y_pred_proba: pd.Series) -> Dict:
        """Детектирование деградации производительности модели"""
        logger.info("Анализ деградации производительности модели...")
        
        # Разделяем данные на временные интервалы
        n_intervals = 4
        interval_size = len(y_true) // n_intervals
        
        performance_metrics = []
        
        for i in range(n_intervals):
            start_idx = i * interval_size
            end_idx = (i + 1) * interval_size if i < n_intervals - 1 else len(y_true)
            
            y_true_interval = y_true.iloc[start_idx:end_idx]
            y_pred_interval = y_pred.iloc[start_idx:end_idx]
            y_pred_proba_interval = y_pred_proba.iloc[start_idx:end_idx]
            
            # Вычисляем метрики для интервала
            accuracy = (y_true_interval == y_pred_interval).mean()
            roc_auc = self._calculate_roc_auc(y_true_interval, y_pred_proba_interval)
            f1_score = self._calculate_f1_score(y_true_interval, y_pred_interval)
            
            performance_metrics.append({
                'interval': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'f1_score': f1_score,
                'sample_size': len(y_true_interval)
            })
        
        # Анализ тренда
        accuracy_trend = np.polyfit(range(n_intervals), 
                                   [m['accuracy'] for m in performance_metrics], 1)[0]
        
        decay_metrics = {
            'performance_intervals': performance_metrics,
            'accuracy_trend_slope': accuracy_trend,
            'is_decaying': accuracy_trend < -0.01,  # Порог деградации
            'average_accuracy': np.mean([m['accuracy'] for m in performance_metrics]),
            'accuracy_std': np.std([m['accuracy'] for m in performance_metrics])
        }
        
        # Визуализация тренда
        self._plot_performance_trend(performance_metrics)
        
        logger.info(f"Тренд точности: {accuracy_trend:.4f}")
        logger.info(f"Деградация обнаружена: {decay_metrics['is_decaying']}")
        
        self.drift_results['performance_decay'] = decay_metrics
        return decay_metrics
    
    def _calculate_roc_auc(self, y_true: pd.Series, y_pred_proba: pd.Series) -> float:
        """Вычисление ROC AUC"""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_pred_proba)
        except:
            return 0.5
    
    def _calculate_f1_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Вычисление F1 score"""
        from sklearn.metrics import f1_score
        try:
            return f1_score(y_true, y_pred)
        except:
            return 0.0
    
    def _plot_performance_trend(self, performance_metrics: list):
        """Визуализация тренда производительности"""
        import matplotlib.pyplot as plt
        
        intervals = [m['interval'] for m in performance_metrics]
        accuracies = [m['accuracy'] for m in performance_metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(intervals, accuracies, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Временной интервал')
        plt.ylabel('Точность')
        plt.title('Тренд производительности модели во времени')
        plt.grid(True, alpha=0.3)
        
        # Добавляем линию тренда
        z = np.polyfit(intervals, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(intervals, p(intervals), "r--", alpha=0.7, label=f'Тренд: {z[0]:.4f}')
        
        plt.legend()
        
        # Сохраняем график
        plot_path = self.report_dir / f"performance_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_drift_analysis(self, reference_path: str, current_path: str, 
                                        y_true: pd.Series = None, y_pred: pd.Series = None,
                                        y_pred_proba: pd.Series = None) -> Dict:
        """Комплексный анализ всех типов дрифта"""
        logger.info("Запуск комплексного анализа дрифта...")
        
        # Загрузка данных
        self.load_reference_data(reference_path)
        self.load_current_data(current_path)
        
        # Анализ дрифта данных
        data_drift_results = self.detect_data_drift()
        
        # Анализ концептуального дрифта (если есть метки)
        concept_drift_results = None
        if y_true is not None and y_pred is not None:
            concept_drift_results = self.detect_concept_drift(y_true, y_pred)
        
        # Анализ деградации производительности
        performance_decay_results = None
        if y_true is not None and y_pred is not None and y_pred_proba is not None:
            performance_decay_results = self.detect_model_performance_decay(y_true, y_pred, y_pred_proba)
        
        # Комплексный отчет
        comprehensive_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_drift': data_drift_results,
            'concept_drift': concept_drift_results,
            'performance_decay': performance_decay_results,
            'summary': self._generate_drift_summary()
        }
        
        # Сохраняем полный отчет
        report_path = self.report_dir / f"comprehensive_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        # Логирование в MLflow
        self._log_to_mlflow(comprehensive_report)
        
        # Генерация HTML дашборда
        self._generate_drift_dashboard(comprehensive_report)
        
        logger.info("Комплексный анализ дрифта завершен")
        return comprehensive_report
    
    def _generate_drift_summary(self) -> Dict:
        """Генерация сводки по дрифту"""
        summary = {
            'overall_status': 'PASS',
            'alerts': [],
            'recommendations': []
        }
        
        # Проверка дрифта данных
        if 'data_drift' in self.drift_results:
            data_drift = self.drift_results['data_drift']
            if data_drift['dataset_drift_detected']:
                summary['overall_status'] = 'WARNING'
                summary['alerts'].append({
                    'type': 'data_drift',
                    'severity': 'warning',
                    'message': f'Дрифт данных обнаружен. Счет: {data_drift["dataset_drift_score"]:.3f}'
                })
                summary['recommendations'].append(
                    "Проверить входные данные и при необходимости обновить модель"
                )
        
        # Проверка концептуального дрифта
        if 'concept_drift' in self.drift_results:
            concept_drift = self.drift_results['concept_drift']
            if concept_drift['target_drift_detected']:
                summary['overall_status'] = 'WARNING'
                summary['alerts'].append({
                    'type': 'concept_drift',
                    'severity': 'warning',
                    'message': f'Концептуальный дрифт обнаружен. Разница в точности: {concept_drift["accuracy_difference"]:.3f}'
                })
                summary['recommendations'].append(
                    "Рассмотреть переобучение модели на новых данных"
                )
        
        # Проверка деградации производительности
        if 'performance_decay' in self.drift_results:
            performance_decay = self.drift_results['performance_decay']
            if performance_decay['is_decaying']:
                summary['overall_status'] = 'CRITICAL'
                summary['alerts'].append({
                    'type': 'performance_decay',
                    'severity': 'critical',
                    'message': f'Деградация производительности обнаружена. Тренд: {performance_decay["accuracy_trend_slope"]:.4f}'
                })
                summary['recommendations'].append(
                    "Срочное переобучение модели требуется"
                )
        
        return summary
    
    def _log_to_mlflow(self, report: Dict):
        """Логирование результатов в MLflow"""
        try:
            with mlflow.start_run(run_name="drift_monitoring"):
                # Логируем метрики
                if report['data_drift']:
                    mlflow.log_metric("data_drift_score", report['data_drift']['dataset_drift_score'])
                    mlflow.log_metric("drifted_columns", report['data_drift']['number_of_drifted_columns'])
                
                if report['concept_drift']:
                    mlflow.log_metric("concept_drift_score", report['concept_drift']['target_drift_score'])
                    mlflow.log_metric("accuracy_difference", report['concept_drift']['accuracy_difference'])
                
                if report['performance_decay']:
                    mlflow.log_metric("performance_decay", report['performance_decay']['is_decaying'])
                    mlflow.log_metric("accuracy_trend", report['performance_decay']['accuracy_trend_slope'])
                
                # Логируем отчет
                mlflow.log_dict(report, "drift_report.json")
                
                # Логируем статус
                mlflow.log_param("drift_status", report['summary']['overall_status'])
        except Exception as e:
            logger.warning(f"Не удалось залогировать в MLflow: {e}")
    
    def _generate_drift_dashboard(self, report: Dict):
        """Генерация HTML дашборда"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Credit Scoring - Drift Monitoring Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .card {{ background: white; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .status {{ font-size: 24px; font-weight: bold; padding: 10px; border-radius: 5px; }}
                .status-pass {{ background: #d4edda; color: #155724; }}
                .status-warning {{ background: #fff3cd; color: #856404; }}
                .status-critical {{ background: #f8d7da; color: #721c24; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #e9ecef; border-radius: 5px; }}
                .alert {{ padding: 15px; margin: 10px 0; border-left: 4px solid; }}
                .alert-warning {{ background: #fff3cd; border-color: #ffc107; }}
                .alert-critical {{ background: #f8d7da; border-color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Credit Scoring - Drift Monitoring Dashboard</h1>
                    <p>Дата анализа: {report['analysis_timestamp']}</p>
                </div>
                
                <div class="card">
                    <h2>📈 Общий статус</h2>
                    <div class="status status-{report['summary']['overall_status'].lower()}">
                        Статус: {report['summary']['overall_status']}
                    </div>
                </div>
                
                <div class="card">
                    <h2>📊 Дрифт данных</h2>
        """
        
        if report['data_drift']:
            data_drift = report['data_drift']
            html_content += f"""
                    <div class="metric">
                        <strong>Счет дрифта:</strong><br>
                        {data_drift['dataset_drift_score']:.3f}
                    </div>
                    <div class="metric">
                        <strong>Дрифтующих колонок:</strong><br>
                        {data_drift['number_of_drifted_columns']}
                    </div>
                    <div class="metric">
                        <strong>Обнаружен дрифт:</strong><br>
                        {'✅ Да' if data_drift['dataset_drift_detected'] else '❌ Нет'}
                    </div>
                    
                    <h3>Детали по колонкам:</h3>
                    <table>
                        <tr>
                            <th>Колонка</th>
                            <th>Дрифт обнаружен</th>
                            <th>Счет дрифта</th>
                        </tr>
            """
            
            for col_drift in data_drift.get('column_drifts', [])[:10]:  # Показываем первые 10
                html_content += f"""
                        <tr>
                            <td>{col_drift['column']}</td>
                            <td>{'✅ Да' if col_drift['drift_detected'] else '❌ Нет'}</td>
                            <td>{col_drift['drift_score']:.3f}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                    <p><a href="data: отчет по дрифту данных">📎 Открыть полный отчет</a></p>
            """
        
        html_content += """
                </div>
                
                <div class="card">
                    <h2>🎯 Концептуальный дрифт</h2>
        """
        
        if report['concept_drift']:
            concept_drift = report['concept_drift']
            html_content += f"""
                    <div class="metric">
                        <strong>Счет дрифта цели:</strong><br>
                        {concept_drift['target_drift_score']:.3f}
                    </div>
                    <div class="metric">
                        <strong>Разница в точности:</strong><br>
                        {concept_drift['accuracy_difference']:.3f}
                    </div>
                    <div class="metric">
                        <strong>Точность (референс):</strong><br>
                        {concept_drift['accuracy_reference']:.3f}
                    </div>
                    <div class="metric">
                        <strong>Точность (текущая):</strong><br>
                        {concept_drift['accuracy_current']:.3f}
                    </div>
                    <p><a href="data: отчет по концептуальному дрифту">📎 Открыть полный отчет</a></p>
            """
        
        html_content += """
                </div>
                
                <div class="card">
                    <h2>📉 Деградация производительности</h2>
        """
        
        if report['performance_decay']:
            performance = report['performance_decay']
            html_content += f"""
                    <div class="metric">
                        <strong>Тренд точности:</strong><br>
                        {performance['accuracy_trend_slope']:.4f}
                    </div>
                    <div class="metric">
                        <strong>Деградация:</strong><br>
                        {'⚠️ Да' if performance['is_decaying'] else '✅ Нет'}
                    </div>
                    <div class="metric">
                        <strong>Средняя точность:</strong><br>
                        {performance['average_accuracy']:.3f}
                    </div>
                    
                    <h3>Метрики по интервалам:</h3>
                    <table>
                        <tr>
                            <th>Интервал</th>
                            <th>Точность</th>
                            <th>ROC AUC</th>
                            <th>F1 Score</th>
                            <th>Размер выборки</th>
                        </tr>
            """
            
            for interval in performance['performance_intervals']:
                html_content += f"""
                        <tr>
                            <td>{interval['interval']}</td>
                            <td>{interval['accuracy']:.3f}</td>
                            <td>{interval['roc_auc']:.3f}</td>
                            <td>{interval['f1_score']:.3f}</td>
                            <td>{interval['sample_size']}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
            """
        
        html_content += f"""
                </div>
                
                <div class="card">
                    <h2>🚨 Оповещения и рекомендации</h2>
        """
        
        for alert in report['summary']['alerts']:
            severity_class = 'alert-warning' if alert['severity'] == 'warning' else 'alert-critical'
            html_content += f"""
                    <div class="alert {severity_class}">
                        <strong>{alert['type'].upper()} - {alert['severity'].upper()}</strong><br>
                        {alert['message']}
                    </div>
            """
        
        html_content += """
                    <h3>Рекомендации:</h3>
                    <ul>
        """
        
        for rec in report['summary']['recommendations']:
            html_content += f"""
                        <li>{rec}</li>
            """
        
        html_content += """
                    </ul>
                </div>
                
                <div class="card">
                    <h2>⚙️ Действия</h2>
                    <button onclick="triggerRetraining()">🔄 Запустить переобучение</button>
                    <button onclick="generateDetailedReport()">📄 Создать детальный отчет</button>
                    <button onclick="notifyTeam()">📢 Уведомить команду</button>
                </div>
            </div>
            
            <script>
                function triggerRetraining() {{
                    fetch('/api/retrain', {{ method: 'POST' }})
                        .then(response => alert('Переобучение запущено'))
                        .catch(error => alert('Ошибка: ' + error));
                }}
                
                function generateDetailedReport() {{
                    window.open('{report['data_drift']['report_path'] if report['data_drift'] else '#'}', '_blank');
                }}
                
                function notifyTeam() {{
                    alert('Уведомление отправлено команде');
                }}
            </script>
        </body>
        </html>
        """
        
        # Сохраняем дашборд
        dashboard_path = self.report_dir / f"drift_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Дашборд сохранен: {dashboard_path}")
    
    def setup_continuous_monitoring(self, interval_hours: int = 24):
        """Настройка непрерывного мониторинга"""
        logger.info(f"Настройка непрерывного мониторинга (интервал: {interval_hours}ч)")
        
        # Создаем конфигурационный файл
        config = {
            'monitoring_interval_hours': interval_hours,
            'reference_data_path': '../data/processed/train.csv',
            'current_data_pattern': '../data/processed/current_*.csv',
            'alert_thresholds': {
                'data_drift_score': 0.3,
                'concept_drift_score': 0.4,
                'accuracy_drop': 0.05
            },
            'notification_channels': ['slack', 'email'],
            'auto_retrain': False,
            'retrain_threshold': 0.5
        }
        
        config_path = self.report_dir / 'monitoring_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Конфигурация сохранена: {config_path}")
        
        # Создаем скрипт для планировщика
        self._create_monitoring_script()


# Пример использования
def main():
    """Основная функция для демонстрации мониторинга дрифта"""
    
    # Инициализация монитора
    monitor = DriftMonitor(project_name="credit-scoring-production")
    
    # Загрузка данных
    reference_data = monitor.load_reference_data('../data/processed/train.csv')
    current_data = monitor.load_current_data('../data/processed/current_week.csv', days_back=7)
    
    # Симуляция меток для демонстрации
    np.random.seed(42)
    n_samples = len(reference_data) + len(current_data)
    y_true = pd.Series(np.random.binomial(1, 0.3, n_samples))
    y_pred = pd.Series(np.random.binomial(1, 0.35, n_samples))  # Немного смещенные предсказания
    y_pred_proba = pd.Series(np.random.uniform(0, 1, n_samples))
    
    # Комплексный анализ
    report = monitor.run_comprehensive_drift_analysis(
        reference_path='../data/processed/train.csv',
        current_path='../data/processed/current_week.csv',
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba
    )
    
    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ МОНИТОРИНГА ДРИФТА")
    print("="*60)
    
    if report['data_drift']:
        print(f"\n📊 Дрифт данных:")
        print(f"  • Обнаружен: {report['data_drift']['dataset_drift_detected']}")
        print(f"  • Счет дрифта: {report['data_drift']['dataset_drift_score']:.3f}")
        print(f"  • Дрифтующих колонок: {report['data_drift']['number_of_drifted_columns']}")
    
    if report['concept_drift']:
        print(f"\n🎯 Концептуальный дрифт:")
        print(f"  • Обнаружен: {report['concept_drift']['target_drift_detected']}")
        print(f"  • Счет дрифта: {report['concept_drift']['target_drift_score']:.3f}")
        print(f"  • Разница в точности: {report['concept_drift']['accuracy_difference']:.3f}")
    
    if report['performance_decay']:
        print(f"\n📉 Деградация производительности:")
        print(f"  • Деградация: {report['performance_decay']['is_decaying']}")
        print(f"  • Тренд точности: {report['performance_decay']['accuracy_trend_slope']:.4f}")
    
    print(f"\n📋 Общий статус: {report['summary']['overall_status']}")
    
    if report['summary']['alerts']:
        print(f"\n🚨 Оповещения:")
        for alert in report['summary']['alerts']:
            print(f"  • {alert['type']}: {alert['message']}")
    
    if report['summary']['recommendations']:
        print(f"\n💡 Рекомендации:")
        for rec in report['summary']['recommendations']:
            print(f"  • {rec}")
    
    print(f"\n📁 Отчеты сохранены в: ../monitoring/reports/")

if __name__ == "__main__":
    main()