"""
Airflow DAG для автоматического переобучения модели кредитного скоринга
Триггеры: дрифт данных, расписание, ручной запуск
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
import logging
import json
from typing import Dict, List

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация DAG
default_args = {
    'owner': 'ml-ops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['ml-ops@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
    'pool': 'model_training_pool',
    'priority_weight': 10,
}

# Создание DAG
dag = DAG(
    'credit_scoring_retraining',
    default_args=default_args,
    description='Автоматическое переобучение модели кредитного скоринга на основе дрифта данных',
    schedule_interval='0 2 * * SAT',  # Каждую субботу в 2:00
    catchup=False,
    tags=['mlops', 'retraining', 'credit-scoring'],
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=4),
)

# Функции для задач
def check_drift_conditions(**context):
    """
    Проверка условий для запуска переобучения
    Возвращает:
      - 'trigger_retraining': если дрифт превышает порог
      - 'skip_retraining': если дрифт в пределах нормы
    """
    try:
        # Получаем метрики дрифта из Prometheus или Evidently
        # В реальной реализации здесь был бы API запрос
        import requests
        
        prometheus_url = Variable.get("prometheus_url", default_var="http://prometheus.monitoring.svc.cluster.local:9090")
        
        # Запрос метрик дрифта
        queries = {
            'data_drift': 'data_drift_score',
            'concept_drift': 'concept_drift_score',
            'accuracy_drop': 'model_accuracy_drop'
        }
        
        drift_metrics = {}
        for name, query in queries.items():
            try:
                response = requests.get(
                    f"{prometheus_url}/api/v1/query",
                    params={'query': query}
                )
                if response.status_code == 200:
                    result = response.json()
                    if result['data']['result']:
                        drift_metrics[name] = float(result['data']['result'][0]['value'][1])
            except Exception as e:
                logger.warning(f"Не удалось получить метрику {name}: {e}")
                drift_metrics[name] = 0.0
        
        # Получаем пороги из переменных Airflow
        thresholds = {
            'data_drift': float(Variable.get("data_drift_threshold", default_var=0.3)),
            'concept_drift': float(Variable.get("concept_drift_threshold", default_var=0.4)),
            'accuracy_drop': float(Variable.get("accuracy_drop_threshold", default_var=0.05))
        }
        
        # Проверяем условия
        conditions_met = []
        retraining_reasons = []
        
        if drift_metrics.get('data_drift', 0) > thresholds['data_drift']:
            conditions_met.append('data_drift')
            retraining_reasons.append(f"Дрифт данных: {drift_metrics['data_drift']:.3f} > {thresholds['data_drift']}")
        
        if drift_metrics.get('concept_drift', 0) > thresholds['concept_drift']:
            conditions_met.append('concept_drift')
            retraining_reasons.append(f"Концептуальный дрифт: {drift_metrics['concept_drift']:.3f} > {thresholds['concept_drift']}")
        
        if abs(drift_metrics.get('accuracy_drop', 0)) > thresholds['accuracy_drop']:
            conditions_met.append('accuracy_drop')
            retraining_reasons.append(f"Падение точности: {drift_metrics['accuracy_drop']:.3f} > {thresholds['accuracy_drop']}")
        
        # Проверяем время с последнего переобучения
        last_retraining_days = Variable.get("last_retraining_days", default_var=0, deserialize_json=False)
        max_days_without_retraining = int(Variable.get("max_days_without_retraining", default_var=30))
        
        if int(last_retraining_days) >= max_days_without_retraining:
            conditions_met.append('scheduled')
            retraining_reasons.append(f"С момента последнего переобучения прошло {last_retraining_days} дней")
        
        # Логируем результаты проверки
        logger.info(f"Метрики дрифта: {drift_metrics}")
        logger.info(f"Пороги: {thresholds}")
        logger.info(f"Условия выполнены: {conditions_met}")
        logger.info(f"Причины переобучения: {retraining_reasons}")
        
        # Сохраняем в XCom для использования в следующих задачах
        context['ti'].xcom_push(key='drift_metrics', value=drift_metrics)
        context['ti'].xcom_push(key='retraining_reasons', value=retraining_reasons)
        context['ti'].xcom_push(key='conditions_met', value=conditions_met)
        
        # Определяем следующий шаг
        if conditions_met:
            logger.info("Условия для переобучения выполнены, запускаем переобучение")
            return 'trigger_retraining'
        else:
            logger.info("Условия для переобучения не выполнены, пропускаем")
            return 'skip_retraining'
            
    except Exception as e:
        logger.error(f"Ошибка при проверке условий дрифта: {e}")
        # В случае ошибки пропускаем переобучение
        return 'skip_retraining'

def prepare_training_data(**context):
    """Подготовка данных для обучения"""
    logger.info("Подготовка данных для обучения...")
    
    try:
        # Здесь должна быть логика подготовки данных:
        # 1. Загрузка свежих данных
        # 2. Предобработка
        # 3. Разделение на train/val/test
        # 4. Сохранение в DVC
        
        # Для демонстрации просто логируем
        retraining_reasons = context['ti'].xcom_pull(task_ids='check_drift', key='retraining_reasons')
        logger.info(f"Причины переобучения: {retraining_reasons}")
        
        # Симуляция подготовки данных
        import pandas as pd
        import numpy as np
        
        # Генерация симулированных данных
        np.random.seed(42)
        n_samples = 10000
        
        data = pd.DataFrame({
            'age': np.random.normal(45, 15, n_samples).astype(int),
            'income': np.random.lognormal(10.5, 0.8, n_samples),
            'loan_amount': np.random.uniform(10000, 100000, n_samples),
            'credit_history_length': np.random.exponential(10, n_samples).astype(int),
            'debt_to_income_ratio': np.random.beta(2, 5, n_samples) * 100,
            'employment_status': np.random.choice(['employed', 'self-employed', 'unemployed'], n_samples),
            'home_ownership': np.random.choice(['mortgage', 'own', 'rent'], n_samples),
            'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'business'], n_samples),
            'marital_status': np.random.choice(['married', 'single', 'divorced'], n_samples),
            'target': np.random.binomial(1, 0.3, n_samples)
        })
        
        # Сохранение данных
        data_path = '/tmp/training_data.csv'
        data.to_csv(data_path, index=False)
        
        logger.info(f"Данные подготовлены: {data.shape}, сохранены в {data_path}")
        
        # Сохраняем информацию в XCom
        context['ti'].xcom_push(key='training_data_path', value=data_path)
        context['ti'].xcom_push(key='data_shape', value=str(data.shape))
        context['ti'].xcom_push(key='class_distribution', value=dict(data['target'].value_counts()))
        
        return "Данные успешно подготовлены"
        
    except Exception as e:
        logger.error(f"Ошибка при подготовке данных: {e}")
        raise

def validate_new_model(**context):
    """Валидация новой модели"""
    logger.info("Валидация новой модели...")
    
    try:
        # Получаем информацию о новой модели
        # В реальной реализации здесь была бы загрузка модели и ее валидация
        
        # Симуляция метрик валидации
        validation_metrics = {
            'accuracy': 0.87,
            'precision': 0.82,
            'recall': 0.79,
            'f1_score': 0.805,
            'roc_auc': 0.92,
            'log_loss': 0.45,
            'inference_latency_ms': 25.3,
            'model_size_mb': 12.7
        }
        
        # Пороги валидации
        validation_thresholds = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'roc_auc': 0.90,
            'inference_latency_max_ms': 50,
            'model_size_max_mb': 50
        }
        
        # Проверяем метрики
        validation_passed = True
        failed_metrics = []
        
        for metric, value in validation_metrics.items():
            threshold = validation_thresholds.get(metric)
            if threshold is not None:
                if metric in ['inference_latency_ms', 'model_size_mb', 'log_loss']:
                    # Для этих метрик меньше = лучше
                    if value > threshold:
                        validation_passed = False
                        failed_metrics.append(f"{metric}: {value} > {threshold}")
                else:
                    # Для этих метрик больше = лучше
                    if value < threshold:
                        validation_passed = False
                        failed_metrics.append(f"{metric}: {value} < {threshold}")
        
        # Проверяем улучшение по сравнению с текущей моделью
        current_model_metrics = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.775
        }
        
        improvements = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in validation_metrics and metric in current_model_metrics:
                improvement = validation_metrics[metric] - current_model_metrics[metric]
                improvements[metric] = improvement
        
        # Определяем, проходит ли модель валидацию
        min_improvement = 0.01  # Минимальное улучшение 1%
        has_improvement = any(imp > min_improvement for imp in improvements.values())
        
        validation_result = {
            'passed': validation_passed and has_improvement,
            'metrics': validation_metrics,
            'improvements': improvements,
            'failed_metrics': failed_metrics,
            'has_improvement': has_improvement
        }
        
        logger.info(f"Результаты валидации: {validation_result}")
        
        # Сохраняем в XCom
        context['ti'].xcom_push(key='validation_result', value=validation_result)
        
        if validation_result['passed']:
            logger.info("✅ Модель прошла валидацию")
            return 'model_passed_validation'
        else:
            logger.warning("❌ Модель не прошла валидацию")
            if not validation_passed:
                logger.warning(f"Не пройдены пороги: {failed_metrics}")
            if not has_improvement:
                logger.warning("Недостаточное улучшение по сравнению с текущей моделью")
            return 'model_failed_validation'
            
    except Exception as e:
        logger.error(f"Ошибка при валидации модели: {e}")
        return 'model_failed_validation'

def deploy_new_model(**context):
    """Деплой новой модели"""
    logger.info("Деплой новой модели...")
    
    try:
        validation_result = context['ti'].xcom_pull(task_ids='validate_model', key='validation_result')
        
        if not validation_result.get('passed', False):
            raise ValueError("Модель не прошла валидацию, деплой невозможен")
        
        # Здесь должна быть логика деплоя:
        # 1. Сохранение модели в реестр моделей (MLflow)
        # 2. Обновление конфигурации Kubernetes
        # 3. Canary или Blue-Green деплой
        
        # Для демонстрации симулируем деплой
        model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        deployment_info = {
            'model_version': model_version,
            'deployment_time': datetime.now().isoformat(),
            'deployment_strategy': 'canary',
            'initial_traffic_percentage': 10,
            'metrics': validation_result['metrics'],
            'improvements': validation_result['improvements']
        }
        
        logger.info(f"Информация о деплое: {deployment_info}")
        
        # Симуляция деплоя в Kubernetes
        logger.info("Обновление конфигурации Kubernetes...")
        logger.info("Запуск canary deployment с 10% трафика...")
        
        # Обновляем переменную времени последнего переобучения
        from airflow.models import Variable
        Variable.set("last_retraining_days", 0)
        
        # Сохраняем информацию в XCom
        context['ti'].xcom_push(key='deployment_info', value=deployment_info)
        context['ti'].xcom_push(key='model_version', value=model_version)
        
        logger.info(f"✅ Модель {model_version} успешно задеплоена")
        return deployment_info
        
    except Exception as e:
        logger.error(f"Ошибка при деплое модели: {e}")
        raise

def monitor_deployment(**context):
    """Мониторинг деплоя новой модели"""
    logger.info("Мониторинг деплоя новой модели...")
    
    try:
        deployment_info = context['ti'].xcom_pull(task_ids='deploy_model', key='deployment_info')
        model_version = deployment_info.get('model_version', 'unknown')
        
        # Здесь должна быть логика мониторинга:
        # 1. Проверка метрик новой модели
        # 2. Сравнение с текущей моделью
        # 3. Принятие решения о полном переходе
        
        # Симуляция мониторинга
        import time
        import random
        
        logger.info(f"Мониторинг модели {model_version}...")
        
        # Симулируем сбор метрик
        monitoring_results = []
        for i in range(5):  # 5 итераций мониторинга
            time.sleep(2)  # Симуляция времени ожидания
            
            metrics = {
                'iteration': i + 1,
                'error_rate': random.uniform(0.01, 0.05),
                'latency_p95_ms': random.uniform(20, 40),
                'throughput_rps': random.uniform(80, 120),
                'success_rate': random.uniform(0.95, 0.99)
            }
            
            monitoring_results.append(metrics)
            logger.info(f"Итерация {i+1}: {metrics}")
        
        # Анализ результатов
        avg_error_rate = sum(r['error_rate'] for r in monitoring_results) / len(monitoring_results)
        avg_latency = sum(r['latency_p95_ms'] for r in monitoring_results) / len(monitoring_results)
        
        # Критерии успеха
        max_error_rate = 0.05
        max_latency_ms = 50
        
        deployment_successful = avg_error_rate <= max_error_rate and avg_latency <= max_latency_ms
        
        monitoring_summary = {
            'model_version': model_version,
            'monitoring_duration_minutes': 10,
            'avg_error_rate': avg_error_rate,
            'avg_latency_ms': avg_latency,
            'deployment_successful': deployment_successful,
            'criteria': {
                'max_error_rate': max_error_rate,
                'max_latency_ms': max_latency_ms
            },
            'recommendation': 'promote_to_100_percent' if deployment_successful else 'rollback'
        }
        
        logger.info(f"Результаты мониторинга: {monitoring_summary}")
        
        # Сохраняем в XCom
        context['ti'].xcom_push(key='monitoring_summary', value=monitoring_summary)
        
        if deployment_successful:
            logger.info("✅ Деплой успешен, можно переходить на 100% трафика")
            return 'deployment_successful'
        else:
            logger.warning("❌ Проблемы с деплоем, требуется откат")
            return 'deployment_failed'
            
    except Exception as e:
        logger.error(f"Ошибка при мониторинге деплоя: {e}")
        return 'deployment_failed'

def promote_model(**context):
    """Полный переход на новую модель"""
    logger.info("Полный переход на новую модель...")
    
    try:
        deployment_info = context['ti'].xcom_pull(task_ids='deploy_model', key='deployment_info')
        model_version = deployment_info.get('model_version')
        
        # Здесь должна быть логика полного перехода:
        # 1. Увеличение трафика до 100%
        # 2. Обновление production тегов
        # 3. Архивирование старой модели
        
        logger.info(f"Переход на модель {model_version} с 100% трафика...")
        
        # Симуляция перехода
        promotion_info = {
            'model_version': model_version,
            'promotion_time': datetime.now().isoformat(),
            'traffic_percentage': 100,
            'previous_model_archived': True,
            'mlflow_production_tag': True
        }
        
        # Отправляем уведомление
        send_notification(
            title="✅ Модель успешно промотирована в production",
            message=f"Модель {model_version} теперь обслуживает 100% трафика",
            level="success"
        )
        
        logger.info(f"Информация о промоушене: {promotion_info}")
        
        # Сохраняем в XCom
        context['ti'].xcom_push(key='promotion_info', value=promotion_info)
        
        return promotion_info
        
    except Exception as e:
        logger.error(f"Ошибка при переходе на новую модель: {e}")
        raise

def rollback_deployment(**context):
    """Откат деплоя"""
    logger.info("Откат деплоя...")
    
    try:
        deployment_info = context['ti'].xcom_pull(task_ids='deploy_model', key='deployment_info', default={})
        model_version = deployment_info.get('model_version', 'unknown')
        
        logger.warning(f"Откат модели {model_version}...")
        
        # Здесь должна быть логика отката:
        # 1. Возврат на предыдущую версию
        # 2. Удаление canary deployment
        # 3. Восстановление предыдущей конфигурации
        
        rollback_info = {
            'rolled_back_model': model_version,
            'rollback_time': datetime.now().isoformat(),
            'current_model': 'previous_version',
            'traffic_percentage': 100,
           