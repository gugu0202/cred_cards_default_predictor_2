# Импорт необходимых библиотек
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
import pickle
import json

# Определение класса нейронной сети для кредитного скоринга
class CreditScoringNN(nn.Module):
    """
    Нейронная сеть с тремя слоями для оценки кредитоспособности заемщиков.
    """
    def __init__(self, input_size=20, hidden_size=64, dropout_rate=0.3):
        super(CreditScoringNN, self).__init__()
        # Последовательность слоев нейронной сети
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),   
            nn.BatchNorm1d(hidden_size),          
            nn.ReLU(),                            
            nn.Dropout(dropout_rate),              
            nn.Linear(hidden_size, 32),           
            nn.ReLU(),                            
            nn.Linear(32, 1),                    
            nn.Sigmoid()                         
        )

    def forward(self, x):
        """
        Прямой проход через сеть.
        """
        return self.network(x)

# Функция загрузки и предварительной обработки данных
def load_and_preprocess_data():
    """
    Генерация синтетического набора данных и нормализация признаков.
    """
    # Фиксируем случайное число для воспроизводимости результатов
    np.random.seed(42)
    n_samples = 10_000  
    n_features = 20     
    
    # Генерируем случайные признаки и целевые переменные
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.7).astype(int)
    
    # Стандартизация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Сохраняем нормализатор для последующего использования
    with open('../models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Разделение данных на тренировочную и тестовую выборки
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Полный процесс обучения модели
def train_model():
    """
    Обучение нейронной сети с использованием MLFlow для отслеживания экспериментов.
    """
    mlflow.set_experiment("credit_scoring")
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
        
        # Создаем экземпляр модели
        model = CreditScoringNN(input_size=20, hidden_size=64, dropout_rate=0.3)
        criterion = nn.BCELoss()       # Бинарная кросс-энтропия
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        n_epochs = 50                  
        batch_size = 32               
        
        # Создание загрузчика данных
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(n_epochs):
            model.train()                 # Переключение модели в режим тренировки
            total_loss = 0
            
            # Проход по каждому пакету данных
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()     # Очистка градиентов перед обратным распространением
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()           # Обратное распространение ошибок
                optimizer.step()          
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                model.eval()                   
                with torch.no_grad():          
                    test_predictions = model(X_test_tensor)
                    test_loss = criterion(test_predictions, y_test_tensor)
                    accuracy = ((test_predictions > 0.5).float() == y_test_tensor).float().mean()
                    
                # Вывод промежуточных результатов
                print(f'Эпоха {epoch+1}/{n_epochs}, '
                      f'Train Loss: {total_loss/len(train_loader):.4f}, '
                      f'Test Loss: {test_loss:.4f}, '
                      f'Accuracy: {accuracy:.4f}')
            
            # Логируем потери и точность в MLFlow
            mlflow.log_metric('train_loss', total_loss/len(train_loader), step=epoch)
            mlflow.log_metric('test_loss', test_loss.item(), step=epoch)
            mlflow.log_metric('accuracy', accuracy.item(), step=epoch)
        
        torch.save(model.state_dict(), '../models/credit_scoring_nn.pth')
        
        # Регистрация гиперпараметров и структуры модели в MLFlow
        mlflow.log_params({
            'input_size': 20,
            'hidden_size': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': n_epochs,
            'batch_size': batch_size
        })
        
        # Логируем саму модель
        mlflow.pytorch.log_model(model, "model")
        
        # Подготавливаем итоговые метрики
        final_metrics = {
            'final_test_loss': float(test_loss.item()),      # Итоговая ошибка на тестовом наборе
            'final_accuracy': float(accuracy.item()),        # Итоговая точность
            'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)  # Размер модели в мегабайтах
        }
        
        # Сохраняем метрики в JSON-файл
        with open('../models/training_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Сообщаем о завершении процесса
        print(f'Обучение завершено. Модель сохранена в ../models/credit_scoring_nn.pth')
        print(f'Итоговые метрики: {final_metrics}')
        
        return model, final_metrics

# Основной запуск программы
if __name__ == '__main__':
    train_model()