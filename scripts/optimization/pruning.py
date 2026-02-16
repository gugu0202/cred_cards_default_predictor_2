import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import json
import time
from pathlib import Path
import copy
from train_nn_model import CreditScoringNN
import onnx
import onnxruntime as ort

class ModelPruner:
    """Класс для прунинга (обрезки) нейронной сети"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.original_model = None
        self.pruned_model = None
        self.pruning_stats = {}
        
    def load_model(self) -> CreditScoringNN:
        """Загрузка оригинальной модели"""
        print(f"Loading model from {self.model_path}")
        model = CreditScoringNN(input_size=20, hidden_size=64, dropout_rate=0.3)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        model.eval()
        self.original_model = model
        return model
    
    def apply_global_pruning(self, amount: float = 0.3) -> CreditScoringNN:
        """
        Применение глобального прунинга (обрезки) к модели
        amount: процент весов для обрезки (0.0 - 1.0)
        """
        print(f"\nApplying global pruning ({amount*100:.0f}% sparsity)")
        
        # Создаем копию модели
        model = copy.deepcopy(self.original_model)
        
        # Определяем параметры для прунинга (только веса, не bias)
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        # Применяем глобальный прунинг
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        
        # Удаляем маски прунинга, делая изменения постоянными
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        self.pruned_model = model
        
        # Сбор статистики
        self._collect_pruning_stats(amount)
        
        return model
    
    def apply_structured_pruning(self, amount: float = 0.4) -> CreditScoringNN:
        """
        Применение структурированного прунинга (удаление целых нейронов)
        """
        print(f"\nApplying structured pruning ({amount*100:.0f}% of neurons)")
        
        model = copy.deepcopy(self.original_model)
        
        # Структурированный прунинг по каналам
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Используем L1 норму для выбора наименее важных нейронов
                weight = module.weight.data
                importance = weight.abs().sum(dim=1)  # L1 норма по выходным измерениям
                
                # Определяем количество нейронов для удаления
                n_prune = int(amount * weight.size(0))
                
                if n_prune > 0:
                    # Индексы наименее важных нейронов
                    prune_indices = torch.argsort(importance)[:n_prune]
                    
                    # Создаем маску
                    mask = torch.ones_like(weight, dtype=torch.bool)
                    mask[prune_indices, :] = 0
                    
                    # Применяем маску
                    module.weight.data *= mask.float()
        
        self.pruned_model = model
        self._collect_pruning_stats(amount, structured=True)
        
        return model
    
    def apply_iterative_pruning(self, target_sparsity: float = 0.7, n_iterations: int = 7) -> CreditScoringNN:
        """
        Итеративный прунинг с постепенным увеличением sparsity
        """
        print(f"\nApplying iterative pruning to {target_sparsity*100:.0f}% sparsity in {n_iterations} iterations")
        
        model = copy.deepcopy(self.original_model)
        current_sparsity = 0.0
        
        for iteration in range(1, n_iterations + 1):
            # Вычисляем sparsity для этой итерации
            sparsity = target_sparsity * (iteration / n_iterations) ** 3  # Кубическое увеличение
            
            # Определяем параметры для прунинга
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    parameters_to_prune.append((module, 'weight'))
            
            # Применяем прунинг
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity - current_sparsity
            )
            
            # Собираем статистику
            total_params = 0
            remaining_params = 0
            
            for module, param_name in parameters_to_prune:
                if param_name == 'weight':
                    weight = getattr(module, param_name)
                    total_params += weight.numel()
                    remaining_params += weight.count_nonzero()
            
            current_sparsity = 1 - (remaining_params / total_params)
            
            print(f"  Iteration {iteration}: "
                  f"Sparsity = {current_sparsity*100:.1f}%, "
                  f"Remaining params = {remaining_params:,}")
        
        # Удаляем маски и делаем изменения постоянными
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        self.pruned_model = model
        self._collect_pruning_stats(target_sparsity, iterative=True)
        
        return model
    
    def _collect_pruning_stats(self, amount: float, structured: bool = False, iterative: bool = False):
        """Сбор статистики по прунингу"""
        if not self.original_model or not self.pruned_model:
            return
        
        original_params = self._count_parameters(self.original_model)
        pruned_params = self._count_parameters(self.pruned_model)
        
        # Подсчет нулевых весов
        original_zeros = self._count_zero_weights(self.original_model)
        pruned_zeros = self._count_zero_weights(self.pruned_model)
        
        # Вычисление сжатия
        size_reduction = (original_params - pruned_params) / original_params * 100
        sparsity_increase = (pruned_zeros - original_zeros) / original_params * 100
        
        self.pruning_stats = {
            'pruning_amount': amount,
            'pruning_type': 'structured' if structured else 'iterative' if iterative else 'global',
            'original_parameters': original_params,
            'pruned_parameters': pruned_params,
            'parameters_removed': original_params - pruned_params,
            'size_reduction_percent': size_reduction,
            'original_zero_weights': original_zeros,
            'pruned_zero_weights': pruned_zeros,
            'sparsity_increase_percent': sparsity_increase,
            'compression_ratio': original_params / pruned_params if pruned_params > 0 else 1
        }
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Подсчет общего количества параметров"""
        return sum(p.numel() for p in model.parameters())
    
    def _count_zero_weights(self, model: nn.Module) -> int:
        """Подсчет нулевых весов"""
        zero_count = 0
        total_count = 0
        
        for param in model.parameters():
            if len(param.shape) >= 2:  # Только веса (не bias)
                zero_count += (param == 0).sum().item()
                total_count += param.numel()
        
        return zero_count
    
    def save_pruned_model(self, output_path: str):
        """Сохранение прунированной модели"""
        if self.pruned_model:
            torch.save(self.pruned_model.state_dict(), output_path)
            print(f"\nPruned model saved to {output_path}")
            
            # Сохранение статистики
            stats_path = output_path.replace('.pth', '_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(self.pruning_stats, f, indent=2)
            print(f"Pruning statistics saved to {stats_path}")
    
    def evaluate_pruning_impact(self, test_samples: int = 1000) -> dict:
        """
        Оценка влияния прунинга на точность и производительность
        """
        print("\nEvaluating pruning impact...")
        
        if not self.original_model or not self.pruned_model:
            raise ValueError("Models not loaded")
        
        # Генерация тестовых данных
        np.random.seed(42)
        test_data = np.random.randn(test_samples, 20).astype(np.float32)
        test_tensor = torch.from_numpy(test_data)
        
        # Инференс на оригинальной модели
        self.original_model.eval()
        with torch.no_grad():
            start_time = time.perf_counter()
            original_outputs = self.original_model(test_tensor)
            original_time = time.perf_counter() - start_time
        
        # Инференс на прунированной модели
        self.pruned_model.eval()
        with torch.no_grad():
            start_time = time.perf_counter()
            pruned_outputs = self.pruned_model(test_tensor)
            pruned_time = time.perf_counter() - start_time
        
        # Вычисление метрик
        output_diff = torch.abs(original_outputs - pruned_outputs)
        
        evaluation_results = {
            'speedup_ratio': original_time / pruned_time,
            'original_inference_time_ms': original_time * 1000,
            'pruned_inference_time_ms': pruned_time * 1000,
            'mean_absolute_difference': output_diff.mean().item(),
            'max_absolute_difference': output_diff.max().item(),
            'std_absolute_difference': output_diff.std().item(),
            'output_correlation': torch.corrcoef(
                torch.stack([original_outputs.flatten(), pruned_outputs.flatten()])
            )[0, 1].item(),
            'test_samples': test_samples
        }
        
        print(f"  Speedup: {evaluation_results['speedup_ratio']:.2f}x")
        print(f"  Mean absolute difference: {evaluation_results['mean_absolute_difference']:.6f}")
        print(f"  Output correlation: {evaluation_results['output_correlation']:.4f}")
        
        return evaluation_results
    
    def convert_to_onnx_after_pruning(self, output_path: str):
        """Конвертация прунированной модели в ONNX"""
        if not self.pruned_model:
            raise ValueError("Pruned model not available")
        
        print(f"\nConverting pruned model to ONNX...")
        
        # Подготовка модели
        self.pruned_model.eval()
        dummy_input = torch.randn(1, 20)
        
        # Экспорт в ONNX
        torch.onnx.export(
            self.pruned_model,
            dummy_input,
            output_path,
            input_names=['features'],
            output_names=['score'],
            dynamic_axes={'features': {0: 'batch_size'}, 'score': {0: 'batch_size'}},
            opset_version=13,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        # Валидация ONNX модели
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"Pruned ONNX model saved to {output_path}")
        
        # Benchmark производительности
        ort_session = ort.InferenceSession(output_path)
        
        # Тестирование на разных размерах батча
        batch_sizes = [1, 8, 32, 64]
        performance_results = {}
        
        for batch_size in batch_sizes:
            test_input = np.random.randn(batch_size, 20).astype(np.float32)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            
            # Warmup
            for _ in range(100):
                _ = ort_session.run(None, ort_inputs)
            
            # Измерение времени
            start_time = time.perf_counter()
            for _ in range(1000):
                _ = ort_session.run(None, ort_inputs)
            elapsed = time.perf_counter() - start_time
            
            performance_results[batch_size] = {
                'avg_inference_time_ms': (elapsed / 1000) * 1000,
                'throughput_rps': 1000 / elapsed
            }
        
        return performance_results

def run_comprehensive_pruning():
    """Запуск комплексного прунинга с оценкой результатов"""
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL PRUNING OPTIMIZATION")
    print("=" * 80)
    
    # Инициализация прунера
    pruner = ModelPruner('../models/credit_scoring_nn.pth')
    pruner.load_model()
    
    all_results = {}
    
    # 1. Глобальный прунинг с разной степенью
    print("\n1. Global Pruning Analysis")
    print("-" * 40)
    
    pruning_levels = [0.2, 0.3, 0.4, 0.5, 0.6]
    global_results = []
    
    for amount in pruning_levels:
        print(f"\nTesting {amount*100:.0f}% global pruning:")
        
        # Применение прунинга
        pruned_model = pruner.apply_global_pruning(amount)
        
        # Оценка влияния
        evaluation = pruner.evaluate_pruning_impact(test_samples=500)
        
        # Сохранение результатов
        result = {
            'pruning_amount': amount,
            **pruner.pruning_stats,
            **evaluation
        }
        global_results.append(result)
        
        # Временное сохранение модели
        temp_path = f"../models/pruned_global_{int(amount*100)}.pth"
        pruner.save_pruned_model(temp_path)
    
    all_results['global_pruning'] = global_results
    
    # 2. Структурированный прунинг
    print("\n\n2. Structured Pruning Analysis")
    print("-" * 40)
    
    structured_amounts = [0.3, 0.4, 0.5]
    structured_results = []
    
    for amount in structured_amounts:
        print(f"\nTesting {amount*100:.0f}% structured pruning:")
        
        # Перезагрузка оригинальной модели
        pruner.load_model()
        
        # Применение структурированного прунинга
        pruned_model = pruner.apply_structured_pruning(amount)
        
        # Оценка влияния
        evaluation = pruner.evaluate_pruning_impact(test_samples=500)
        
        result = {
            'pruning_amount': amount,
            **pruner.pruning_stats,
            **evaluation
        }
        structured_results.append(result)
        
        # Временное сохранение
        temp_path = f"../models/pruned_structured_{int(amount*100)}.pth"
        pruner.save_pruned_model(temp_path)
    
    all_results['structured_pruning'] = structured_results
    
    # 3. Итеративный прунинг
    print("\n\n3. Iterative Pruning Analysis")
    print("-" * 40)
    
    iterative_targets = [0.6, 0.7, 0.8]
    iterative_results = []
    
    for target in iterative_targets:
        print(f"\nTesting iterative pruning to {target*100:.0f}% sparsity:")
        
        # Перезагрузка оригинальной модели
        pruner.load_model()
        
        # Применение итеративного прунинга
        pruned_model = pruner.apply_iterative_pruning(target_sparsity=target, n_iterations=7)
        
        # Оценка влияния
        evaluation = pruner.evaluate_pruning_impact(test_samples=500)
        
        result = {
            'target_sparsity': target,
            **pruner.pruning_stats,
            **evaluation
        }
        iterative_results.append(result)
        
        # Сохранение лучшей модели (50% прунинг)
        if target == 0.5:
            pruner.save_pruned_model('../models/credit_scoring_pruned.pth')
    
    all_results['iterative_pruning'] = iterative_results
    
    # 4. Выбор лучшей стратегии и конвертация в ONNX
    print("\n\n4. Selecting Best Pruning Strategy")
    print("-" * 40)
    
    # Анализ всех результатов
    best_global = max(global_results, 
                     key=lambda x: x['speedup_ratio'] / (x['mean_absolute_difference'] + 1e-10))
    best_structured = max(structured_results, 
                         key=lambda x: x['speedup_ratio'] / (x['mean_absolute_difference'] + 1e-10))
    best_iterative = max(iterative_results, 
                        key=lambda x: x['speedup_ratio'] / (x['mean_absolute_difference'] + 1e-10))
    
    # Выбор лучшей стратегии
    candidates = [
        ('global', best_global),
        ('structured', best_structured),
        ('iterative', best_iterative)
    ]
    
    best_strategy = max(candidates, 
                       key=lambda x: x[1]['speedup_ratio'] / (x[1]['mean_absolute_difference'] * 10 + 1e-10))
    
    print(f"\nBest pruning strategy: {best_strategy[0].upper()}")
    print(f"  Speedup: {best_strategy[1]['speedup_ratio']:.2f}x")
    print(f"  Size reduction: {best_strategy[1]['size_reduction_percent']:.1f}%")
    print(f"  Mean difference: {best_strategy[1]['mean_absolute_difference']:.6f}")
    
    # 5. Применение выбранной стратегии и конвертация в ONNX
    print("\n\n5. Applying Best Strategy and Converting to ONNX")
    print("-" * 40)
    
    pruner.load_model()
    
    if best_strategy[0] == 'global':
        final_model = pruner.apply_global_pruning(best_strategy[1]['pruning_amount'])
    elif best_strategy[0] == 'structured':
        final_model = pruner.apply_structured_pruning(best_strategy[1]['pruning_amount'])
    else:  # iterative
        final_model = pruner.apply_iterative_pruning(
            target_sparsity=best_strategy[1]['target_sparsity']
        )
    
    # Сохранение финальной прунированной модели
    pruner.save_pruned_model('../models/credit_scoring_pruned_final.pth')
    
    # Конвертация в ONNX
    onnx_performance = pruner.convert_to_onnx_after_pruning(
        '../models/credit_scoring_pruned.onnx'
    )
    
    # 6. Генерация финального отчета
    print("\n\n6. Generating Final Report")
    print("-" * 40)
    
    final_report = {
        'pruning_analysis_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'original_model': '../models/credit_scoring_nn.pth',
        'pruned_model': '../models/credit_scoring_pruned_final.pth',
        'pruned_onnx_model': '../models/credit_scoring_pruned.onnx',
        'best_strategy': {
            'name': best_strategy[0],
            'parameters': best_strategy[1],
            'selected_reason': 'Best trade-off between speedup and accuracy loss'
        },
        'detailed_results': all_results,
        'onnx_performance': onnx_performance,
        'final_comparison': {
            'model_size_reduction_mb': (
                Path('../models/credit_scoring_nn.pth').stat().st_size -
                Path('../models/credit_scoring_pruned_final.pth').stat().st_size
            ) / (1024 * 1024),
            'expected_speedup': best_strategy[1]['speedup_ratio'],
            'expected_accuracy_drop': best_strategy[1]['mean_absolute_difference'],
            'sparsity_achieved': best_strategy[1]['sparsity_increase_percent'],
            'recommended_use_cases': [
                'Real-time inference with latency constraints',
                'Edge deployment with limited resources',
                'High-throughput batch processing'
            ],
            'limitations': [
                f'Potential accuracy drop: {best_strategy[1]["mean_absolute_difference"]:.4f}',
                'May require fine-tuning for optimal results',
                'Structured pruning can affect model capacity'
            ]
        },
        'recommendations': [
            {
                'scenario': 'Production deployment with latency sensitivity',
                'recommendation': 'Use pruned ONNX model',
                'expected_improvement': f"{best_strategy[1]['speedup_ratio']:.1f}x speedup"
            },
            {
                'scenario': 'Batch processing with throughput focus',
                'recommendation': 'Use pruned model with batch size 32-64',
                'expected_improvement': f"{max(onnx_performance.values(), key=lambda x: x['throughput_rps'])['throughput_rps']:.0f} RPS"
            },
            {
                'scenario': 'Retraining after pruning',
                'recommendation': 'Apply iterative pruning + fine-tuning',
                'expected_improvement': 'Recover accuracy loss while maintaining speedup'
            }
        ]
    }
    
    # Сохранение отчета
    with open('../models/pruning_optimization_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n{'='*80}")
    print("PRUNING OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    
    print(f"\nSummary of improvements:")
    print(f"  • Model size reduction: {final_report['final_comparison']['model_size_reduction_mb']:.2f} MB")
    print(f"  • Expected speedup: {final_report['final_comparison']['expected_speedup']:.2f}x")
    print(f"  • Achieved sparsity: {final_report['final_comparison']['sparsity_achieved']:.1f}%")
    
    print(f"\nGenerated files:")
    print(f"  • Pruned PyTorch model: ../models/credit_scoring_pruned_final.pth")
    print(f"  • Pruned ONNX model: ../models/credit_scoring_pruned.onnx")
    print(f"  • Optimization report: ../models/pruning_optimization_report.json")
    
    return final_report

if __name__ == "__main__":
    print("Starting comprehensive model pruning optimization...")
    report = run_comprehensive_pruning()
    print(f"\nFull pruning report saved to ../models/pruning_optimization_report.json")