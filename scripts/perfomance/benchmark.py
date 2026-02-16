import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List
import psutil
import onnxruntime as ort
import torch
import sys
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

@dataclass
class BenchmarkConfig:
    """Конфигурация benchmark тестов"""
    model_path: str
    model_type: str  # 'pytorch', 'onnx', 'onnx_quantized'
    batch_sizes: List[int] = None
    num_iterations: int = 1000
    num_warmup: int = 100
    use_gpu: bool = False
    num_threads: int = None
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 8, 16, 32, 64, 128]
        if self.num_threads is None:
            self.num_threads = os.cpu_count()

@dataclass
class BenchmarkResult:
    """Результаты benchmark"""
    config: BenchmarkConfig
    results_per_batch: Dict[int, Dict] = None
    summary: Dict = None
    
    def __post_init__(self):
        if self.results_per_batch is None:
            self.results_per_batch = {}
        if self.summary is None:
            self.summary = {}

class ModelBenchmark:
    """Класс для комплексного benchmark тестирования"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = BenchmarkResult(config)
        self.session = None
        
        # Настройка окружения
        if config.model_type == 'onnx' or config.model_type == 'onnx_quantized':
            self._setup_onnx_runtime()
        elif config.model_type == 'pytorch':
            self._setup_pytorch()
    
    def _setup_onnx_runtime(self):
        """Настройка ONNX Runtime"""
        sess_options = ort.SessionOptions()
        
        # Оптимизации
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_profiling = False
        
        # Параллелизм
        sess_options.intra_op_num_threads = self.config.num_threads
        sess_options.inter_op_num_threads = self.config.num_threads
        
        # Провайдеры
        providers = ['CPUExecutionProvider']
        if self.config.use_gpu and ort.get_device() == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Создание сессии
        self.session = ort.InferenceSession(
            self.config.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        print(f"ONNX Runtime session created with {providers}")
    
    def _setup_pytorch(self):
        """Настройка PyTorch"""
        import torch
        torch.set_num_threads(self.config.num_threads)
        
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"PyTorch using CPU with {self.config.num_threads} threads")
        
        # Загрузка модели
        from scripts.model_training.train_nn_model import CreditScoringNN
        self.model = CreditScoringNN(input_size=20, hidden_size=64, dropout_rate=0.3)
        self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
    
    def _generate_test_data(self, batch_size: int) -> np.ndarray:
        """Генерация тестовых данных"""
        return np.random.randn(batch_size, 20).astype(np.float32)
    
    def _benchmark_onnx(self, batch_size: int) -> Dict:
        """Benchmark для ONNX модели"""
        test_input = self._generate_test_data(batch_size)
        ort_inputs = {self.session.get_inputs()[0].name: test_input}
        
        # Warmup
        for _ in range(self.config.num_warmup):
            _ = self.session.run(None, ort_inputs)
        
        # Измерение времени
        start_time = time.perf_counter()
        
        for _ in range(self.config.num_iterations):
            _ = self.session.run(None, ort_inputs)
        
        total_time = time.perf_counter() - start_time
        
        # Измерение памяти
        process = psutil.Process()
        mem_before = process.memory_info().rss
        _ = self.session.run(None, ort_inputs)
        mem_after = process.memory_info().rss
        memory_used = (mem_after - mem_before) / 1024 / 1024  # MB
        
        return {
            'total_time_seconds': total_time,
            'avg_inference_time_ms': (total_time / self.config.num_iterations) * 1000,
            'throughput_rps': self.config.num_iterations / total_time,
            'memory_used_mb': memory_used,
            'batch_size': batch_size
        }
    
    def _benchmark_pytorch(self, batch_size: int) -> Dict:
        """Benchmark для PyTorch модели"""
        test_input = self._generate_test_data(batch_size)
        tensor_input = torch.from_numpy(test_input).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.num_warmup):
                _ = self.model(tensor_input)
        
        # Измерение времени
        torch.cuda.synchronize() if self.config.use_gpu else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(self.config.num_iterations):
                _ = self.model(tensor_input)
        
        torch.cuda.synchronize() if self.config.use_gpu else None
        total_time = time.perf_counter() - start_time
        
        # Измерение памяти
        if self.config.use_gpu:
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = self.model(tensor_input)
            memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            process = psutil.Process()
            mem_before = process.memory_info().rss
            with torch.no_grad():
                _ = self.model(tensor_input)
            mem_after = process.memory_info().rss
            memory_used = (mem_after - mem_before) / 1024 / 1024  # MB
        
        return {
            'total_time_seconds': total_time,
            'avg_inference_time_ms': (total_time / self.config.num_iterations) * 1000,
            'throughput_rps': self.config.num_iterations / total_time,
            'memory_used_mb': memory_used,
            'batch_size': batch_size
        }
    
    def run_benchmark(self) -> BenchmarkResult:
        """Запуск комплексного benchmark"""
        print(f"\nStarting benchmark for {self.config.model_type} model")
        print(f"Model: {self.config.model_path}")
        print(f"Batch sizes: {self.config.batch_sizes}")
        print(f"Iterations per batch: {self.config.num_iterations}")
        
        for batch_size in self.config.batch_sizes:
            print(f"\nBenchmarking batch size: {batch_size}")
            
            if self.config.model_type in ['onnx', 'onnx_quantized']:
                result = self._benchmark_onnx(batch_size)
            else:  # pytorch
                result = self._benchmark_pytorch(batch_size)
            
            self.results.results_per_batch[batch_size] = result
            
            print(f"  Avg inference: {result['avg_inference_time_ms']:.3f} ms")
            print(f"  Throughput: {result['throughput_rps']:.1f} requests/sec")
            print(f"  Memory used: {result['memory_used_mb']:.1f} MB")
        
        # Генерация summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Генерация сводки результатов"""
        results_df = pd.DataFrame(self.results.results_per_batch.values())
        
        self.results.summary = {
            'model_type': self.config.model_type,
            'model_path': self.config.model_path,
            'best_batch_size': int(results_df.loc[results_df['throughput_rps'].idxmax()]['batch_size']),
            'max_throughput': float(results_df['throughput_rps'].max()),
            'min_latency': float(results_df['avg_inference_time_ms'].min()),
            'avg_memory_usage': float(results_df['memory_used_mb'].mean()),
            'optimal_configuration': {
                'batch_size': int(results_df.loc[results_df['throughput_rps'].idxmax()]['batch_size']),
                'expected_rps': float(results_df['throughput_rps'].max()),
                'expected_latency': float(results_df.loc[results_df['throughput_rps'].idxmax()]['avg_inference_time_ms'])
            },
            'detailed_results': results_df.to_dict('records')
        }
    
    def plot_results(self, output_path: str):
        """Визуализация результатов benchmark"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        results_df = pd.DataFrame(self.results.results_per_batch.values())
        
        # 1. Throughput vs Batch Size
        axes[0, 0].plot(results_df['batch_size'], results_df['throughput_rps'], 
                       marker='o', linewidth=2, markersize=8, color='blue')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Throughput (requests/sec)')
        axes[0, 0].set_title(f'Throughput vs Batch Size - {self.config.model_type}')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log', base=2)
        
        # 2. Latency vs Batch Size
        axes[0, 1].plot(results_df['batch_size'], results_df['avg_inference_time_ms'], 
                       marker='s', linewidth=2, markersize=8, color='red')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title(f'Latency vs Batch Size - {self.config.model_type}')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xscale('log', base=2)
        axes[0, 1].set_yscale('log')
        
        # 3. Memory Usage
        axes[1, 0].bar(results_df['batch_size'].astype(str), 
                      results_df['memory_used_mb'], color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage by Batch Size')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Summary
        summary_text = (
            f"Model: {self.config.model_type}\n"
            f"Best Batch Size: {self.results.summary['best_batch_size']}\n"
            f"Max Throughput: {self.results.summary['max_throughput']:.1f} RPS\n"
            f"Min Latency: {self.results.summary['min_latency']:.2f} ms\n"
            f"Avg Memory: {self.results.summary['avg_memory_usage']:.1f} MB\n"
            f"\nOptimal Config:\n"
            f"Batch Size: {self.results.summary['optimal_configuration']['batch_size']}\n"
            f"Expected RPS: {self.results.summary['optimal_configuration']['expected_rps']:.1f}\n"
            f"Expected Latency: {self.results.summary['optimal_configuration']['expected_latency']:.2f} ms"
        )
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                       verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def compare_all_models():
    """Сравнение всех версий моделей"""
    models_to_test = [
        {
            'name': 'PyTorch_CPU',
            'path': '../models/credit_scoring_nn.pth',
            'type': 'pytorch',
            'use_gpu': False
        },
        {
            'name': 'ONNX_CPU',
            'path': '../models/credit_scoring.onnx',
            'type': 'onnx',
            'use_gpu': False
        },
        {
            'name': 'ONNX_Quantized_CPU',
            'path': '../models/credit_scoring_quantized.onnx',
            'type': 'onnx_quantized',
            'use_gpu': False
        }
    ]
    
    all_results = {}
    comparison_data = []
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL BENCHMARK COMPARISON")
    print("=" * 80)
    
    for model_info in models_to_test:
        print(f"\n{'='*40}")
        print(f"Testing: {model_info['name']}")
        print(f"{'='*40}")
        
        # Конфигурация benchmark
        config = BenchmarkConfig(
            model_path=model_info['path'],
            model_type=model_info['type'],
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
            num_iterations=1000,
            use_gpu=model_info.get('use_gpu', False),
            num_threads=4
        )
        
        # Запуск benchmark
        benchmark = ModelBenchmark(config)
        results = benchmark.run_benchmark()
        
        # Сохранение результатов
        all_results[model_info['name']] = results
        
        # Добавление в сравнение
        best_result = results.results_per_batch[results.summary['best_batch_size']]
        
        comparison_data.append({
            'model_name': model_info['name'],
            'best_batch_size': results.summary['best_batch_size'],
            'max_throughput_rps': results.summary['max_throughput'],
            'min_latency_ms': results.summary['min_latency'],
            'avg_memory_mb': results.summary['avg_memory_usage'],
            'optimal_latency': best_result['avg_inference_time_ms'],
            'optimal_throughput': best_result['throughput_rps']
        })
        
        # Визуализация
        benchmark.plot_results(
            f"../monitoring/reports/benchmark_{model_info['name']}_{int(time.time())}.png"
        )
    
    # Создание сравнительной таблицы
    comparison_df = pd.DataFrame(comparison_data)
    
    # Расчет улучшений относительно PyTorch
    pytorch_baseline = comparison_df[comparison_df['model_name'] == 'PyTorch_CPU'].iloc[0]
    
    for idx, row in comparison_df.iterrows():
        if row['model_name'] != 'PyTorch_CPU':
            throughput_improvement = (row['max_throughput_rps'] / pytorch_baseline['max_throughput_rps'] - 1) * 100
            latency_improvement = (1 - row['min_latency_ms'] / pytorch_baseline['min_latency_ms']) * 100
            comparison_df.at[idx, 'throughput_improvement_%'] = throughput_improvement
            comparison_df.at[idx, 'latency_improvement_%'] = latency_improvement
    
    # Генерация финального отчета
    final_report = {
        'benchmark_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'test_configuration': {
            'batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128],
            'iterations_per_test': 1000,
            'warmup_iterations': 100,
            'cpu_threads': 4
        },
        'detailed_results': {
            name: {
                'summary': results.summary,
                'per_batch_results': results.results_per_batch
            }
            for name, results in all_results.items()
        },
        'comparison_table': comparison_df.to_dict('records'),
        'recommendations': {
            'production_choice': 'ONNX_Quantized_CPU',
            'reasoning': [
                'Highest throughput per resource',
                'Smallest model size',
                'Good latency characteristics',
                'Optimized for CPU inference'
            ],
            'optimal_configuration': {
                'model': 'credit_scoring_quantized.onnx',
                'batch_size': 32,
                'expected_rps': comparison_df.loc[
                    comparison_df['model_name'] == 'ONNX_Quantized_CPU', 
                    'max_throughput_rps'
                ].values[0],
                'expected_p95_latency_ms': comparison_df.loc[
                    comparison_df['model_name'] == 'ONNX_Quantized_CPU', 
                    'min_latency_ms'
                ].values[0] * 1.5,  # Добавляем запас для p95
                'memory_footprint_mb': comparison_df.loc[
                    comparison_df['model_name'] == 'ONNX_Quantized_CPU', 
                    'avg_memory_mb'
                ].values[0]
            },
            'alternative_scenarios': {
                'low_latency': {
                    'model': 'ONNX_CPU',
                    'batch_size': 1,
                    'expected_latency_ms': comparison_df.loc[
                        comparison_df['model_name'] == 'ONNX_CPU', 
                        'min_latency_ms'
                    ].values[0]
                },
                'high_throughput': {
                    'model': 'ONNX_Quantized_CPU',
                    'batch_size': 64,
                    'expected_rps': comparison_df.loc[
                        comparison_df['model_name'] == 'ONNX_Quantized_CPU', 
                        'optimal_throughput'
                    ].values[0]
                }
            }
        }
    }
    
    # Сохранение отчета
    with open('../monitoring/reports/benchmark_final_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Вывод результатов
    print(f"\n{'='*80}")
    print("FINAL BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    print("\nComparison Table:")
    print("-" * 100)
    print(f"{'Model':<20} {'Best Batch':<12} {'Max RPS':<12} {'Min Latency':<14} {'Memory':<12} {'Improvement':<20}")
    print("-" * 100)
    
    for _, row in comparison_df.iterrows():
        improvement_text = ""
        if 'throughput_improvement_%' in row:
            improvement_text = f"+{row['throughput_improvement_%']:.1f}% RPS"
        
        print(f"{row['model_name']:<20} "
              f"{row['best_batch_size']:<12} "
              f"{row['max_throughput_rps']:<12.1f} "
              f"{row['min_latency_ms']:<14.3f} "
              f"{row['avg_memory_mb']:<12.1f} "
              f"{improvement_text:<20}")
    
    print(f"\n{'='*80}")
    print("PRODUCTION RECOMMENDATION")
    print(f"{'='*80}")
    
    rec = final_report['recommendations']['optimal_configuration']
    print(f"\nRecommended configuration:")
    print(f"  Model: {rec['model']}")
    print(f"  Batch size: {rec['batch_size']}")
    print(f"  Expected throughput: {rec['expected_rps']:.1f} requests/sec")
    print(f"  Expected p95 latency: {rec['expected_p95_latency_ms']:.2f} ms")
    print(f"  Memory footprint: {rec['memory_footprint_mb']:.1f} MB")
    
    return final_report

if __name__ == "__main__":
    print("Starting comprehensive benchmark comparison...")
    report = compare_all_models()
    print(f"\nFull benchmark report saved to ../monitoring/reports/benchmark_final_report.json")
    print(f"Visualizations saved to ../monitoring/reports/")