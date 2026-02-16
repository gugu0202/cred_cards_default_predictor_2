import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
import time
import json
import psutil
import os

def quantize_onnx_model():
    """Квантование ONNX модели для уменьшения размера и ускорения инференса"""
    print("Начало квантования ONNX модели")
    
    # Исходная и целевая модели
    input_model_path = '../models/credit_scoring.onnx'
    quantized_model_path = '../models/credit_scoring_quantized.onnx'
    
    # Динамическое квантование (подходит для CPU)
    quantize_dynamic(
        input_model_path,
        quantized_model_path,
        weight_type=QuantType.QUInt8,
        optimize_model=True,
        use_external_data_format=False
    )
    
    print(f"Квантованная модель сохранена в {quantized_model_path}")
    
    # Сравнение размеров
    original_size = os.path.getsize(input_model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)
    
    print(f"\nСравнение размеров:")
    print(f"  Оригинальная модель: {original_size:.2f} MB")
    print(f"  Квантованная модель: {quantized_size:.2f} MB")
    print(f"  Сжатие: {original_size/quantized_size:.2f}x (уменьшение на {((original_size-quantized_size)/original_size*100):.1f}%)")
    
    return input_model_path, quantized_model_path

def validate_quantization(original_path, quantized_path):
    """Валидация квантованной модели"""
    print("\nВалидация квантованной модели:")
    
    # Загрузка сессий
    original_session = ort.InferenceSession(original_path)
    quantized_session = ort.InferenceSession(quantized_path)
    
    # Тестовые данные
    test_inputs = [
        np.random.randn(1, 20).astype(np.float32),
        np.random.randn(10, 20).astype(np.float32),
        np.random.randn(100, 20).astype(np.float32)
    ]
    
    validation_results = []
    
    for i, test_input in enumerate(test_inputs):
        # Инференс на оригинальной модели
        orig_inputs = {original_session.get_inputs()[0].name: test_input}
        orig_output = original_session.run(None, orig_inputs)[0]
        
        # Инференс на квантованной модели
        quant_inputs = {quantized_session.get_inputs()[0].name: test_input}
        quant_output = quantized_session.run(None, quant_inputs)[0]
        
        # Вычисление метрик
        mae = np.mean(np.abs(orig_output - quant_output))
        mse = np.mean((orig_output - quant_output) ** 2)
        max_diff = np.max(np.abs(orig_output - quant_output))
        
        validation_results.append({
            "batch_size": test_input.shape[0],
            "mae": float(mae),
            "mse": float(mse),
            "max_diff": float(max_diff),
            "original_mean": float(np.mean(orig_output)),
            "quantized_mean": float(np.mean(quant_output))
        })
        
        print(f"  Batch size {test_input.shape[0]:3d}: "
              f"MAE={mae:.6f}, MSE={mse:.8f}, MaxDiff={max_diff:.6f}")
    
    return validation_results

def benchmark_models(original_path, quantized_path, n_iterations=5000):
    """Сравнение производительности до/после квантования"""
    print("\nBenchmark производительности:")
    
    # Загрузка сессий
    original_session = ort.InferenceSession(original_path)
    quantized_session = ort.InferenceSession(quantized_path)
    
    # Тестовые данные разных размеров
    batch_sizes = [1, 8, 32, 64]
    benchmark_results = {}
    
    for batch_size in batch_sizes:
        test_input = np.random.randn(batch_size, 20).astype(np.float32)
        
        # Измерение использования памяти
        process = psutil.Process()
        
        # Benchmark оригинальной модели
        orig_inputs = {original_session.get_inputs()[0].name: test_input}
        
        # Warmup
        for _ in range(100):
            _ = original_session.run(None, orig_inputs)
        
        # Измерение времени
        mem_before = process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()
        
        for _ in range(n_iterations):
            _ = original_session.run(None, orig_inputs)
        
        orig_time = time.perf_counter() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024
        
        # Benchmark квантованной модели
        quant_inputs = {quantized_session.get_inputs()[0].name: test_input}
        
        # Warmup
        for _ in range(100):
            _ = quantized_session.run(None, quant_inputs)
        
        # Измерение времени
        start_time = time.perf_counter()
        
        for _ in range(n_iterations):
            _ = quantized_session.run(None, quant_inputs)
        
        quant_time = time.perf_counter() - start_time
        
        benchmark_results[batch_size] = {
            "original": {
                "time_ms": orig_time * 1000,
                "inference_per_sec": n_iterations / orig_time,
                "memory_mb": mem_after - mem_before,
                "latency_ms": (orig_time / n_iterations) * 1000
            },
            "quantized": {
                "time_ms": quant_time * 1000,
                "inference_per_sec": n_iterations / quant_time,
                "latency_ms": (quant_time / n_iterations) * 1000
            },
            "speedup": orig_time / quant_time
        }
        
        print(f"\n  Batch size {batch_size}:")
        print(f"    Оригинальная: {benchmark_results[batch_size]['original']['latency_ms']:.3f} ms/inf, "
              f"{benchmark_results[batch_size]['original']['inference_per_sec']:.0f} inf/sec")
        print(f"    Квантованная: {benchmark_results[batch_size]['quantized']['latency_ms']:.3f} ms/inf, "
              f"{benchmark_results[batch_size]['quantized']['inference_per_sec']:.0f} inf/sec")
        print(f"    Ускорение: {benchmark_results[batch_size]['speedup']:.2f}x")
    
    return benchmark_results

def measure_accuracy_drop(original_path, quantized_path, n_samples=1000):
    """Измерение потери точности после квантования"""
    print("\nИзмерение потери точности:")
    
    # Загрузка сессий
    original_session = ort.InferenceSession(original_path)
    quantized_session = ort.InferenceSession(quantized_path)
    
    # Генерация тестовых данных
    test_data = np.random.randn(n_samples, 20).astype(np.float32)
    
    # Инференс
    orig_inputs = {original_session.get_inputs()[0].name: test_data}
    orig_outputs = original_session.run(None, orig_inputs)[0]
    
    quant_inputs = {quantized_session.get_inputs()[0].name: test_data}
    quant_outputs = quantized_session.run(None, quant_inputs)[0]
    
    # Порог для бинарной классификации
    threshold = 0.5
    
    # Классификация
    orig_predictions = (orig_outputs > threshold).astype(int)
    quant_predictions = (quant_outputs > threshold).astype(int)
    
    # Сравнение предсказаний
    agreement = np.mean(orig_predictions == quant_predictions)
    disagreement_count = np.sum(orig_predictions != quant_predictions)
    
    # Точность на симулированных метках
    true_labels = (np.random.rand(n_samples, 1) > 0.7).astype(int)
    orig_accuracy = np.mean(orig_predictions == true_labels)
    quant_accuracy = np.mean(quant_predictions == true_labels)
    accuracy_drop = orig_accuracy - quant_accuracy
    
    accuracy_results = {
        "agreement_rate": float(agreement),
        "disagreement_count": int(disagreement_count),
        "original_accuracy": float(orig_accuracy),
        "quantized_accuracy": float(quant_accuracy),
        "accuracy_drop": float(accuracy_drop),
        "threshold": threshold,
        "n_samples": n_samples
    }
    
    print(f"  Согласие предсказаний: {agreement*100:.2f}%")
    print(f"  Количество расхождений: {disagreement_count}")
    print(f"  Точность оригинальной: {orig_accuracy*100:.2f}%")
    print(f"  Точность квантованной: {quant_accuracy*100:.2f}%")
    print(f"  Потеря точности: {accuracy_drop*100:.2f}%")
    
    return accuracy_results

if __name__ == "__main__":
    # Шаг 1: Квантование
    original_path, quantized_path = quantize_onnx_model()
    
    # Шаг 2: Валидация
    validation_results = validate_quantization(original_path, quantized_path)
    
    # Шаг 3: Benchmark
    benchmark_results = benchmark_models(original_path, quantized_path, n_iterations=3000)
    
    # Шаг 4: Измерение точности
    accuracy_results = measure_accuracy_drop(original_path, quantized_path)
    
    # Сохранение полного отчета
    optimization_report = {
        "optimization_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_model": original_path,
        "quantized_model": quantized_path,
        "file_sizes": {
            "original_mb": os.path.getsize(original_path) / (1024 * 1024),
            "quantized_mb": os.path.getsize(quantized_path) / (1024 * 1024),
            "compression_ratio": os.path.getsize(original_path) / os.path.getsize(quantized_path)
        },
        "validation": validation_results,
        "benchmark": benchmark_results,
        "accuracy": accuracy_results,
        "summary": {
            "size_reduction_percent": ((os.path.getsize(original_path) - os.path.getsize(quantized_path)) / 
                                      os.path.getsize(original_path) * 100),
            "avg_speedup": np.mean([results["speedup"] for results in benchmark_results.values()]),
            "accuracy_drop_percent": accuracy_results["accuracy_drop"] * 100,
            "recommended_for_production": accuracy_results["accuracy_drop"] < 0.01  # Потеря <1%
        }
    }
    
    with open('../models/quantization_report.json', 'w') as f:
        json.dump(optimization_report, f, indent=2)
    
    print(f"\nПолный отчет сохранен в ../models/quantization_report.json")
    
    # Рекомендации
    print("\n" + "="*60)
    print("РЕКОМЕНДАЦИИ:")
    print("="*60)
    
    if optimization_report["summary"]["recommended_for_production"]:
        print("✅ Квантованную модель МОЖНО использовать в продакшене:")
        print(f"   - Ускорение: {optimization_report['summary']['avg_speedup']:.2f}x")
        print(f"   - Уменьшение размера: {optimization_report['summary']['size_reduction_percent']:.1f}%")
        print(f"   - Потеря точности: {optimization_report['summary']['accuracy_drop_percent']:.2f}% (<1%)")
    else:
        print("⚠️  Квантованную модель НЕ рекомендуется использовать в продакшене:")
        print(f"   - Потеря точности слишком высока: {optimization_report['summary']['accuracy_drop_percent']:.2f}%")
        print("   - Рассмотрите другие методы оптимизации или дообучение после квантования")