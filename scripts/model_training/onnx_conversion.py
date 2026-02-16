import torch
import onnx
import onnxruntime as ort
import numpy as np
import time
import json
from train_nn_model import CreditScoringNN

def validate_onnx_conversion(pytorch_model, onnx_model_path):
    """Валидация корректности конвертации"""
    # Тестовые данные
    dummy_input = torch.randn(1, 20)
    
    # PyTorch инференс
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()
    
    # ONNX Runtime инференс
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Сравнение результатов
    diff = np.abs(pytorch_output - ort_output).max()
    
    # Проверка структуры ONNX
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    
    validation_results = {
        "max_difference": float(diff),
        "pytorch_output": float(pytorch_output[0][0]),
        "onnx_output": float(ort_output[0][0]),
        "onnx_ir_version": onnx_model.ir_version,
        "onnx_producer_name": onnx_model.producer_name,
        "onnx_inputs": str([input.name for input in onnx_model.graph.input]),
        "onnx_outputs": str([output.name for output in onnx_model.graph.output]),
        "validation_status": "PASS" if diff < 1e-5 else "FAIL"
    }
    
    print(f"\nВалидация конвертации:")
    print(f"  Max разница: {diff:.10f}")
    print(f"  PyTorch результат: {pytorch_output[0][0]:.6f}")
    print(f"  ONNX результат: {ort_output[0][0]:.6f}")
    print(f"  Статус: {validation_results['validation_status']}")
    
    return validation_results

def benchmark_performance(pytorch_model, onnx_model_path, n_iterations=1000):
    """Сравнение производительности PyTorch vs ONNX на CPU"""
    # Тестовые данные
    dummy_input = torch.randn(1, 20)
    dummy_input_np = dummy_input.numpy()
    
    # PyTorch benchmark
    pytorch_model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = pytorch_model(dummy_input)
    
    # Измерение времени PyTorch
    torch_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = pytorch_model(dummy_input)
    torch_time = time.perf_counter() - torch_start
    
    # ONNX Runtime benchmark
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
    
    # Warmup
    for _ in range(100):
        _ = ort_session.run(None, ort_inputs)
    
    # Измерение времени ONNX
    onnx_start = time.perf_counter()
    for _ in range(n_iterations):
        _ = ort_session.run(None, ort_inputs)
    onnx_time = time.perf_counter() - onnx_start
    
    benchmark_results = {
        "pytorch_time_ms": torch_time * 1000,
        "onnx_time_ms": onnx_time * 1000,
        "speedup": torch_time / onnx_time,
        "iterations": n_iterations,
        "pytorch_inference_per_sec": n_iterations / torch_time,
        "onnx_inference_per_sec": n_iterations / onnx_time
    }
    
    print(f"\nBenchmark производительности ({n_iterations} итераций):")
    print(f"  PyTorch CPU: {torch_time*1000:.2f} ms")
    print(f"  ONNX Runtime: {onnx_time*1000:.2f} ms")
    print(f"  Ускорение: {benchmark_results['speedup']:.2f}x")
    
    return benchmark_results

def convert_to_onnx():
    """Полный процесс конвертации PyTorch → ONNX"""
    print("Начало конвертации PyTorch → ONNX")
    
    # Загрузка обученной модели
    model = CreditScoringNN(input_size=20, hidden_size=64, dropout_rate=0.3)
    model.load_state_dict(torch.load('../models/credit_scoring_nn.pth', map_location='cpu'))
    model.eval()
    
    # Конвертация
    dummy_input = torch.randn(1, 20)
    onnx_path = '../models/credit_scoring.onnx'
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['features'],
        output_names=['score'],
        dynamic_axes={'features': {0: 'batch_size'}, 'score': {0: 'batch_size'}},
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"Модель успешно экспортирована в {onnx_path}")
    
    # Валидация
    validation_results = validate_onnx_conversion(model, onnx_path)
    
    # Benchmark
    benchmark_results = benchmark_performance(model, onnx_path, n_iterations=5000)
    
    # Сохранение результатов
    conversion_report = {
        "conversion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_model": "../models/credit_scoring_nn.pth",
        "onnx_model": onnx_path,
        "validation": validation_results,
        "benchmark": benchmark_results,
        "onnx_export_settings": {
            "opset_version": 13,
            "dynamic_axes": True,
            "do_constant_folding": True
        }
    }
    
    with open('../models/conversion_report.json', 'w') as f:
        json.dump(conversion_report, f, indent=2)
    
    print(f"\nПолный отчет сохранен в ../models/conversion_report.json")
    
    # Проверка на разных размерах батча
    print("\nПроверка на разных размерах батча:")
    batch_sizes = [1, 8, 32, 64]
    ort_session = ort.InferenceSession(onnx_path)
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 20)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        
        start_time = time.perf_counter()
        for _ in range(100):
            _ = ort_session.run(None, ort_inputs)
        elapsed = time.perf_counter() - start_time
        
        print(f"  Batch size {batch_size:2d}: {elapsed*1000/100:.2f} ms per inference")

if __name__ == "__main__":
    convert_to_onnx()