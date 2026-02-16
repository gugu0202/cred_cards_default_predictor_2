import torch
import onnx
import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
import time
from typing import Dict, Tuple
import pandas as pd

def validate_pytorch_to_onnx() -> Dict:
    """
    Полная валидация конвертации PyTorch → ONNX
    Проверяет корректность, точность и производительность
    """
    
    print("=" * 80)
    print("VALIDATION OF PyTorch → ONNX CONVERSION")
    print("=" * 80)
    
    # Пути к моделям
    pytorch_model_path = Path("../models/credit_scoring_nn.pth")
    onnx_model_path = Path("../models/credit_scoring.onnx")
    
    # Проверка существования файлов
    if not pytorch_model_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pytorch_model_path}")
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    
    validation_results = {
        "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "pytorch": str(pytorch_model_path),
            "onnx": str(onnx_model_path),
            "pytorch_size_mb": pytorch_model_path.stat().st_size / (1024 * 1024),
            "onnx_size_mb": onnx_model_path.stat().st_size / (1024 * 1024)
        },
        "tests": {},
        "summary": {}
    }
    
    # 1. Проверка структуры ONNX модели
    print("\n1. Checking ONNX model structure...")
    try:
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        
        validation_results["tests"]["onnx_structure"] = {
            "status": "PASS",
            "ir_version": onnx_model.ir_version,
            "producer_name": onnx_model.producer_name,
            "producer_version": onnx_model.producer_version,
            "opset_import": [str(opset) for opset in onnx_model.opset_import],
            "inputs": [
                {
                    "name": input.name,
                    "type": str(input.type),
                    "shape": str([dim.dim_value for dim in input.type.tensor_type.shape.dim])
                }
                for input in onnx_model.graph.input
            ],
            "outputs": [
                {
                    "name": output.name,
                    "type": str(output.type),
                    "shape": str([dim.dim_value for dim in output.type.tensor_type.shape.dim])
                }
                for output in onnx_model.graph.output
            ]
        }
        print("   ✓ ONNX structure is valid")
    except Exception as e:
        validation_results["tests"]["onnx_structure"] = {
            "status": "FAIL",
            "error": str(e)
        }
        print(f"   ✗ ONNX structure error: {e}")
    
    # 2. Загрузка PyTorch модели
    print("\n2. Loading PyTorch model...")
    try:
        from train_nn_model import CreditScoringNN
        pytorch_model = CreditScoringNN(input_size=20, hidden_size=64, dropout_rate=0.3)
        pytorch_model.load_state_dict(torch.load(str(pytorch_model_path), map_location='cpu'))
        pytorch_model.eval()
        print("   ✓ PyTorch model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch model: {e}")
    
    # 3. Проверка точности на тестовых данных
    print("\n3. Testing numerical accuracy...")
    
    # Генерация тестовых данных
    np.random.seed(42)
    test_cases = [
        ("single_sample", np.random.randn(1, 20).astype(np.float32)),
        ("small_batch", np.random.randn(8, 20).astype(np.float32)),
        ("medium_batch", np.random.randn(32, 20).astype(np.float32)),
        ("large_batch", np.random.randn(128, 20).astype(np.float32)),
        ("edge_case_zeros", np.zeros((1, 20), dtype=np.float32)),
        ("edge_case_ones", np.ones((1, 20), dtype=np.float32))
    ]
    
    accuracy_results = []
    
    # Инициализация ONNX Runtime сессии
    ort_session = ort.InferenceSession(str(onnx_model_path))
    
    for test_name, test_input in test_cases:
        # PyTorch inference
        with torch.no_grad():
            pytorch_input = torch.from_numpy(test_input)
            pytorch_output = pytorch_model(pytorch_input).numpy()
        
        # ONNX inference
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Вычисление метрик
        mae = np.mean(np.abs(pytorch_output - ort_output))
        mse = np.mean((pytorch_output - ort_output) ** 2)
        max_diff = np.max(np.abs(pytorch_output - ort_output))
        relative_error = np.mean(np.abs(pytorch_output - ort_output) / (np.abs(pytorch_output) + 1e-10))
        
        accuracy_results.append({
            "test_case": test_name,
            "input_shape": list(test_input.shape),
            "pytorch_mean": float(np.mean(pytorch_output)),
            "onnx_mean": float(np.mean(ort_output)),
            "mae": float(mae),
            "mse": float(mse),
            "max_diff": float(max_diff),
            "relative_error": float(relative_error),
            "status": "PASS" if max_diff < 1e-5 else "WARNING" if max_diff < 1e-3 else "FAIL"
        })
        
        status_icon = "✓" if max_diff < 1e-5 else "⚠" if max_diff < 1e-3 else "✗"
        print(f"   {status_icon} {test_name}: "
              f"max_diff={max_diff:.2e}, "
              f"mae={mae:.2e}, "
              f"rel_error={relative_error:.2e}")
    
    validation_results["tests"]["numerical_accuracy"] = accuracy_results
    
    # 4. Проверка производительности
    print("\n4. Testing performance consistency...")
    
    # Тестирование на разных размерах батча
    batch_sizes = [1, 8, 16, 32, 64, 128]
    n_iterations = 1000
    
    performance_results = []
    
    for batch_size in batch_sizes:
        test_input = np.random.randn(batch_size, 20).astype(np.float32)
        
        # PyTorch время
        torch_input = torch.from_numpy(test_input)
        
        # Warmup
        with torch.no_grad():
            for _ in range(100):
                _ = pytorch_model(torch_input)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = pytorch_model(torch_input)
        pytorch_time = time.perf_counter() - start_time
        
        # ONNX время
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        
        # Warmup
        for _ in range(100):
            _ = ort_session.run(None, ort_inputs)
        
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            _ = ort_session.run(None, ort_inputs)
        onnx_time = time.perf_counter() - start_time
        
        performance_results.append({
            "batch_size": batch_size,
            "pytorch_time_ms": pytorch_time * 1000,
            "onnx_time_ms": onnx_time * 1000,
            "speedup": pytorch_time / onnx_time,
            "pytorch_rps": n_iterations / pytorch_time,
            "onnx_rps": n_iterations / onnx_time,
            "time_difference_percent": ((pytorch_time - onnx_time) / pytorch_time) * 100
        })
        
        print(f"   Batch {batch_size:3d}: "
              f"PyTorch={pytorch_time*1000/n_iterations:.3f}ms, "
              f"ONNX={onnx_time*1000/n_iterations:.3f}ms, "
              f"Speedup={pytorch_time/onnx_time:.2f}x")
    
    validation_results["tests"]["performance"] = performance_results
    
    # 5. Проверка градиентов (для тренировочных сценариев)
    print("\n5. Checking gradient compatibility...")
    
    try:
        # Создаем ONNX модель с поддержкой градиентов
        torch.onnx.export(
            pytorch_model,
            torch.randn(1, 20),
            "../models/credit_scoring_training.onnx",
            input_names=['features'],
            output_names=['score'],
            training=torch.onnx.TrainingMode.TRAINING,
            do_constant_folding=False,
            export_params=True,
            opset_version=13
        )
        
        validation_results["tests"]["gradient_compatibility"] = {
            "status": "PASS",
            "training_model_exported": True,
            "path": "../models/credit_scoring_training.onnx"
        }
        print("   ✓ Training model with gradients exported successfully")
    except Exception as e:
        validation_results["tests"]["gradient_compatibility"] = {
            "status": "WARNING",
            "error": str(e),
            "note": "Gradient export not required for inference-only deployment"
        }
        print(f"   ⚠ Gradient export warning: {e}")
    
    # 6. Проверка сериализации/десериализации
    print("\n6. Testing serialization/deserialization...")
    
    serialization_tests = []
    
    try:
        # Пересохранение ONNX модели
        onnx.save(onnx_model, "../models/credit_scoring_reloaded.onnx")
        
        # Загрузка пересохраненной модели
        reloaded_model = onnx.load("../models/credit_scoring_reloaded.onnx")
        onnx.checker.check_model(reloaded_model)
        
        # Проверка эквивалентности
        reloaded_session = ort.InferenceSession("../models/credit_scoring_reloaded.onnx")
        test_input = np.random.randn(1, 20).astype(np.float32)
        
        original_output = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input})[0]
        reloaded_output = reloaded_session.run(None, {reloaded_session.get_inputs()[0].name: test_input})[0]
        
        diff = np.max(np.abs(original_output - reloaded_output))
        
        serialization_tests.append({
            "test": "reload_onnx",
            "status": "PASS" if diff < 1e-10 else "FAIL",
            "max_difference": float(diff)
        })
        
        print(f"   ✓ ONNX reload test: diff={diff:.2e}")
    except Exception as e:
        serialization_tests.append({
            "test": "reload_onnx",
            "status": "FAIL",
            "error": str(e)
        })
        print(f"   ✗ ONNX reload test failed: {e}")
    
    validation_results["tests"]["serialization"] = serialization_tests
    
    # 7. Генерация сводки
    print("\n7. Generating validation summary...")
    
    # Подсчет статусов
    all_tests = []
    for category, tests in validation_results["tests"].items():
        if isinstance(tests, list):
            for test in tests:
                if "status" in test:
                    all_tests.append(test["status"])
        elif isinstance(tests, dict) and "status" in tests:
            all_tests.append(tests["status"])
    
    passed = sum(1 for status in all_tests if status == "PASS")
    warnings = sum(1 for status in all_tests if status == "WARNING")
    failed = sum(1 for status in all_tests if status == "FAIL")
    
    # Проверка ключевых критериев
    numerical_tests = validation_results["tests"]["numerical_accuracy"]
    max_numerical_diff = max(test["max_diff"] for test in numerical_tests)
    
    performance_tests = validation_results["tests"]["performance"]
    avg_speedup = np.mean([test["speedup"] for test in performance_tests])
    
    validation_results["summary"] = {
        "total_tests": len(all_tests),
        "passed": passed,
        "warnings": warnings,
        "failed": failed,
        "success_rate": (passed / len(all_tests)) * 100 if all_tests else 0,
        "key_metrics": {
            "max_numerical_difference": float(max_numerical_diff),
            "average_speedup": float(avg_speedup),
            "size_reduction_percent": (
                (validation_results["models"]["pytorch_size_mb"] - 
                 validation_results["models"]["onnx_size_mb"]) / 
                validation_results["models"]["pytorch_size_mb"] * 100
            )
        },
        "validation_status": (
            "PASS" if failed == 0 and max_numerical_diff < 1e-4 
            else "WARNING" if max_numerical_diff < 1e-3 
            else "FAIL"
        ),
        "recommendations": [
            "Use ONNX model for production deployment" if max_numerical_diff < 1e-4 
            else "Review conversion parameters" if max_numerical_diff < 1e-3 
            else "Fix conversion errors before deployment"
        ]
    }
    
    # Сохранение результатов
    output_path = Path("../models/conversion_validation_report.json")
    with open(output_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Вывод финального вердикта
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    
    status = validation_results["summary"]["validation_status"]
    status_color = "\033[92m" if status == "PASS" else "\033[93m" if status == "WARNING" else "\033[91m"
    print(f"\nOverall Status: {status_color}{status}\033[0m")
    
    print(f"\nSummary:")
    print(f"  Tests: {passed} passed, {warnings} warnings, {failed} failed")
    print(f"  Max numerical difference: {max_numerical_diff:.2e}")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Size reduction: {validation_results['summary']['key_metrics']['size_reduction_percent']:.1f}%")
    
    print(f"\nRecommendations:")
    for rec in validation_results["summary"]["recommendations"]:
        print(f"  • {rec}")
    
    print(f"\nFull report saved to: {output_path}")
    
    return validation_results

def generate_validation_report() -> None:
    """Генерация HTML отчета для валидации"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Загрузка результатов
    with open('../models/conversion_validation_report.json', 'r') as f:
        results = json.load(f)
    
    # Создание фигуры
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Numerical Accuracy by Test Case',
            'Performance Comparison',
            'Batch Size vs Inference Time',
            'Error Distribution',
            'Validation Summary'
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"colspan": 2, "type": "table"}, None]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Numerical Accuracy
    accuracy_data = results["tests"]["numerical_accuracy"]
    test_cases = [test["test_case"] for test in accuracy_data]
    max_diffs = [test["max_diff"] for test in accuracy_data]
    
    colors = []
    for test in accuracy_data:
        if test["max_diff"] < 1e-5:
            colors.append('green')
        elif test["max_diff"] < 1e-3:
            colors.append('orange')
        else:
            colors.append('red')
    
    fig.add_trace(
        go.Bar(
            x=test_cases,
            y=[-np.log10(diff + 1e-10) for diff in max_diffs],
            marker_color=colors,
            name='-log10(Max Diff)',
            text=[f"{diff:.2e}" for diff in max_diffs],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    fig.update_yaxes(title_text="-log10(Max Difference)", row=1, col=1)
    
    # 2. Performance Comparison
    perf_data = results["tests"]["performance"]
    batch_sizes = [test["batch_size"] for test in perf_data]
    pytorch_times = [test["pytorch_time_ms"] for test in perf_data]
    onnx_times = [test["onnx_time_ms"] for test in perf_data]
    
    fig.add_trace(
        go.Scatter(
            x=batch_sizes,
            y=pytorch_times,
            mode='lines+markers',
            name='PyTorch',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=batch_sizes,
            y=onnx_times,
            mode='lines+markers',
            name='ONNX',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Batch Size", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Total Time (ms)", type="log", row=1, col=2)
    
    # 3. Batch Size vs Inference Time
    inference_per_batch = [test["onnx_time_ms"] / 1000 for test in perf_data]
    
    fig.add_trace(
        go.Scatter(
            x=batch_sizes,
            y=inference_per_batch,
            mode='lines+markers',
            name='Inference Time',
            fill='tozeroy',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Batch Size", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Time per Inference (ms)", row=2, col=1)
    
    # 4. Error Distribution
    all_errors = []
    for test in accuracy_data:
        all_errors.extend([test["mae"], test["mse"], test["max_diff"]])
    
    fig.add_trace(
        go.Histogram(
            x=all_errors,
            nbinsx=50,
            name='Error Distribution',
            marker_color='lightblue'
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Error Value", type="log", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    # 5. Validation Summary Table
    summary = results["summary"]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value', 'Status'],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[
                    [
                        'Total Tests', 'Passed', 'Warnings', 'Failed',
                        'Success Rate', 'Max Numerical Diff',
                        'Average Speedup', 'Size Reduction', 'Overall Status'
                    ],
                    [
                        summary['total_tests'],
                        summary['passed'],
                        summary['warnings'],
                        summary['failed'],
                        f"{summary['success_rate']:.1f}%",
                        f"{summary['key_metrics']['max_numerical_difference']:.2e}",
                        f"{summary['key_metrics']['average_speedup']:.2f}x",
                        f"{summary['key_metrics']['size_reduction_percent']:.1f}%",
                        summary['validation_status']
                    ],
                    [
                        '', '', '', '',
                        '✓' if summary['success_rate'] > 90 else '⚠' if summary['success_rate'] > 70 else '✗',
                        '✓' if summary['key_metrics']['max_numerical_difference'] < 1e-5 else '⚠' if summary['key_metrics']['max_numerical_difference'] < 1e-3 else '✗',
                        '✓' if summary['key_metrics']['average_speedup'] > 1.0 else '⚠',
                        '✓' if summary['key_metrics']['size_reduction_percent'] > 0 else '✗',
                        '✓' if summary['validation_status'] == 'PASS' else '⚠' if summary['validation_status'] == 'WARNING' else '✗'
                    ]
                ],
                fill_color=[['white', 'lightgrey'] * 5],
                align='left',
                font=dict(size=11)
            )
        ),
        row=3, col=1
    )
    
    # Обновление layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="PyTorch to ONNX Conversion Validation Report",
        title_font_size=20
    )
    
    # Сохранение HTML
    fig.write_html("../monitoring/reports/conversion_validation_report.html")
    print("HTML report saved to ../monitoring/reports/conversion_validation_report.html")

if __name__ == "__main__":
    # Запуск валидации
    print("Starting comprehensive validation of PyTorch to ONNX conversion...")
    results = validate_pytorch_to_onnx()
    
    # Генерация HTML отчета
    generate_validation_report()