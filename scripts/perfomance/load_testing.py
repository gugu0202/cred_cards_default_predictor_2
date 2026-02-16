import asyncio
import aiohttp
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import json
import psutil
import requests
from dataclasses import dataclass
from typing import List, Dict
import statistics

@dataclass
class LoadTestConfig:
    """Конфигурация нагрузочного теста"""
    url: str = "http://localhost:8000/predict"
    max_concurrent: int = 100
    total_requests: int = 10000
    request_timeout: int = 30
    payload_size: int = 1  # batch size
    ramp_up_time: int = 60  # seconds

@dataclass
class TestResult:
    """Результаты тестирования"""
    config: LoadTestConfig
    total_time: float = 0.0
    requests_per_second: float = 0.0
    avg_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    cpu_usage: List[float] = None
    memory_usage: List[float] = None
    response_times: List[float] = None
    
    def __post_init__(self):
        if self.cpu_usage is None:
            self.cpu_usage = []
        if self.memory_usage is None:
            self.memory_usage = []
        if self.response_times is None:
            self.response_times = []

class LoadTester:
    """Класс для нагрузочного тестирования"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = TestResult(config)
        self.latencies = []
        
    def generate_payload(self):
        """Генерация тестовых данных"""
        # 20 фичей как в модели
        return {
            "features": np.random.randn(self.config.payload_size, 20).tolist(),
            "model_version": "quantized"
        }
    
    async def make_request(self, session: aiohttp.ClientSession, request_id: int):
        """Выполнение одного запроса"""
        start_time = time.perf_counter()
        
        try:
            payload = self.generate_payload()
            
            async with session.post(
                self.config.url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            ) as response:
                end_time = time.perf_counter()
                
                if response.status == 200:
                    latency = (end_time - start_time) * 1000  # ms
                    self.latencies.append(latency)
                    self.results.success_count += 1
                    return True, latency
                else:
                    self.results.error_count += 1
                    return False, 0
                    
        except Exception as e:
            self.results.error_count += 1
            return False, 0
    
    async def monitor_resources(self, duration: int):
        """Мониторинг использования ресурсов"""
        cpu_readings = []
        mem_readings = []
        
        for _ in range(duration):
            cpu_readings.append(psutil.cpu_percent(interval=1))
            mem_readings.append(psutil.virtual_memory().percent)
            await asyncio.sleep(1)
        
        self.results.cpu_usage = cpu_readings
        self.results.memory_usage = mem_readings
    
    async def run_test(self):
        """Запуск нагрузочного теста"""
        print(f"Starting load test: {self.config.total_requests} requests, "
              f"{self.config.max_concurrent} concurrent")
        
        # Запуск мониторинга ресурсов в фоне
        monitor_task = asyncio.create_task(
            self.monitor_resources(self.config.total_requests // 100 + 10)
        )
        
        # Подготовка сессии
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            start_time = time.perf_counter()
            
            # Создание задач
            tasks = []
            for i in range(self.config.total_requests):
                if i % 100 == 0 and i > 0:
                    print(f"  Progress: {i}/{self.config.total_requests} requests")
                
                task = asyncio.create_task(self.make_request(session, i))
                tasks.append(task)
                
                # Контролируем ramp-up
                if i < self.config.max_concurrent:
                    await asyncio.sleep(self.config.ramp_up_time / self.config.max_concurrent)
            
            # Ожидание завершения
            results = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
        
        # Ожидание завершения мониторинга
        await monitor_task
        
        # Расчет метрик
        self.results.total_time = end_time - start_time
        self.results.requests_per_second = self.results.success_count / self.results.total_time
        
        if self.latencies:
            self.results.avg_response_time = statistics.mean(self.latencies)
            self.results.min_response_time = min(self.latencies)
            self.results.max_response_time = max(self.latencies)
            self.results.response_times = self.latencies
            
            # Перцентили
            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)
            self.results.p50_response_time = sorted_latencies[int(n * 0.5)]
            self.results.p95_response_time = sorted_latencies[int(n * 0.95)]
            self.results.p99_response_time = sorted_latencies[int(n * 0.99)]
        
        return self.results
    
    def generate_report(self, instance_type: str):
        """Генерация отчета"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "instance_type": instance_type,
            "config": {
                "total_requests": self.config.total_requests,
                "max_concurrent": self.config.max_concurrent,
                "payload_size": self.config.payload_size
            },
            "results": {
                "total_time_seconds": round(self.results.total_time, 2),
                "requests_per_second": round(self.results.requests_per_second, 2),
                "success_rate": round(self.results.success_count / 
                                    (self.results.success_count + self.results.error_count) * 100, 2),
                "avg_response_time_ms": round(self.results.avg_response_time, 2),
                "p50_response_time_ms": round(self.results.p50_response_time, 2),
                "p95_response_time_ms": round(self.results.p95_response_time, 2),
                "p99_response_time_ms": round(self.results.p99_response_time, 2),
                "min_response_time_ms": round(self.results.min_response_time, 2),
                "max_response_time_ms": round(self.results.max_response_time, 2),
                "success_count": self.results.success_count,
                "error_count": self.results.error_count
            },
            "resource_usage": {
                "avg_cpu_percent": round(statistics.mean(self.results.cpu_usage), 2) 
                                   if self.results.cpu_usage else 0,
                "max_cpu_percent": round(max(self.results.cpu_usage), 2) 
                                 if self.results.cpu_usage else 0,
                "avg_memory_percent": round(statistics.mean(self.results.memory_usage), 2) 
                                    if self.results.memory_usage else 0,
                "max_memory_percent": round(max(self.results.memory_usage), 2) 
                                    if self.results.memory_usage else 0
            }
        }
        
        # Визуализация
        self.plot_results(instance_type)
        
        return report
    
    def plot_results(self, instance_type: str):
        """Визуализация результатов"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Распределение latency
        axes[0, 0].hist(self.latencies, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(self.results.p50_response_time, color='red', linestyle='--', 
                          label=f'p50: {self.results.p50_response_time:.2f}ms')
        axes[0, 0].axvline(self.results.p95_response_time, color='orange', linestyle='--', 
                          label=f'p95: {self.results.p95_response_time:.2f}ms')
        axes[0, 0].set_xlabel('Response Time (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Latency Distribution - {instance_type}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. CPU и Memory usage
        if self.results.cpu_usage and self.results.memory_usage:
            time_axis = list(range(len(self.results.cpu_usage)))
            axes[0, 1].plot(time_axis, self.results.cpu_usage, label='CPU %', color='green')
            axes[0, 1].plot(time_axis, self.results.memory_usage, label='Memory %', color='purple')
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Usage %')
            axes[0, 1].set_title('Resource Usage Over Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Throughput over time
        window_size = 100
        throughput = []
        for i in range(0, len(self.latencies), window_size):
            window = self.latencies[i:i+window_size]
            if window:
                throughput.append(len(window) / (sum(window) / 1000))  # requests per second
        
        axes[1, 0].plot(range(len(throughput)), throughput, color='red')
        axes[1, 0].set_xlabel('Time Window')
        axes[1, 0].set_ylabel('Throughput (req/sec)')
        axes[1, 0].set_title('Throughput Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary metrics
        metrics_text = (
            f"Instance: {instance_type}\n"
            f"Total Requests: {self.config.total_requests}\n"
            f"RPS: {self.results.requests_per_second:.1f}\n"
            f"Success Rate: {self.results.success_count/(self.results.success_count+self.results.error_count)*100:.1f}%\n"
            f"Avg Latency: {self.results.avg_response_time:.2f}ms\n"
            f"p95 Latency: {self.results.p95_response_time:.2f}ms\n"
            f"Max Latency: {self.results.max_response_time:.2f}ms"
        )
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'../monitoring/reports/load_test_{instance_type}_{int(time.time())}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

def test_different_configs():
    """Тестирование разных конфигураций инстансов"""
    # Эмулируем разные типы инстансов через разные настройки
    configs = {
        "cpu_small": LoadTestConfig(max_concurrent=50, total_requests=5000),
        "cpu_large": LoadTestConfig(max_concurrent=200, total_requests=20000),
        "gpu_small": LoadTestConfig(max_concurrent=100, total_requests=10000, 
                                   payload_size=32),  # Большие батчи для GPU
        "gpu_large": LoadTestConfig(max_concurrent=500, total_requests=50000, 
                                   payload_size=64)
    }
    
    all_reports = {}
    
    for instance_type, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing {instance_type} configuration")
        print(f"{'='*60}")
        
        tester = LoadTester(config)
        
        # Запуск теста
        asyncio.run(tester.run_test())
        
        # Генерация отчета
        report = tester.generate_report(instance_type)
        all_reports[instance_type] = report
        
        print(f"\nResults for {instance_type}:")
        print(f"  RPS: {report['results']['requests_per_second']:.1f}")
        print(f"  Avg Latency: {report['results']['avg_response_time_ms']:.2f}ms")
        print(f"  p95 Latency: {report['results']['p95_response_time_ms']:.2f}ms")
        print(f"  Success Rate: {report['results']['success_rate']}%")
    
    # Сравнительный анализ
    comparison = {
        "best_rps": max(all_reports.items(), 
                       key=lambda x: x[1]['results']['requests_per_second']),
        "best_latency": min(all_reports.items(), 
                          key=lambda x: x[1]['results']['avg_response_time_ms']),
        "most_cost_effective": None,  # Можно добавить цены инстансов
        "recommended_for_production": "cpu_large"  # По умолчанию
    }
    
    # Сохранение всех отчетов
    final_report = {
        "load_testing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configurations": configs,
        "detailed_results": all_reports,
        "comparison": comparison,
        "recommendations": {
            "production_config": {
                "instance_type": "cpu_large",
                "reason": "Balance of throughput and latency with good resource utilization",
                "expected_rps": all_reports["cpu_large"]["results"]["requests_per_second"],
                "expected_latency_p95": all_reports["cpu_large"]["results"]["p95_response_time_ms"],
                "concurrent_connections": 200,
                "auto_scaling_threshold": 150  # RPS для триггера масштабирования
            },
            "development_config": {
                "instance_type": "cpu_small",
                "reason": "Cost-effective for development and testing"
            },
            "batch_processing_config": {
                "instance_type": "gpu_large",
                "reason": "Optimal for large batch predictions",
                "optimal_batch_size": 64
            }
        }
    }
    
    with open('../monitoring/reports/load_testing_final_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FINAL RECOMMENDATIONS:")
    print(f"{'='*60}")
    print(f"Production: {final_report['recommendations']['production_config']['instance_type']}")
    print(f"  Expected RPS: {final_report['recommendations']['production_config']['expected_rps']:.1f}")
    print(f"  p95 Latency: {final_report['recommendations']['production_config']['expected_latency_p95']:.2f}ms")
    print(f"  Concurrent connections: {final_report['recommendations']['production_config']['concurrent_connections']}")
    
    return final_report

if __name__ == "__main__":
    # Запуск тестирования всех конфигураций
    print("Starting comprehensive load testing...")
    report = test_different_configs()
    print(f"\nFull report saved to ../monitoring/reports/load_testing_final_report.json")