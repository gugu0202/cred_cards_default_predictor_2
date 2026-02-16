# Prometheus Stack
resource "helm_release" "kube_prometheus_stack" {
  name       = "kube-prometheus-stack"
  namespace  = "monitoring"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "45.0.0"

  create_namespace = true

  values = [
    templatefile("${path.module}/values/prometheus-values.yaml", {
      environment    = var.environment
      storage_class  = "fast-ssd"
      retention_size = "50Gi"
      retention_time = "30d"
    })
  ]

  set {
    name  = "grafana.adminPassword"
    value = random_password.grafana_password.result
  }

  set {
    name  = "alertmanager.config.global.slack_api_url"
    value = var.slack_webhook_url
  }

  depends_on = [kubernetes_namespace.monitoring]
}

# Loki для логов
resource "helm_release" "loki" {
  name       = "loki"
  namespace  = "logging"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "loki"
  version    = "5.0.0"

  create_namespace = true

  values = [
    templatefile("${path.module}/values/loki-values.yaml", {
      environment   = var.environment
      storage_class = "standard-hdd"
      bucket_name   = var.logs_bucket_name
    })
  ]

  depends_on = [kubernetes_namespace.logging]
}

# Promtail для сбора логов
resource "helm_release" "promtail" {
  name       = "promtail"
  namespace  = "logging"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "promtail"
  version    = "6.0.0"

  values = [templatefile("${path.module}/values/promtail-values.yaml", {
    environment = var.environment
    loki_url    = "http://loki.logging:3100"
  })]

  depends_on = [helm_release.loki]
}

# Jaeger для трассировки
resource "helm_release" "jaeger" {
  name       = "jaeger"
  namespace  = "monitoring"
  repository = "https://jaegertracing.github.io/helm-charts"
  chart      = "jaeger"
  version    = "0.71.0"

  values = [templatefile("${path.module}/values/jaeger-values.yaml", {
    environment = var.environment
  })]

  depends_on = [kubernetes_namespace.monitoring]
}

# Custom Prometheus exporters
resource "kubernetes_deployment" "mlflow_exporter" {
  metadata {
    name      = "mlflow-exporter"
    namespace = "monitoring"
    labels = {
      app = "mlflow-exporter"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "mlflow-exporter"
      }
    }

    template {
      metadata {
        labels = {
          app = "mlflow-exporter"
        }
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "8000"
        }
      }

      spec {
        container {
          name  = "mlflow-exporter"
          image = "prometheuscommunity/postgres-exporter:v0.11.1"

          env {
            name  = "DATA_SOURCE_NAME"
            value = "postgresql://mlflow_user:${var.mlflow_db_password}@${var.postgresql_host}:6432/mlflow?sslmode=disable"
          }

          port {
            container_port = 9187
            name           = "http"
          }

          resources {
            requests = {
              memory = "64Mi"
              cpu    = "50m"
            }
            limits = {
              memory = "128Mi"
              cpu    = "200m"
            }
          }
        }
      }
    }
  }
}

# ServiceMonitor для кастомных метрик
resource "kubernetes_manifest" "credit_scoring_service_monitor" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "credit-scoring-monitor"
      namespace = "monitoring"
      labels = {
        release = "kube-prometheus-stack"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "app.kubernetes.io/name" = "credit-scoring-api"
        }
      }
      endpoints = [{
        port     = "metrics"
        interval = "30s"
        path     = "/metrics"
      }]
    }
  }
}

# Alertmanager правила
resource "kubernetes_config_map" "alertmanager_rules" {
  metadata {
    name      = "alertmanager-rules"
    namespace = "monitoring"
  }

  data = {
    "credit-scoring-rules.yml" = templatefile("${path.module}/rules/alert-rules.yaml", {
      environment   = var.environment
      alert_email   = var.alert_email
      slack_channel = var.slack_channel
    })
  }
}

# Grafana дашборды
resource "kubernetes_config_map" "grafana_dashboards" {
  metadata {
    name      = "grafana-dashboards"
    namespace = "monitoring"
  }

  data = {
    "credit-scoring-overview.json" = file("${path.module}/dashboards/credit-scoring-overview.json")
    "model-metrics.json"           = file("${path.module}/dashboards/model-metrics.json")
    "infrastructure.json"          = file("${path.module}/dashboards/infrastructure.json")
    "drift-detection.json"         = file("${path.module}/dashboards/drift-detection.json")
  }
}

# Prometheus правила для записи метрик
resource "kubernetes_config_map" "prometheus_rules" {
  metadata {
    name      = "prometheus-rules"
    namespace = "monitoring"
  }

  data = {
    "recording-rules.yml" = file("${path.module}/rules/recording-rules.yaml")
  }
}

# Service для доступа к Grafana
resource "kubernetes_service" "grafana_external" {
  metadata {
    name      = "grafana-external"
    namespace = "monitoring"
    annotations = {
      "yandex.cloud/load-balancer-type" = "external"
    }
  }

  spec {
    selector = {
      "app.kubernetes.io/name" = "grafana"
    }

    port {
      name        = "http"
      port        = 80
      target_port = 3000
    }

    type = "LoadBalancer"
  }
}

# Ingress для мониторинга (если нужен)
resource "kubernetes_ingress_v1" "monitoring_ingress" {
  count = var.enable_ingress ? 1 : 0

  metadata {
    name      = "monitoring-ingress"
    namespace = "monitoring"
    annotations = {
      "cert-manager.io/cluster-issuer" = "letsencrypt-prod"
      "nginx.ingress.kubernetes.io/ssl-redirect" = "true"
    }
  }

  spec {
    ingress_class_name = "nginx"

    tls {
      hosts       = ["monitoring.${var.domain}"]
      secret_name = "monitoring-tls"
    }

    rule {
      host = "monitoring.${var.domain}"
      http {
        path {
          path = "/"
          backend {
            service {
              name = "kube-prometheus-stack-grafana"
              port {
                number = 80
              }
            }
          }
        }
      }
    }
  }
}

# Генерация пароля для Grafana
resource "random_password" "grafana_password" {
  length  = 16
  special = true
}

# Выводы
output "grafana_url" {
  value       = "http://${kubernetes_service.grafana_external.status.0.load_balancer.0.ingress.0.ip}"
  description = "Grafana URL"
}

output "prometheus_url" {
  value       = "http://${helm_release.kube_prometheus_stack.name}-prometheus.${helm_release.kube_prometheus_stack.namespace}.svc.cluster.local:9090"
  description = "Prometheus internal URL"
}

output "alertmanager_url" {
  value       = "http://${helm_release.kube_prometheus_stack.name}-alertmanager.${helm_release.kube_prometheus_stack.namespace}.svc.cluster.local:9093"
  description = "Alertmanager internal URL"
}

output "loki_url" {
  value       = "http://${helm_release.loki.name}.${helm_release.loki.namespace}.svc.cluster.local:3100"
  description = "Loki internal URL"
}

output "jaeger_url" {
  value       = "http://${helm_release.jaeger.name}-query.${helm_release.jaeger.namespace}.svc.cluster.local:16686"
  description = "Jaeger Query UI internal URL"
}

output "monitoring_credentials" {
  value = {
    grafana = {
      username = "admin"
      password = random_password.grafana_password.result
    }
  }
  description = "Monitoring system credentials"
  sensitive   = true
}