# Kubernetes
output "kubernetes_cluster_endpoint" {
  value       = module.kubernetes.cluster_endpoint
  description = "Kubernetes API Server endpoint"
}

output "kubernetes_cluster_name" {
  value       = module.kubernetes.cluster_name
  description = "Kubernetes cluster name"
}

output "kubernetes_cluster_ca_certificate" {
  value       = module.kubernetes.cluster_ca_certificate
  description = "Kubernetes cluster CA certificate"
  sensitive   = true
}

output "kubernetes_node_groups" {
  value       = module.kubernetes.node_group_names
  description = "List of Kubernetes node groups"
}

# Сеть
output "vpc_id" {
  value       = module.network.vpc_id
  description = "VPC ID"
}

output "subnet_id" {
  value       = module.network.subnet_id
  description = "Primary subnet ID"
}

output "subnet_cidr" {
  value       = module.network.subnet_cidr
  description = "Primary subnet CIDR"
}

# Хранилище
output "storage_bucket_urls" {
  value       = module.storage.bucket_urls
  description = "Object Storage bucket URLs"
}

output "mlflow_tracking_uri" {
  value       = module.storage.mlflow_tracking_uri
  description = "MLflow tracking URI"
  sensitive   = true
}

output "dvc_remote_url" {
  value       = module.storage.dvc_remote_url
  description = "DVC remote storage URL"
}

output "postgresql_connection" {
  value = {
    host = module.storage.postgresql_host
    port = module.storage.postgresql_port
  }
  description = "PostgreSQL connection details"
}

# Мониторинг
output "monitoring_endpoints" {
  value = {
    grafana       = module.monitoring.grafana_url
    prometheus    = module.monitoring.prometheus_url
    alertmanager  = module.monitoring.alertmanager_url
    loki          = module.monitoring.loki_url
    jaeger        = module.monitoring.jaeger_url
  }
  description = "Monitoring system endpoints"
}

output "grafana_credentials" {
  value       = module.monitoring.monitoring_credentials.grafana
  description = "Grafana admin credentials"
  sensitive   = true
}

# Безопасность
output "kms_key_id" {
  value       = yandex_kms_symmetric_key.kms_key.id
  description = "KMS key ID for encryption"
}

output "security_group_ids" {
  value = {
    master   = yandex_vpc_security_group.k8s_master_sg.id
    nodes    = yandex_vpc_security_group.k8s_nodes_sg.id
  }
  description = "Security group IDs"
}

# Service Accounts
output "service_accounts" {
  value = {
    terraform = yandex_iam_service_account.terraform_sa.id
    cluster   = module.kubernetes.service_accounts.cluster
    nodes     = module.kubernetes.service_accounts.nodes
    storage   = module.storage.storage_access_key
  }
  description = "Service account IDs and keys"
}

# Namespaces
output "kubernetes_namespaces" {
  value = [
    kubernetes_namespace.credit_scoring.metadata[0].name,
    kubernetes_namespace.monitoring.metadata[0].name,
    kubernetes_namespace.logging.metadata[0].name,
    kubernetes_namespace.ml_training.metadata[0].name
  ]
  description = "Created Kubernetes namespaces"
}

# Конфигурация приложения
output "app_config" {
  value = {
    namespace   = kubernetes_namespace.credit_scoring.metadata[0].name
    config_map  = kubernetes_config_map.app_config.metadata[0].name
    secret      = kubernetes_secret.app_secrets.metadata[0].name
    environment = var.environment
  }
  description = "Application configuration details"
}

# Инструкции
output "deployment_instructions" {
  value = <<-EOT

  🚀 Credit Scoring MLOps Infrastructure Deployed Successfully!

  📋 Next Steps:

  1. Configure kubectl:
     export KUBECONFIG=./kubeconfig.yaml
     echo '${base64encode(local.kubeconfig_content)}' | base64 --decode > kubeconfig.yaml

  2. Verify cluster access:
     kubectl cluster-info
     kubectl get nodes

  3. Deploy the application:
     kubectl apply -f deployment/kubernetes/ -n ${kubernetes_namespace.credit_scoring.metadata[0].name}

  4. Access monitoring:
     Grafana: ${module.monitoring.grafana_url}
     Username: admin
     Password: ${module.monitoring.monitoring_credentials.grafana.password}

  5. Configure CI/CD:
     - Set GitHub Secrets with outputs above
     - Configure kubectl in GitHub Actions

  🔧 Infrastructure Details:
  - Environment: ${var.environment}
  - Cluster: ${module.kubernetes.cluster_name}
  - VPC: ${module.network.vpc_id}
  - Storage: ${join(", ", keys(module.storage.bucket_urls))}

  EOT

  description = "Deployment instructions and next steps"
}

# Локальные вычисления для kubeconfig
locals {
  kubeconfig_content = templatefile("${path.module}/templates/kubeconfig.tpl", {
    server                 = module.kubernetes.cluster_endpoint
    cluster_ca_certificate = module.kubernetes.cluster_ca_certificate
    namespace              = kubernetes_namespace.credit_scoring.metadata[0].name
    service_account_token  = data.yandex_client_config.client.iam_token
  })
}

output "kubeconfig_file" {
  value       = local.kubeconfig_content
  description = "Kubeconfig file content"
  sensitive   = true
}

# Сводка развертывания
output "deployment_summary" {
  value = {
    timestamp           = timestamp()
    environment         = var.environment
    kubernetes_cluster  = module.kubernetes.cluster_name
    vpc_network        = module.network.vpc_id
    object_storage     = keys(module.storage.bucket_urls)
    monitoring         = keys(module.monitoring.monitoring_endpoints)
    security           = {
      kms_key          = yandex_kms_symmetric_key.kms_key.id
      security_groups  = keys(output.security_group_ids.value)
    }
    cost_estimate = {
      monthly_estimate_usd = local.cost_estimate
      resources_count      = local.resource_count
    }
  }
  description = "Complete deployment summary"
}

# Оценка стоимости (примерная)
locals {
  resource_count = 4 + length(var.node_groups) + length(module.storage.bucket_urls)
  cost_estimate = {
    dev       = "~$150/month"
    staging   = "~$300/month"
    production = "~$800/month"
  }[var.environment]
}