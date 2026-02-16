terraform {
  required_version = ">= 1.3.0"

  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.89.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.20.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.9.0"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.14.0"
    }
  }

  backend "s3" {
    endpoint   = "storage.yandexcloud.net"
    bucket     = "credit-scoring-tfstate-${var.environment}"
    key        = "terraform.tfstate"
    region     = "ru-central1"
    access_key = var.yc_storage_access_key
    secret_key = var.yc_storage_secret_key

    skip_region_validation      = true
    skip_credentials_validation = true
  }
}

provider "yandex" {
  token     = var.yc_token
  cloud_id  = var.yc_cloud_id
  folder_id = var.yc_folder_id
  zone      = var.zone
}

provider "kubernetes" {
  host                   = yandex_kubernetes_cluster.credit_scoring_cluster.master[0].external_v4_endpoint
  cluster_ca_certificate = base64decode(yandex_kubernetes_cluster.credit_scoring_cluster.master[0].cluster_ca_certificate)
  token                  = data.yandex_client_config.client.iam_token
}

provider "helm" {
  kubernetes {
    host                   = yandex_kubernetes_cluster.credit_scoring_cluster.master[0].external_v4_endpoint
    cluster_ca_certificate = base64decode(yandex_kubernetes_cluster.credit_scoring_cluster.master[0].cluster_ca_certificate)
    token                  = data.yandex_client_config.client.iam_token
  }
}

provider "kubectl" {
  host                   = yandex_kubernetes_cluster.credit_scoring_cluster.master[0].external_v4_endpoint
  cluster_ca_certificate = base64decode(yandex_kubernetes_cluster.credit_scoring_cluster.master[0].cluster_ca_certificate)
  token                  = data.yandex_client_config.client.iam_token
  load_config_file       = false
}

# Получение IAM токена
data "yandex_client_config" "client" {}

# Создание Service Account для Terraform
resource "yandex_iam_service_account" "terraform_sa" {
  name        = "terraform-sa-${var.environment}"
  description = "Service account for Terraform"
}

# Роли для Service Account
resource "yandex_resourcemanager_folder_iam_binding" "editor" {
  folder_id = var.yc_folder_id
  role      = "editor"
  members   = [
    "serviceAccount:${yandex_iam_service_account.terraform_sa.id}"
  ]
}

resource "yandex_resourcemanager_folder_iam_binding" "k8s_admin" {
  folder_id = var.yc_folder_id
  role      = "k8s.admin"
  members   = [
    "serviceAccount:${yandex_iam_service_account.terraform_sa.id}"
  ]
}

resource "yandex_resourcemanager_folder_iam_binding" "vpc_admin" {
  folder_id = var.yc_folder_id
  role      = "vpc.admin"
  members   = [
    "serviceAccount:${yandex_iam_service_account.terraform_sa.id}"
  ]
}

# Ключи доступа для Service Account
resource "yandex_iam_service_account_static_access_key" "terraform_sa_key" {
  service_account_id = yandex_iam_service_account.terraform_sa.id
  description        = "Static access key for Terraform"
}

# KMS ключ для шифрования секретов
resource "yandex_kms_symmetric_key" "kms_key" {
  name              = "kms-key-${var.environment}"
  description       = "KMS key for Kubernetes secrets"
  default_algorithm = "AES_128"
  rotation_period   = "8760h" # 1 год
}

# Сеть и подсети
module "network" {
  source = "./modules/network"

  environment = var.environment
  zone        = var.zone
  vpc_cidr    = var.vpc_cidr
}

# Kubernetes кластер
module "kubernetes" {
  source = "./modules/kubernetes"

  environment     = var.environment
  zone           = var.zone
  network_id     = module.network.vpc_id
  subnet_id      = module.network.subnet_id
  kms_key_id     = yandex_kms_symmetric_key.kms_key.id
  service_account_id = yandex_iam_service_account.terraform_sa.id

  node_groups = var.node_groups
}

# Хранилище для данных
module "storage" {
  source = "./modules/storage"

  environment = var.environment
  zone        = var.zone

  # MLflow artifacts storage
  mlflow_bucket_name = "mlflow-artifacts-${var.environment}"
  
  # DVC data storage
  dvc_bucket_name = "dvc-data-${var.environment}"
  
  # Model registry storage
  models_bucket_name = "models-registry-${var.environment}"
}

# Мониторинг и логирование
module "monitoring" {
  source = "./modules/monitoring"

  environment = var.environment
  zone        = var.zone
  
  cluster_id  = module.kubernetes.cluster_id
  network_id  = module.network.vpc_id
  
  # Настройки алертинга
  slack_webhook_url = var.slack_webhook_url
  alert_email       = var.alert_email
}

# Namespace для приложения
resource "kubernetes_namespace" "credit_scoring" {
  metadata {
    name = "credit-scoring-${var.environment}"
    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }
  }
}

# Конфигурация для приложения
resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "credit-scoring-config"
    namespace = kubernetes_namespace.credit_scoring.metadata[0].name
  }

  data = {
    "environment"    = var.environment
    "model_version"  = var.model_version
    "batch_size"     = var.batch_size
    "log_level"      = var.log_level
    "max_workers"    = var.max_workers
  }
}

# Secrets для приложения
resource "kubernetes_secret" "app_secrets" {
  metadata {
    name      = "credit-scoring-secrets"
    namespace = kubernetes_namespace.credit_scoring.metadata[0].name
  }

  data = {
    "mlflow_tracking_uri" = module.storage.mlflow_tracking_uri
    "dvc_remote_url"      = module.storage.dvc_remote_url
    "database_url"        = base64encode(var.database_url)
    "api_key"             = base64encode(var.api_key)
  }

  type = "Opaque"
}

# Network Policies для безопасности
resource "kubernetes_network_policy" "allow_namespace" {
  metadata {
    name      = "allow-namespace-traffic"
    namespace = kubernetes_namespace.credit_scoring.metadata[0].name
  }

  spec {
    pod_selector {}
    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = kubernetes_namespace.credit_scoring.metadata[0].name
          }
        }
      }
    }

    egress {
      to {
        namespace_selector {
          match_labels = {
            name = kubernetes_namespace.credit_scoring.metadata[0].name
          }
        }
      }
    }

    policy_types = ["Ingress", "Egress"]
  }
}

resource "kubernetes_network_policy" "allow_monitoring" {
  metadata {
    name      = "allow-monitoring"
    namespace = kubernetes_namespace.credit_scoring.metadata[0].name
  }

  spec {
    pod_selector {}
    
    ingress {
      ports {
        port     = "9090"
        protocol = "TCP"
      }
      ports {
        port     = "9100"
        protocol = "TCP"
      }
      
      from {
        namespace_selector {
          match_labels = {
            name = "monitoring"
          }
        }
      }
    }

    policy_types = ["Ingress"]
  }
}

# Security Groups в Yandex Cloud
resource "yandex_vpc_security_group" "k8s_master_sg" {
  name        = "k8s-master-sg-${var.environment}"
  description = "Security group for Kubernetes master nodes"
  network_id  = module.network.vpc_id

  ingress {
    protocol       = "TCP"
    description    = "Kubernetes API"
    port           = 6443
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    protocol       = "TCP"
    description    = "SSH access"
    port           = 22
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol       = "ANY"
    description    = "Full egress"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "yandex_vpc_security_group" "k8s_nodes_sg" {
  name        = "k8s-nodes-sg-${var.environment}"
  description = "Security group for Kubernetes worker nodes"
  network_id  = module.network.vpc_id

  ingress {
    protocol          = "ANY"
    description       = "Internal cluster communication"
    predefined_target = "self_security_group"
  }

  ingress {
    protocol       = "TCP"
    description    = "NodePort services"
    port           = 30000-32767
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    protocol       = "ICMP"
    description    = "ICMP"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol       = "ANY"
    description    = "Full egress"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}

# Выводы
output "cluster_endpoint" {
  value       = yandex_kubernetes_cluster.credit_scoring_cluster.master[0].external_v4_endpoint
  description = "Kubernetes API endpoint"
}

output "cluster_ca_certificate" {
  value       = yandex_kubernetes_cluster.credit_scoring_cluster.master[0].cluster_ca_certificate
  description = "Kubernetes cluster CA certificate"
  sensitive   = true
}

output "storage_buckets" {
  value       = module.storage.bucket_urls
  description = "Object Storage bucket URLs"
}

output "kubeconfig" {
  value = templatefile("${path.module}/templates/kubeconfig.tpl", {
    server                     = yandex_kubernetes_cluster.credit_scoring_cluster.master[0].external_v4_endpoint
    cluster_ca_certificate     = base64decode(yandex_kubernetes_cluster.credit_scoring_cluster.master[0].cluster_ca_certificate)
    service_account_namespace  = "kube-system"
    service_account_name       = "terraform-admin"
    service_account_token      = data.yandex_client_config.client.iam_token
  })
  description = "Kubeconfig file content"
  sensitive   = true
}

output "terraform_sa_key" {
  value       = yandex_iam_service_account_static_access_key.terraform_sa_key.secret_key
  description = "Terraform service account secret key"
  sensitive   = true
}

output "infrastructure_summary" {
  value = {
    environment          = var.environment
    cluster_name         = module.kubernetes.cluster_name
    vpc_id              = module.network.vpc_id
    subnet_cidr         = module.network.subnet_cidr
    node_groups         = module.kubernetes.node_group_names
    storage_buckets     = keys(module.storage.bucket_urls)
    kms_key_id         = yandex_kms_symmetric_key.kms_key.id
    namespace          = kubernetes_namespace.credit_scoring.metadata[0].name
    security_groups    = [yandex_vpc_security_group.k8s_master_sg.name, yandex_vpc_security_group.k8s_nodes_sg.name]
  }
  description = "Infrastructure deployment summary"
}