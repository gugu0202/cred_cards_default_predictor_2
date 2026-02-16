# S3-совместимое Object Storage (Yandex Cloud)
resource "yandex_storage_bucket" "mlflow_artifacts" {
  bucket     = var.mlflow_bucket_name
  access_key = yandex_iam_service_account_static_access_key.sa_static_key.access_key
  secret_key = yandex_iam_service_account_static_access_key.sa_static_key.secret_key

  anonymous_access_flags {
    read = false
    list = false
  }

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = var.kms_key_id
        sse_algorithm     = "aws:kms"
      }
    }
  }

  lifecycle_rule {
    id      = "mlflow-artifacts-rotation"
    enabled = true

    expiration {
      days = 365
    }

    noncurrent_version_expiration {
      days = 30
    }

    abort_incomplete_multipart_upload_days = 7
  }

  tags = {
    environment = var.environment
    purpose     = "mlflow-artifacts"
    managed-by  = "terraform"
  }
}

resource "yandex_storage_bucket" "dvc_data" {
  bucket     = var.dvc_bucket_name
  access_key = yandex_iam_service_account_static_access_key.sa_static_key.access_key
  secret_key = yandex_iam_service_account_static_access_key.sa_static_key.secret_key

  anonymous_access_flags {
    read = false
    list = false
  }

  versioning {
    enabled = true
  }

  lifecycle_rule {
    id      = "dvc-data-retention"
    enabled = true

    transition {
      days          = 30
      storage_class = "COLD"
    }

    expiration {
      days = 730 # 2 года
    }
  }

  tags = {
    environment = var.environment
    purpose     = "dvc-data"
    managed-by  = "terraform"
  }
}

resource "yandex_storage_bucket" "models_registry" {
  bucket     = var.models_bucket_name
  access_key = yandex_iam_service_account_static_access_key.sa_static_key.access_key
  secret_key = yandex_iam_service_account_static_access_key.sa_static_key.secret_key

  anonymous_access_flags {
    read = false
    list = false
  }

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = var.kms_key_id
        sse_algorithm     = "aws:kms"
      }
    }
  }

  lifecycle_rule {
    id      = "models-cleanup"
    enabled = true

    expiration {
      days = 180
    }

    noncurrent_version_expiration {
      days = 90
    }
  }

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }

  tags = {
    environment = var.environment
    purpose     = "models-registry"
    managed-by  = "terraform"
  }
}

# Service Account для доступа к хранилищу
resource "yandex_iam_service_account" "storage_sa" {
  name        = "storage-sa-${var.environment}"
  description = "Service account for storage access"
}

resource "yandex_resourcemanager_folder_iam_binding" "storage_admin" {
  folder_id = var.folder_id
  role      = "storage.admin"
  members   = [
    "serviceAccount:${yandex_iam_service_account.storage_sa.id}"
  ]
}

resource "yandex_iam_service_account_static_access_key" "sa_static_key" {
  service_account_id = yandex_iam_service_account.storage_sa.id
  description        = "Static access key for storage"
}

# Bucket для логов
resource "yandex_storage_bucket" "logs" {
  bucket     = "logs-${var.environment}-${random_id.bucket_suffix.hex}"
  access_key = yandex_iam_service_account_static_access_key.sa_static_key.access_key
  secret_key = yandex_iam_service_account_static_access_key.sa_static_key.secret_key

  lifecycle_rule {
    id      = "logs-lifecycle"
    enabled = true

    transition {
      days          = 30
      storage_class = "COLD"
    }

    expiration {
      days = 365
    }
  }

  tags = {
    environment = var.environment
    purpose     = "logs"
    managed-by  = "terraform"
  }
}

# Random suffix для уникальности имен bucket
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Managed PostgreSQL для метаданных MLflow и Airflow
resource "yandex_mdb_postgresql_cluster" "metadata_db" {
  name        = "mlflow-metadata-${var.environment}"
  description = "PostgreSQL cluster for MLflow and Airflow metadata"
  environment = "PRODUCTION"
  network_id  = var.network_id

  config {
    version = 15
    resources {
      resource_preset_id = "s2.micro"
      disk_type_id       = "network-ssd"
      disk_size          = 20
    }

    postgresql_config = {
      max_connections                   = 200
      enable_parallel_hash              = true
      vacuum_cleanup_index_scale_factor = 0.1
      autovacuum_vacuum_scale_factor    = 0.01
      shared_preload_libraries          = "pg_stat_statements"
    }
  }

  maintenance_window {
    type = "WEEKLY"
    day  = "SAT"
    hour = 3
  }

  database {
    name  = "mlflow"
    owner = "mlflow_user"
  }

  database {
    name  = "airflow"
    owner = "airflow_user"
  }

  user {
    name     = "mlflow_user"
    password = random_password.mlflow_db_password.result
    permission {
      database_name = "mlflow"
    }
  }

  user {
    name     = "airflow_user"
    password = random_password.airflow_db_password.result
    permission {
      database_name = "airflow"
    }
  }

  host {
    zone      = var.zone
    subnet_id = var.subnet_id
  }

  security_group_ids = [var.security_group_ids.database]

  deletion_protection = var.environment == "production"

  labels = {
    environment = var.environment
    purpose     = "metadata-database"
    managed-by  = "terraform"
  }
}

# Генерация паролей для БД
resource "random_password" "mlflow_db_password" {
  length  = 16
  special = false
}

resource "random_password" "airflow_db_password" {
  length  = 16
  special = false
}

# Secret для хранения паролей БД в Kubernetes
resource "kubernetes_secret" "database_passwords" {
  metadata {
    name      = "database-passwords"
    namespace = "default"
  }

  data = {
    mlflow-db-password   = random_password.mlflow_db_password.result
    airflow-db-password  = random_password.airflow_db_password.result
  }

  type = "Opaque"
}

# Выводы
output "bucket_urls" {
  value = {
    mlflow_artifacts = "https://${yandex_storage_bucket.mlflow_artifacts.bucket}.storage.yandexcloud.net"
    dvc_data         = "https://${yandex_storage_bucket.dvc_data.bucket}.storage.yandexcloud.net"
    models_registry  = "https://${yandex_storage_bucket.models_registry.bucket}.storage.yandexcloud.net"
    logs             = "https://${yandex_storage_bucket.logs.bucket}.storage.yandexcloud.net"
  }
  description = "Object Storage bucket URLs"
}

output "storage_access_key" {
  value       = yandex_iam_service_account_static_access_key.sa_static_key.access_key
  description = "Storage access key"
  sensitive   = true
}

output "storage_secret_key" {
  value       = yandex_iam_service_account_static_access_key.sa_static_key.secret_key
  description = "Storage secret key"
  sensitive   = true
}

output "mlflow_tracking_uri" {
  value = "postgresql://mlflow_user:${random_password.mlflow_db_password.result}@${yandex_mdb_postgresql_cluster.metadata_db.host[0].fqdn}:6432/mlflow"
  description = "MLflow tracking database URI"
  sensitive   = true
}

output "dvc_remote_url" {
  value = "s3://${yandex_storage_bucket.dvc_data.bucket}/"
  description = "DVC remote storage URL"
}

output "postgresql_host" {
  value       = yandex_mdb_postgresql_cluster.metadata_db.host[0].fqdn
  description = "PostgreSQL host FQDN"
}

output "postgresql_port" {
  value       = 6432
  description = "PostgreSQL port"
}

output "database_connections" {
  value = {
    mlflow  = "postgresql://mlflow_user:${random_password.mlflow_db_password.result}@${yandex_mdb_postgresql_cluster.metadata_db.host[0].fqdn}:6432/mlflow"
    airflow = "postgresql://airflow_user:${random_password.airflow_db_password.result}@${yandex_mdb_postgresql_cluster.metadata_db.host[0].fqdn}:6432/airflow"
  }
  description = "Database connection strings"
  sensitive   = true
}