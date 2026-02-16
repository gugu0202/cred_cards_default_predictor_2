resource "yandex_vpc_network" "vpc" {
  name        = "credit-scoring-vpc-${var.environment}"
  description = "VPC for Credit Scoring MLOps system"
  labels = {
    environment = var.environment
    managed-by  = "terraform"
    project     = "credit-scoring"
  }
}

resource "yandex_vpc_subnet" "subnet" {
  name           = "credit-scoring-subnet-${var.environment}"
  description    = "Main subnet for Credit Scoring MLOps"
  zone           = var.zone
  network_id     = yandex_vpc_network.vpc.id
  v4_cidr_blocks = [var.vpc_cidr]
  
  route_table_id = yandex_vpc_route_table.nat_route_table.id
  
  labels = {
    environment = var.environment
    managed-by  = "terraform"
    purpose     = "kubernetes"
  }
}

# NAT Gateway для исходящего трафика
resource "yandex_vpc_gateway" "nat_gateway" {
  name = "nat-gateway-${var.environment}"
  shared_egress_gateway {}
}

resource "yandex_vpc_route_table" "nat_route_table" {
  name       = "nat-route-table-${var.environment}"
  network_id = yandex_vpc_network.vpc.id

  static_route {
    destination_prefix = "0.0.0.0/0"
    gateway_id         = yandex_vpc_gateway.nat_gateway.id
  }
}

# Secondary subnets для высокой доступности
resource "yandex_vpc_subnet" "secondary_subnets" {
  for_each = {
    b = "${var.zone}b"
    c = "${var.zone}c"
  }

  name           = "credit-scoring-subnet-${var.environment}-${each.key}"
  description    = "Secondary subnet for HA in ${each.value}"
  zone           = each.value
  network_id     = yandex_vpc_network.vpc.id
  v4_cidr_blocks = [cidrsubnet(var.vpc_cidr, 8, index(keys({ b = "b", c = "c" }), each.key) + 1)]
  
  labels = {
    environment = var.environment
    managed-by  = "terraform"
    purpose     = "high-availability"
  }
}

# VPC Peering для подключения к другим сетям (если нужно)
resource "yandex_vpc_address" "public_ip" {
  name = "nat-public-ip-${var.environment}"
  
  external_ipv4_address {
    zone_id = var.zone
  }
}

# Выводы
output "vpc_id" {
  value       = yandex_vpc_network.vpc.id
  description = "VPC ID"
}

output "subnet_id" {
  value       = yandex_vpc_subnet.subnet.id
  description = "Primary subnet ID"
}

output "subnet_cidr" {
  value       = var.vpc_cidr
  description = "Primary subnet CIDR"
}

output "secondary_subnets" {
  value = {
    for k, subnet in yandex_vpc_subnet.secondary_subnets :
    k => {
      id   = subnet.id
      zone = subnet.zone
      cidr = subnet.v4_cidr_blocks[0]
    }
  }
  description = "Secondary subnets for HA"
}

output "nat_gateway_id" {
  value       = yandex_vpc_gateway.nat_gateway.id
  description = "NAT Gateway ID"
}