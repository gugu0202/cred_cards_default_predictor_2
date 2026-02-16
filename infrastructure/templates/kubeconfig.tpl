apiVersion: v1
kind: Config
clusters:
- name: ${cluster_name}
  cluster:
    server: ${server}
    certificate-authority-data: ${cluster_ca_certificate}
contexts:
- name: ${cluster_name}
  context:
    cluster: ${cluster_name}
    namespace: ${namespace}
    user: ${service_account}
current-context: ${cluster_name}
users:
- name: ${service_account}
  user:
    token: ${service_account_token}