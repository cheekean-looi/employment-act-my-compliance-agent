#!/usr/bin/env bash
# Install Docker Engine + NVIDIA Container Toolkit on Ubuntu
# Usage: sudo ./scripts/install_docker_nvidia.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run with sudo/root." >&2
  echo "Try: sudo $0" >&2
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "/etc/os-release not found; unsupported OS." >&2
  exit 1
fi

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This installer targets Ubuntu. Detected: ${ID:-unknown}" >&2
  exit 1
fi

echo "==> Installing Docker Engine and plugins"
apt-get update -y
apt-get remove -y docker docker-engine docker.io containerd runc || true
apt-get install -y ca-certificates curl gnupg lsb-release
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker

echo "==> Installing NVIDIA Container Toolkit"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "==> Post-install steps"
if id -nG "${SUDO_USER:-$USER}" | grep -qw docker; then
  echo "User ${SUDO_USER:-$USER} already in docker group"
else
  usermod -aG docker "${SUDO_USER:-$USER}"
  echo "Added ${SUDO_USER:-$USER} to docker group. You must re-login or run 'newgrp docker' to use docker without sudo."
fi

echo "==> Done. Verify with:"
echo "  ./scripts/check_gpu_docker.sh"

