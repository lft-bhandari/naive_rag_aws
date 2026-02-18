#!/usr/bin/env bash
# =============================================================================
# provision_ec2.sh – Bootstrap an EC2 instance for the RAG Microservices stack
#
# Usage:
#   chmod +x provision_ec2.sh
#   ./provision_ec2.sh [--repo <git-url>] [--branch <branch>]
#
# Tested on: Ubuntu 22.04 LTS (ami-0c7217cdde317cfec in us-east-1)
# Recommended instance: t3.xlarge (4 vCPU / 16 GB) or g4dn.xlarge for GPU
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/lft-bhandari/naive_rag_aws}"
BRANCH="${BRANCH:-main}"
APP_DIR="/opt/rag-microservices"
COMPOSE_VERSION="v2.29.7"
LOG_FILE="/var/log/rag-provision.log"

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "$LOG_FILE"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "$LOG_FILE"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"; exit 1; }

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)   REPO_URL="$2"; shift 2 ;;
        --branch) BRANCH="$2";   shift 2 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

# ── Must be root ───────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || error "Run this script as root (sudo ./provision_ec2.sh)"

info "=== RAG Microservices EC2 Provisioning ==="
info "Repo:   $REPO_URL"
info "Branch: $BRANCH"
info "Dir:    $APP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# ── 1. System packages ─────────────────────────────────────────────────────────
info "Updating system packages…"
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    curl wget git unzip jq \
    ca-certificates gnupg lsb-release \
    htop tmux tree net-tools \
    awscli

apt-get install -y python3-pip
pip3 install awscli --break-system-packages

# ── 2. Docker Engine ───────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    info "Installing Docker Engine…"
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io
    systemctl enable --now docker
    usermod -aG docker ubuntu
    info "Docker installed: $(docker --version)"
else
    info "Docker already installed: $(docker --version)"
fi

# ── 3. Docker Compose v2 ───────────────────────────────────────────────────────
if ! docker compose version &>/dev/null 2>&1; then
    info "Installing Docker Compose ${COMPOSE_VERSION}…"
    ARCH=$(uname -m)
    [[ "$ARCH" == "x86_64" ]] && ARCH="x86_64"
    [[ "$ARCH" == "aarch64" ]] && ARCH="aarch64"
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -fsSL \
        "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-${ARCH}" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    info "Docker Compose installed: $(docker compose version)"
else
    info "Docker Compose already installed: $(docker compose version)"
fi

# ── 4. Clone / update the application repo ────────────────────────────────────
if [[ -d "$APP_DIR/.git" ]]; then
    info "Updating existing repo in $APP_DIR…"
    git -C "$APP_DIR" fetch origin
    git -C "$APP_DIR" reset --hard "origin/$BRANCH"
else
    info "Cloning repo into $APP_DIR…"
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$APP_DIR"
fi
chown -R ubuntu:ubuntu "$APP_DIR"

# ── 5. Environment file ────────────────────────────────────────────────────────
ENV_FILE="$APP_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    info "Creating default .env file…"
    cat > "$ENV_FILE" <<'ENVEOF'
# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=rag_documents

# ── Models ────────────────────────────────────────────────────────────────────
EMBED_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct

# ── RAG parameters ────────────────────────────────────────────────────────────
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5
MAX_NEW_TOKENS=512

# ── Frontend ──────────────────────────────────────────────────────────────────
API_BASE_URL=http://backend:8000
ENVEOF
    chown ubuntu:ubuntu "$ENV_FILE"
    info ".env created at $ENV_FILE (edit before first start if needed)"
else
    info ".env already exists – skipping."
fi

# ── 6. UFW Firewall ────────────────────────────────────────────────────────────
info "Configuring UFW firewall…"
ufw --force enable
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8000/tcp   # direct API access (restrict in prod)
ufw allow 8501/tcp   # direct UI access  (restrict in prod)
ufw reload
info "Firewall rules applied."

# ── 7. CloudWatch Logs agent ───────────────────────────────────────────────────
if ! command -v amazon-cloudwatch-agent &>/dev/null; then
    info "Installing CloudWatch Logs agent…"
    ARCH=$(uname -m)
    [[ "$ARCH" == "x86_64" ]] && DEB_ARCH="amd64"
    [[ "$ARCH" == "aarch64" ]] && DEB_ARCH="arm64"
    wget -q \
        "https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/${DEB_ARCH}/latest/amazon-cloudwatch-agent.deb" \
        -O /tmp/cwa.deb
    dpkg -i /tmp/cwa.deb
    rm /tmp/cwa.deb

    # Minimal config – ships Docker container logs to CloudWatch
    cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'CWEOF'
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/lib/docker/containers/**/*-json.log",
            "log_group_name": "/rag-microservices/docker",
            "log_stream_name": "{instance_id}/{container_name}",
            "timezone": "UTC",
            "timestamp_format": "%Y-%m-%dT%H:%M:%S"
          },
          {
            "file_path": "/var/log/rag-provision.log",
            "log_group_name": "/rag-microservices/provision",
            "log_stream_name": "{instance_id}",
            "timezone": "UTC"
          }
        ]
      }
    }
  }
}
CWEOF
    systemctl enable amazon-cloudwatch-agent
    systemctl start amazon-cloudwatch-agent
    info "CloudWatch agent installed and started."
else
    info "CloudWatch agent already installed."
fi

# ── 8. Systemd service for auto-start ─────────────────────────────────────────
info "Creating systemd service for RAG stack…"
cat > /etc/systemd/system/rag-stack.service <<SVCEOF
[Unit]
Description=RAG Microservices Docker Compose Stack
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${APP_DIR}
ExecStart=/usr/local/lib/docker/cli-plugins/docker-compose up -d --remove-orphans
ExecStop=/usr/local/lib/docker/cli-plugins/docker-compose down
TimeoutStartSec=300
User=ubuntu
Group=ubuntu

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable rag-stack.service
info "systemd service 'rag-stack' enabled (starts on boot)."

# ── 9. Build & start the stack ────────────────────────────────────────────────
info "Building Docker images (this may take several minutes)…"
cd "$APP_DIR"
docker compose build --no-cache 2>&1 | tee -a "$LOG_FILE"

info "Starting all services…"
docker compose up -d 2>&1 | tee -a "$LOG_FILE"

# ── 10. Summary ───────────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4 || echo "unknown")

info ""
info "════════════════════════════════════════════════════════"
info " Provisioning complete! 🎉"
info "════════════════════════════════════════════════════════"
info " Public IP  : $PUBLIC_IP"
info " UI (Nginx) : http://$PUBLIC_IP/"
info " UI (direct): http://$PUBLIC_IP:8501/"
info " API        : http://$PUBLIC_IP:8000/docs"
info " Qdrant     : http://$PUBLIC_IP:6333/dashboard"
info " Log file   : $LOG_FILE"
info ""
info " To tail logs:  docker compose -f $APP_DIR/docker-compose.yml logs -f"
info " To stop stack: sudo systemctl stop rag-stack"
info "════════════════════════════════════════════════════════"