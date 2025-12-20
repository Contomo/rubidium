#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

KLIPPER_PATH="${HOME}/klipper"
INSTALL_PATH="${HOME}/rubidium"
REPO_URL="https://github.com/contomo/rubidium.git"

preflight_checks() {
    if [ "${EUID}" -eq 0 ]; then
        echo "[PRE-CHECK] This script must not be run as root!"
        exit 1
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "[ERROR] git not found. Install git first."
        exit 1
    fi

    if ! command -v systemctl >/dev/null 2>&1; then
        echo "[ERROR] systemctl not found. This installer expects a systemd-based setup."
        exit 1
    fi

    if sudo systemctl is-enabled --quiet klipper.service 2>/dev/null || sudo systemctl status klipper.service >/dev/null 2>&1; then
        echo "[PRE-CHECK] Klipper service found! Continuing..."
        echo
    else
        echo "[ERROR] klipper.service not found. Install Klipper first."
        exit 1
    fi
}

check_download() {
    local installdirname installbasename
    installdirname="$(dirname "${INSTALL_PATH}")"
    installbasename="$(basename "${INSTALL_PATH}")"

    if [ ! -d "${INSTALL_PATH}" ]; then
        echo "[DOWNLOAD] Downloading repository..."
        git -C "${installdirname}" clone "${REPO_URL}" "${installbasename}"
        chmod +x "${INSTALL_PATH}/install.sh" || true
        echo "[DOWNLOAD] Download complete!"
        echo
    else
        echo "[DOWNLOAD] Repository already found locally. Continuing..."
        echo
    fi
}

link_extension() {
    echo "[INSTALL] Linking Rubidium package into Klipper extras..."

    local src="${INSTALL_PATH}/rubidium"
    local dst="${KLIPPER_PATH}/klippy/extras/rubidium"

    if [ ! -d "${src}" ]; then
        echo "[ERROR] Rubidium package dir not found at: ${src}"
        exit 1
    fi
    if [ ! -f "${src}/__init__.py" ]; then
        echo "[ERROR] ${src} exists but has no __init__.py (not a Python package)"
        exit 1
    fi
    if [ ! -d "${KLIPPER_PATH}/klippy/extras" ]; then
        echo "[ERROR] Klipper extras dir not found at: ${KLIPPER_PATH}/klippy/extras"
        exit 1
    fi

    ln -sfn "${src}" "${dst}"
    echo "  → rubidium (package)"
    echo
}


restart_klipper() {
    echo "[POST-INSTALL] Restarting Klipper..."
    sudo systemctl restart klipper.service
}

printf "\n======================================\n"
echo   "     - Rubidium install script -      "
printf "======================================\n\n"

preflight_checks
check_download
link_extension
restart_klipper
