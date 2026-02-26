#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Bluesky-Grokker · One-Shot Setup Script
#
#  What this does:
#    1. Detects OS (Linux / macOS / WSL)
#    2. Installs Python 3.11+ if missing
#    3. Installs PostgreSQL + pgvector, Redis (or checks they exist)
#    4. Creates a Python virtual environment
#    5. Installs pip requirements + Playwright browsers
#    6. Creates the database, user, and tables
#    7. Launches an interactive Playwright walkthrough to:
#         • Create a Bluesky account (or log in)
#         • Generate an App Password
#         • Write credentials to .env
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD='\033[1m'
CYAN='\033[96m'
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
RESET='\033[0m'

info()    { echo -e "${CYAN}ℹ  $*${RESET}"; }
success() { echo -e "${GREEN}✔  $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠  $*${RESET}"; }
err()     { echo -e "${RED}✖  $*${RESET}"; }
header()  {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}  $*${RESET}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
    echo ""
}

# ── Resolve project root (where this script lives) ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/bluesky_grokker"
VENV_DIR="$SCRIPT_DIR/venv"

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Detect Operating System
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 1 · Detecting Operating System"

OS="unknown"
DISTRO="unknown"
PKG_MGR=""

case "$(uname -s)" in
    Linux*)
        OS="linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            DISTRO="${ID:-unknown}"
        fi
        # Check for WSL
        if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
            OS="wsl"
            info "Detected: Windows Subsystem for Linux ($DISTRO)"
        else
            info "Detected: Linux ($DISTRO)"
        fi
        # Determine package manager
        if command -v apt-get &>/dev/null; then
            PKG_MGR="apt"
        elif command -v dnf &>/dev/null; then
            PKG_MGR="dnf"
        elif command -v yum &>/dev/null; then
            PKG_MGR="yum"
        elif command -v pacman &>/dev/null; then
            PKG_MGR="pacman"
        elif command -v zypper &>/dev/null; then
            PKG_MGR="zypper"
        fi
        ;;
    Darwin*)
        OS="macos"
        info "Detected: macOS $(sw_vers -productVersion 2>/dev/null || echo '')"
        if command -v brew &>/dev/null; then
            PKG_MGR="brew"
        fi
        ;;
    CYGWIN*|MINGW*|MSYS*)
        OS="windows"
        info "Detected: Windows (Git Bash / MSYS)"
        ;;
    *)
        warn "Unknown OS: $(uname -s). Will attempt best-effort setup."
        ;;
esac

success "OS=$OS  DISTRO=$DISTRO  PKG_MGR=${PKG_MGR:-none detected}"

# ── Helper: install a system package ────────────────────────────────────────
install_pkg() {
    local pkg="$1"
    info "Installing system package: $pkg"
    case "$PKG_MGR" in
        apt)    sudo apt-get update -qq && sudo apt-get install -y -qq "$pkg" ;;
        dnf)    sudo dnf install -y -q "$pkg" ;;
        yum)    sudo yum install -y -q "$pkg" ;;
        pacman) sudo pacman -S --noconfirm "$pkg" ;;
        zypper) sudo zypper install -y "$pkg" ;;
        brew)   brew install "$pkg" ;;
        *)      err "No supported package manager found. Please install '$pkg' manually."; return 1 ;;
    esac
}

# ═══════════════════════════════════════════════════════════════════════════════
#  2. Ensure Python 3.11+
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 2 · Checking Python"

PYTHON=""

# Look for python3.12, python3.11, python3 in order
for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PY_VER=$("$candidate" --version 2>&1 | awk '{print $2}')
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -n "$PYTHON" ]; then
    success "Found $PYTHON ($($PYTHON --version))"
else
    warn "Python 3.11+ not found. Attempting to install…"
    case "$PKG_MGR" in
        apt)
            sudo apt-get update -qq
            # Try the deadsnakes PPA on Ubuntu
            if command -v add-apt-repository &>/dev/null; then
                sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
                sudo apt-get update -qq
            fi
            sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev 2>/dev/null \
                || sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev 2>/dev/null \
                || sudo apt-get install -y -qq python3 python3-venv python3-dev
            ;;
        dnf)
            sudo dnf install -y python3.12 python3.12-devel 2>/dev/null \
                || sudo dnf install -y python3.11 python3.11-devel 2>/dev/null \
                || sudo dnf install -y python3 python3-devel
            ;;
        yum)
            sudo yum install -y python3 python3-devel
            ;;
        pacman)
            sudo pacman -S --noconfirm python
            ;;
        zypper)
            sudo zypper install -y python312 python312-devel 2>/dev/null \
                || sudo zypper install -y python311 python311-devel 2>/dev/null \
                || sudo zypper install -y python3 python3-devel
            ;;
        brew)
            brew install python@3.12 || brew install python@3.11
            ;;
        *)
            err "Cannot auto-install Python. Please install Python 3.11+ manually."
            err "  https://www.python.org/downloads/"
            exit 1
            ;;
    esac

    # Re-detect
    for candidate in python3.12 python3.11 python3 python; do
        if command -v "$candidate" &>/dev/null; then
            PY_VER=$("$candidate" --version 2>&1 | awk '{print $2}')
            PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
            PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
            if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
                PYTHON="$candidate"
                break
            fi
        fi
    done

    if [ -z "$PYTHON" ]; then
        err "Failed to install Python 3.11+. Please install it manually."
        exit 1
    fi
    success "Installed $PYTHON ($($PYTHON --version))"
fi

# Make sure venv module is available
if ! "$PYTHON" -m venv --help &>/dev/null; then
    warn "Python venv module missing. Installing…"
    PY_VER_SHORT=$("$PYTHON" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    case "$PKG_MGR" in
        apt) sudo apt-get install -y -qq "python${PY_VER_SHORT}-venv" 2>/dev/null || true ;;
        *)   warn "Please ensure the python venv module is installed." ;;
    esac
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  3. Install / Check PostgreSQL & Redis
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 3 · Checking PostgreSQL & Redis"

# ── PostgreSQL ───────────────────────────────────────────────────────────────
if command -v psql &>/dev/null; then
    success "PostgreSQL client found: $(psql --version)"
else
    warn "PostgreSQL not found. Attempting to install…"
    case "$PKG_MGR" in
        apt)
            sudo apt-get update -qq
            sudo apt-get install -y -qq postgresql postgresql-contrib libpq-dev
            ;;
        dnf)    sudo dnf install -y postgresql-server postgresql-devel; sudo postgresql-setup --initdb 2>/dev/null || true ;;
        yum)    sudo yum install -y postgresql-server postgresql-devel; sudo postgresql-setup initdb 2>/dev/null || true ;;
        pacman) sudo pacman -S --noconfirm postgresql ;;
        zypper) sudo zypper install -y postgresql-server postgresql-devel ;;
        brew)   brew install postgresql@16 || brew install postgresql ;;
        *)      err "Please install PostgreSQL manually."; exit 1 ;;
    esac
    success "PostgreSQL installed"
fi

# ── pgvector extension ───────────────────────────────────────────────────────
PGVECTOR_INSTALLED=false
if command -v psql &>/dev/null; then
    # Check if pgvector extension files exist
    if psql -U postgres -tc "SELECT 1 FROM pg_available_extensions WHERE name='vector'" 2>/dev/null | grep -q 1; then
        PGVECTOR_INSTALLED=true
        success "pgvector extension is available"
    fi
fi

if [ "$PGVECTOR_INSTALLED" = false ]; then
    warn "pgvector extension not found. Attempting to install…"
    case "$PKG_MGR" in
        apt)
            # Try the official package first
            sudo apt-get install -y -qq postgresql-16-pgvector 2>/dev/null \
                || sudo apt-get install -y -qq postgresql-15-pgvector 2>/dev/null \
                || sudo apt-get install -y -qq postgresql-14-pgvector 2>/dev/null \
                || {
                    # Build from source as fallback
                    info "Building pgvector from source…"
                    sudo apt-get install -y -qq build-essential postgresql-server-dev-all git
                    TMPDIR=$(mktemp -d)
                    git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git "$TMPDIR/pgvector"
                    (cd "$TMPDIR/pgvector" && make && sudo make install)
                    rm -rf "$TMPDIR"
                }
            ;;
        dnf|yum)
            sudo "$PKG_MGR" install -y pgvector 2>/dev/null || {
                info "Building pgvector from source…"
                sudo "$PKG_MGR" install -y gcc make postgresql-devel git
                TMPDIR=$(mktemp -d)
                git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git "$TMPDIR/pgvector"
                (cd "$TMPDIR/pgvector" && make && sudo make install)
                rm -rf "$TMPDIR"
            }
            ;;
        brew)
            brew install pgvector
            ;;
        *)
            warn "Please install pgvector manually: https://github.com/pgvector/pgvector"
            ;;
    esac
fi

# ── Ensure PostgreSQL is running ─────────────────────────────────────────────
if command -v systemctl &>/dev/null && systemctl is-enabled postgresql &>/dev/null 2>&1; then
    if ! systemctl is-active --quiet postgresql; then
        info "Starting PostgreSQL service…"
        sudo systemctl start postgresql
    fi
    success "PostgreSQL service is running"
elif [ "$OS" = "macos" ] && command -v brew &>/dev/null; then
    brew services start postgresql@16 2>/dev/null || brew services start postgresql 2>/dev/null || true
    success "PostgreSQL service started (Homebrew)"
fi

# ── Redis ────────────────────────────────────────────────────────────────────
if command -v redis-cli &>/dev/null; then
    success "Redis client found: $(redis-cli --version)"
else
    warn "Redis not found. Attempting to install…"
    case "$PKG_MGR" in
        apt)    sudo apt-get install -y -qq redis-server ;;
        dnf)    sudo dnf install -y redis ;;
        yum)    sudo yum install -y redis ;;
        pacman) sudo pacman -S --noconfirm redis ;;
        zypper) sudo zypper install -y redis ;;
        brew)   brew install redis ;;
        *)      err "Please install Redis manually."; exit 1 ;;
    esac
    success "Redis installed"
fi

# ── Ensure Redis is running ──────────────────────────────────────────────────
if command -v systemctl &>/dev/null; then
    if systemctl is-enabled redis-server &>/dev/null 2>&1 || systemctl is-enabled redis &>/dev/null 2>&1; then
        sudo systemctl start redis-server 2>/dev/null || sudo systemctl start redis 2>/dev/null || true
    fi
elif [ "$OS" = "macos" ] && command -v brew &>/dev/null; then
    brew services start redis 2>/dev/null || true
fi

if redis-cli ping 2>/dev/null | grep -q PONG; then
    success "Redis is running"
else
    warn "Redis may not be running. Start it manually if needed: redis-server"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  4. Create Virtual Environment
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 4 · Creating Python Virtual Environment"

if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
success "Virtual environment activated ($VENV_DIR)"
info "Python: $(python --version)  pip: $(pip --version | awk '{print $2}')"

# ═══════════════════════════════════════════════════════════════════════════════
#  5. Install Python Dependencies
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 5 · Installing Python Dependencies"

pip install --upgrade pip setuptools wheel -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
success "Core requirements installed"

# Install Playwright
info "Installing Playwright Python package…"
pip install playwright -q
success "Playwright package installed"

info "Installing Playwright browsers (Chromium)…"
python -m playwright install chromium
# Install system deps needed by Playwright
info "Installing Playwright system dependencies…"
python -m playwright install-deps chromium 2>/dev/null || {
    warn "Could not auto-install Playwright system deps. You may need to install them manually."
    warn "Run: python -m playwright install-deps"
}
success "Playwright browsers ready"

# ═══════════════════════════════════════════════════════════════════════════════
#  6. Create Database & Tables
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 6 · Setting Up Database"

DB_NAME="bluesky_grokker"
DB_USER="grokker"
DB_PASS="grokker"

# Try to create role and database (may already exist)
info "Creating PostgreSQL user '$DB_USER' and database '$DB_NAME'…"

# Function to run psql as the postgres superuser
run_psql() {
    if [ "$OS" = "macos" ]; then
        # macOS Homebrew PostgreSQL runs as the current user
        psql -d postgres -c "$1" 2>/dev/null || true
    else
        sudo -u postgres psql -c "$1" 2>/dev/null || psql -U postgres -c "$1" 2>/dev/null || true
    fi
}

run_psql "CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASS';" 
run_psql "ALTER ROLE $DB_USER CREATEDB;"
run_psql "CREATE DATABASE $DB_NAME OWNER $DB_USER;" 

# Enable pgvector extension
run_psql_db() {
    if [ "$OS" = "macos" ]; then
        psql -d "$DB_NAME" -c "$1" 2>/dev/null || true
    else
        sudo -u postgres psql -d "$DB_NAME" -c "$1" 2>/dev/null || psql -U postgres -d "$DB_NAME" -c "$1" 2>/dev/null || true
    fi
}

run_psql_db "CREATE EXTENSION IF NOT EXISTS vector;"

success "Database '$DB_NAME' is ready"

# ── Create tables via a small Python script using our storage module ─────────
info "Creating tables via storage.py…"
(
    cd "$PROJECT_DIR"
    python -c "
import asyncio
import sys
sys.path.insert(0, '.')
from storage import init_storage, close_storage

async def setup():
    try:
        await init_storage()
        print('Tables created successfully')
        await close_storage()
    except Exception as e:
        print(f'Warning: Could not auto-create tables: {e}')
        print('Tables will be created on first run of main.py')

asyncio.run(setup())
"
)

success "Database schema initialised"

# ═══════════════════════════════════════════════════════════════════════════════
#  7. Bluesky Account & App Password Walkthrough
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 7 · Bluesky Account Setup"

echo ""
echo -e "${BOLD}This step will open a browser and walk you through:${RESET}"
echo "  • Creating a Bluesky account (or logging in)"
echo "  • Generating an App Password for Grokker to use"
echo "  • Saving your credentials to .env"
echo ""

read -rp "$(echo -e "${BOLD}Ready to begin? (y/n): ${RESET}")" DO_WALKTHROUGH

if [[ "$DO_WALKTHROUGH" =~ ^[Yy] ]]; then
    (cd "$PROJECT_DIR" && python setup_walkthrough.py)
else
    warn "Skipping Bluesky walkthrough."
    warn "You can run it later:  source venv/bin/activate && cd bluesky_grokker && python setup_walkthrough.py"

    # Create a minimal .env from the example if it doesn't exist
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        if [ -f "$PROJECT_DIR/.env.example" ]; then
            cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
            info "Created .env from .env.example – edit it with your credentials."
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  Done!
# ═══════════════════════════════════════════════════════════════════════════════
header "Setup Complete!"

echo -e "${GREEN}Bluesky-Grokker is ready to run.${RESET}"
echo ""
echo -e "  ${BOLD}To start:${RESET}"
echo "    source venv/bin/activate"
echo "    cd bluesky_grokker"
echo "    python main.py"
echo ""
echo -e "  ${BOLD}To re-run the Bluesky setup wizard:${RESET}"
echo "    source venv/bin/activate"
echo "    cd bluesky_grokker"
echo "    python setup_walkthrough.py"
echo ""
echo -e "  ${BOLD}Configuration:${RESET}  bluesky_grokker/.env"
echo -e "  ${BOLD}Docs:${RESET}           bluesky_grokker/README.md"
echo ""
success "Happy grokking! 🚀"
