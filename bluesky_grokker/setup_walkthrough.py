#!/usr/bin/env python3
"""Interactive Playwright walkthrough – helps the user create a Bluesky account
and generate an App Password, then writes credentials to .env."""

from __future__ import annotations

import os
import sys
import time

# ── Colours for terminal output ──────────────────────────────────────────────

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def info(msg: str) -> None:
    print(f"{CYAN}ℹ {msg}{RESET}")

def success(msg: str) -> None:
    print(f"{GREEN}✔ {msg}{RESET}")

def warn(msg: str) -> None:
    print(f"{YELLOW}⚠ {msg}{RESET}")

def error(msg: str) -> None:
    print(f"{RED}✖ {msg}{RESET}")

def header(msg: str) -> None:
    print(f"\n{BOLD}{'═' * 60}")
    print(f"  {msg}")
    print(f"{'═' * 60}{RESET}\n")

def prompt(msg: str) -> str:
    return input(f"{BOLD}{msg}{RESET} ").strip()


# ── Bluesky walkthrough ─────────────────────────────────────────────────────

def walkthrough() -> None:
    header("Bluesky-Grokker  ·  Account & App Password Setup")

    info("This wizard will open a browser and guide you through:")
    print("  1. Creating a Bluesky account (or logging in if you have one)")
    print("  2. Generating an App Password for Grokker")
    print("  3. Writing credentials to your .env file")
    print()

    # Ask if user already has an account
    has_account = prompt("Do you already have a Bluesky account? (y/n):").lower()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        error("Playwright is not installed. Run: pip install playwright && playwright install")
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=300)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        if has_account in ("y", "yes"):
            handle, app_password = _login_and_create_app_password(page)
        else:
            handle, app_password = _create_account_and_app_password(page)

        browser.close()

    if handle and app_password:
        _write_env(handle, app_password)
        success("Setup complete! Your .env file has been updated.")
    else:
        warn("Setup was not completed. You can re-run this script anytime.")


# ── Flow: Create a new account ──────────────────────────────────────────────

def _create_account_and_app_password(page) -> tuple[str, str]:
    header("Step 1 · Create a Bluesky Account")

    info("Opening Bluesky sign-up page…")
    page.goto("https://bsky.app")
    time.sleep(2)

    info("Look for the 'Sign up' or 'Create Account' button in the browser.")
    print()
    print(f"  {BOLD}Follow the on-screen instructions to:{RESET}")
    print("    • Choose a hosting provider (Bluesky Social is the default)")
    print("    • Enter your email address")
    print("    • Create a password")
    print("    • Choose your handle (e.g., yourname.bsky.social)")
    print("    • Complete any verification (date of birth, etc.)")
    print()

    # Try to click the sign-up / create account button
    try:
        signup_btn = page.get_by_role("button", name="Create a new account")
        if signup_btn.is_visible(timeout=5000):
            signup_btn.click()
    except Exception:
        try:
            signup_link = page.get_by_text("Sign up")
            if signup_link.is_visible(timeout=3000):
                signup_link.click()
        except Exception:
            pass

    warn("Complete the sign-up process in the browser window.")
    prompt("Press ENTER here once you have finished creating your account and are logged in…")

    # Now get the handle
    handle = _detect_or_ask_handle(page)

    # Proceed to app password
    app_password = _create_app_password(page)
    return handle, app_password


# ── Flow: Log in to existing account ────────────────────────────────────────

def _login_and_create_app_password(page) -> tuple[str, str]:
    header("Step 1 · Log In to Bluesky")

    info("Opening Bluesky login page…")
    page.goto("https://bsky.app")
    time.sleep(2)

    # Try to click Sign In
    try:
        signin_btn = page.get_by_role("button", name="Sign in")
        if signin_btn.is_visible(timeout=5000):
            signin_btn.click()
    except Exception:
        pass

    warn("Log in to your account in the browser window.")
    prompt("Press ENTER here once you are logged in…")

    handle = _detect_or_ask_handle(page)
    app_password = _create_app_password(page)
    return handle, app_password


# ── Create App Password ─────────────────────────────────────────────────────

def _create_app_password(page) -> str:
    header("Step 2 · Create an App Password")

    info("Navigating to App Passwords settings…")
    page.goto("https://bsky.app/settings/app-passwords")
    time.sleep(3)

    print(f"""
  {BOLD}What is an App Password?{RESET}
  An App Password is a special password that lets Grokker
  access your Bluesky account without using your main password.
  It can be revoked at any time from Settings.

  {BOLD}In the browser window:{RESET}
    1. Click "{BOLD}Add App Password{RESET}"
    2. Name it something like "{BOLD}grokker{RESET}"
    3. Click "{BOLD}Create App Password{RESET}"
    4. {RED}IMPORTANT: Copy the generated password!{RESET}
       It will only be shown once.
""")

    # Try to click "Add App Password" automatically
    try:
        add_btn = page.get_by_role("button", name="Add App Password")
        if add_btn.is_visible(timeout=5000):
            info("Found 'Add App Password' button – clicking it for you…")
            add_btn.click()
            time.sleep(1)

            # Try to fill in the name
            try:
                name_input = page.get_by_placeholder("Name")
                if not name_input.is_visible(timeout=2000):
                    name_input = page.locator("input[type='text']").first
                name_input.fill("grokker")
                info("Pre-filled the name as 'grokker'.")
            except Exception:
                warn("Could not auto-fill the name. Please type 'grokker' manually.")
    except Exception:
        warn("Could not find the 'Add App Password' button. Please click it manually.")

    warn("Complete the App Password creation in the browser.")
    print()
    app_password = prompt("Paste the generated App Password here:")

    if not app_password:
        error("No app password entered.")
        return ""

    # Validate format (app passwords are like xxxx-xxxx-xxxx-xxxx)
    if len(app_password.replace("-", "")) < 12:
        warn("That doesn't look like a standard app password, but we'll use it anyway.")

    success(f"App password received: {app_password[:4]}-****-****-****")
    return app_password


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_or_ask_handle(page) -> str:
    """Try to detect the user's handle from the page, or ask."""
    handle = ""

    # Try navigating to profile to read handle
    try:
        page.goto("https://bsky.app/profile/me", wait_until="networkidle", timeout=10000)
        time.sleep(2)
        url = page.url
        # URL should be like https://bsky.app/profile/handle.bsky.social
        if "/profile/" in url and url != "https://bsky.app/profile/me":
            handle = url.split("/profile/")[-1].strip("/")
            if handle:
                success(f"Detected your handle: @{handle}")
    except Exception:
        pass

    if not handle:
        handle = prompt("Enter your Bluesky handle (e.g. yourname.bsky.social):")

    if not handle:
        error("No handle provided.")
    return handle


PROVIDERS = {
    "1": {
        "name": "OpenAI",
        "key_env": "OPENAI_API_KEY",
        "provider": "openai",
        "default_model": "gpt-4o",
        "embed_model": "text-embedding-3-small",
        "embed_dim": "1536",
        "key_prefix": "sk-",
        "key_hint": "Starts with sk-…",
    },
    "2": {
        "name": "Anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "provider": "anthropic",
        "default_model": "claude-sonnet-4-20250514",
        "embed_model": "text-embedding-3-small",
        "embed_dim": "1536",
        "embed_note": "Anthropic has no embedding API – embeddings will use OpenAI if a key is set, otherwise Ollama.",
        "key_prefix": "sk-ant-",
        "key_hint": "Starts with sk-ant-…",
    },
    "3": {
        "name": "Google (Gemini)",
        "key_env": "GOOGLE_API_KEY",
        "provider": "google",
        "default_model": "gemini-1.5-flash",
        "embed_model": "models/text-embedding-004",
        "embed_dim": "768",
        "key_prefix": "",
        "key_hint": "From Google AI Studio",
    },
    "4": {
        "name": "OpenRouter",
        "key_env": "OPENROUTER_API_KEY",
        "provider": "openrouter",
        "default_model": "openai/gpt-4o",
        "embed_model": "openai/text-embedding-3-small",
        "embed_dim": "1536",
        "key_prefix": "sk-or-",
        "key_hint": "Starts with sk-or-…",
    },
    "5": {
        "name": "Ollama (local)",
        "key_env": None,
        "provider": "ollama",
        "default_model": "llama3",
        "embed_model": "nomic-embed-text",
        "embed_dim": "768",
        "key_prefix": "",
        "key_hint": "No API key needed – runs locally",
    },
}


def _configure_llm_provider(existing: dict[str, str]) -> None:
    """Interactive menu to choose LLM provider and enter API key."""
    header("Step 3 · Choose Your LLM Provider")

    info("Grokker needs an LLM for embeddings and agent reasoning.")
    info("Pick the provider you'd like to use:\n")

    for num, p in PROVIDERS.items():
        note = f"  ({p.get('key_hint', '')})" if p.get("key_hint") else ""
        print(f"  {BOLD}[{num}]{RESET}  {p['name']}{note}")
    print(f"  {BOLD}[s]{RESET}  Skip for now\n")

    choice = prompt("Enter choice (1-5, or s to skip):").strip().lower()

    if choice == "s" or choice not in PROVIDERS:
        if choice != "s":
            warn(f"Unknown choice '{choice}'. Skipping LLM setup.")
        info("You can configure LLM_PROVIDER and API keys in .env later.")
        return

    prov = PROVIDERS[choice]
    existing["LLM_PROVIDER"] = prov["provider"]
    existing["LLM_MODEL"] = prov["default_model"]
    existing["EMBEDDING_MODEL"] = prov["embed_model"]
    existing["EMBEDDING_DIM"] = prov["embed_dim"]

    success(f"Selected: {prov['name']}")
    print(f"  LLM model:       {BOLD}{prov['default_model']}{RESET}")
    print(f"  Embedding model:  {BOLD}{prov['embed_model']}{RESET}")

    if prov.get("embed_note"):
        print()
        warn(prov["embed_note"])
        # If Anthropic, also ask for an OpenAI key for embeddings
        oai_key = prompt("  Enter an OpenAI API key for embeddings (or ENTER to use Ollama):")
        if oai_key:
            existing["OPENAI_API_KEY"] = oai_key
            existing["EMBEDDING_PROVIDER"] = "openai"
            existing["EMBEDDING_MODEL"] = "text-embedding-3-small"
            existing["EMBEDDING_DIM"] = "1536"
            success("OpenAI key saved for embeddings.")
        else:
            existing["EMBEDDING_PROVIDER"] = "ollama"
            info("Embeddings will use Ollama. Make sure it's running locally.")

    # Collect API key
    if prov["key_env"] is not None:
        print()
        api_key = prompt(f"Enter your {prov['name']} API key ({prov['key_hint']}):")
        if api_key:
            existing[prov["key_env"]] = api_key
            success(f"{prov['name']} API key saved.")
        else:
            warn(f"No API key entered. Set {prov['key_env']} in .env before running Grokker.")
    else:
        info("No API key needed. Make sure Ollama is running: ollama serve")

    print()


def _write_env(handle: str, app_password: str) -> None:
    """Write or update the .env file with Bluesky credentials."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    header("Step 4 · Writing .env File")

    existing: dict[str, str] = {}

    # Read existing .env if present
    if os.path.exists(env_path):
        info(f"Updating existing .env at {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()
    else:
        # Copy from .env.example if available
        example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.example")
        if os.path.exists(example_path):
            info("Creating .env from .env.example")
            with open(example_path) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and "=" in stripped:
                        k, v = stripped.split("=", 1)
                        existing[k.strip()] = v.strip()

    # Set Bluesky credentials
    existing["BLUESKY_HANDLE"] = handle
    existing["BLUESKY_PASSWORD"] = app_password

    # Prompt for LLM provider and API key
    _configure_llm_provider(existing)

    # Write .env
    with open(env_path, "w") as f:
        f.write("# Bluesky-Grokker configuration\n")
        f.write("# Generated by setup_walkthrough.py\n\n")
        for k, v in existing.items():
            f.write(f"{k}={v}\n")

    success(f"Credentials written to {env_path}")
    print()
    print(f"  {BOLD}BLUESKY_HANDLE{RESET}    = {handle}")
    print(f"  {BOLD}BLUESKY_PASSWORD{RESET}  = {app_password[:4]}-****-****-****")
    print()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    walkthrough()
