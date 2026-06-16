# Local Config Folder

This folder documents the machine-local settings that should stay out of the public repository.

## Usage

1. Create a `local_config` folder at the repo root.
2. Copy `.env.example` into that folder as `.env`.
3. Replace the placeholder IPs and runtime settings with your own.

The app auto-loads `local_config/.env` when `python-dotenv` is available.
