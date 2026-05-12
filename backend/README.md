# Backend

## Demo Seed Data (Dev Only)

This repo supports seeding a deterministic demo dataset into the backend SQLite DB.

### Storage Locations

- `TB_DATA_DIR`: data root (default: `<repo>/data`)
- `TB_DB_PATH`: SQLite DB path (default: `$TB_DATA_DIR/tb_cdss.sqlite3`)
- X-rays are stored under `$TB_DATA_DIR/xrays/`

### Seed Command

Seed demo patients (refuses if DB already has patients):

```bash
python -m backend.seed_demo
```

Seed demo patients plus a couple of tiny demo X-ray files:

```bash
python -m backend.seed_demo --include-xrays
```

### Reset + Seed (Destructive)

Reset is intentionally guarded. You must opt in via:

- `TB_ALLOW_DEV_RESET=1`

Then run:

```bash
TB_ALLOW_DEV_RESET=1 python -m backend.seed_demo --reset --include-xrays
```

You can also reset only one part:

```bash
TB_ALLOW_DEV_RESET=1 python -m backend.seed_demo --reset-db
TB_ALLOW_DEV_RESET=1 python -m backend.seed_demo --reset-xrays
```
