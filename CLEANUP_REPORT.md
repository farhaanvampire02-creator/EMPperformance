# EMPperformance Cleanup Report

Generated during final UI polish phase. Only clearly safe temporary/cache files were auto-removed.

## Auto-removed (safe)

| Path | Reason | Referenced? | Safe? | Action |
|------|--------|-------------|-------|--------|
| `__pycache__/`, `*.pyc` | Python bytecode cache | No | Yes | Removed |

## Intentionally preserved (required / never-delete list)

| Path | Reason |
|------|--------|
| `emphr.py` | Main Streamlit application |
| `EMPperformance.ipynb` | Project notebook |
| `requirements.txt` | Deployment dependencies |
| `employee_performance_500_rows.csv` | Seed / fallback dataset |
| `employee_performance_master.csv` | Active normal-employee master dataset |
| `field_sales_master.csv` | Field Sales master dataset |
| `employee_performance_model.pkl` | Random Forest artifact (training pipeline) |
| `employee_performance_scaler.pkl` | Scaler for federated prediction |
| `federated_employee_model.keras` | Federated Neural Network model |
| `federated_metrics.pkl` | Federated evaluation metrics |
| `federated_training_history.csv` | Federated training history |
| `field_sales_performance_model.pkl` | Field Sales model |
| `field_sales_preprocessor.pkl` | Field Sales preprocessor |
| `field_sales_training_meta.json` | Field Sales training metadata |
| `field_sales_training_history.csv` | Field Sales training history |
| `model_training_meta.json` | Normal employee training metadata |
| `dataset_backups/` | Recovery artifacts (~174 KB) |
| `model_backups/` | Pre-retrain recovery artifacts (~6.9 MB) |
| `.devcontainer/devcontainer.json` | Dev / Codespaces configuration |

## Candidates requiring approval (NOT deleted)

| Path | Size | Reason it appears unused | Referenced by app/notebook? | Recommendation |
|------|------|--------------------------|-----------------------------|----------------|
| `emphr_backup_before_FL.py` | ~11 KB | Older code snapshot before FL work | No | Keep locally for recovery; ignored by git via `emphr_backup_*.py` |
| `emphr_backup_before_fieldsales.py` | ~68 KB | Older code snapshot before Field Sales | No | Keep locally; ignored by git |
| `emphr_backup_before_final_polish.py` | ~106 KB | Backup created for this polish phase | No | Keep locally until polish is confirmed; then optional delete |
| `dataset_backups/*.csv` | ~174 KB | Pre-import CSV snapshots | Written by app; not loaded at runtime | Preserve; now gitignored |
| `model_backups/**` | ~6.9 MB | Pre-retrain model snapshots | Written by app; not loaded at runtime | Preserve; now gitignored |

## Notes

- `matplotlib` remains in `requirements.txt` because `EMPperformance.ipynb` may use it; it is not imported by `emphr.py`.
- No duplicate datasets or obsolete training scripts were permanently deleted.
- `.gitignore` now excludes `dataset_backups/`, `model_backups/`, and `emphr_backup_*.py`.
