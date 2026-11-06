import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="EMPperformance",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "employee_performance_500_rows.csv"
MASTER_DATA_PATH = BASE_DIR / "employee_performance_master.csv"
MODEL_PATH = BASE_DIR / "employee_performance_model.pkl"
FEDERATED_MODEL_PATH = BASE_DIR / "federated_employee_model.keras"
SCALER_PATH = BASE_DIR / "employee_performance_scaler.pkl"
FEDERATED_HISTORY_PATH = BASE_DIR / "federated_training_history.csv"
FEDERATED_METRICS_PATH = BASE_DIR / "federated_metrics.pkl"
TRAINING_META_PATH = BASE_DIR / "model_training_meta.json"
BACKUP_DIR = BASE_DIR / "model_backups"
IMPORT_BACKUP_DIR = BASE_DIR / "dataset_backups"

FEATURES = [
    "attendance_percentage",
    "task_completion_rate",
    "quality_score",
    "average_work_hours",
    "training_hours",
    "workload_score",
    "tasks_completed",
    "experience_years"
]

REQUIRED_COLUMNS = [
    "employee_id",
    "department",
    "job_role",
    "age",
    "experience_years",
    "attendance_percentage",
    "tasks_assigned",
    "tasks_completed",
    "task_completion_rate",
    "quality_score",
    "average_work_hours",
    "training_hours",
    "workload_score",
    "performance_score",
    "performance_status"
]

NUMERIC_COLUMNS = [
    "age",
    "experience_years",
    "attendance_percentage",
    "tasks_assigned",
    "tasks_completed",
    "task_completion_rate",
    "quality_score",
    "average_work_hours",
    "training_hours",
    "workload_score",
    "performance_score"
]

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .emp-hero {
        background: linear-gradient(135deg, #0f3d5e 0%, #1a6b8a 55%, #2a8f9e 100%);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        color: #ffffff;
        margin-bottom: 1.1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .emp-hero h2 {
        margin: 0 0 0.35rem 0;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .emp-hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 0.98rem;
    }
    .emp-card {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        height: 100%;
    }
    .emp-kpi {
        background: linear-gradient(180deg, #f7fbfd 0%, #eef6f8 100%);
        border: 1px solid #c9dde6;
        border-radius: 14px;
        padding: 1.25rem 1.1rem;
        text-align: center;
    }
    .emp-kpi .label {
        color: #486581;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .emp-kpi .value {
        color: #0f3d5e;
        font-size: 2.6rem;
        font-weight: 750;
        line-height: 1.15;
        margin: 0.25rem 0;
    }
    .emp-kpi .unit {
        color: #627d98;
        font-size: 0.95rem;
    }
    .emp-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-weight: 650;
        font-size: 0.92rem;
        border: 1px solid transparent;
    }
    .badge-on-track {
        background: #e3f8ef;
        color: #0b6b45;
        border-color: #b7ebcf;
    }
    .badge-monitor {
        background: #fff6e0;
        color: #8a5b00;
        border-color: #ffe2a8;
    }
    .badge-needs-support {
        background: #fdecec;
        color: #9b1c1c;
        border-color: #f5c2c2;
    }
    .emp-section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #102a43;
        margin: 0.2rem 0 0.65rem 0;
    }
    .emp-soft {
        color: #486581;
        font-size: 0.92rem;
        margin-bottom: 0.75rem;
    }
    .suggestion-item {
        background: #f8fafc;
        border-left: 4px solid #1a6b8a;
        border-radius: 8px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.55rem;
        color: #243b53;
        font-size: 0.94rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HELPERS ----------------
def get_performance_category(score):
    if score < 60:
        return "Needs Support"
    if score < 75:
        return "Monitor"
    return "On Track"


def badge_class(category):
    mapping = {
        "On Track": "badge-on-track",
        "Monitor": "badge-monitor",
        "Needs Support": "badge-needs-support"
    }
    return mapping.get(category, "badge-monitor")


def get_active_data_path():
    if MASTER_DATA_PATH.exists():
        return MASTER_DATA_PATH
    return DATA_PATH


def load_training_meta():
    if not TRAINING_META_PATH.exists():
        return {}
    try:
        return json.loads(TRAINING_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_training_meta(meta):
    TRAINING_META_PATH.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )


def ensure_dirs():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    IMPORT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def atomic_replace(src_tmp: Path, dest: Path):
    os.replace(str(src_tmp), str(dest))


def atomic_joblib_dump(obj, dest: Path):
    ensure_dirs()
    tmp = dest.with_name(dest.name + ".tmp")
    joblib.dump(obj, tmp)
    atomic_replace(tmp, dest)


def atomic_dataframe_to_csv(df: pd.DataFrame, dest: Path):
    ensure_dirs()
    tmp = dest.with_name(dest.name + ".tmp")
    df.to_csv(tmp, index=False)
    atomic_replace(tmp, dest)


def atomic_keras_save(model, dest: Path):
    ensure_dirs()
    fd, tmp_name = tempfile.mkstemp(suffix=".keras", dir=str(dest.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        model.save(tmp_path)
        atomic_replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def backup_existing_artifacts(tag: str):
    ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BACKUP_DIR / f"{tag}_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    for path in [
        MODEL_PATH,
        FEDERATED_MODEL_PATH,
        SCALER_PATH,
        FEDERATED_HISTORY_PATH,
        FEDERATED_METRICS_PATH,
        TRAINING_META_PATH
    ]:
        if path.exists():
            shutil.copy2(path, target / path.name)
    return target


def backup_dataset(df: pd.DataFrame, label: str):
    ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = IMPORT_BACKUP_DIR / f"{label}_{stamp}.csv"
    df.to_csv(path, index=False)
    return path


def evaluate_regression(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }


def create_fl_model(input_dim=8, learning_rate=0.005, output_bias=None):
    """Build the shared FL architecture. output_bias should be y_train mean."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.initializers import Constant

    bias_init = Constant(float(output_bias)) if output_bias is not None else "zeros"

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear", bias_initializer=bias_init)
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def average_weights(weight_list, sizes):
    """FedAvg: size-weighted mean of client weights."""
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("Client sample counts must be positive for FedAvg.")
    avg = []
    for layer_idx in range(len(weight_list[0])):
        layer_sum = None
        for weights, size in zip(weight_list, sizes):
            scaled = weights[layer_idx] * (float(size) / total)
            layer_sum = scaled if layer_sum is None else layer_sum + scaled
        avg.append(layer_sum)
    return avg


def build_department_clients(X_scaled, y, departments):
    """Partition scaled features and aligned targets by department."""
    X_arr = np.asarray(X_scaled, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    dept_arr = np.asarray(pd.Series(departments).astype(str))

    if len(X_arr) != len(y_arr) or len(X_arr) != len(dept_arr):
        raise ValueError(
            f"Client partition alignment error: X={len(X_arr)}, "
            f"y={len(y_arr)}, departments={len(dept_arr)}"
        )

    clients = []
    for dept in sorted(pd.unique(dept_arr)):
        mask = dept_arr == dept
        n = int(mask.sum())
        if n == 0:
            continue
        clients.append((X_arr[mask], y_arr[mask], str(dept), n))
    return clients


def train_federated_model(
    X_train,
    y_train,
    departments_train,
    X_test,
    y_test,
    rounds=8,
    local_epochs=8,
    learning_rate=0.005,
    progress_callback=None
):
    """
    Department-wise FedAvg training.

    Critical fix: initialize the output bias to the training target mean.
    Without this, Adam starts near prediction ~0 while performance_score is
    ~54-100, so short federated rounds stay stuck with MAE≈target_mean.
    """
    import tensorflow as tf

    # Align and freeze arrays so client masks cannot desync X/y
    X_train_arr = np.asarray(X_train, dtype=np.float64)
    y_train_arr = np.asarray(y_train, dtype=np.float64).ravel()
    X_test_arr = np.asarray(X_test, dtype=np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.float64).ravel()
    dept_arr = np.asarray(pd.Series(departments_train).astype(str))

    if X_train_arr.shape[0] != y_train_arr.shape[0]:
        raise ValueError("X_train and y_train length mismatch.")
    if X_train_arr.shape[0] != dept_arr.shape[0]:
        raise ValueError("X_train and departments_train length mismatch.")
    if X_test_arr.shape[0] != y_test_arr.shape[0]:
        raise ValueError("X_test and y_test length mismatch.")

    target_diag = {
        "min": float(np.min(y_train_arr)),
        "max": float(np.max(y_train_arr)),
        "mean": float(np.mean(y_train_arr)),
        "std": float(np.std(y_train_arr))
    }
    if target_diag["min"] < 0 or target_diag["max"] > 100:
        raise ValueError(
            f"Unexpected performance_score scale in training target: {target_diag}"
        )

    # Scaler fitted ONLY on training features (never on target / test)
    scaler = StandardScaler()
    X_train_df = pd.DataFrame(X_train_arr, columns=FEATURES)
    X_test_df = pd.DataFrame(X_test_arr, columns=FEATURES)
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # Validation split from training only (untouched test reserved for final eval)
    (
        X_fit, X_val,
        y_fit, y_val,
        dept_fit, _dept_val
    ) = train_test_split(
        X_train_scaled,
        y_train_arr,
        dept_arr,
        test_size=0.15,
        random_state=42
    )

    clients = build_department_clients(X_fit, y_fit, dept_fit)
    if len(clients) < 2:
        raise ValueError(
            "Federated training requires at least 2 department clients."
        )

    # Verify each client has matching X/y rows and positive counts
    for X_client, y_client, dept_name, n in clients:
        if len(X_client) != len(y_client) or len(X_client) != n:
            raise ValueError(
                f"Misaligned client partition for department '{dept_name}'."
            )

    output_bias = float(np.mean(y_fit))
    tf.random.set_seed(42)
    np.random.seed(42)

    global_model = create_fl_model(
        input_dim=X_train_scaled.shape[1],
        learning_rate=learning_rate,
        output_bias=output_bias
    )

    history_rows = []
    best_val_mae = float("inf")
    best_weights = global_model.get_weights()
    best_round = 0

    for round_idx in range(1, int(rounds) + 1):
        client_weights = []
        client_sizes = []

        for X_client, y_client, _dept_name, n_client in clients:
            # Same architecture + start from current global weights
            local_model = create_fl_model(
                input_dim=X_train_scaled.shape[1],
                learning_rate=learning_rate,
                output_bias=output_bias
            )
            local_model.set_weights(global_model.get_weights())
            batch_size = int(min(16, max(1, len(X_client))))
            local_model.fit(
                X_client,
                y_client,
                epochs=int(local_epochs),
                batch_size=batch_size,
                verbose=0,
                shuffle=True
            )
            client_weights.append(local_model.get_weights())
            client_sizes.append(int(n_client))

        if not client_weights:
            raise ValueError("Federated round received no client updates.")
        if sum(client_sizes) != len(X_fit):
            raise ValueError(
                "FedAvg client sample counts do not cover the fit partition."
            )

        global_model.set_weights(
            average_weights(client_weights, client_sizes)
        )

        val_pred = global_model.predict(X_val, verbose=0).ravel()
        loss_value = float(mean_squared_error(y_val, val_pred))
        mae_value = float(mean_absolute_error(y_val, val_pred))
        history_rows.append({
            "round": round_idx,
            "loss": loss_value,
            "mae": mae_value
        })

        if mae_value < best_val_mae:
            best_val_mae = mae_value
            best_weights = [w.copy() for w in global_model.get_weights()]
            best_round = round_idx

        if progress_callback is not None:
            progress_callback(round_idx, rounds, loss_value, mae_value)

    # Restore best validation round, then evaluate once on untouched test
    global_model.set_weights(best_weights)
    y_pred = global_model.predict(X_test_scaled, verbose=0).ravel()
    metrics = evaluate_regression(y_test_arr, y_pred)
    metrics["best_round"] = int(best_round)
    metrics["val_mae_best"] = float(best_val_mae)

    prediction_diag = {
        "min": float(np.min(y_pred)),
        "max": float(np.max(y_pred)),
        "mean": float(np.mean(y_pred)),
        "std": float(np.std(y_pred))
    }
    diagnostics = {
        "target_train": target_diag,
        "prediction_test": prediction_diag,
        "output_bias_init": output_bias,
        "client_sizes": {dept: n for _, _, dept, n in clients},
        "learning_rate": float(learning_rate),
        "local_epochs": int(local_epochs),
        "rounds": int(rounds),
        "best_round": int(best_round)
    }

    history_df = pd.DataFrame(history_rows)
    return global_model, scaler, metrics, history_df, diagnostics


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)
    return model, metrics, y_pred


def validate_employee_dataframe(df: pd.DataFrame):
    report = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "missing_columns": [],
        "missing_values": {},
        "duplicate_rows": 0,
        "duplicate_employee_ids": 0,
        "rows": int(len(df)),
        "columns": int(df.shape[1])
    }

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing_cols
    if missing_cols:
        report["ok"] = False
        report["errors"].append(
            "Missing required columns: " + ", ".join(missing_cols)
        )
        return report

    working = df[REQUIRED_COLUMNS].copy()

    na_counts = working.isna().sum()
    report["missing_values"] = {
        col: int(count) for col, count in na_counts.items() if count > 0
    }
    if report["missing_values"]:
        report["ok"] = False
        report["errors"].append(
            "Missing values found in: "
            + ", ".join(
                f"{k} ({v})" for k, v in report["missing_values"].items()
            )
        )

    report["duplicate_rows"] = int(working.duplicated().sum())
    report["duplicate_employee_ids"] = int(
        working["employee_id"].duplicated().sum()
    )
    if report["duplicate_rows"] > 0:
        report["warnings"].append(
            f"{report['duplicate_rows']} fully duplicate row(s) detected."
        )
    if report["duplicate_employee_ids"] > 0:
        report["warnings"].append(
            f"{report['duplicate_employee_ids']} duplicate employee_id value(s) detected."
        )

    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(working[col], errors="coerce")
        invalid = int(coerced.isna().sum() - working[col].isna().sum())
        if invalid > 0:
            report["ok"] = False
            report["errors"].append(
                f"Column '{col}' has {invalid} non-numeric value(s)."
            )

    if report["ok"]:
        numeric_view = working.copy()
        for col in NUMERIC_COLUMNS:
            numeric_view[col] = pd.to_numeric(numeric_view[col], errors="coerce")

        if (numeric_view["attendance_percentage"] < 0).any() or (
            numeric_view["attendance_percentage"] > 100
        ).any():
            report["ok"] = False
            report["errors"].append(
                "attendance_percentage must be between 0 and 100."
            )

        if (numeric_view["task_completion_rate"] < 0).any() or (
            numeric_view["task_completion_rate"] > 100
        ).any():
            report["ok"] = False
            report["errors"].append(
                "task_completion_rate must be between 0 and 100."
            )

        if (numeric_view["quality_score"] < 0).any() or (
            numeric_view["quality_score"] > 100
        ).any():
            report["ok"] = False
            report["errors"].append("quality_score must be between 0 and 100.")

        if (numeric_view["performance_score"] < 0).any() or (
            numeric_view["performance_score"] > 100
        ).any():
            report["ok"] = False
            report["errors"].append(
                "performance_score must be between 0 and 100."
            )

        if (numeric_view["tasks_completed"] > numeric_view["tasks_assigned"]).any():
            report["warnings"].append(
                "Some rows have tasks_completed greater than tasks_assigned."
            )

    return report


def prepare_training_frame(df: pd.DataFrame):
    clean = df[REQUIRED_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=FEATURES + ["performance_score", "department"])
    clean = clean.drop_duplicates()
    clean = clean.drop_duplicates(subset=["employee_id"], keep="last")

    if len(clean) < 30:
        raise ValueError(
            "Not enough valid rows to train models (need at least 30)."
        )
    if clean["department"].nunique() < 2:
        raise ValueError(
            "Federated training requires at least 2 departments in the dataset."
        )
    return clean


def generate_improvement_suggestions(values: dict, predicted_score: float):
    attendance = values["attendance_percentage"]
    completion = values["task_completion_rate"]
    quality = values["quality_score"]
    work_hours = values["average_work_hours"]
    training = values["training_hours"]
    workload = values["workload_score"]
    tasks = values["tasks_completed"]
    experience = values["experience_years"]

    personalized = []
    training_recs = []
    attendance_recs = []
    productivity_recs = []
    quality_recs = []
    workload_recs = []

    if attendance < 75:
        attendance_recs.append(
            f"Attendance is {attendance:.1f}%, below 75%. Prioritize punctuality tracking and short check-ins to stabilize presence."
        )
        personalized.append(
            "Current attendance is a primary risk factor for performance consistency."
        )
    elif attendance < 85:
        attendance_recs.append(
            f"Attendance is {attendance:.1f}%. Improve schedule reliability and reduce unplanned absences to move above 85%."
        )
    else:
        attendance_recs.append(
            f"Attendance is strong at {attendance:.1f}%. Maintain current attendance habits."
        )

    if completion < 70:
        productivity_recs.append(
            f"Task completion is {completion:.1f}%, below 70%. Break work into weekly milestones and review blockers twice a week."
        )
        training_recs.append(
            "Recommend Task Planning and Time Management training based on low completion rate."
        )
        personalized.append(
            "Low task completion is currently limiting predicted performance."
        )
    elif completion < 85:
        productivity_recs.append(
            f"Task completion is {completion:.1f}%. Focus on backlog reduction and clearer prioritization to reach 85%+."
        )
    else:
        productivity_recs.append(
            f"Task completion is healthy at {completion:.1f}%. Keep delivery cadence consistent."
        )

    if quality < 70:
        quality_recs.append(
            f"Quality score is {quality:.1f}, below 70%. Add peer review checkpoints and a pre-delivery quality checklist."
        )
        training_recs.append(
            "Recommend Work Quality Improvement training based on the current quality score."
        )
        personalized.append(
            "Quality gaps are contributing to a lower predicted score."
        )
    elif quality < 85:
        quality_recs.append(
            f"Quality score is {quality:.1f}. Strengthen documentation standards and reduce rework cycles."
        )
    else:
        quality_recs.append(
            f"Quality score is strong at {quality:.1f}. Maintain review discipline."
        )

    if training < 5:
        training_recs.append(
            f"Training hours are only {training:.1f}. Schedule structured skill development to reach at least 5 hours."
        )
        personalized.append(
            "Limited recent training reduces readiness for higher performance outcomes."
        )
    elif training < 10:
        training_recs.append(
            f"Training hours are {training:.1f}. Add targeted upskilling for role-critical competencies."
        )
    else:
        training_recs.append(
            f"Training investment is solid at {training:.1f} hours. Continue role-aligned learning."
        )

    if work_hours > 10:
        workload_recs.append(
            f"Average work hours are high at {work_hours:.1f}. Introduce workload redistribution and recovery planning."
        )
        personalized.append(
            "Elevated work hours indicate possible burnout risk that can hurt sustained performance."
        )
    elif work_hours > 9:
        workload_recs.append(
            f"Average work hours are {work_hours:.1f}. Monitor overtime and protect focus blocks."
        )
    else:
        workload_recs.append(
            f"Work hours are balanced at {work_hours:.1f}. Maintain sustainable pacing."
        )

    if workload > 80:
        workload_recs.append(
            f"Workload score is high at {workload:.1f}. Rebalance assignments and clarify priority tiers."
        )
        training_recs.append(
            "Recommend Workload Management and Support based on elevated workload."
        )
        personalized.append(
            "High workload pressure is a key area for managerial intervention."
        )
    elif workload > 65:
        workload_recs.append(
            f"Workload score is {workload:.1f}. Review capacity weekly to prevent overload."
        )
    else:
        workload_recs.append(
            f"Workload score is manageable at {workload:.1f}. Capacity can support additional development goals."
        )

    if tasks < 25 and experience >= 3:
        productivity_recs.append(
            f"Tasks completed ({int(tasks)}) appear low relative to {experience:.1f} years of experience. Align goals with measurable output targets."
        )

    if predicted_score < 60:
        personalized.append(
            f"Predicted score {predicted_score:.1f} is in Needs Support. Focus first on attendance, completion, and quality fundamentals."
        )
    elif predicted_score < 75:
        personalized.append(
            f"Predicted score {predicted_score:.1f} is in Monitor. Targeted coaching on weakest input metrics can move performance On Track."
        )
    else:
        personalized.append(
            f"Predicted score {predicted_score:.1f} is On Track. Maintain strengths and invest in advanced capability building."
        )

    if not training_recs:
        training_recs.append(
            "Continue advanced skill development and career-growth learning paths."
        )

    return {
        "personalized": personalized,
        "training": training_recs,
        "attendance": attendance_recs,
        "productivity": productivity_recs,
        "quality": quality_recs,
        "workload": workload_recs
    }


def clear_model_caches():
    load_data.clear()
    load_model.clear()
    load_federated_model.clear()
    load_scaler.clear()
    load_federated_history.clear()
    load_federated_metrics.clear()


# ---------------- LOAD DATA / MODELS ----------------
@st.cache_data
def load_data(path_str: str):
    return pd.read_csv(path_str)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_federated_model():
    from tensorflow import keras
    return keras.models.load_model(FEDERATED_MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_federated_history():
    return pd.read_csv(FEDERATED_HISTORY_PATH)


@st.cache_data
def load_federated_metrics():
    if FEDERATED_METRICS_PATH.exists():
        return joblib.load(FEDERATED_METRICS_PATH)
    return None


active_data_path = get_active_data_path()
if not active_data_path.exists():
    st.error("Dataset CSV file not found!")
    st.stop()

df = load_data(str(active_data_path))
training_meta = load_training_meta()

# ---------------- SIDEBAR ----------------
st.sidebar.title("EMPperformance")
st.sidebar.caption("HR Intelligence System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Employee Analytics",
        "Attendance Analytics",
        "Productivity Analytics",
        "ML Performance Prediction",
        "Federated Learning Prediction",
        "Model Comparison",
        "Federated Training History",
        "Data Import & Model Training",
        "Training Recommendation",
        "Employee Data"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Active dataset: `{active_data_path.name}`")
st.sidebar.caption(f"Records: {len(df)}")
if training_meta.get("last_trained_at"):
    st.sidebar.caption(f"Last trained: {training_meta['last_trained_at']}")

# ---------------- HEADER ----------------
st.title("Employee Performance Intelligence")
st.caption("Productivity Analytics | ML Prediction | Training Support")

# =================================================
# 1. DASHBOARD OVERVIEW
# =================================================
if page == "Dashboard Overview":

    st.header("HR Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Employees", df["employee_id"].nunique())
    c2.metric("Average Performance",
              round(df["performance_score"].mean(), 2))
    c3.metric("Average Attendance",
              f"{df['attendance_percentage'].mean():.2f}%")
    c4.metric("Task Completion",
              f"{df['task_completion_rate'].mean():.2f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        dept = df.groupby("department", as_index=False)[
            "performance_score"
        ].mean()

        fig = px.bar(
            dept,
            x="department",
            y="performance_score",
            title="Department Performance",
            color="performance_score"
        )

        st.plotly_chart(fig, width="stretch")

    with col2:
        status = df["performance_status"].value_counts().reset_index()
        status.columns = ["Status", "Count"]

        fig = px.pie(
            status,
            names="Status",
            values="Count",
            title="Performance Status Distribution",
            hole=0.4
        )

        st.plotly_chart(fig, width="stretch")

    fig = px.histogram(
        df,
        x="performance_score",
        nbins=20,
        title="Performance Score Distribution"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Top 10 Performance Scores")
    st.dataframe(
        df.nlargest(10, "performance_score"),
        width="stretch"
    )

# =================================================
# 2. EMPLOYEE ANALYTICS
# =================================================
elif page == "Employee Analytics":

    st.header("Employee Analytics")

    departments = ["All"] + sorted(df["department"].unique())

    selected = st.selectbox("Select Department", departments)

    filtered = df if selected == "All" else df[
        df["department"] == selected
    ]

    st.metric("Employees", len(filtered))

    fig = px.box(
        filtered,
        x="department",
        y="performance_score",
        color="department",
        title="Department Performance Comparison"
    )

    st.plotly_chart(fig, width="stretch")

    fig = px.scatter(
        filtered,
        x="experience_years",
        y="performance_score",
        color="department",
        hover_data=["employee_id"],
        title="Experience vs Performance"
    )

    st.plotly_chart(fig, width="stretch")

    st.dataframe(filtered, width="stretch")

# =================================================
# 3. ATTENDANCE ANALYTICS
# =================================================
elif page == "Attendance Analytics":

    st.header("Attendance Analytics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Attendance",
        f"{df['attendance_percentage'].mean():.2f}%"
    )

    c2.metric(
        "Attendance Below 75%",
        int((df["attendance_percentage"] < 75).sum())
    )

    c3.metric(
        "Attendance Above 90%",
        int((df["attendance_percentage"] >= 90).sum())
    )

    fig = px.histogram(
        df,
        x="attendance_percentage",
        nbins=20,
        title="Attendance Distribution"
    )

    st.plotly_chart(fig, width="stretch")

    dept_att = df.groupby("department", as_index=False)[
        "attendance_percentage"
    ].mean()

    fig = px.bar(
        dept_att,
        x="department",
        y="attendance_percentage",
        title="Department Attendance"
    )

    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        df[["employee_id", "department", "attendance_percentage"]],
        width="stretch"
    )

# =================================================
# 4. PRODUCTIVITY ANALYTICS
# =================================================
elif page == "Productivity Analytics":

    st.header("Productivity Analytics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Task Completion",
        f"{df['task_completion_rate'].mean():.2f}%"
    )

    c2.metric(
        "Average Quality",
        round(df["quality_score"].mean(), 2)
    )

    c3.metric(
        "Average Workload",
        round(df["workload_score"].mean(), 2)
    )

    fig = px.scatter(
        df,
        x="task_completion_rate",
        y="quality_score",
        color="department",
        title="Task Completion vs Quality"
    )

    st.plotly_chart(fig, width="stretch")

    fig = px.box(
        df,
        x="department",
        y="workload_score",
        color="department",
        title="Workload Distribution"
    )

    st.plotly_chart(fig, width="stretch")

    dept_productivity = df.groupby("department", as_index=False)[
        "task_completion_rate"
    ].mean()

    fig = px.bar(
        dept_productivity,
        x="department",
        y="task_completion_rate",
        title="Department Productivity"
    )

    st.plotly_chart(fig, width="stretch")

# =================================================
# 5. ML PERFORMANCE PREDICTION
# =================================================
elif page == "ML Performance Prediction":

    st.header("ML Performance Prediction")

    st.warning(
        "Demo prediction using synthetic dataset. "
        "This is not a validated real-world employee assessment."
    )

    if not MODEL_PATH.exists():
        st.error("Model file not found!")
        st.stop()

    model = load_model()

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            attendance = st.slider(
                "Attendance Percentage", 0.0, 100.0, 90.0
            )

            completion = st.slider(
                "Task Completion Rate", 0.0, 100.0, 80.0
            )

            quality = st.slider(
                "Quality Score", 0.0, 100.0, 80.0
            )

            work_hours = st.number_input(
                "Average Work Hours", 0.0, 24.0, 8.0
            )

        with col2:
            training = st.number_input(
                "Training Hours", 0.0, 100.0, 5.0
            )

            workload = st.slider(
                "Workload Score", 0.0, 100.0, 50.0
            )

            tasks = st.number_input(
                "Tasks Completed", 0, 1000, 40
            )

            experience = st.number_input(
                "Experience Years", 0.0, 50.0, 3.0
            )

        submit = st.form_submit_button("Predict Performance")

    if submit:

        new_employee = pd.DataFrame([{
            "attendance_percentage": attendance,
            "task_completion_rate": completion,
            "quality_score": quality,
            "average_work_hours": work_hours,
            "training_hours": training,
            "workload_score": workload,
            "tasks_completed": tasks,
            "experience_years": experience
        }])

        new_employee = new_employee[FEATURES]

        prediction = model.predict(new_employee)[0]

        prediction = np.clip(prediction, 0, 100)

        st.success(
            f"Predicted Performance Score: {prediction:.2f}/100"
        )

        st.info(
            "Prediction is an estimate. Use it only as supportive information."
        )

# =================================================
# 5b. FEDERATED LEARNING PREDICTION
# =================================================
elif page == "Federated Learning Prediction":

    st.markdown(
        """
        <div class="emp-hero">
            <h2>Employee Performance Intelligence</h2>
            <p>AI-powered performance estimation and personalized development insights</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    missing = []
    if not FEDERATED_MODEL_PATH.exists():
        missing.append(FEDERATED_MODEL_PATH.name)
    if not SCALER_PATH.exists():
        missing.append(SCALER_PATH.name)

    if missing:
        st.error(
            "Required Federated Learning file(s) not found: "
            + ", ".join(missing)
        )
        st.stop()

    federated_model = load_federated_model()
    scaler = load_scaler()

    st.markdown(
        '<div class="emp-section-title">Employee Feature Inputs</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="emp-soft">Enter operational metrics to generate a federated neural network performance estimate.</div>',
        unsafe_allow_html=True
    )

    with st.form("federated_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            fl_attendance = st.slider(
                "Attendance Percentage", 0.0, 100.0, 90.0
            )

            fl_completion = st.slider(
                "Task Completion Rate", 0.0, 100.0, 80.0
            )

            fl_quality = st.slider(
                "Quality Score", 0.0, 100.0, 80.0
            )

            fl_work_hours = st.number_input(
                "Average Work Hours", 0.0, 24.0, 8.0
            )

        with col2:
            fl_training = st.number_input(
                "Training Hours", 0.0, 100.0, 5.0
            )

            fl_workload = st.slider(
                "Workload Score", 0.0, 100.0, 50.0
            )

            fl_tasks = st.number_input(
                "Tasks Completed", 0, 1000, 40
            )

            fl_experience = st.number_input(
                "Experience Years", 0.0, 50.0, 3.0
            )

        fl_submit = st.form_submit_button("Generate Performance Insights")

    if fl_submit:

        values = {
            "attendance_percentage": fl_attendance,
            "task_completion_rate": fl_completion,
            "quality_score": fl_quality,
            "average_work_hours": fl_work_hours,
            "training_hours": fl_training,
            "workload_score": fl_workload,
            "tasks_completed": fl_tasks,
            "experience_years": fl_experience
        }

        fl_input = pd.DataFrame([values])[FEATURES]
        fl_scaled = scaler.transform(fl_input)
        fl_prediction = float(
            federated_model.predict(fl_scaled, verbose=0)[0][0]
        )
        fl_prediction = float(np.clip(fl_prediction, 0, 100))
        fl_category = get_performance_category(fl_prediction)
        suggestions = generate_improvement_suggestions(values, fl_prediction)

        kpi_col, badge_col, indicator_col = st.columns([1.2, 1, 1.2])

        with kpi_col:
            st.markdown(
                f"""
                <div class="emp-kpi">
                    <div class="label">Predicted Performance Score</div>
                    <div class="value">{fl_prediction:.1f}</div>
                    <div class="unit">out of 100</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with badge_col:
            st.markdown('<div class="emp-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="emp-section-title">Performance Category</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<span class="emp-badge {badge_class(fl_category)}">{fl_category}</span>',
                unsafe_allow_html=True
            )
            st.caption(
                "Category thresholds: Needs Support < 60, "
                "Monitor 60-74.9, On Track >= 75"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with indicator_col:
            st.markdown('<div class="emp-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="emp-section-title">Employee Performance Indicator</div>',
                unsafe_allow_html=True
            )
            st.progress(min(max(fl_prediction / 100.0, 0.0), 1.0))
            if fl_prediction >= 75:
                indicator_text = "Strong projected performance trajectory"
            elif fl_prediction >= 60:
                indicator_text = "Stable with targeted improvement opportunity"
            else:
                indicator_text = "At-risk projection requiring focused support"
            st.write(indicator_text)
            st.markdown("</div>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        feature_chart_df = pd.DataFrame({
            "Feature": [
                "Attendance",
                "Task Completion",
                "Quality",
                "Training Hours",
                "Workload",
                "Work Hours"
            ],
            "Value": [
                fl_attendance,
                fl_completion,
                fl_quality,
                min(fl_training * 5, 100),
                fl_workload,
                min(fl_work_hours * 10, 100)
            ]
        })

        with chart_col1:
            fig = px.bar(
                feature_chart_df,
                x="Feature",
                y="Value",
                title="Input Profile (Normalized View)",
                color="Value",
                color_continuous_scale="Teal"
            )
            fig.update_layout(yaxis_title="Indexed Value", showlegend=False)
            st.plotly_chart(fig, width="stretch")

        with chart_col2:
            radar_df = pd.DataFrame({
                "Metric": [
                    "Attendance",
                    "Completion",
                    "Quality",
                    "Training",
                    "Workload Control",
                    "Hour Balance"
                ],
                "Score": [
                    fl_attendance,
                    fl_completion,
                    fl_quality,
                    min(fl_training * 10, 100),
                    max(0, 100 - fl_workload),
                    max(0, 100 - abs(fl_work_hours - 8) * 12)
                ]
            })
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_df["Score"].tolist() + [radar_df["Score"].iloc[0]],
                theta=radar_df["Metric"].tolist() + [radar_df["Metric"].iloc[0]],
                fill="toself",
                name="Employee Profile",
                line_color="#1a6b8a"
            ))
            fig.update_layout(
                title="Capability Balance Radar",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown(
            '<div class="emp-section-title">Personalized Employee Improvement Suggestions</div>',
            unsafe_allow_html=True
        )
        for item in suggestions["personalized"]:
            st.markdown(
                f'<div class="suggestion-item">{item}</div>',
                unsafe_allow_html=True
            )

        sug_c1, sug_c2 = st.columns(2)
        with sug_c1:
            st.markdown(
                '<div class="emp-section-title">Training Recommendations</div>',
                unsafe_allow_html=True
            )
            for item in suggestions["training"]:
                st.markdown(
                    f'<div class="suggestion-item">{item}</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="emp-section-title">Attendance Improvement Suggestions</div>',
                unsafe_allow_html=True
            )
            for item in suggestions["attendance"]:
                st.markdown(
                    f'<div class="suggestion-item">{item}</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="emp-section-title">Productivity Improvement Suggestions</div>',
                unsafe_allow_html=True
            )
            for item in suggestions["productivity"]:
                st.markdown(
                    f'<div class="suggestion-item">{item}</div>',
                    unsafe_allow_html=True
                )

        with sug_c2:
            st.markdown(
                '<div class="emp-section-title">Quality Improvement Suggestions</div>',
                unsafe_allow_html=True
            )
            for item in suggestions["quality"]:
                st.markdown(
                    f'<div class="suggestion-item">{item}</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="emp-section-title">Workload Balance Suggestions</div>',
                unsafe_allow_html=True
            )
            for item in suggestions["workload"]:
                st.markdown(
                    f'<div class="suggestion-item">{item}</div>',
                    unsafe_allow_html=True
                )

# =================================================
# 5c. MODEL COMPARISON
# =================================================
elif page == "Model Comparison":

    st.header("Model Comparison")
    st.info(
        "Federated Learning in this project is simulated across department "
        "partitions using FedAvg. Each department acts as a client; local "
        "updates are averaged into a global neural network."
    )

    missing = []
    if not MODEL_PATH.exists():
        missing.append(MODEL_PATH.name)
    if not FEDERATED_MODEL_PATH.exists():
        missing.append(FEDERATED_MODEL_PATH.name)
    if not SCALER_PATH.exists():
        missing.append(SCALER_PATH.name)

    if missing:
        st.error(
            "Required model file(s) not found: " + ", ".join(missing)
        )
        st.stop()

    rf_model = load_model()
    federated_model = load_federated_model()
    scaler = load_scaler()

    X = df[FEATURES]
    y = df["performance_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_pred = rf_model.predict(X_test)
    fl_pred = federated_model.predict(
        scaler.transform(X_test), verbose=0
    ).ravel()

    comparison_df = pd.DataFrame({
        "Model": ["Random Forest", "Federated Neural Network"],
        "MAE": [
            mean_absolute_error(y_test, rf_pred),
            mean_absolute_error(y_test, fl_pred),
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, rf_pred)),
            np.sqrt(mean_squared_error(y_test, fl_pred)),
        ],
        "R2": [
            r2_score(y_test, rf_pred),
            r2_score(y_test, fl_pred),
        ],
    })

    comparison_display = comparison_df.copy()
    comparison_display["MAE"] = comparison_display["MAE"].round(2)
    comparison_display["RMSE"] = comparison_display["RMSE"].round(2)
    comparison_display["R2"] = comparison_display["R2"].round(3)

    st.subheader("Metrics Comparison")
    st.dataframe(comparison_display, width="stretch", hide_index=True)

    best_idx = comparison_df["MAE"].idxmin()
    best_model = comparison_df.loc[best_idx, "Model"]
    best_mae = comparison_df.loc[best_idx, "MAE"]
    st.success(
        f"Best model by test MAE: {best_model} "
        f"(MAE = {best_mae:.2f})"
    )

    metrics_long = comparison_df.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE", "R2"],
        var_name="Metric",
        value_name="Value"
    )

    fig = px.bar(
        metrics_long,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="MAE, RMSE, and R2 by Model"
    )
    st.plotly_chart(fig, width="stretch")

    pred_compare = pd.DataFrame({
        "Actual": y_test.values,
        "Random Forest": rf_pred,
        "Federated Neural Network": fl_pred,
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            pred_compare,
            x="Actual",
            y="Random Forest",
            title="Random Forest: Actual vs Predicted"
        )
        fig.add_shape(
            type="line",
            x0=pred_compare["Actual"].min(),
            y0=pred_compare["Actual"].min(),
            x1=pred_compare["Actual"].max(),
            y1=pred_compare["Actual"].max(),
            line=dict(dash="dash", color="gray")
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.scatter(
            pred_compare,
            x="Actual",
            y="Federated Neural Network",
            title="Federated NN: Actual vs Predicted"
        )
        fig.add_shape(
            type="line",
            x0=pred_compare["Actual"].min(),
            y0=pred_compare["Actual"].min(),
            x1=pred_compare["Actual"].max(),
            y1=pred_compare["Actual"].max(),
            line=dict(dash="dash", color="gray")
        )
        st.plotly_chart(fig, width="stretch")

# =================================================
# 5d. FEDERATED TRAINING HISTORY
# =================================================
elif page == "Federated Training History":

    st.header("Federated Training History")

    if not FEDERATED_HISTORY_PATH.exists():
        st.error(
            f"Training history file not found: {FEDERATED_HISTORY_PATH.name}"
        )
        st.stop()

    history_df = load_federated_history()

    st.subheader("Training Rounds")
    st.dataframe(history_df, width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rounds", len(history_df))
    c2.metric("Final MAE", round(float(history_df["mae"].iloc[-1]), 2))
    c3.metric("Final Loss", round(float(history_df["loss"].iloc[-1]), 2))

    fig = px.line(
        history_df,
        x="round",
        y="mae",
        markers=True,
        title="Federated Training MAE Across Rounds"
    )
    st.plotly_chart(fig, width="stretch")

    fig = px.line(
        history_df,
        x="round",
        y="loss",
        markers=True,
        title="Federated Training Loss Across Rounds"
    )
    st.plotly_chart(fig, width="stretch")

# =================================================
# 5e. DATA IMPORT & MODEL TRAINING
# =================================================
elif page == "Data Import & Model Training":

    st.header("Data Import & Model Training")
    st.caption(
        "Upload employee CSV data, manage the master dataset, and retrain "
        "Random Forest plus Federated Learning models safely."
    )

    meta_c1, meta_c2, meta_c3 = st.columns(3)
    meta_c1.metric("Active Dataset", active_data_path.name)
    meta_c2.metric("Dataset Size", len(df))
    meta_c3.metric(
        "Latest Training",
        training_meta.get("last_trained_at", "Not trained yet")
    )

    st.subheader("1) CSV Data Import Center")
    uploaded = st.file_uploader(
        "Upload employee performance CSV",
        type=["csv"],
        help="Drag and drop a CSV file with the required employee columns."
    )

    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Unable to read CSV: {exc}")
            st.stop()

        st.write(
            f"Uploaded file preview | Rows: {uploaded_df.shape[0]} | "
            f"Columns: {uploaded_df.shape[1]}"
        )
        st.dataframe(uploaded_df.head(20), width="stretch")

        validation = validate_employee_dataframe(uploaded_df)

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Rows", validation["rows"])
        v2.metric("Columns", validation["columns"])
        v3.metric("Duplicate Rows", validation["duplicate_rows"])
        v4.metric("Duplicate IDs", validation["duplicate_employee_ids"])

        if validation["missing_columns"]:
            st.error(
                "Missing required columns: "
                + ", ".join(validation["missing_columns"])
            )
        if validation["missing_values"]:
            st.error("Missing values detected:")
            st.json(validation["missing_values"])
        for warning in validation["warnings"]:
            st.warning(warning)
        for error in validation["errors"]:
            if "Missing required columns" not in error and "Missing values" not in error:
                st.error(error)

        if validation["ok"]:
            st.success("CSV validation passed.")
            st.subheader("Dataset Statistics")
            st.dataframe(
                uploaded_df[NUMERIC_COLUMNS].describe().T,
                width="stretch"
            )

            mode = st.radio(
                "How should this CSV update the master dataset?",
                ["Append to master dataset", "Replace master dataset"],
                horizontal=True
            )

            current_master = (
                pd.read_csv(MASTER_DATA_PATH)
                if MASTER_DATA_PATH.exists()
                else df.copy()
            )

            st.download_button(
                "Download safety backup of current working dataset",
                data=current_master.to_csv(index=False).encode("utf-8"),
                file_name="employee_dataset_backup.csv",
                mime="text/csv"
            )

            if st.button("Apply Dataset Update", type="primary"):
                try:
                    backup_path = backup_dataset(
                        current_master, "pre_import_backup"
                    )
                    incoming = uploaded_df[REQUIRED_COLUMNS].copy()

                    if mode.startswith("Append"):
                        combined = pd.concat(
                            [current_master[REQUIRED_COLUMNS], incoming],
                            ignore_index=True
                        )
                        combined = combined.drop_duplicates()
                        combined = combined.drop_duplicates(
                            subset=["employee_id"], keep="last"
                        )
                        action = "append"
                    else:
                        combined = incoming.drop_duplicates()
                        combined = combined.drop_duplicates(
                            subset=["employee_id"], keep="last"
                        )
                        action = "replace"

                    atomic_dataframe_to_csv(combined, MASTER_DATA_PATH)
                    clear_model_caches()
                    st.success(
                        f"Master dataset updated via {action}. "
                        f"Saved to `{MASTER_DATA_PATH.name}` "
                        f"({len(combined)} rows). "
                        f"Original `{DATA_PATH.name}` was not modified. "
                        f"Backup: `{backup_path.name}`."
                    )
                    st.info(
                        "CSV import does not train models. "
                        "Use Train / Update Models below when ready."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Dataset update failed: {exc}")

            st.download_button(
                "Download imported CSV",
                data=uploaded_df.to_csv(index=False).encode("utf-8"),
                file_name="imported_employee_data.csv",
                mime="text/csv",
                key="download_imported_csv"
            )
        else:
            st.error(
                "Validation failed. Fix the CSV issues before updating "
                "the master dataset."
            )

    st.divider()
    st.subheader("2) Master Dataset Management")

    master_exists = MASTER_DATA_PATH.exists()
    master_df = pd.read_csv(MASTER_DATA_PATH) if master_exists else None

    m1, m2 = st.columns(2)
    with m1:
        st.write(
            f"Master dataset present: {'Yes' if master_exists else 'No'}"
        )
        if master_exists:
            st.write(f"Master rows: {len(master_df)}")
            st.dataframe(master_df.head(10), width="stretch")
            st.download_button(
                "Download master dataset",
                data=master_df.to_csv(index=False).encode("utf-8"),
                file_name="employee_performance_master.csv",
                mime="text/csv",
                key="download_master_csv"
            )
        else:
            st.info(
                "No master dataset yet. Upload and apply a CSV to create "
                "`employee_performance_master.csv`."
            )

    with m2:
        st.write(f"Original dataset: `{DATA_PATH.name}` (read-only)")
        st.write(f"Original rows: {len(pd.read_csv(DATA_PATH))}")
        if st.button("Initialize master from original dataset"):
            try:
                original_df = pd.read_csv(DATA_PATH)
                atomic_dataframe_to_csv(original_df, MASTER_DATA_PATH)
                clear_model_caches()
                st.success(
                    "Master dataset initialized from original CSV. "
                    "Original file remains unchanged."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Initialization failed: {exc}")

    st.divider()
    st.subheader("3) Train / Update Models")
    st.warning(
        "Training will create backups of current models first. "
        "If training fails, existing models are preserved."
    )

    train_source = (
        MASTER_DATA_PATH if MASTER_DATA_PATH.exists() else active_data_path
    )
    st.caption(f"Training source: `{train_source.name}`")

    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        fl_rounds = st.number_input(
            "Federated rounds", min_value=1, max_value=50, value=8, step=1
        )
    with cfg2:
        local_epochs = st.number_input(
            "Local epochs per round", min_value=1, max_value=20, value=8, step=1
        )
    with cfg3:
        fl_learning_rate = st.number_input(
            "FL learning rate",
            min_value=0.0001,
            max_value=0.05,
            value=0.005,
            step=0.001,
            format="%.4f"
        )
    with cfg4:
        test_size = st.selectbox("Test size", options=[0.2, 0.25, 0.3], index=0)

    if st.button("Train / Update Models", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Validating training dataset...")
            train_df_raw = pd.read_csv(train_source)
            validation = validate_employee_dataframe(train_df_raw)
            if not validation["ok"]:
                raise ValueError(
                    "Training dataset validation failed: "
                    + " | ".join(validation["errors"])
                )

            train_df = prepare_training_frame(train_df_raw)
            progress.progress(8)

            X = train_df[FEATURES]
            y = train_df["performance_score"]
            departments = train_df["department"].astype(str)

            X_train, X_test, y_train, y_test, dept_train, _dept_test = (
                train_test_split(
                    X,
                    y,
                    departments,
                    test_size=float(test_size),
                    random_state=42
                )
            )

            st.write("Target diagnostics (train split)")
            st.json({
                "min": float(np.min(y_train)),
                "max": float(np.max(y_train)),
                "mean": float(np.mean(y_train)),
                "std": float(np.std(y_train))
            })

            # Capture before metrics from currently deployed models
            before_metrics = {"Random Forest": None, "Federated Neural Network": None}
            status.info("Evaluating current models (before)...")
            try:
                if MODEL_PATH.exists():
                    current_rf = joblib.load(MODEL_PATH)
                    before_metrics["Random Forest"] = evaluate_regression(
                        y_test, current_rf.predict(X_test)
                    )
                if FEDERATED_MODEL_PATH.exists() and SCALER_PATH.exists():
                    from tensorflow import keras
                    current_fl = keras.models.load_model(FEDERATED_MODEL_PATH)
                    current_scaler = joblib.load(SCALER_PATH)
                    before_pred = current_fl.predict(
                        current_scaler.transform(X_test), verbose=0
                    ).ravel()
                    before_metrics["Federated Neural Network"] = (
                        evaluate_regression(y_test, before_pred)
                    )
            except Exception:
                pass

            progress.progress(15)
            status.info("Creating model backups...")
            backup_path = backup_existing_artifacts("pre_retrain")

            status.info("Training Random Forest...")
            rf_model, rf_metrics, _rf_pred = train_random_forest(
                X_train, y_train, X_test, y_test
            )
            progress.progress(40)

            def fl_progress(round_idx, total_rounds, loss_value, mae_value):
                pct = 40 + int(50 * (round_idx / total_rounds))
                progress.progress(min(pct, 90))
                status.info(
                    f"Federated round {round_idx}/{total_rounds} | "
                    f"loss={loss_value:.4f} | mae={mae_value:.4f}"
                )

            status.info(
                "Training Federated Neural Network with department-wise FedAvg..."
            )
            (
                fl_model,
                fl_scaler,
                fl_metrics,
                history_df,
                fl_diagnostics
            ) = train_federated_model(
                X_train,
                y_train,
                dept_train,
                X_test,
                y_test,
                rounds=int(fl_rounds),
                local_epochs=int(local_epochs),
                learning_rate=float(fl_learning_rate),
                progress_callback=fl_progress
            )

            # Honest baseline comparison on the same untouched test split
            mean_baseline_pred = np.full(
                shape=len(y_test),
                fill_value=float(np.mean(y_train)),
                dtype=float
            )
            mean_metrics = evaluate_regression(y_test, mean_baseline_pred)
            fl_test_pred = fl_model.predict(
                fl_scaler.transform(pd.DataFrame(X_test, columns=FEATURES)),
                verbose=0
            ).ravel()

            status.info("Saving updated artifacts atomically...")
            atomic_joblib_dump(rf_model, MODEL_PATH)
            atomic_keras_save(fl_model, FEDERATED_MODEL_PATH)
            atomic_joblib_dump(fl_scaler, SCALER_PATH)
            atomic_dataframe_to_csv(history_df, FEDERATED_HISTORY_PATH)
            atomic_joblib_dump(fl_metrics, FEDERATED_METRICS_PATH)

            meta = {
                "last_trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_file": train_source.name,
                "dataset_rows": int(len(train_df)),
                "test_size": float(test_size),
                "fl_rounds": int(fl_rounds),
                "local_epochs": int(local_epochs),
                "fl_learning_rate": float(fl_learning_rate),
                "backup_dir": str(backup_path),
                "random_forest_metrics": rf_metrics,
                "federated_metrics": fl_metrics,
                "mean_baseline_metrics": mean_metrics,
                "fl_diagnostics": fl_diagnostics
            }
            save_training_meta(meta)

            clear_model_caches()
            progress.progress(100)
            status.success("Training completed successfully.")

            st.subheader("Training Completion Status")
            st.success(
                "Random Forest and Federated Neural Network were retrained "
                "and saved. Caches cleared so prediction pages use the "
                "updated models."
            )
            st.caption(f"Backup location: `{backup_path}`")

            st.subheader("Prediction Diagnostics")
            st.json({
                "target_train": fl_diagnostics["target_train"],
                "prediction_test": fl_diagnostics["prediction_test"],
                "output_bias_init": fl_diagnostics["output_bias_init"],
                "best_round": fl_diagnostics["best_round"],
                "client_sizes": fl_diagnostics["client_sizes"]
            })
            st.caption(
                f"Test prediction stats — min={float(np.min(fl_test_pred)):.2f}, "
                f"max={float(np.max(fl_test_pred)):.2f}, "
                f"mean={float(np.mean(fl_test_pred)):.2f}"
            )

            st.subheader("Model Comparison vs Mean Baseline")
            compare_rows = pd.DataFrame([
                {
                    "Model": "Mean Baseline",
                    "MAE": round(mean_metrics["MAE"], 3),
                    "RMSE": round(mean_metrics["RMSE"], 3),
                    "R2": round(mean_metrics["R2"], 3)
                },
                {
                    "Model": "Random Forest",
                    "MAE": round(rf_metrics["MAE"], 3),
                    "RMSE": round(rf_metrics["RMSE"], 3),
                    "R2": round(rf_metrics["R2"], 3)
                },
                {
                    "Model": "Federated Neural Network",
                    "MAE": round(fl_metrics["MAE"], 3),
                    "RMSE": round(fl_metrics["RMSE"], 3),
                    "R2": round(fl_metrics["R2"], 3)
                }
            ])
            st.dataframe(compare_rows, width="stretch", hide_index=True)

            st.subheader("Before vs After Model Performance")
            rows = []
            for model_name, before, after in [
                ("Random Forest", before_metrics["Random Forest"], rf_metrics),
                (
                    "Federated Neural Network",
                    before_metrics["Federated Neural Network"],
                    fl_metrics
                )
            ]:
                rows.append({
                    "Model": model_name,
                    "Before MAE": None if before is None else round(before["MAE"], 2),
                    "After MAE": round(after["MAE"], 2),
                    "Before RMSE": None if before is None else round(before["RMSE"], 2),
                    "After RMSE": round(after["RMSE"], 2),
                    "Before R2": None if before is None else round(before["R2"], 3),
                    "After R2": round(after["R2"], 3)
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            st.subheader("Updated Federated Training History")
            st.dataframe(history_df, width="stretch", hide_index=True)

        except Exception as exc:
            progress.progress(0)
            status.error("Training failed. Existing models were preserved.")
            st.exception(exc)

# =================================================
# 6. TRAINING RECOMMENDATION
# =================================================
elif page == "Training Recommendation":

    st.header("Training Recommendation")

    employee_id = st.selectbox(
        "Select Employee",
        df["employee_id"].tolist()
    )

    employee = df[
        df["employee_id"] == employee_id
    ].iloc[0]

    st.write("Department:", employee["department"])
    st.write("Job Role:", employee["job_role"])

    st.metric(
        "Task Completion",
        f"{employee['task_completion_rate']:.2f}%"
    )

    st.metric(
        "Quality Score",
        employee["quality_score"]
    )

    st.metric(
        "Training Hours",
        employee["training_hours"]
    )

    recommendations = []

    if employee["task_completion_rate"] < 70:
        recommendations.append(
            "Task Planning and Time Management Training"
        )

    if employee["quality_score"] < 70:
        recommendations.append(
            "Work Quality Improvement Training"
        )

    if employee["training_hours"] < 5:
        recommendations.append(
            "Skill Development Program"
        )

    if employee["workload_score"] > 80:
        recommendations.append(
            "Workload Management and Support"
        )

    if not recommendations:
        recommendations.append(
            "Advanced Skill Development and Career Growth"
        )

    st.subheader("Suggested Support")

    for recommendation in recommendations:
        st.info("📘 " + recommendation)

# =================================================
# 7. EMPLOYEE DATA + CSV DOWNLOAD
# =================================================
elif page == "Employee Data":

    st.header("Employee Data Management")

    departments = ["All"] + sorted(df["department"].unique())

    selected = st.selectbox(
        "Filter Department",
        departments
    )

    search = st.text_input("Search Employee ID")

    filtered = df.copy()

    if selected != "All":
        filtered = filtered[
            filtered["department"] == selected
        ]

    if search:
        filtered = filtered[
            filtered["employee_id"].astype(str).str.contains(
                search, case=False, na=False
            )
        ]

    st.write(f"Total Records: {len(filtered)}")

    st.dataframe(
        filtered,
        width="stretch"
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Employee CSV",
        data=csv_data,
        file_name="employee_performance_report.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "EMPperformance | Employee Performance Intelligence "
    "and Productivity Enhancement System"
)
