
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="EMPperformance",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "employee_performance_500_rows.csv"
MODEL_PATH = BASE_DIR / "employee_performance_model.pkl"

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

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not DATA_PATH.exists():
    st.error("Dataset CSV file not found!")
    st.stop()

df = load_data()

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
        "Training Recommendation",
        "Employee Data"
    ]
)

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
