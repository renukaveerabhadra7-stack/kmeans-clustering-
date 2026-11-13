"""
Energy Cluster Predictor
Machine Learning: K-Means Clustering
Framework: Streamlit

Run:
python -m streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Energy Cluster Predictor",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# =====================================================
# 2. FILE HANDLING
# =====================================================

def find_file(filename):

    possible_paths = [
        BASE_DIR / filename,
        BASE_DIR / "model" / filename,
        BASE_DIR / "data" / filename
    ]

    for path in possible_paths:
        if path.is_file():
            return path

    st.error(f"❌ Required file not found: {filename}")

    st.info(
        "Please check your GitHub repository. "
        "The required file must be uploaded."
    )

    st.write("Files detected in your project:")

    for path in BASE_DIR.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            st.write(str(path.relative_to(BASE_DIR)))

    st.stop()


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    model_path = find_file("kmeans_model.joblib")

    saved = joblib.load(model_path)

    model = saved["model"]
    scaler = saved["scaler"]
    features = saved["features"]

    return model, scaler, features


try:

    model, scaler, FEATURES = load_model()

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# =====================================================
# 4. LOAD DATASET
# =====================================================

@st.cache_data
def load_data():

    clustered_path = BASE_DIR / "energy_clustered.csv"

    if not clustered_path.is_file():
        clustered_path = BASE_DIR / "data" / "energy_clustered.csv"

    if clustered_path.is_file():

        data = pd.read_csv(clustered_path)

    else:

        data = pd.read_csv(find_file("energy_data.csv"))

        data["cluster"] = model.predict(
            scaler.transform(data[FEATURES])
        )

    return data


try:

    data = load_data()

    # Validate dataset
    required_columns = FEATURES

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        st.error(f"Missing dataset columns: {missing_columns}")
        st.stop()

    if "cluster" not in data.columns:
        data["cluster"] = model.predict(
            scaler.transform(data[FEATURES])
        )

except Exception as e:

    st.error(f"Dataset loading failed: {e}")
    st.stop()


# =====================================================
# 5. CLUSTER INTERPRETATION
# =====================================================

centres = pd.DataFrame(
    scaler.inverse_transform(model.cluster_centers_),
    columns=FEATURES
)

avg_energy = data["energy_consumption"].mean()
avg_temp = data["temperature"].mean()

COLORS = [
    "#00B894",
    "#0984E3",
    "#FDCB6E",
    "#E17055",
    "#6C5CE7",
    "#FD79A8"
]

names = {}
advice = {}

for i, row in centres.iterrows():

    energy_level = (
        "High"
        if row["energy_consumption"] > avg_energy
        else "Low"
    )

    weather = (
        "Hot"
        if row["temperature"] > avg_temp
        else "Cool"
    )

    names[i] = f"{energy_level} Usage · {weather} Weather"

    advice[i] = (
        "Heavy energy consumption detected. "
        "Consider shifting usage away from peak hours "
        "and using energy-efficient equipment."
        if energy_level == "High"
        else
        "Lower energy consumption detected. "
        "This represents relatively efficient energy usage."
    )

data["segment"] = data["cluster"].map(names)


# =====================================================
# 6. CUSTOM PAGE DESIGN
# =====================================================

st.markdown("""
<style>

.hero {
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    background: linear-gradient(
        120deg,
        #00B894,
        #0984E3 60%,
        #6C5CE7
    );
}

.hero h1 {
    margin: 0;
    font-size: 36px;
}

.hero p {
    margin-top: 10px;
    font-size: 16px;
}

.result {
    padding: 25px;
    border-radius: 16px;
    background: #1A1F2B;
    border-left: 7px solid var(--c);
}

.result h2 {
    margin: 0;
    color: var(--c);
}

.result p {
    color: #D0D6E0;
}

</style>

<div class="hero">

<h1>⚡ Energy Cluster Predictor</h1>

<p>
Analyze energy consumption patterns using
Machine Learning and K-Means Clustering.
</p>

</div>

""", unsafe_allow_html=True)


# =====================================================
# 7. SIDEBAR INPUT
# =====================================================

st.sidebar.header("⚙️ Enter Energy Details")

energy = st.sidebar.slider(
    "Energy Consumption (kWh)",
    min_value=50.0,
    max_value=500.0,
    value=275.0,
    step=1.0
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=15.0,
    max_value=35.0,
    value=25.0,
    step=0.1
)

st.sidebar.caption(
    "Adjust the values to predict the energy cluster."
)


# =====================================================
# 8. SINGLE PREDICTION
# =====================================================

new_data = pd.DataFrame(
    [[energy, temperature]],
    columns=FEATURES
)

new_scaled = scaler.transform(new_data)

cluster = int(model.predict(new_scaled)[0])

color = COLORS[cluster % len(COLORS)]


# =====================================================
# 9. DISPLAY PREDICTION
# =====================================================

left, right = st.columns([1, 1.4])

with left:

    st.subheader("🎯 Prediction Result")

    st.markdown(
        f"""
        <div class="result" style="--c:{color}">

        <h2>Cluster {cluster}</h2>

        <h3>{names[cluster]}</h3>

        <p>{advice[cluster]}</p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2 = st.columns(2)

    c1.metric(
        "Energy Consumption",
        f"{energy:.1f} kWh",
        f"{energy - avg_energy:+.1f} vs average"
    )

    c2.metric(
        "Temperature",
        f"{temperature:.1f} °C",
        f"{temperature - avg_temp:+.1f} vs average"
    )

    # Distance calculation
    distances = (
        (new_scaled - model.cluster_centers_) ** 2
    ).sum(axis=1) ** 0.5

    st.subheader("Distance from Cluster Centres")

    distance_df = pd.DataFrame({
        "Cluster": [
            f"Cluster {i}"
            for i in range(len(distances))
        ],
        "Distance": distances
    })

    st.bar_chart(
        distance_df.set_index("Cluster")
    )

    st.caption(
        "A smaller distance means your reading "
        "is closer to that cluster centre."
    )


# =====================================================
# 10. CLUSTER VISUALIZATION
# =====================================================

with right:

    st.subheader("📊 Energy Consumption Clusters")

    plot_data = (
        data.sample(
            min(2500, len(data)),
            random_state=1
        )
    )

    fig = px.scatter(
        plot_data,
        x="temperature",
        y="energy_consumption",
        color="segment",
        opacity=0.55,
        template="plotly_dark",
        labels={
            "temperature": "Temperature (°C)",
            "energy_consumption": "Energy Consumption (kWh)",
            "segment": "Energy Group"
        },
        title="K-Means Clustering Visualization"
    )

    # User prediction
    fig.add_trace(
        go.Scatter(
            x=[temperature],
            y=[energy],
            mode="markers",
            name="Your Input",
            marker=dict(
                size=20,
                color="white",
                symbol="star",
                line=dict(
                    width=2,
                    color="black"
                )
            )
        )
    )

    # Cluster centres
    fig.add_trace(
        go.Scatter(
            x=centres["temperature"],
            y=centres["energy_consumption"],
            mode="markers",
            name="Cluster Centres",
            marker=dict(
                size=15,
                color="red",
                symbol="x"
            )
        )
    )

    fig.update_layout(
        height=500,
        legend_title_text="",
        margin=dict(
            l=0,
            r=0,
            t=40,
            b=0
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =====================================================
# 11. BATCH PREDICTION USING CSV
# =====================================================

st.divider()

st.subheader("📁 Bulk Energy Cluster Prediction")

st.write(
    "Upload a CSV file containing energy consumption "
    "and temperature values to predict clusters for "
    "multiple readings."
)

st.caption(
    "Required columns: energy_consumption, temperature"
)

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df_in = pd.read_csv(uploaded_file)

        if set(FEATURES).issubset(df_in.columns):

            df_in["cluster"] = model.predict(
                scaler.transform(df_in[FEATURES])
            )

            df_in["segment"] = df_in["cluster"].map(names)

            st.success(
                "Predictions generated successfully!"
            )

            st.dataframe(
                df_in,
                use_container_width=True
            )

            csv_output = df_in.to_csv(index=False)

            st.download_button(
                label="⬇️ Download Predictions",
                data=csv_output,
                file_name="energy_cluster_predictions.csv",
                mime="text/csv"
            )

        else:

            st.error(
                f"Your CSV must contain these columns: {FEATURES}"
            )

    except Exception as e:

        st.error(f"CSV processing error: {e}")


# =====================================================
# 12. CLUSTER SUMMARY
# =====================================================

st.divider()

with st.expander("📋 View Cluster Summary"):

    summary = (
        data.groupby("segment")
        .agg(
            readings=("cluster", "size"),
            avg_energy=("energy_consumption", "mean"),
            avg_temp=("temperature", "mean")
        )
        .round(2)
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.subheader("Cluster Distribution")

    cluster_counts = (
        data["segment"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Readings"
    ]

    fig2 = px.pie(
        cluster_counts,
        names="Cluster",
        values="Readings",
        hole=0.4,
        title="Energy Cluster Distribution"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">

    ⚡ Energy Cluster Predictor

    <br>

    Powered by Machine Learning | K-Means Clustering

    </div>
    """,
    unsafe_allow_html=True
)
