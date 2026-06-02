"""Streamlit dashboard for Student Learning Segment Predictor.

File ini memuat dashboard prediksi segmentasi siswa berdasarkan model yang
tersimpan di student_segment_model_bundle.pkl.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# =============================================================================
# Konfigurasi halaman
# =============================================================================

APP_TITLE = "Student Learning Segment Predictor"
MODEL_PATH = Path("student_segment_model_bundle.pkl")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Styling dashboard
# =============================================================================

CUSTOM_CSS = """
<style>
    .main { background-color: #f7f8fb; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    .hero-card {
        background: linear-gradient(135deg, #152238 0%, #23395d 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 22px;
        margin-bottom: 20px;
        box-shadow: 0 12px 30px rgba(21, 34, 56, 0.18);
    }

    .hero-card h1 {
        font-size: 34px;
        margin-bottom: 6px;
    }

    .hero-card p {
        font-size: 16px;
        color: #d8e0ec;
        margin-bottom: 0;
    }

    .metric-card {
        background: white;
        border: 1px solid #e7e9ef;
        border-radius: 18px;
        padding: 20px 22px;
        min-height: 140px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        letter-spacing: .02em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #111827;
        font-size: 24px;
        font-weight: 700;
        line-height: 1.25;
    }

    .metric-caption {
        color: #64748b;
        font-size: 13px;
        margin-top: 8px;
        line-height: 1.45;
    }

    .note-box {
        background: #eef4ff;
        color: #1e3a5f;
        border-left: 5px solid #3b82f6;
        padding: 14px 16px;
        border-radius: 12px;
        margin: 12px 0 20px 0;
    }

    .section-title {
        color: #111827;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .small-muted {
        color: #6b7280;
        font-size: 13px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Label input dashboard
# =============================================================================

YES_NO_OPTIONS = ["yes", "no"]
YES_NO_LABELS = {"yes": "Ya", "no": "Tidak"}

EDU_LABELS = {
    0: "0 - Tidak ada",
    1: "1 - Pendidikan dasar",
    2: "2 - SMP",
    3: "3 - SMA",
    4: "4 - Pendidikan tinggi",
}

STUDYTIME_LABELS = {
    1: "1 - < 2 jam/minggu",
    2: "2 - 2 sampai 5 jam/minggu",
    3: "3 - 5 sampai 10 jam/minggu",
    4: "4 - > 10 jam/minggu",
}

FAILURES_LABELS = {
    0: "0 - Tidak ada",
    1: "1 kali",
    2: "2 kali",
    3: "3 kali atau lebih",
}


# =============================================================================
# Utility untuk mengambil isi bundle secara aman
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_bundle(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Load bundle model dari file pickle."""
    if not model_path.exists():
        st.error(
            "File student_segment_model_bundle.pkl tidak ditemukan. "
            "Pastikan file .pkl berada dalam folder yang sama dengan app.py."
        )
        st.stop()

    with model_path.open("rb") as file:
        return pickle.load(file)


def get_first_available(bundle: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Ambil nilai pertama dari beberapa kemungkinan nama key."""
    for key in keys:
        if key in bundle and bundle[key] is not None:
            return bundle[key]
    return default


bundle = load_bundle()

categorical_features = bundle["categorical_features"]
numeric_features = bundle["numeric_features"]
selected_features = bundle["selected_features"]

cluster_labels = get_first_available(bundle, "cluster_labels", "cluster_names", default={})
cluster_labels = {int(key): value for key, value in cluster_labels.items()}

cluster_descriptions = get_first_available(bundle, "cluster_descriptions", default={})
cluster_descriptions = {int(key): value for key, value in cluster_descriptions.items()}

metrics = get_first_available(bundle, "model_metrics", "metrics", default={})
classifier_model = get_first_available(bundle, "classification_model", "classifier_model")
regressor_model = get_first_available(bundle, "regression_model", "regressor_model")


# =============================================================================
# Preprocessing dan prediksi
# =============================================================================

def get_absences_bounds() -> Dict[str, float] | None:
    """Ambil batas capping absences dari bundle."""
    bounds = get_first_available(bundle, "absences_cap_bounds", "absences_cap")
    if bounds is not None:
        return bounds

    outlier_metrics = metrics.get("outlier", {})
    lower = outlier_metrics.get("absences_lower_bound")
    upper = outlier_metrics.get("absences_upper_bound")
    if lower is not None and upper is not None:
        return {"lower": lower, "upper": upper}

    return None


def apply_absences_capping(input_df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan capping IQR pada variabel absences sesuai preprocessing training."""
    df_input = input_df.copy()
    bounds = get_absences_bounds()

    if bounds is None or "absences" not in df_input.columns:
        return df_input

    lower = float(bounds["lower"])
    upper = float(bounds["upper"])
    df_input["absences"] = df_input["absences"].clip(lower=lower, upper=upper)

    return df_input


def make_kprototype_input(input_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Siapkan matriks input untuk model K-Prototypes."""
    processed_input = apply_absences_capping(input_df)

    numeric_scaled = pd.DataFrame(
        bundle["kproto_numeric_scaler"].transform(processed_input[numeric_features]),
        columns=numeric_features,
    )

    kprototype_input = pd.concat(
        [
            processed_input[categorical_features].astype(str).reset_index(drop=True),
            numeric_scaled.reset_index(drop=True),
        ],
        axis=1,
    )

    return kprototype_input.to_numpy(), processed_input


def get_prediction_confidence(model: Any, input_df: pd.DataFrame, predicted_class: int) -> float | None:
    """Hitung confidence prediksi classification jika model mendukung predict_proba."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(input_df)[0]
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        classes = getattr(final_step, "classes_", None)

    if classes is None:
        return float(np.max(probabilities))

    class_probability = dict(zip([int(class_id) for class_id in classes], probabilities))
    return float(class_probability.get(predicted_class, np.max(probabilities)))


def predict_student(input_row: Dict[str, Any]) -> Dict[str, Any]:
    """Prediksi segmentasi siswa menggunakan K-Prototypes, classification, dan regression."""
    input_df = pd.DataFrame([input_row])[selected_features]
    kprototype_matrix, processed_input = make_kprototype_input(input_df)

    kprototype_cluster = int(
        bundle["kproto_model"].predict(
            kprototype_matrix,
            categorical=bundle["kproto_categorical_index"],
        )[0]
    )

    classification_cluster = int(classifier_model.predict(processed_input)[0])
    confidence = get_prediction_confidence(
        classifier_model,
        processed_input,
        classification_cluster,
    )

    regression_score = float(regressor_model.predict(processed_input)[0])
    regression_cluster = 1 if regression_score >= 0.5 else 0

    return {
        "processed_input": processed_input,
        "kprototype_cluster": kprototype_cluster,
        "classification_cluster": classification_cluster,
        "confidence": confidence,
        "regression_score": regression_score,
        "regression_cluster": regression_cluster,
    }


# =============================================================================
# Fungsi visualisasi
# =============================================================================

def plot_cluster_distribution() -> plt.Figure:
    """Visualisasi jumlah siswa pada setiap cluster K-Prototypes."""
    df_result = bundle["df_result"]
    counts = df_result["cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Distribusi Cluster K-Prototypes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah siswa")
    ax.tick_params(axis="x", rotation=0)
    return fig


def plot_correlation_heatmap() -> plt.Figure:
    """Visualisasi heatmap korelasi."""
    correlation_matrix = get_first_available(
        bundle,
        "correlation_selected",
        "correlation_matrix",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Heatmap Korelasi Variabel Numerik")
    return fig


def plot_pca_cluster(cluster_col: str) -> plt.Figure:
    """Visualisasi sebaran cluster menggunakan PCA."""
    pca_df = get_first_available(bundle, "pca_cluster_points", "pca_df").copy()

    if cluster_col == "cluster_kproto" and cluster_col not in pca_df.columns:
        cluster_col = "cluster"
    if cluster_col == "cluster_kmeans" and cluster_col not in pca_df.columns:
        cluster_col = "kmeans_cluster"

    title = (
        "Visualisasi PCA K-Prototypes"
        if cluster_col in {"cluster_kproto", "cluster"}
        else "Visualisasi PCA K-Means Pembanding"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=cluster_col,
        s=55,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    return fig


def plot_confusion_matrix() -> plt.Figure:
    """Visualisasi confusion matrix classification."""
    cm = np.array(metrics["classification"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix Classification")
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("Actual cluster")
    return fig


# =============================================================================
# Komponen tampilan
# =============================================================================

def render_hero() -> None:
    """Tampilkan header utama dashboard."""
    st.markdown(
        """
        <div class="hero-card">
            <h1>🎓 Student Learning Segment Predictor</h1>
            <p>Dashboard segmentasi siswa berdasarkan dukungan pendidikan,
            motivasi belajar, kedisiplinan, dan riwayat akademik.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(label: str, value: str, caption: str) -> None:
    """Tampilkan card metrik custom."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Tuple[str, Dict[str, Any], bool]:
    """Sidebar menu dan form input siswa."""
    with st.sidebar:
        st.markdown("### Input Data Siswa")

        page = st.radio(
            "Menu Dashboard",
            ["Prediksi Segmentasi", "Analisis Data", "Evaluasi Model"],
            label_visibility="collapsed",
        )

        st.divider()

        with st.form("student_input_form"):
            schoolsup = st.radio(
                "Dukungan tambahan dari sekolah",
                YES_NO_OPTIONS,
                format_func=lambda value: YES_NO_LABELS[value],
                horizontal=True,
            )
            famsup = st.radio(
                "Dukungan pendidikan dari keluarga",
                YES_NO_OPTIONS,
                format_func=lambda value: YES_NO_LABELS[value],
                horizontal=True,
            )
            paid = st.radio(
                "Mengikuti kelas tambahan berbayar",
                YES_NO_OPTIONS,
                format_func=lambda value: YES_NO_LABELS[value],
                horizontal=True,
            )
            internet = st.radio(
                "Akses internet di rumah",
                YES_NO_OPTIONS,
                format_func=lambda value: YES_NO_LABELS[value],
                horizontal=True,
            )
            higher = st.radio(
                "Ingin melanjutkan pendidikan tinggi",
                YES_NO_OPTIONS,
                index=0,
                format_func=lambda value: YES_NO_LABELS[value],
                horizontal=True,
            )

            medu = st.selectbox(
                "Pendidikan ibu",
                list(EDU_LABELS.keys()),
                format_func=lambda value: EDU_LABELS[value],
                index=4,
            )
            fedu = st.selectbox(
                "Pendidikan ayah",
                list(EDU_LABELS.keys()),
                format_func=lambda value: EDU_LABELS[value],
                index=3,
            )
            studytime = st.selectbox(
                "Waktu belajar mingguan",
                list(STUDYTIME_LABELS.keys()),
                format_func=lambda value: STUDYTIME_LABELS[value],
                index=1,
            )
            absences = st.number_input(
                "Jumlah ketidakhadiran",
                min_value=0,
                max_value=100,
                value=4,
                step=1,
            )
            failures = st.selectbox(
                "Jumlah kegagalan kelas sebelumnya",
                list(FAILURES_LABELS.keys()),
                format_func=lambda value: FAILURES_LABELS[value],
                index=0,
            )

            submitted = st.form_submit_button("Analisis Siswa", use_container_width=True)

    input_row = {
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "internet": internet,
        "higher": higher,
        "Medu": int(medu),
        "Fedu": int(fedu),
        "studytime": int(studytime),
        "absences": float(absences),
        "failures": int(failures),
    }

    return page, input_row, submitted


def render_prediction_page(input_row: Dict[str, Any], submitted: bool) -> None:
    """Halaman prediksi segmentasi siswa."""
    st.markdown(
        '<h3 class="section-title">Prediksi Segmentasi Siswa</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="note-box">
            Model utama menggunakan K-Prototypes. Model classification digunakan
            untuk memprediksi label cluster, sedangkan regression digunakan sebagai
            skor kecenderungan cluster.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if submitted or "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = predict_student(input_row)
        st.session_state["last_input_row"] = input_row

    result = st.session_state["prediction_result"]
    last_input = st.session_state["last_input_row"]

    kprototype_cluster = result["kprototype_cluster"]
    classification_cluster = result["classification_cluster"]
    regression_cluster = result["regression_cluster"]
    regression_score = result["regression_score"]
    confidence = result["confidence"]

    confidence_text = "Tidak tersedia"
    if confidence is not None:
        confidence_text = f"{confidence * 100:.2f}%"

    col1, col2, col3 = st.columns(3)

    with col1:
        render_prediction_card(
            label="Segmentasi utama",
            value=f"Cluster {kprototype_cluster}",
            caption=(
                f"<b>{cluster_labels.get(kprototype_cluster, 'Cluster tidak dikenal')}</b>"
                f"<br>{cluster_descriptions.get(kprototype_cluster, '')}"
            ),
        )

    with col2:
        render_prediction_card(
            label="Classification model",
            value=f"Cluster {classification_cluster}",
            caption=(
                f"Confidence: <b>{confidence_text}</b>"
                f"<br>{cluster_labels.get(classification_cluster, '')}"
            ),
        )

    with col3:
        render_prediction_card(
            label="Regression score",
            value=f"{regression_score:.3f}",
            caption=(
                f"Skor mendekati Cluster {regression_cluster}. Regression ini "
                "bukan nilai akademik, tetapi skor kecenderungan cluster."
            ),
        )

    st.markdown("### Data input yang diproses")
    shown_input = result["processed_input"].copy()
    shown_input[categorical_features] = shown_input[categorical_features].replace(YES_NO_LABELS)
    st.dataframe(shown_input, use_container_width=True)

    cap_bounds = get_absences_bounds() or {}
    upper_bound = cap_bounds.get("upper", np.inf)
    if last_input["absences"] > upper_bound:
        st.info(
            f"Nilai absences {last_input['absences']} diproses menggunakan "
            f"capping IQR menjadi {upper_bound:.2f}, sesuai preprocessing saat training."
        )


def render_analysis_page() -> None:
    """Halaman analisis data, cluster, korelasi, outlier, dan PCA."""
    st.markdown(
        '<h3 class="section-title">Analisis Data dan Hasil Clustering</h3>',
        unsafe_allow_html=True,
    )

    identification = get_first_available(bundle, "data_identification", default={})
    metrics_identification = metrics.get("data_identification", {})

    row_count = identification.get("row_count", metrics_identification.get("record_count", "-"))
    column_count = identification.get("column_count", metrics_identification.get("variable_count", "-"))
    missing_total = identification.get("missing_total", metrics_identification.get("missing_total", "-"))
    duplicate_count = identification.get("duplicate_count", metrics_identification.get("duplicate_total_after_drop", "-"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah record", row_count)
    col2.metric("Jumlah variabel", column_count)
    col3.metric("Missing value", missing_total)
    col4.metric("Duplikasi", duplicate_count)

    st.markdown("### Profil Cluster K-Prototypes")
    profile_display = bundle["cluster_profile"].copy()
    profile_display.insert(
        0,
        "nama_cluster",
        [cluster_labels.get(int(index), "") for index in profile_display.index],
    )
    st.dataframe(profile_display, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(plot_cluster_distribution(), clear_figure=True)
    with col_b:
        st.pyplot(plot_correlation_heatmap(), clear_figure=True)

    st.markdown("### Outlier Report")
    outlier_before = get_first_available(bundle, "outlier_report_before", "outlier_before")
    outlier_after = get_first_available(bundle, "outlier_report_after", "outlier_after")

    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.caption("Sebelum capping")
        st.dataframe(outlier_before, use_container_width=True)
    with col_out2:
        st.caption("Sesudah capping")
        st.dataframe(outlier_after, use_container_width=True)

    st.markdown("### Visualisasi PCA")
    col_pca1, col_pca2 = st.columns(2)
    with col_pca1:
        st.pyplot(plot_pca_cluster("cluster_kproto"), clear_figure=True)
    with col_pca2:
        st.pyplot(plot_pca_cluster("cluster_kmeans"), clear_figure=True)


def render_evaluation_page() -> None:
    """Halaman evaluasi classification, regression, dan pembanding clustering."""
    st.markdown(
        '<h3 class="section-title">Evaluasi Model</h3>',
        unsafe_allow_html=True,
    )

    classification_metrics = metrics["classification"]
    regression_metrics = metrics["regression"]
    clustering_metrics = get_first_available(
        metrics,
        "clustering_comparison",
        "clustering",
        default={},
    )

    st.markdown("### Classification")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{classification_metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{classification_metrics['precision_weighted']:.4f}")
    c3.metric("Recall", f"{classification_metrics['recall_weighted']:.4f}")
    c4.metric("F1-score", f"{classification_metrics['f1_weighted']:.4f}")

    st.pyplot(plot_confusion_matrix(), clear_figure=True)

    st.markdown("### Regression")
    mae = get_first_available(regression_metrics, "mae", "MAE")
    mse = get_first_available(regression_metrics, "mse", "MSE")
    rmse = get_first_available(regression_metrics, "rmse", "RMSE")
    r2_score = get_first_available(regression_metrics, "r2", "R2")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("MAE", f"{mae:.4f}")
    r2.metric("MSE", f"{mse:.4f}")
    r3.metric("RMSE", f"{rmse:.4f}")
    r4.metric("R²", f"{r2_score:.4f}")
    st.caption(
        "Regression menggunakan target label cluster, sehingga hasilnya ditafsirkan "
        "sebagai skor kecenderungan cluster, bukan nilai akademik siswa."
    )

    st.markdown("### Perbandingan K-Prototypes dan K-Means")
    silhouette_kproto = get_first_available(
        clustering_metrics,
        "silhouette_kprototypes_proxy",
        "silhouette_kprototypes_on_encoded_space",
    )
    silhouette_kmeans = clustering_metrics.get("silhouette_kmeans")
    adjusted_rand = clustering_metrics.get("adjusted_rand_index")

    k1, k2, k3 = st.columns(3)
    k1.metric("Silhouette K-Prototypes", f"{silhouette_kproto:.4f}")
    k2.metric("Silhouette K-Means", f"{silhouette_kmeans:.4f}")
    k3.metric("Adjusted Rand Index", f"{adjusted_rand:.4f}")

    st.markdown("#### Tabel silang cluster")
    comparison_table = get_first_available(
        bundle,
        "cluster_comparison",
        "comparison_table",
    )
    st.dataframe(comparison_table, use_container_width=True)


# =============================================================================
# Main app
# =============================================================================

page, input_row, submitted = render_sidebar()
render_hero()

if page == "Prediksi Segmentasi":
    render_prediction_page(input_row, submitted)
elif page == "Analisis Data":
    render_analysis_page()
elif page == "Evaluasi Model":
    render_evaluation_page()

st.markdown(
    '<div class="small-muted">Model utama: K-Prototypes. Pembanding: K-Means. '
    "Target classification dan regression berasal dari hasil clustering.</div>",
    unsafe_allow_html=True,
)
