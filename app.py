import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import FitnessPipeline
from src.a4_recommendations.visualizer import (
    plot_ctl_atl_tsb, plot_steps_forecast, plot_readiness_distribution,
    plot_cluster_scatter, plot_model_comparison, plot_feature_importance,
    plot_confusion_matrix,
)
from src.a4_recommendations.recommender import format_recommendation_text
from src.a4_recommendations.report_generator import generate_report
st.set_page_config(
    page_title="TrainAI",
    page_icon="https://img.icons8.com/color/96/combo-chart.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.metric-card {
    background: #F0F0F0;
    border: 1px solid #3B444B;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 700;
    color: #666552;
}
.metric-card .lbl {
    font-size: 0.82rem;
    color: #666552;
    margin-top: 2px;
}
.readiness-high   { color: #10B981; font-weight: 700; font-size: 1.2rem; }
.readiness-medium { color: #F59E0B; font-weight: 700; font-size: 1.2rem; }
.readiness-low    { color: #EF4444; font-weight: 700; font-size: 1.2rem; }
.rec-box {
    background: #fbf9db;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.95rem;
    white-space: pre-line;
    color: #3B444B;
}
/* ─── TAB BASE STYLE ─── */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #3B444B !important;   
    font-weight: 500;
    border-radius: 10px;
    transition: all 0.2s ease-in-out;
}

/* ─── HOVER (замість червоного) ─── */
button[data-baseweb="tab"]:hover {
    background: #E0E0E0 !important;
    color: #3B444B !important;
    transform: translateY(-1px);
}

/* ─── ACTIVE TAB ─── */
button[data-baseweb="tab"][aria-selected="true"] {
    background: #fbf9db !important;
    padding: 5px 7px;
    color: #3B444B !important;
}
button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span {
    font-size: 16px !important;
    font-weight: 700 !important;
}
/* ─── ACTIVE UNDERLINE (замість червоної) ─── */
div[role="tablist"] > div {
    background-color: #3B444B !important;
}

button[data-baseweb="tab"][aria-selected="true"]::after {
    content: "";
    position: absolute;
    bottom: 0px;
    height: 2px;
    background: #3B444B !important;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)


if "pipeline" not in st.session_state:
    st.session_state.pipeline = FitnessPipeline()
if "trained" not in st.session_state:
    st.session_state.trained = False

pipeline: FitnessPipeline = st.session_state.pipeline

st.markdown("""
<style>

/* Sidebar фон */
section[data-testid="stSidebar"] {
    background-color: #fbf9db; 
    color: #3B444B;
}

/* Текст у sidebar */
section[data-testid="stSidebar"] * {
    color: #3B444B !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stButton > button {
    background-color:  #F8F8F8;
    color:  #3B444B !important;
    border-radius: 10px;
    border: none;
    padding: 10px 16px;
    font-weight: 500;
}

.stButton > button:hover {
    background-color: #E0E0E0
}
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/128/10/10699.png", width=60)
    st.title("TrainAI")
    st.caption("Аналіз і прогнозування ефективності тренувань")
    st.divider()
    st.image("https://cdn-icons-png.flaticon.com/128/8903/8903498.png", width=30)

    uploaded = st.file_uploader(
        "Завантажте CSV-датасет", type=["csv"],
        help="Очікувані колонки: user_id, date, age, gender, height_cm, weight_kg, "
             "steps, calories_burned, sleep_hours, heart_rate_avg, workout_type, "
             "workout_duration_minutes, water_intake_liters, stress_level, mood",
    )

    if uploaded:
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        save_path = data_dir / "uploaded.csv"
        save_path.write_bytes(uploaded.read())

        if st.button("Запустити аналіз", type="primary", use_container_width=True):
            with st.spinner("Обробка даних та навчання моделей..."):
                try:
                    pipeline.run_full_pipeline(str(save_path))
                    st.session_state.trained = True
                    st.markdown("""
                    <div style="
                        background: #F5F5F7;
                        color: #111827;
                        padding: 12px 14px;
                        font-size: 14px;
                    ">
                    ✔ Аналіз завершено
                    </div>
                    """, unsafe_allow_html=True)                
                except Exception as e:
                    st.error(f"✖ Помилка: {e}")
                    st.exception(e)

    if st.session_state.trained:
        st.divider()
        st.image("https://cdn-icons-png.flaticon.com/128/10628/10628940.png", width=30)
        st.subheader("Вибір користувача")
        user_list = pipeline.get_user_list()
        selected_user = st.selectbox("User ID:", user_list)
        st.session_state.selected_user = selected_user


st.markdown("""
<style>
.stApp {
    background-color: #F8F8F8;
    color: #3B444B !important;
}
</style>
""", unsafe_allow_html=True)
if not st.session_state.trained:
    st.title("Система аналізу та прогнозування ефективності спортивних тренувань")
    st.markdown("""
    ### Як розпочати роботу:
    1. Завантажте CSV-файл датасету через панель ліворуч
    2. Натисніть **«Запустити аналіз»**
    3. Оберіть користувача для перегляду персонального аналізу

    ### Можливості системи:
    - Завантаження, валідація та очищення даних 
    - Розрахунок CTL, ATL, TSB та рівня готовності 
    - Прогнозування кроків 
    - Класифікація рівня готовності 
    - Кластеризація тренувальної поведінки 
    - Рекомендації, дашборд, PDF-звіт 
    """)
    st.markdown("   ")
    st.markdown("""
    <div style="
        background: #E8E8E8;
        border-radius: 12px;
        padding: 14px 16px;
        color: #3B444B;
        font-size: 0.95rem;
    ">
    Завантажте датасет у лівій панелі, щоб розпочати.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


user_id = st.session_state.get("selected_user", pipeline.get_user_list()[0])
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Дашборд", "Прогноз та рекомендація",
    "Моделі", "Кластери", "Звіт"
])

user_df   = pipeline.get_user_dataframe(user_id)
user_pred = pipeline.predict_for_user(user_id)

with tab1:
    st.markdown(
        f"""
        <h2>
            <img src="https://cdn-icons-png.flaticon.com/128/14093/14093022.png"
                width="30"
                style="vertical-align: middle; margin-right: 8px;">
            Дашборд - Користувач {user_id}
        </h2>
        """,
        unsafe_allow_html=True
    )

    last = user_df.iloc[-1] if not user_df.empty else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "CTL",    f"{float(last.get('ctl', 0)):.1f}",    "Хронічне навантаження"),
        (c2, "ATL",    f"{float(last.get('atl', 0)):.1f}",    "Гостре навантаження"),
        (c3, "TSB",    f"{float(last.get('tsb', 0)):.1f}",    "Баланс"),
        (c4, "Кроки",  f"{int(last.get('steps', 0)):,}",      "Вчора"),
        (c5, "Сон",    f"{float(last.get('sleep_hours', 0)):.1f} год", "Вчора"),
    ]
    for col, title, val, sub in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val">{val}</div>'
                f'<div class="lbl">{title}<br><small>{sub}</small></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.plotly_chart(
        plot_ctl_atl_tsb(user_df, user_id, last_n_days=90),
        use_container_width=True,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(
            plot_readiness_distribution(user_df),
            use_container_width=True,
        )
    with col_r:
        st.subheader("Останні 14 днів")
        show_cols = ["date", "steps", "sleep_hours", "heart_rate_avg",
                     "ctl", "atl", "tsb", "readiness_level"]
        avail = [c for c in show_cols if c in user_df.columns]
        st.dataframe(
            user_df[avail].tail(14).sort_values("date", ascending=False)
            .reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

with tab2:
    st.markdown(
        f"""
        <h2>
            <img src="https://cdn-icons-png.flaticon.com/128/681/681575.png"
                width="22"
                style="vertical-align: middle; margin-right: 8px;">
            Прогноз та рекомендація - Користувач {user_id}
        </h2>
        """,
        unsafe_allow_html=True
    )
    reg  = user_pred.get("regression", {})
    clf  = user_pred.get("classification", {})
    rec  = user_pred.get("recommendation", {})

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Прогноз кроків на завтра")
        pred_steps = reg.get("predicted_steps", 0)
        ci_lo = reg.get("ci_low", 0)
        ci_hi = reg.get("ci_high", 0)
        st.metric(
            "Прогноз", f"{pred_steps:,} кроків",
            help=f"Довірчий інтервал: {ci_lo:,} – {ci_hi:,}",
        )
        st.caption(f"Довірчий інтервал: **{ci_lo:,} – {ci_hi:,}** кроків")
        st.plotly_chart(
            plot_steps_forecast(user_df, user_id, pred_steps, ci_lo, ci_hi),
            use_container_width=True,
        )

    with col_b:
        st.subheader("Рівень готовності")
        rl = clf.get("readiness_level", "Середня")
        css_class = {
            "Висока": "readiness-high",
            "Середня": "readiness-medium",
            "Низька": "readiness-low",
        }.get(rl, "readiness-medium")
        st.markdown(f'<span class="{css_class}">{rl}</span>', unsafe_allow_html=True)

        proba = clf.get("probabilities", {})
        if proba:
            st.bar_chart(pd.Series(proba, name="Ймовірність"))

        st.subheader("Поточні показники навантаження")
        st.metric("TSB", f"{user_pred.get('current_tsb', 0):.2f}")
        st.metric("CTL", f"{user_pred.get('current_ctl', 0):.2f}")
        st.metric("ATL", f"{user_pred.get('current_atl', 0):.2f}")

    st.divider()
    st.subheader("Рекомендація щодо наступного тренування")
    rec_text = format_recommendation_text(rec)
    st.markdown(f'<div class="rec-box">{rec_text}</div>', unsafe_allow_html=True)

with tab3:
    st.markdown(
        f"""
        <h2>
            <img src="https://cdn-icons-png.flaticon.com/128/4616/4616895.png"
                width="22"
                style="vertical-align: middle; margin-right: 8px;">
            Результати навчання моделей - Користувач {user_id}
        </h2>
        """,
        unsafe_allow_html=True
    )
    reg_res = pipeline.regression_results
    clf_res = pipeline.classification_results

    st.subheader("Регресія (прогноз кроків)")
    st.markdown(
        f"""
        <div style="
            background-color: #fbf9db; 
            color: #3B444B;
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 20px;
        ">
            <b>Найкраща модель:</b> {reg_res['best_name']}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("   ")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_model_comparison(reg_res["results"], "regression"),
            use_container_width=True,
        )
    with col2:
        if reg_res.get("feature_importance") is not None:
            st.plotly_chart(
                plot_feature_importance(reg_res["feature_importance"]),
                use_container_width=True,
            )

    metrics_data = []
    for name, r in reg_res["results"].items():
        metrics_data.append({
            "Модель": name, "MAE": r["mae"], "RMSE": r["rmse"],
            "R²": r["r2"],
        })
    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Класифікація (рівень готовності)")
    st.markdown(
        f"""
        <div style="
            background-color: #fbf9db; 
            color: #3B444B;
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 20px;
        ">
            <b>Найкраща модель:</b> {clf_res['best_name']}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("   ")
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            plot_model_comparison(clf_res["results"], "classification"),
            use_container_width=True,
        )
    with col4:
        best_cm = clf_res["results"][clf_res["best_name"]]["confusion_matrix"]
        st.plotly_chart(
            plot_confusion_matrix(best_cm),
            use_container_width=True,
        )

    metrics_clf = []
    for name, r in clf_res["results"].items():
        metrics_clf.append({
            "Модель": name, "Accuracy": r["accuracy"],
            "F1": r["f1"],
        })
    st.dataframe(pd.DataFrame(metrics_clf), hide_index=True, use_container_width=True)

    with st.expander("Детальний звіт найкращого класифікатора"):
        st.code(clf_res["results"][clf_res["best_name"]]["report"])

with tab4:
    st.markdown(
        f"""
        <h2>
            <img src="https://cdn-icons-png.flaticon.com/128/16786/16786354.png"
                width="22"
                style="vertical-align: middle; margin-right: 8px;">
            Кластеризація тренувальної поведінки - Користувач {user_id}
        </h2>
        """,
        unsafe_allow_html=True
    )

    clust = pipeline.clustering_result

    c1, c2, c3 = st.columns(3)
    c1.metric("Кількість кластерів (k)", clust["best_k"])
    c2.metric("Силуетний коефіцієнт", f"{clust['silhouette']:.4f}")
    c3.metric("Профіль користувача", user_pred.get("cluster", "N/A"))

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            plot_cluster_scatter(clust["result_df"]),
            use_container_width=True,
        )
    with col_b:
        k_data = clust["k_analysis"]
        sil_df = pd.DataFrame({
            "k": k_data["k_range"],
            "Силуетний коефіцієнт": k_data["silhouettes"],
        })
        st.line_chart(sil_df.set_index("k"))
        st.caption("Метод аналізу силуетного коефіцієнта для вибору k")

    st.subheader("Центроїди кластерів")
    st.dataframe(
        clust["centroids_df"].round(2),
        hide_index=True, use_container_width=True,
    )

    st.subheader("Розподіл користувачів по кластерах")
    dist = clust["result_df"]["cluster_name"].value_counts().reset_index()
    dist.columns = ["Кластер", "Кількість користувачів"]
    st.dataframe(dist, hide_index=True, use_container_width=True)

with tab5:
    st.markdown(
        f"""
        <h2>
            <img src="https://cdn-icons-png.flaticon.com/128/1508/1508130.png"
                width="22"
                style="vertical-align: middle; margin-right: 8px;">
            Формування PDF-звіту - Користувач {user_id}
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div style="
            background-color: #fbf9db; 
            color: #3B444B;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
        ">
            Буде сформовано звіт для користувача <b>{user_id}</b>
            з результатами всіх трьох ML-задач та рекомендацією.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("   ")
    clust_name = user_pred.get("cluster", "N/A")
    pipeline.clustering_result["user_cluster_name"] = clust_name

    if st.button("Згенерувати PDF-звіт", type="primary"):
        with st.spinner("Формування звіту..."):
            try:
                pdf_path = generate_report(
                    user_id=user_id,
                    user_df=user_df,
                    regression_results=pipeline.regression_results,
                    classification_results=pipeline.classification_results,
                    clustering_result=pipeline.clustering_result,
                    recommendation=user_pred.get("recommendation", {}),
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇Завантажити PDF",
                        data=f.read(),
                        file_name=Path(pdf_path).name,
                        mime="application/pdf",
                    )
                st.markdown(
                    """
                    <div style="
                        background: #fbf9db
                        color: #3B444B;
                        padding: 12px 16px;
                        border-radius: 10px;
                        font-weight: 500;
                        font-size: 16px;
                    ">
                        Звіт готовий!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Помилка: {e}")
