import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Système de Détection des Barrages",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CONSTANTES ====================
MODEL_PATH = "model_final_sn.pt"
CLASS_NAMES = ["Low", "Critical", "Normal"]
INPUT_SIZE = (64, 64)

class_colors = {"Critical": "#EF4444", "Low": "#F59E0B", "Normal": "#22C55E"}

class_descriptions = {
    "Critical": "⚠️ État critique - Intervention urgente requise",
    "Low": "⚡ État de vigilance - Surveillance recommandée",
    "Normal": "✅ État normal - Aucune action requise",
}

# ==================== CSS PERSONNALISÉ ====================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Police globale */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fond principal */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(129, 140, 248, 0.2) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeInDown 0.8s ease-out;
    }
    
    .main-header h1 {
        font-size: 3rem;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.5);
    }
    
    .main-header p {
        color: #cbd5e1;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Cartes d'information */
    .info-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(56, 189, 248, 0.3);
        border-color: rgba(56, 189, 248, 0.5);
    }
    
    .info-card h3 {
        color: #38bdf8;
        margin-top: 0;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    .info-card p {
        color: #cbd5e1;
        line-height: 1.7;
        margin-bottom: 0;
    }
    
    /* Cartes de résultats */
    .result-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-critical {
        background: rgba(239, 68, 68, 0.15);
        border-left: 5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        color: #fecaca;
        font-weight: 500;
    }
    
    .result-normal {
        background: rgba(34, 197, 94, 0.15);
        border-left: 5px solid #22C55E;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        color: #bbf7d0;
        font-weight: 500;
    }
    
    .result-low {
        background: rgba(245, 158, 11, 0.15);
        border-left: 5px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        color: #fde68a;
        font-weight: 500;
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(56, 189, 248, 0.6);
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    }
    
    /* Métriques */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #cbd5e1;
    }
    
    /* Upload de fichiers */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        border: 2px dashed rgba(56, 189, 248, 0.3);
        border-radius: 15px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.6);
        background: rgba(30, 41, 59, 0.7);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    }
    
    /* Texte général */
    p, li, span {
        color: #cbd5e1;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess {
        background: rgba(34, 197, 94, 0.15);
        border-left: 4px solid #22C55E;
        color: #bbf7d0;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.15);
        border-left: 4px solid #F59E0B;
        color: #fde68a;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15);
        border-left: 4px solid #EF4444;
        color: #fecaca;
    }
    
    .stInfo {
        background: rgba(56, 189, 248, 0.15);
        border-left: 4px solid #38bdf8;
        color: #bae6fd;
    }
    
    /* Animation */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.6;
        }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==================== INITIALISATION SESSION STATE ====================
if "classification_history" not in st.session_state:
    st.session_state.classification_history = []
if "daily_stats" not in st.session_state:
    st.session_state.daily_stats = {
        "total": 0,
        "critical": 0,
        "low": 0,
        "normal": 0,
        "avg_confidence": 0,
    }
if "total_classifications" not in st.session_state:
    st.session_state.total_classifications = 0


# ==================== FONCTIONS ====================
@st.cache_resource
def load_model():
    """Charger le modèle TorchScript"""
    try:
        model = torch.jit.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None


def preprocess_image(image):
    """Prétraiter l'image pour l'inférence"""
    transform = transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image).unsqueeze(0)


def predict_image(model, image):
    """Faire une prédiction sur l'image"""
    try:
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_pct = confidence.item() * 100
        all_probs = {
            CLASS_NAMES[i]: probabilities[i].item() * 100
            for i in range(len(CLASS_NAMES))
        }

        return predicted_class, confidence_pct, all_probs
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None, None, None


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(129, 140, 248, 0.2) 100%); border-radius: 15px; margin-bottom: 1.5rem; border: 1px solid rgba(56, 189, 248, 0.3);">
            <h2 style="color: #38bdf8; margin: 0; font-size: 1.5rem;">🌊 Dam Monitor</h2>
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0.3rem 0 0 0;">AI-Powered Detection</p>
        </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "📋 Navigation",
        [
            "🏠 Accueil",
            "🔍 Classification",
            "📊 Tableau de Bord",
            "📚 Documentation",
            "ℹ️ À propos",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")

    confidence_threshold = st.slider(
        "Seuil de confiance (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Seuil minimum de confiance pour accepter une prédiction",
    )

    show_probabilities = st.checkbox("Afficher les probabilités", value=True)

    st.markdown("---")
    st.markdown("### 📈 Statistiques")
    st.metric("Classifications", st.session_state.total_classifications)

    if st.session_state.daily_stats["critical"] > 0:
        st.metric(
            "⚠️ Critiques", st.session_state.daily_stats["critical"], delta="Attention!"
        )

# ==================== PAGE: ACCUEIL ====================
if page == "🏠 Accueil":
    st.markdown(
        """
        <div class="main-header">
            <h1>🌊 Système de Détection des Barrages</h1>
            <p>Intelligence Artificielle pour la surveillance et l'évaluation de l'état des infrastructures hydrauliques</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>🎯 Précision</h3>
                <p>Modèle optimisé avec architecture personnalisée offrant une détection rapide et fiable des anomalies structurelles</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>⚡ Performance</h3>
                <p>Analyse ultra-rapide grâce au format TorchScript permettant une évaluation en temps réel</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>🔒 Fiabilité</h3>
                <p>Système éprouvé pour identifier les états critiques et garantir la sécurité des infrastructures</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("🚀 Fonctionnalités principales")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Classification Multi-Images** 📸
        - Téléchargement simultané de plusieurs images
        - Analyse batch optimisée
        - Résultats structurés sous forme de tableau
        
        **Visualisations Avancées** 📊
        - Graphiques interactifs des probabilités
        - Comparaison visuelle des résultats
        - Export des données au format CSV
        """
        )

    with col2:
        st.markdown(
            """
        **Analyses Détaillées** 🔬
        - Niveau de confiance pour chaque prédiction
        - Distribution des probabilités par classe
        - Recommandations basées sur les résultats
        
        **Surveillance Continue** 📈
        - Tableau de bord en temps réel
        - Historique des classifications
        - Alertes automatiques pour états critiques
        """
        )

    st.markdown("---")
    st.info(
        "💡 **Conseil:** Commencez par la section 'Classification' pour analyser vos images de barrages."
    )

# ==================== PAGE: CLASSIFICATION ====================
elif page == "🔍 Classification":
    st.markdown(
        """
        <div class="main-header">
            <h1>🔍 Classification des Barrages</h1>
            <p>Téléchargez une ou plusieurs images pour une analyse intelligente</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Charger le modèle
    model = load_model()

    if model is None:
        st.error(
            "Le modèle n'a pas pu être chargé. Veuillez vérifier que le fichier 'model_final_sn.pt' est présent."
        )
        st.stop()

    # Upload multiple images
    uploaded_files = st.file_uploader(
        "📤 Sélectionnez une ou plusieurs images",
        type=["jpg", "jpeg", "png", "tif"],
        accept_multiple_files=True,
        help="Formats acceptés: JPG, JPEG, PNG",
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} image(s) téléchargée(s)")

        if st.button("🚀 Lancer l'analyse", type="primary"):
            results = []

            # Barre de progression
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(
                    f"Analyse en cours: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})"
                )

                try:
                    # Charger l'image
                    image = Image.open(uploaded_file).convert("RGB")

                    # Prédiction
                    time.sleep(0.3)  # Simulation effet visuel
                    predicted_class, confidence, all_probs = predict_image(model, image)

                    if predicted_class is not None:
                        results.append(
                            {
                                "image": image,
                                "filename": uploaded_file.name,
                                "prediction": predicted_class,
                                "confidence": confidence,
                                "probabilities": all_probs,
                            }
                        )

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse de {uploaded_file.name}: {e}")

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("✅ Analyse terminée!")
            st.session_state.total_classifications += len(uploaded_files)

            # Mettre à jour les statistiques
            for result in results:
                st.session_state.classification_history.append(
                    {
                        "filename": result["filename"],
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                st.session_state.daily_stats["total"] += 1
                if result["prediction"] == "Critical":
                    st.session_state.daily_stats["critical"] += 1
                elif result["prediction"] == "Low":
                    st.session_state.daily_stats["low"] += 1
                else:
                    st.session_state.daily_stats["normal"] += 1

            # Calculer la confiance moyenne
            if st.session_state.daily_stats["total"] > 0:
                total_confidence = sum(r["confidence"] for r in results)
                current_avg = st.session_state.daily_stats["avg_confidence"]
                previous_total = st.session_state.daily_stats["total"] - len(results)

                if previous_total > 0:
                    st.session_state.daily_stats["avg_confidence"] = (
                        current_avg * previous_total + total_confidence
                    ) / st.session_state.daily_stats["total"]
                else:
                    st.session_state.daily_stats["avg_confidence"] = (
                        total_confidence / len(results)
                    )

            # Affichage des résultats
            st.markdown("---")
            st.subheader("📊 Résultats de l'analyse")

            # Grille d'images
            num_cols = min(3, len(results))

            for i in range(0, len(results), num_cols):
                cols = st.columns(num_cols)

                for j, col in enumerate(cols):
                    if i + j < len(results):
                        result = results[i + j]

                        with col:
                            # Image
                            st.image(result["image"], use_container_width=True)

                            # Nom du fichier
                            st.markdown(f"**📁 {result['filename']}**")

                            # Résultat
                            pred_class = result["prediction"]
                            confidence = result["confidence"]

                            if pred_class == "Critical":
                                st.markdown(
                                    f"""
                                    <div class="result-critical">
                                        <strong>⚠️ {pred_class}</strong><br>
                                        Confiance: {confidence:.1f}%<br>
                                        {class_descriptions[pred_class]}
                                    </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                            elif pred_class == "Normal":
                                st.markdown(
                                    f"""
                                    <div class="result-normal">
                                        <strong>✅ {pred_class}</strong><br>
                                        Confiance: {confidence:.1f}%<br>
                                        {class_descriptions[pred_class]}
                                    </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div class="result-low">
                                        <strong>⚡ {pred_class}</strong><br>
                                        Confiance: {confidence:.1f}%<br>
                                        {class_descriptions[pred_class]}
                                    </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                            # Graphique
                            if show_probabilities:
                                probs = result["probabilities"]

                                fig = go.Figure(
                                    data=[
                                        go.Bar(
                                            x=list(probs.keys()),
                                            y=list(probs.values()),
                                            marker_color=[
                                                class_colors[k] for k in probs.keys()
                                            ],
                                            text=[f"{v:.1f}%" for v in probs.values()],
                                            textposition="auto",
                                        )
                                    ]
                                )

                                fig.update_layout(
                                    title="Probabilités",
                                    xaxis_title="Classe",
                                    yaxis_title="Probabilité (%)",
                                    height=250,
                                    margin=dict(l=20, r=20, t=40, b=20),
                                    showlegend=False,
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(30,41,59,0.5)",
                                    font=dict(color="#cbd5e1"),
                                )

                                st.plotly_chart(fig, use_container_width=True)

            # Tableau récapitulatif
            st.markdown("---")
            st.subheader("📋 Tableau récapitulatif")

            df_results = pd.DataFrame(
                [
                    {
                        "Fichier": r["filename"],
                        "Prédiction": r["prediction"],
                        "Confiance (%)": f"{r['confidence']:.2f}",
                        "Critical (%)": f"{r['probabilities']['Critical']:.2f}",
                        "Low (%)": f"{r['probabilities']['Low']:.2f}",
                        "Normal (%)": f"{r['probabilities']['Normal']:.2f}",
                    }
                    for r in results
                ]
            )

            st.dataframe(df_results, use_container_width=True, height=300)

            # Statistiques globales
            st.markdown("---")
            st.subheader("📈 Statistiques globales")

            col1, col2, col3, col4 = st.columns(4)

            critical_count = sum(1 for r in results if r["prediction"] == "Critical")
            low_count = sum(1 for r in results if r["prediction"] == "Low")
            normal_count = sum(1 for r in results if r["prediction"] == "Normal")

            with col1:
                st.metric("Total analysé", len(results))
            with col2:
                st.metric(
                    "Critical",
                    critical_count,
                    delta=None if critical_count == 0 else "⚠️",
                )
            with col3:
                st.metric("Low", low_count)
            with col4:
                st.metric(
                    "Normal", normal_count, delta=None if normal_count == 0 else "✅"
                )

            # Graphique de distribution
            dist_fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Critical", "Low", "Normal"],
                        values=[critical_count, low_count, normal_count],
                        marker_colors=[
                            class_colors["Critical"],
                            class_colors["Low"],
                            class_colors["Normal"],
                        ],
                        hole=0.4,
                    )
                ]
            )

            dist_fig.update_layout(
                title="Distribution des classifications",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
            )

            st.plotly_chart(dist_fig, use_container_width=True)

            # Export CSV
            st.markdown("---")
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv,
                file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    else:
        st.info("👆 Veuillez télécharger au moins une image pour commencer l'analyse.")

# ==================== PAGE: TABLEAU DE BORD ====================
elif page == "📊 Tableau de Bord":
    st.markdown(
        """
        <div class="main-header">
            <h1>📊 Tableau de Bord</h1>
            <p>Vue d'ensemble et statistiques en temps réel</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    stats = st.session_state.daily_stats
    history = st.session_state.classification_history

    if stats["total"] == 0:
        st.info(
            "📌 Aucune classification effectuée pour le moment. Allez dans la section 'Classification' pour commencer à analyser des images."
        )
    else:
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['total']}</div>
                    <div class="metric-label">Classifications totales</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #EF4444;">{stats['critical']}</div>
                    <div class="metric-label">Barrages critiques</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #F59E0B;">{stats['low']}</div>
                    <div class="metric-label">Barrages en vigilance</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['avg_confidence']:.1f}%</div>
                    <div class="metric-label">Confiance moyenne</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Graphiques
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Distribution des classifications")

            dist_fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Critical", "Low", "Normal"],
                        values=[stats["critical"], stats["low"], stats["normal"]],
                        marker_colors=[
                            class_colors["Critical"],
                            class_colors["Low"],
                            class_colors["Normal"],
                        ],
                        hole=0.4,
                        textinfo="label+percent",
                        textfont_size=14,
                    )
                ]
            )

            dist_fig.update_layout(
                height=350,
                showlegend=True,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
            )

            st.plotly_chart(dist_fig, use_container_width=True)

        with col2:
            st.subheader("📈 Répartition par état")

            bar_fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Critical", "Low", "Normal"],
                        y=[stats["critical"], stats["low"], stats["normal"]],
                        marker_color=[
                            class_colors["Critical"],
                            class_colors["Low"],
                            class_colors["Normal"],
                        ],
                        text=[stats["critical"], stats["low"], stats["normal"]],
                        textposition="auto",
                        textfont_size=16,
                    )
                ]
            )

            bar_fig.update_layout(
                height=350,
                yaxis_title="Nombre",
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,41,59,0.5)",
                font=dict(color="#cbd5e1"),
            )

            st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("---")

        # Historique
        st.subheader("📋 Historique des classifications récentes")

        if len(history) > 0:
            recent_history = history[-10:][::-1]

            df_history = pd.DataFrame(recent_history)
            df_display = df_history[
                ["timestamp", "filename", "prediction", "confidence"]
            ].copy()
            df_display.columns = [
                "Date/Heure",
                "Fichier",
                "Prédiction",
                "Confiance (%)",
            ]
            df_display["Confiance (%)"] = df_display["Confiance (%)"].apply(
                lambda x: f"{x:.2f}"
            )
            df_display["Statut"] = df_display["Prédiction"].apply(
                lambda x: (
                    "⚠️ Critical"
                    if x == "Critical"
                    else ("⚡ Low" if x == "Low" else "✅ Normal")
                )
            )

            st.dataframe(
                df_display[["Date/Heure", "Fichier", "Statut", "Confiance (%)"]],
                use_container_width=True,
                height=400,
            )

            st.markdown("---")
            full_history_df = pd.DataFrame(history)
            csv = full_history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Télécharger l'historique complet (CSV)",
                data=csv,
                file_name=f"historique_classifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        # Alertes
        if stats["critical"] > 0:
            st.markdown("---")
            st.error(
                f"⚠️ **Attention:** {stats['critical']} barrage(s) en état critique détecté(s). Une inspection urgente est recommandée."
            )

        if stats["low"] > 0:
            st.warning(
                f"⚡ **Vigilance:** {stats['low']} barrage(s) en état de surveillance. Une maintenance préventive est conseillée."
            )

        # Statistiques détaillées
        st.markdown("---")
        st.subheader("📈 Statistiques détaillées")

        col1, col2, col3 = st.columns(3)

        with col1:
            if stats["total"] > 0:
                critical_pct = (stats["critical"] / stats["total"]) * 100
                st.metric("Taux de criticité", f"{critical_pct:.1f}%")

        with col2:
            if stats["total"] > 0:
                normal_pct = (stats["normal"] / stats["total"]) * 100
                st.metric("Taux de normalité", f"{normal_pct:.1f}%")

        with col3:
            st.metric("Total analysé", f"{stats['total']} images")

# ==================== PAGE: DOCUMENTATION ====================
elif page == "📚 Documentation":
    st.markdown(
        """
        <div class="main-header">
            <h1>📚 Documentation</h1>
            <p>Guide complet sur les types de barrages et leur classification</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🏗️ Types de barrages",
            "🎯 Classes de risque",
            "💡 Guide d'utilisation",
            "🔧 Informations techniques",
        ]
    )

    with tab1:
        st.header("Types de barrages")

        st.markdown(
            """
        ### 1. Barrage-poids 🏔️
        
        **Description:** Structure massive qui résiste à la pression de l'eau par son propre poids.
        
        **Caractéristiques:**
        - Construction en béton ou en maçonnerie
        - Profil triangulaire
        - Très stable et durable
        - Nécessite des fondations rocheuses solides
        
        **Exemples célèbres:** Barrage Hoover (États-Unis), Barrage Grand Coulee (États-Unis)
        """
        )

        st.image(
            "https://www.edf.fr/sites/groupe/files/styles/img_465x260/public/2025-08/edfgroup_comprendre_hydro_barrage_type-poids_1066x595.jpg?itok=EaUq5BrS",
            caption="Exemple de barrage-poids: Barrage Hoover",
            use_container_width=True,
        )

        st.markdown("---")

        st.markdown(
            """
        ### 2. Barrage-voûte 🌉
        
        **Description:** Structure arquée qui transfère la pression de l'eau vers les flancs de la vallée.
        
        **Caractéristiques:**
        - Forme incurvée vers l'amont
        - Économie de matériaux
        - Convient aux vallées étroites
        - Nécessite des appuis rocheux de qualité
        
        **Exemples célèbres:** Barrage de Vajont (Italie), Barrage de Monteynard (France)
        """
        )

        st.image(
            "https://www.edf.fr/sites/groupe/files/styles/img_465x260/public/2025-08/edfgroup_comprendre_hydro_barrage_type-voute_1066x595.jpg?itok=F3OgLhyA",
            caption="Exemple de barrage-voûte: Barrage de Monteynard",
            use_container_width=True,
        )

        st.markdown("---")

        st.markdown(
            """
        ### 3. Barrage en remblai 🌊
        
        **Description:** Construction en terre ou enrochement avec un noyau imperméable.
        
        **Caractéristiques:**
        - Utilisation de matériaux locaux
        - Adapté à tout type de fondation
        - Plus économique pour les grandes hauteurs
        - Nécessite un entretien régulier
        
        **Exemples célèbres:** Barrage de Tarbela (Pakistan), Barrage des Trois Gorges (Chine)
        """
        )

        st.image(
            "https://www.edf.fr/sites/groupe/files/styles/img_465x260/public/2025-08/edfgroup_comprendre_hydro_barrage_type-contreforts_1066x595.jpg?itok=lo8bOpoC",
            caption="Exemple de barrage en remblai: Barrage des Trois Gorges",
            use_container_width=True,
        )

    with tab2:
        st.header("Classes de risque")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(
                """
                <div class="result-critical" style="margin: 1rem 0;">
                    <h3 style="margin-top: 0;">⚠️ Critical</h3>
                    <p><strong>Niveau de risque:</strong> Élevé</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            **Caractéristiques identifiées:**
            - Fissures importantes visibles
            - Déformations structurelles
            - Infiltrations majeures
            - Signes de dégradation avancée
            
            **Actions requises:**
            - Inspection immédiate par des experts
            - Évaluation structurelle complète
            - Mise en place de mesures d'urgence
            - Surveillance renforcée
            """
            )

        st.markdown("---")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(
                """
                <div class="result-low" style="margin: 1rem 0;">
                    <h3 style="margin-top: 0;">⚡ Low</h3>
                    <p><strong>Niveau de risque:</strong> Modéré</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            **Caractéristiques identifiées:**
            - Fissures mineures localisées
            - Début d'altération des matériaux
            - Infiltrations légères
            - Signes de vieillissement normal
            
            **Actions recommandées:**
            - Surveillance régulière
            - Maintenance préventive
            - Documentation photographique
            - Planification d'interventions
            """
            )

        st.markdown("---")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(
                """
                <div class="result-normal" style="margin: 1rem 0;">
                    <h3 style="margin-top: 0;">✅ Normal</h3>
                    <p><strong>Niveau de risque:</strong> Faible</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            **Caractéristiques identifiées:**
            - Structure intègre
            - Absence de fissures significatives
            - Bon état général
            - Fonctionnement nominal
            
            **Actions requises:**
            - Inspections de routine
            - Maintenance standard
            - Surveillance continue
            - Documentation régulière
            """
            )

    with tab3:
        st.header("Guide d'utilisation")

        st.markdown(
            """
        ### 🚀 Démarrage rapide
        
        1. **Accédez à la section Classification** 🔍
           - Cliquez sur "Classification" dans le menu latéral
        
        2. **Téléchargez vos images** 📤
           - Cliquez sur "Parcourir les fichiers"
           - Sélectionnez une ou plusieurs images
           - Formats acceptés: JPG, JPEG, PNG
        
        3. **Lancez l'analyse** 🚀
           - Cliquez sur "Lancer l'analyse"
           - Attendez le traitement (moins d'une seconde par image)
        
        4. **Consultez les résultats** 📊
           - Visualisez les prédictions pour chaque image
           - Analysez les probabilités et la confiance
           - Téléchargez le rapport CSV si nécessaire
        
        ---
        
        ### 💡 Conseils pour de meilleurs résultats
        
        **Qualité des images:**
        - ✅ Utilisez des images nettes et bien éclairées
        - ✅ Privilégiez une résolution minimale de 200x200 pixels
        - ✅ Cadrez correctement la structure du barrage
        - ❌ Évitez les images floues ou sous-exposées
        - ❌ Évitez les obstructions (arbres, brouillard)
        
        **Analyse des résultats:**
        - Vérifiez le niveau de confiance (>70% recommandé)
        - Comparez les probabilités entre classes
        - Considérez le contexte et l'historique du barrage
        """
        )

    with tab4:
        st.header("Informations techniques")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            ### 🧠 Architecture du modèle
            
            **Format:** TorchScript (JIT)
            - Modèle compilé et optimisé
            - Inférence ultra-rapide
            - Compatible CPU
            
            **Prétraitement:**
            - Redimensionnement: 64x64 pixels
            - Normalisation: mean=[0.5, 0.5, 0.5]
            - std=[0.5, 0.5, 0.5]
            
            **Sortie:** Probabilités softmax
            """
            )

        with col2:
            st.markdown(
                """
            ### 📊 Spécifications
            
            **Taille d'entrée:** 64x64 pixels RGB
            
            **Classes de sortie:**
            1. Low (Faible)
            2. Critical (Critique)
            3. Normal (Normal)
            
            **Performance:**
            - Temps d'inférence: ~0.2-0.5s (CPU)
            - Format optimisé TorchScript
            """
            )

        st.markdown("---")

        st.markdown(
            """
        ### 🔬 Processus de traitement
        
        1. **Prétraitement**
           - Redimensionnement à 64x64
           - Conversion en tenseur
           - Normalisation
        
        2. **Inférence**
           - Propagation avant
           - Calcul des probabilités
           - Application du softmax
        
        3. **Post-traitement**
           - Extraction de la classe prédite
           - Génération des visualisations
        """
        )

# ==================== PAGE: À PROPOS ====================
elif page == "ℹ️ À propos":
    st.markdown(
        """
        <div class="main-header">
            <h1>ℹ️ À propos</h1>
            <p>Information sur le système et l'équipe</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        ### 🎯 Mission
        
        Le **Système de Détection des Barrages** utilise l'intelligence artificielle pour améliorer 
        la surveillance et l'évaluation de l'état des infrastructures hydrauliques. Notre objectif est 
        de fournir un outil rapide, précis et accessible pour aider les ingénieurs et les responsables 
        dans la gestion préventive des barrages.
        
        ### 🔬 Technologie
        
        Cette application utilise:
        - **PyTorch** pour le deep learning
        - **TorchScript (JIT)** pour l'optimisation
        - **Streamlit** pour l'interface utilisateur
        - **Plotly** pour les visualisations interactives
        
        ### 🌟 Fonctionnalités clés
        
        - Classification multi-classe (Low, Critical, Normal)
        - Analyse batch de plusieurs images
        - Visualisations interactives des résultats
        - Export des données au format CSV
        - Interface intuitive et moderne
        
        ### 📞 Contact
        
        Pour toute question, suggestion ou collaboration:
        - 📧 Email: contact@dam-monitor.com
        - 🌐 Site web: https://www.edf.fr/groupe-edf/comprendre/production/hydraulique/formes-de-barrages
        """
        )

    with col2:
        st.markdown(
            """
        ### 📦 Version
        
        **v2.0.0**
        
        *Dernière mise à jour:*  
        Février 2026
        
        ---
        
        ### 📝 Licence
        
        © 2026 Dam Monitor  
        Tous droits réservés
        
        ---
        
        ### 🙏 Remerciements
        
        Merci aux contributeurs  
        et à la communauté  
        open-source
        """
        )

        st.markdown("---")

        st.info(
            """
        **Note:** Ce système est un outil d'aide à la décision. 
        Les résultats doivent toujours être validés par des experts qualifiés.
        """
        )

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; padding: 2rem 0;">
        <p style="font-size: 0.9rem;">Développé avec ❤️ par l'équipe Dam Monitor</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">Version 2.0.0 | Propulsé par PyTorch & Streamlit</p>
        <p style="font-size: 0.75rem; margin-top: 0.5rem;">© 2026 Tous droits réservés</p>
    </div>
""",
    unsafe_allow_html=True,
)
