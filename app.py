import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Guidance System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Model & Encoder ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/career_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# ── Skills & Roadmaps ─────────────────────────────────────────────────────────
SKILLS = [
    "python", "javascript", "sql", "machine_learning", "statistics",
    "react", "nodejs", "docker", "kubernetes", "networking",
    "linux", "swift", "kotlin", "cloud", "data_analysis",
    "deep_learning", "communication", "project_management", "uiux_design", "git"
]

SKILL_LABELS = {
    "python": "Python", "javascript": "JavaScript", "sql": "SQL",
    "machine_learning": "Machine Learning", "statistics": "Statistics",
    "react": "React", "nodejs": "Node.js", "docker": "Docker",
    "kubernetes": "Kubernetes", "networking": "Networking",
    "linux": "Linux", "swift": "Swift", "kotlin": "Kotlin",
    "cloud": "Cloud (AWS/GCP/Azure)", "data_analysis": "Data Analysis",
    "deep_learning": "Deep Learning", "communication": "Communication",
    "project_management": "Project Management", "uiux_design": "UI/UX Design",
    "git": "Git & Version Control"
}

ROADMAPS = {
    "Data Scientist": {
        "emoji": "📊",
        "description": "Analyze complex data to extract insights and drive business decisions.",
        "steps": [
            ("🟢 Beginner",   "Master Python basics, Pandas, NumPy, and SQL for data wrangling."),
            ("🟡 Intermediate","Learn statistics, data visualization (Matplotlib, Seaborn), and EDA."),
            ("🟠 Advanced",   "Study Machine Learning with Scikit-learn, model evaluation & tuning."),
            ("🔴 Expert",     "Deep Learning (TensorFlow/PyTorch), MLOps, and deploying models."),
        ],
        "resources": ["Kaggle Learn", "fast.ai", "Coursera ML Specialization", "Towards Data Science"],
        "tools": ["Jupyter Notebook", "Pandas", "Scikit-learn", "TensorFlow", "Tableau"],
        "avg_salary": "₹8–20 LPA (India) | $95K–$140K (US)"
    },
    "Web Developer": {
        "emoji": "🌐",
        "description": "Build responsive, modern websites and web applications.",
        "steps": [
            ("🟢 Beginner",   "Learn HTML, CSS, and JavaScript fundamentals."),
            ("🟡 Intermediate","Master React.js, Node.js, REST APIs, and databases."),
            ("🟠 Advanced",   "TypeScript, system design, performance optimization, CI/CD."),
            ("🔴 Expert",     "Microservices, Web Security, Full-stack architecture at scale."),
        ],
        "resources": ["freeCodeCamp", "The Odin Project", "MDN Web Docs", "Frontend Masters"],
        "tools": ["VS Code", "React", "Node.js", "MongoDB", "Figma"],
        "avg_salary": "₹5–18 LPA (India) | $80K–$130K (US)"
    },
    "Mobile Developer": {
        "emoji": "📱",
        "description": "Create iOS and Android apps used by millions of people.",
        "steps": [
            ("🟢 Beginner",   "Learn Swift (iOS) or Kotlin (Android) basics and UI fundamentals."),
            ("🟡 Intermediate","Build apps with APIs, local storage, and navigation."),
            ("🟠 Advanced",   "State management, animations, push notifications, testing."),
            ("🔴 Expert",     "App Store deployment, performance tuning, cross-platform (Flutter)."),
        ],
        "resources": ["Apple Developer Docs", "Android Developers", "Ray Wenderlich", "Flutter Docs"],
        "tools": ["Xcode", "Android Studio", "Flutter", "Firebase", "Figma"],
        "avg_salary": "₹6–20 LPA (India) | $90K–$135K (US)"
    },
    "DevOps Engineer": {
        "emoji": "⚙️",
        "description": "Bridge development and operations to deliver software faster and reliably.",
        "steps": [
            ("🟢 Beginner",   "Learn Linux, Bash scripting, Git, and basic networking."),
            ("🟡 Intermediate","Master Docker, CI/CD pipelines (GitHub Actions, Jenkins)."),
            ("🟠 Advanced",   "Kubernetes, Infrastructure as Code (Terraform), monitoring."),
            ("🔴 Expert",     "Cloud architecture (AWS/GCP), SRE practices, disaster recovery."),
        ],
        "resources": ["Linux Foundation", "KodeKloud", "A Cloud Guru", "DevOps Roadmap"],
        "tools": ["Docker", "Kubernetes", "Terraform", "Prometheus", "Jenkins"],
        "avg_salary": "₹8–25 LPA (India) | $100K–$150K (US)"
    },
    "Cybersecurity Analyst": {
        "emoji": "🔐",
        "description": "Protect systems, networks, and data from cyber threats.",
        "steps": [
            ("🟢 Beginner",   "Learn networking fundamentals, Linux, and security basics."),
            ("🟡 Intermediate","Study ethical hacking, vulnerability assessment, and SIEM tools."),
            ("🟠 Advanced",   "Penetration testing, incident response, cloud security."),
            ("🔴 Expert",     "Threat intelligence, red teaming, security architecture & compliance."),
        ],
        "resources": ["TryHackMe", "Hack The Box", "SANS Institute", "CompTIA Security+"],
        "tools": ["Wireshark", "Metasploit", "Nmap", "Burp Suite", "Splunk"],
        "avg_salary": "₹6–22 LPA (India) | $90K–$140K (US)"
    },
    "AI/ML Engineer": {
        "emoji": "🤖",
        "description": "Build and deploy intelligent systems and machine learning pipelines.",
        "steps": [
            ("🟢 Beginner",   "Python, Math (Linear Algebra, Calculus), and ML fundamentals."),
            ("🟡 Intermediate","Scikit-learn, feature engineering, model evaluation, NLP basics."),
            ("🟠 Advanced",   "Deep Learning (CNNs, RNNs, Transformers), experiment tracking."),
            ("🔴 Expert",     "LLMs, MLOps, model serving at scale, research & publications."),
        ],
        "resources": ["fast.ai", "Deep Learning Specialization", "Papers With Code", "Hugging Face"],
        "tools": ["PyTorch", "TensorFlow", "MLflow", "Hugging Face", "CUDA"],
        "avg_salary": "₹10–30 LPA (India) | $110K–$160K (US)"
    },
    "Cloud Architect": {
        "emoji": "☁️",
        "description": "Design and manage scalable, secure cloud infrastructure.",
        "steps": [
            ("🟢 Beginner",   "Learn cloud basics (AWS/GCP/Azure), networking, and Linux."),
            ("🟡 Intermediate","Docker, Kubernetes, cloud storage, IAM, and serverless."),
            ("🟠 Advanced",   "Multi-cloud strategy, cost optimization, high availability design."),
            ("🔴 Expert",     "Enterprise architecture, FinOps, security compliance, certifications."),
        ],
        "resources": ["AWS Training", "Google Cloud Skills Boost", "A Cloud Guru", "Cloud Guru"],
        "tools": ["AWS", "Terraform", "Kubernetes", "Ansible", "Datadog"],
        "avg_salary": "₹12–35 LPA (India) | $120K–$170K (US)"
    },
    "Product Manager": {
        "emoji": "📋",
        "description": "Lead product strategy, roadmap, and cross-functional teams.",
        "steps": [
            ("🟢 Beginner",   "Learn product thinking, user research, and basic SQL/analytics."),
            ("🟡 Intermediate","Roadmapping, A/B testing, stakeholder management, Agile/Scrum."),
            ("🟠 Advanced",   "Metrics & KPIs, go-to-market strategy, pricing, growth loops."),
            ("🔴 Expert",     "Company strategy, P&L ownership, building & scaling PM teams."),
        ],
        "resources": ["Reforge", "Lenny's Newsletter", "Product School", "Intercom on PM"],
        "tools": ["Jira", "Notion", "Mixpanel", "Figma", "Amplitude"],
        "avg_salary": "₹10–30 LPA (India) | $110K–$160K (US)"
    }
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 2rem;
    }
    .career-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea55;
        border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
    }
    .step-card {
        background: #1e1e2e; border-radius: 10px;
        padding: 0.9rem 1.2rem; margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .metric-box {
        background: #1e1e2e; border-radius: 12px;
        padding: 1rem; text-align: center;
    }
    .tag {
        display: inline-block;
        background: #667eea33; color: #a78bfa;
        border-radius: 20px; padding: 0.2rem 0.8rem;
        font-size: 0.85rem; margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 Career Guidance System")
    st.markdown("---")
    st.markdown("### 📌 How it works")
    st.markdown("""
    1. **Rate your skills** (0–10) on the main page
    2. Click **Predict My Career**
    3. View your **top career matches**
    4. Explore your **personalized roadmap**
    """)
    st.markdown("---")
    st.markdown("### 🎯 Skill Scale Guide")
    scale_data = {
        "Score": ["0–2", "3–4", "5–6", "7–8", "9–10"],
        "Level": ["No knowledge", "Beginner", "Intermediate", "Advanced", "Expert"]
    }
    st.dataframe(pd.DataFrame(scale_data), hide_index=True, use_container_width=True)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.success("✅ Random Forest Classifier")
    st.info("🔁 5-Fold Cross Validation")
    st.warning("🎯 90%+ Accuracy")

# ── Main Page ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚀 AI Career Guidance System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Rate your skills → Get your ideal career track + personalized roadmap</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Skill Assessment", "📊 Model Insights", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Skill Assessment
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 Rate Your Skills (0 = None, 10 = Expert)")
    st.markdown("---")

    skill_values = {}
    cols_per_row = 4
    skill_list = list(SKILL_LABELS.items())

    for i in range(0, len(skill_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (skill_key, skill_label) in enumerate(skill_list[i:i+cols_per_row]):
            with cols[j]:
                skill_values[skill_key] = st.slider(
                    skill_label, min_value=0, max_value=10, value=0, key=skill_key
                )

    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🔍 Predict My Career Track", use_container_width=True, type="primary")

    if predict_btn:
        input_data = np.array([[skill_values[s] for s in SKILLS]])
        prediction_encoded = model.predict(input_data)[0]
        prediction = le.inverse_transform([prediction_encoded])[0]
        probabilities = model.predict_proba(input_data)[0]

        # Top 3 careers
        top3_idx = np.argsort(probabilities)[::-1][:3]
        top3_careers = [(le.inverse_transform([i])[0], probabilities[i]) for i in top3_idx]

        st.markdown("---")
        st.markdown("## 🎉 Your Results")

        # ── Top Match ──────────────────────────────────────────────────────────
        rd = ROADMAPS[prediction]
        st.markdown(f"""
        <div class="career-card">
            <h2>{rd['emoji']} {prediction} — Best Match!</h2>
            <p style="color:#ccc; font-size:1.05rem">{rd['description']}</p>
            <p><strong>💰 Avg Salary:</strong> {rd['avg_salary']}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Top 3 Probabilities ───────────────────────────────────────────────
        st.markdown("### 🏆 Top 3 Career Matches")
        c1, c2, c3 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for idx, (col, (career, prob)) in enumerate(zip([c1, c2, c3], top3_careers)):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size:2rem">{medals[idx]}</div>
                    <div style="font-size:1rem; font-weight:700; margin:0.4rem 0">{career}</div>
                    <div style="font-size:1.5rem; color:#a78bfa">{prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Skill Radar Chart ─────────────────────────────────────────────────
        st.markdown("### 📡 Your Skill Profile")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#a78bfa" if v >= 7 else "#667eea" if v >= 4 else "#374151"
                  for v in skill_values.values()]
        ax.barh(
            [SKILL_LABELS[s] for s in SKILLS],
            [skill_values[s] for s in SKILLS],
            color=colors
        )
        ax.set_xlim(0, 10)
        ax.set_xlabel("Skill Level", color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")
        legend_elements = [
            mpatches.Patch(color="#a78bfa", label="Expert (7–10)"),
            mpatches.Patch(color="#667eea", label="Intermediate (4–6)"),
            mpatches.Patch(color="#374151", label="Beginner (0–3)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right",
                  facecolor="#1e1e2e", labelcolor="white")
        st.pyplot(fig)

        # ── Roadmap ───────────────────────────────────────────────────────────
        st.markdown(f"### 🗺️ Your Personalized Roadmap — {prediction}")
        for step_title, step_desc in rd["steps"]:
            st.markdown(f"""
            <div class="step-card">
                <strong>{step_title}</strong><br>
                <span style="color:#ccc">{step_desc}</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Tools & Resources ─────────────────────────────────────────────────
        col_t, col_r = st.columns(2)
        with col_t:
            st.markdown("### 🛠️ Key Tools to Learn")
            for tool in rd["tools"]:
                st.markdown(f'<span class="tag">🔧 {tool}</span>', unsafe_allow_html=True)
        with col_r:
            st.markdown("### 📚 Learning Resources")
            for res in rd["resources"]:
                st.markdown(f'<span class="tag">📖 {res}</span>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance & Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Algorithm", "Random Forest")
    with col2:
        st.metric("CV Accuracy", "90%+")
    with col3:
        st.metric("Training Samples", "2,000")

    st.markdown("---")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("#### 🔥 Feature Importance")
        try:
            st.image("data/feature_importance.png", use_column_width=True)
        except:
            st.info("Run train_model.py first to generate this chart.")
    with img_col2:
        st.markdown("#### 🎯 Confusion Matrix")
        try:
            st.image("data/confusion_matrix.png", use_column_width=True)
        except:
            st.info("Run train_model.py first to generate this chart.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    **AI-Driven Career Guidance & Roadmap System** is a machine learning web application
    that helps individuals discover the most suitable tech career path based on their current skills.

    #### 🧠 How the ML Model Works
    - **Algorithm:** Random Forest Classifier (200 trees)
    - **Input:** 20 skill ratings (0–10 scale)
    - **Output:** Predicted career track + confidence probabilities
    - **Validation:** 5-Fold Stratified Cross Validation
    - **Accuracy:** 90%+ on held-out test data

    #### 🏗️ Tech Stack
    | Layer | Technology |
    |-------|-----------|
    | Language | Python 3.x |
    | ML Library | Scikit-learn |
    | Dashboard | Streamlit |
    | Data | Pandas, NumPy |
    | Visualization | Matplotlib, Seaborn |

    #### 👨‍💻 Built for
    This project demonstrates end-to-end ML development skills:
    data generation → model training → evaluation → deployment via web app.
    """)