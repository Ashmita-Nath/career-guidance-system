import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import re
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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

# ── Skills ────────────────────────────────────────────────────────────────────
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

# ── Resume Keyword Map ────────────────────────────────────────────────────────
RESUME_KEYWORDS = {
    "python": ["python", "django", "flask", "fastapi", "pandas", "numpy", "scipy"],
    "javascript": ["javascript", "js", "typescript", "es6", "jquery", "vue", "angular"],
    "sql": ["sql", "mysql", "postgresql", "sqlite", "oracle", "database", "nosql", "mongodb"],
    "machine_learning": ["machine learning", "ml", "scikit", "sklearn", "xgboost", "random forest", "classification", "regression"],
    "statistics": ["statistics", "statistical", "probability", "hypothesis", "regression", "anova", "correlation"],
    "react": ["react", "reactjs", "react.js", "redux", "next.js", "gatsby"],
    "nodejs": ["node", "nodejs", "node.js", "express", "npm", "backend"],
    "docker": ["docker", "container", "containerization", "dockerfile"],
    "kubernetes": ["kubernetes", "k8s", "helm", "orchestration"],
    "networking": ["networking", "tcp/ip", "dns", "firewall", "vpn", "cisco", "network"],
    "linux": ["linux", "ubuntu", "bash", "shell", "unix", "centos", "debian"],
    "swift": ["swift", "ios", "xcode", "swiftui", "objective-c", "apple"],
    "kotlin": ["kotlin", "android", "android studio", "gradle", "java"],
    "cloud": ["aws", "gcp", "azure", "cloud", "s3", "ec2", "lambda", "serverless"],
    "data_analysis": ["data analysis", "analytics", "tableau", "power bi", "excel", "visualization", "pandas"],
    "deep_learning": ["deep learning", "neural network", "tensorflow", "pytorch", "keras", "cnn", "rnn", "transformer", "llm"],
    "communication": ["communication", "presentation", "stakeholder", "leadership", "collaboration", "team"],
    "project_management": ["project management", "agile", "scrum", "jira", "kanban", "sprint", "pmp"],
    "uiux_design": ["ui", "ux", "figma", "sketch", "adobe xd", "wireframe", "prototype", "design"],
    "git": ["git", "github", "gitlab", "version control", "bitbucket", "ci/cd", "devops"]
}

# ── Roadmaps ──────────────────────────────────────────────────────────────────
ROADMAPS = {
    "Data Scientist": {
        "emoji": "📊",
        "description": "Analyze complex data to extract insights and drive business decisions.",
        "steps": [
            ("🟢 Beginner",    "Master Python basics, Pandas, NumPy, and SQL for data wrangling."),
            ("🟡 Intermediate","Learn statistics, data visualization (Matplotlib, Seaborn), and EDA."),
            ("🟠 Advanced",    "Study Machine Learning with Scikit-learn, model evaluation & tuning."),
            ("🔴 Expert",      "Deep Learning (TensorFlow/PyTorch), MLOps, and deploying models."),
        ],
        "resources": ["Kaggle Learn", "fast.ai", "Coursera ML Specialization", "Towards Data Science"],
        "tools": ["Jupyter Notebook", "Pandas", "Scikit-learn", "TensorFlow", "Tableau"],
        "avg_salary": "₹8–20 LPA (India) | $95K–$140K (US)",
        "job_keyword": "data scientist"
    },
    "Web Developer": {
        "emoji": "🌐",
        "description": "Build responsive, modern websites and web applications.",
        "steps": [
            ("🟢 Beginner",    "Learn HTML, CSS, and JavaScript fundamentals."),
            ("🟡 Intermediate","Master React.js, Node.js, REST APIs, and databases."),
            ("🟠 Advanced",    "TypeScript, system design, performance optimization, CI/CD."),
            ("🔴 Expert",      "Microservices, Web Security, Full-stack architecture at scale."),
        ],
        "resources": ["freeCodeCamp", "The Odin Project", "MDN Web Docs", "Frontend Masters"],
        "tools": ["VS Code", "React", "Node.js", "MongoDB", "Figma"],
        "avg_salary": "₹5–18 LPA (India) | $80K–$130K (US)",
        "job_keyword": "web developer"
    },
    "Mobile Developer": {
        "emoji": "📱",
        "description": "Create iOS and Android apps used by millions of people.",
        "steps": [
            ("🟢 Beginner",    "Learn Swift (iOS) or Kotlin (Android) basics and UI fundamentals."),
            ("🟡 Intermediate","Build apps with APIs, local storage, and navigation."),
            ("🟠 Advanced",    "State management, animations, push notifications, testing."),
            ("🔴 Expert",      "App Store deployment, performance tuning, cross-platform (Flutter)."),
        ],
        "resources": ["Apple Developer Docs", "Android Developers", "Ray Wenderlich", "Flutter Docs"],
        "tools": ["Xcode", "Android Studio", "Flutter", "Firebase", "Figma"],
        "avg_salary": "₹6–20 LPA (India) | $90K–$135K (US)",
        "job_keyword": "mobile developer"
    },
    "DevOps Engineer": {
        "emoji": "⚙️",
        "description": "Bridge development and operations to deliver software faster and reliably.",
        "steps": [
            ("🟢 Beginner",    "Learn Linux, Bash scripting, Git, and basic networking."),
            ("🟡 Intermediate","Master Docker, CI/CD pipelines (GitHub Actions, Jenkins)."),
            ("🟠 Advanced",    "Kubernetes, Infrastructure as Code (Terraform), monitoring."),
            ("🔴 Expert",      "Cloud architecture (AWS/GCP), SRE practices, disaster recovery."),
        ],
        "resources": ["Linux Foundation", "KodeKloud", "A Cloud Guru", "DevOps Roadmap"],
        "tools": ["Docker", "Kubernetes", "Terraform", "Prometheus", "Jenkins"],
        "avg_salary": "₹8–25 LPA (India) | $100K–$150K (US)",
        "job_keyword": "devops engineer"
    },
    "Cybersecurity Analyst": {
        "emoji": "🔐",
        "description": "Protect systems, networks, and data from cyber threats.",
        "steps": [
            ("🟢 Beginner",    "Learn networking fundamentals, Linux, and security basics."),
            ("🟡 Intermediate","Study ethical hacking, vulnerability assessment, and SIEM tools."),
            ("🟠 Advanced",    "Penetration testing, incident response, cloud security."),
            ("🔴 Expert",      "Threat intelligence, red teaming, security architecture & compliance."),
        ],
        "resources": ["TryHackMe", "Hack The Box", "SANS Institute", "CompTIA Security+"],
        "tools": ["Wireshark", "Metasploit", "Nmap", "Burp Suite", "Splunk"],
        "avg_salary": "₹6–22 LPA (India) | $90K–$140K (US)",
        "job_keyword": "cybersecurity analyst"
    },
    "AI/ML Engineer": {
        "emoji": "🤖",
        "description": "Build and deploy intelligent systems and machine learning pipelines.",
        "steps": [
            ("🟢 Beginner",    "Python, Math (Linear Algebra, Calculus), and ML fundamentals."),
            ("🟡 Intermediate","Scikit-learn, feature engineering, model evaluation, NLP basics."),
            ("🟠 Advanced",    "Deep Learning (CNNs, RNNs, Transformers), experiment tracking."),
            ("🔴 Expert",      "LLMs, MLOps, model serving at scale, research & publications."),
        ],
        "resources": ["fast.ai", "Deep Learning Specialization", "Papers With Code", "Hugging Face"],
        "tools": ["PyTorch", "TensorFlow", "MLflow", "Hugging Face", "CUDA"],
        "avg_salary": "₹10–30 LPA (India) | $110K–$160K (US)",
        "job_keyword": "machine learning engineer"
    },
    "Cloud Architect": {
        "emoji": "☁️",
        "description": "Design and manage scalable, secure cloud infrastructure.",
        "steps": [
            ("🟢 Beginner",    "Learn cloud basics (AWS/GCP/Azure), networking, and Linux."),
            ("🟡 Intermediate","Docker, Kubernetes, cloud storage, IAM, and serverless."),
            ("🟠 Advanced",    "Multi-cloud strategy, cost optimization, high availability design."),
            ("🔴 Expert",      "Enterprise architecture, FinOps, security compliance, certifications."),
        ],
        "resources": ["AWS Training", "Google Cloud Skills Boost", "A Cloud Guru", "Cloud Guru"],
        "tools": ["AWS", "Terraform", "Kubernetes", "Ansible", "Datadog"],
        "avg_salary": "₹12–35 LPA (India) | $120K–$170K (US)",
        "job_keyword": "cloud architect"
    },
    "Product Manager": {
        "emoji": "📋",
        "description": "Lead product strategy, roadmap, and cross-functional teams.",
        "steps": [
            ("🟢 Beginner",    "Learn product thinking, user research, and basic SQL/analytics."),
            ("🟡 Intermediate","Roadmapping, A/B testing, stakeholder management, Agile/Scrum."),
            ("🟠 Advanced",    "Metrics & KPIs, go-to-market strategy, pricing, growth loops."),
            ("🔴 Expert",      "Company strategy, P&L ownership, building & scaling PM teams."),
        ],
        "resources": ["Reforge", "Lenny's Newsletter", "Product School", "Intercom on PM"],
        "tools": ["Jira", "Notion", "Mixpanel", "Figma", "Amplitude"],
        "avg_salary": "₹10–30 LPA (India) | $110K–$160K (US)",
        "job_keyword": "product manager"
    },
    "UI/UX Designer": {
        "emoji": "🎨",
        "description": "Design intuitive, beautiful digital experiences that delight users.",
        "steps": [
            ("🟢 Beginner",    "Learn design principles, color theory, typography, and Figma basics."),
            ("🟡 Intermediate","Build wireframes, prototypes, and conduct user research."),
            ("🟠 Advanced",    "Design systems, accessibility, usability testing, and handoff."),
            ("🔴 Expert",      "Strategic UX leadership, cross-platform design, and design ops."),
        ],
        "resources": ["Google UX Design Certificate", "Nielsen Norman Group", "Figma Academy", "Dribbble"],
        "tools": ["Figma", "Adobe XD", "Sketch", "InVision", "Maze"],
        "avg_salary": "₹5–18 LPA (India) | $80K–$130K (US)",
        "job_keyword": "ux designer"
    },
    "Data Engineer": {
        "emoji": "🔧",
        "description": "Build and maintain the data pipelines that power analytics and ML.",
        "steps": [
            ("🟢 Beginner",    "Learn Python, SQL, and data warehousing fundamentals."),
            ("🟡 Intermediate","Master ETL pipelines, Apache Spark, Airflow, and cloud data tools."),
            ("🟠 Advanced",    "Real-time streaming (Kafka), dbt, data modeling at scale."),
            ("🔴 Expert",      "Data platform architecture, DataOps, ML infrastructure."),
        ],
        "resources": ["Data Engineering Zoomcamp", "dbt Learn", "Databricks Academy", "Towards Data Engineering"],
        "tools": ["Apache Spark", "Airflow", "dbt", "Kafka", "Snowflake"],
        "avg_salary": "₹8–25 LPA (India) | $100K–$150K (US)",
        "job_keyword": "data engineer"
    },
    "Blockchain Developer": {
        "emoji": "⛓️",
        "description": "Build decentralized applications and smart contracts on blockchain networks.",
        "steps": [
            ("🟢 Beginner",    "Learn blockchain fundamentals, cryptography basics, and Solidity."),
            ("🟡 Intermediate","Build smart contracts on Ethereum, Web3.js, and testing frameworks."),
            ("🟠 Advanced",    "DeFi protocols, NFT platforms, Layer 2 solutions, security audits."),
            ("🔴 Expert",      "Protocol design, cross-chain interoperability, and DAO governance."),
        ],
        "resources": ["CryptoZombies", "Ethereum Docs", "Buildspace", "Patrick Collins Solidity Course"],
        "tools": ["Solidity", "Hardhat", "Web3.js", "MetaMask", "IPFS"],
        "avg_salary": "₹8–28 LPA (India) | $100K–$160K (US)",
        "job_keyword": "blockchain developer"
    }
}

# ── Resume Scorer ─────────────────────────────────────────────────────────────
def score_resume(resume_text):
    text = resume_text.lower()
    scores = {}
    for skill, keywords in RESUME_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text)
        scores[skill] = min(count * 2, 10)
    return scores

# ── Job Listings (Remotive Free API) ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_jobs(keyword):
    try:
        url = f"https://remotive.com/api/remote-jobs?search={keyword}&limit=5"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            jobs = response.json().get("jobs", [])
            return jobs[:5]
    except:
        pass
    return []

# ── PDF Generator ─────────────────────────────────────────────────────────────
def generate_pdf(career, skill_values, top3_careers):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Title style
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=24, textColor=colors.HexColor("#667eea"),
        spaceAfter=6, alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"],
        fontSize=14, textColor=colors.HexColor("#764ba2"),
        spaceBefore=12, spaceAfter=6
    )
    normal_style = ParagraphStyle(
        "CustomNormal", parent=styles["Normal"],
        fontSize=10, spaceAfter=4, leading=14
    )
    step_style = ParagraphStyle(
        "StepStyle", parent=styles["Normal"],
        fontSize=10, spaceAfter=4, leading=14,
        leftIndent=12
    )

    rd = ROADMAPS[career]

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("🚀 AI Career Guidance System", title_style))
    story.append(Paragraph("Personalized Career Roadmap Report", styles["Normal"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#667eea")))
    story.append(Spacer(1, 0.15*inch))

    # ── Career Match ─────────────────────────────────────────────────────────
    story.append(Paragraph(f"{rd['emoji']} Predicted Career: {career}", heading_style))
    story.append(Paragraph(rd["description"], normal_style))
    story.append(Paragraph(f"💰 Average Salary: {rd['avg_salary']}", normal_style))
    story.append(Spacer(1, 0.1*inch))

    # ── Top 3 Matches ────────────────────────────────────────────────────────
    story.append(Paragraph("🏆 Top 3 Career Matches", heading_style))
    medals = ["🥇", "🥈", "🥉"]
    match_data = [["Rank", "Career Track", "Confidence"]]
    for i, (c, p) in enumerate(top3_careers):
        match_data.append([medals[i], c, f"{p*100:.1f}%"])
    t = Table(match_data, colWidths=[0.8*inch, 3*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9ff"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",    (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15*inch))

    # ── Roadmap Steps ────────────────────────────────────────────────────────
    story.append(Paragraph(f"🗺️ Your Learning Roadmap", heading_style))
    for step_title, step_desc in rd["steps"]:
        story.append(Paragraph(f"<b>{step_title}</b>", step_style))
        story.append(Paragraph(step_desc, step_style))
        story.append(Spacer(1, 0.05*inch))

    # ── Tools & Resources ────────────────────────────────────────────────────
    story.append(Spacer(1, 0.1*inch))
    col_data = [
        [Paragraph("🛠️ Key Tools", heading_style),
         Paragraph("📚 Resources", heading_style)],
        [Paragraph("<br/>".join([f"• {t}" for t in rd["tools"]]), normal_style),
         Paragraph("<br/>".join([f"• {r}" for r in rd["resources"]]), normal_style)]
    ]
    col_table = Table(col_data, colWidths=[3.2*inch, 3.2*inch])
    col_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(col_table)

    # ── Skill Profile ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("📡 Your Skill Profile", heading_style))
    skill_data = [["Skill", "Score", "Level"]]
    for skill in SKILLS:
        val = skill_values[skill]
        level = "Expert" if val >= 7 else "Intermediate" if val >= 4 else "Beginner"
        skill_data.append([SKILL_LABELS[skill], str(val), level])
    skill_table = Table(skill_data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
    skill_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#764ba2")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9ff"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ("PADDING",        (0, 0), (-1, -1), 6),
    ]))
    story.append(skill_table)

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Paragraph("Generated by AI Career Guidance System", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle { text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 2rem; }
    .career-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea55; border-radius: 16px;
        padding: 1.5rem; margin: 1rem 0;
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
        display: inline-block; background: #667eea33; color: #a78bfa;
        border-radius: 20px; padding: 0.2rem 0.8rem;
        font-size: 0.85rem; margin: 0.2rem;
    }
    .job-card {
        background: #1e1e2e; border-radius: 10px;
        padding: 1rem; margin: 0.5rem 0;
        border-left: 4px solid #764ba2;
    }
    .score-bar { height: 8px; border-radius: 4px; background: #374151; margin: 2px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 Career Guidance System")
    st.markdown("---")
    st.markdown("### 📌 How it works")
    st.markdown("""
    1. **Rate your skills** (0–10) on the Assessment tab
    2. OR **paste your resume** to auto-score
    3. Click **Predict My Career**
    4. View your **top career matches**
    5. Explore your **roadmap & job listings**
    6. **Download PDF** report
    """)
    st.markdown("---")
    st.markdown("### 🎯 Skill Scale")
    st.dataframe(pd.DataFrame({
        "Score": ["0–2", "3–4", "5–6", "7–8", "9–10"],
        "Level": ["None", "Beginner", "Intermediate", "Advanced", "Expert"]
    }), hide_index=True, use_container_width=True)
    st.markdown("---")
    st.success("✅ Random Forest Classifier")
    st.info("🔁 5-Fold Cross Validation")
    st.warning("🎯 90%+ Accuracy | 11 Careers")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚀 AI Career Guidance System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover your ideal tech career path + personalized roadmap + live job listings</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Skill Assessment", "📄 Resume Scorer", "📊 Model Insights", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Skill Assessment
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 Rate Your Skills (0 = None, 10 = Expert)")
    st.markdown("---")

    skill_values = {}
    for i in range(0, len(SKILLS), 4):
        cols = st.columns(4)
        for j, skill_key in enumerate(SKILLS[i:i+4]):
            with cols[j]:
                skill_values[skill_key] = st.slider(
                    SKILL_LABELS[skill_key], 0, 10, 0, key=f"slider_{skill_key}"
                )

    st.markdown("---")
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        predict_btn = st.button("🔍 Predict My Career Track", use_container_width=True, type="primary")

    if predict_btn:
        input_data = np.array([[skill_values[s] for s in SKILLS]])
        pred_encoded = model.predict(input_data)[0]
        prediction = le.inverse_transform([pred_encoded])[0]
        probs = model.predict_proba(input_data)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(le.inverse_transform([i])[0], probs[i]) for i in top3_idx]

        st.markdown("---")
        st.markdown("## 🎉 Your Results")

        rd = ROADMAPS[prediction]
        st.markdown(f"""
        <div class="career-card">
            <h2>{rd['emoji']} {prediction} — Best Match!</h2>
            <p style="color:#ccc">{rd['description']}</p>
            <p><strong>💰 Avg Salary:</strong> {rd['avg_salary']}</p>
        </div>""", unsafe_allow_html=True)

        # Top 3
        st.markdown("### 🏆 Top 3 Career Matches")
        c1, c2, c3 = st.columns(3)
        for col, (career, prob), medal in zip([c1,c2,c3], top3, ["🥇","🥈","🥉"]):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size:2rem">{medal}</div>
                    <div style="font-weight:700;margin:0.4rem 0">{career}</div>
                    <div style="font-size:1.5rem;color:#a78bfa">{prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        # Skill chart
        st.markdown("### 📡 Your Skill Profile")
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_colors = ["#a78bfa" if v>=7 else "#667eea" if v>=4 else "#374151"
                      for v in skill_values.values()]
        ax.barh([SKILL_LABELS[s] for s in SKILLS],
                [skill_values[s] for s in SKILLS], color=bar_colors)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Skill Level", color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")
        ax.legend(handles=[
            mpatches.Patch(color="#a78bfa", label="Expert (7–10)"),
            mpatches.Patch(color="#667eea", label="Intermediate (4–6)"),
            mpatches.Patch(color="#374151", label="Beginner (0–3)"),
        ], loc="lower right", facecolor="#1e1e2e", labelcolor="white")
        st.pyplot(fig)

        # Roadmap
        st.markdown(f"### 🗺️ Personalized Roadmap — {prediction}")
        for step_title, step_desc in rd["steps"]:
            st.markdown(f"""
            <div class="step-card">
                <strong>{step_title}</strong><br>
                <span style="color:#ccc">{step_desc}</span>
            </div>""", unsafe_allow_html=True)

        # Tools & Resources
        col_t, col_r = st.columns(2)
        with col_t:
            st.markdown("### 🛠️ Key Tools")
            for tool in rd["tools"]:
                st.markdown(f'<span class="tag">🔧 {tool}</span>', unsafe_allow_html=True)
        with col_r:
            st.markdown("### 📚 Resources")
            for res in rd["resources"]:
                st.markdown(f'<span class="tag">📖 {res}</span>', unsafe_allow_html=True)

        # Job Listings
        st.markdown(f"### 💼 Live Remote Job Listings — {prediction}")
        with st.spinner("Fetching live jobs..."):
            jobs = fetch_jobs(rd["job_keyword"])
        if jobs:
            for job in jobs:
                st.markdown(f"""
                <div class="job-card">
                    <strong>🏢 {job.get('company_name','N/A')}</strong> —
                    <span style="color:#a78bfa">{job.get('title','N/A')}</span><br>
                    📍 {job.get('candidate_required_location','Remote')} &nbsp;|&nbsp;
                    🏷️ {job.get('job_type','Full-time')}<br>
                    <a href="{job.get('url','#')}" target="_blank"
                       style="color:#667eea">🔗 View Job Posting</a>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No live listings found right now. Check back later!")

        # PDF Download
        st.markdown("### 📥 Download Your Report")
        pdf_buffer = generate_pdf(prediction, skill_values, top3)
        st.download_button(
            label="📄 Download PDF Roadmap Report",
            data=pdf_buffer,
            file_name=f"career_roadmap_{prediction.replace(' ','_')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Resume Scorer
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📄 Resume Skill Scorer")
    st.markdown("Paste your resume text below and we'll automatically score your skills and predict your best career match.")
    st.markdown("---")

    resume_text = st.text_area(
        "Paste your resume here...",
        height=300,
        placeholder="Paste the full text of your resume here. Include skills, experience, projects, education..."
    )

    _, col_rb, _ = st.columns([1, 2, 1])
    with col_rb:
        score_btn = st.button("🔍 Score My Resume", use_container_width=True, type="primary")

    if score_btn and resume_text.strip():
        resume_scores = score_resume(resume_text)

        st.markdown("---")
        st.markdown("### 📊 Detected Skill Scores from Your Resume")

        # Show skill scores as a grid
        score_cols = st.columns(4)
        for i, (skill, score) in enumerate(resume_scores.items()):
            with score_cols[i % 4]:
                level = "🟣 Expert" if score>=7 else "🔵 Mid" if score>=4 else "⚪ Basic"
                st.metric(SKILL_LABELS[skill], f"{score}/10", level)

        # Predict from resume scores
        input_arr = np.array([[resume_scores[s] for s in SKILLS]])
        pred_enc = model.predict(input_arr)[0]
        resume_prediction = le.inverse_transform([pred_enc])[0]
        probs = model.predict_proba(input_arr)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_resume = [(le.inverse_transform([i])[0], probs[i]) for i in top3_idx]

        rd = ROADMAPS[resume_prediction]
        st.markdown("---")
        st.markdown(f"""
        <div class="career-card">
            <h2>{rd['emoji']} Best Career Match: {resume_prediction}</h2>
            <p style="color:#ccc">{rd['description']}</p>
            <p><strong>💰 Avg Salary:</strong> {rd['avg_salary']}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 🏆 Top 3 Matches")
        r1, r2, r3 = st.columns(3)
        for col, (career, prob), medal in zip([r1,r2,r3], top3_resume, ["🥇","🥈","🥉"]):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size:2rem">{medal}</div>
                    <div style="font-weight:700;margin:0.4rem 0">{career}</div>
                    <div style="font-size:1.5rem;color:#a78bfa">{prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        # Skill gap analysis
        st.markdown("### 🔍 Skill Gap Analysis")
        st.markdown(f"Skills you should improve for **{resume_prediction}**:")
        gaps = {s: v for s, v in resume_scores.items() if v < 5}
        if gaps:
            gap_cols = st.columns(4)
            for i, (skill, score) in enumerate(gaps.items()):
                with gap_cols[i % 4]:
                    needed = 7 - score
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size:0.85rem">{SKILL_LABELS[skill]}</div>
                        <div style="color:#f87171">+{needed} needed</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.success("🎉 Your skills look strong for this career path!")

        # PDF for resume prediction
        st.markdown("### 📥 Download Your Report")
        pdf_buf = generate_pdf(resume_prediction, resume_scores, top3_resume)
        st.download_button(
            label="📄 Download PDF Roadmap Report",
            data=pdf_buf,
            file_name=f"resume_roadmap_{resume_prediction.replace(' ','_')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    elif score_btn:
        st.warning("⚠️ Please paste your resume text first!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Performance & Insights")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", "Random Forest")
    c2.metric("CV Accuracy", "90%+")
    c3.metric("Training Samples", "2,750")
    c4.metric("Career Tracks", "11")
    st.markdown("---")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("#### 🔥 Feature Importance")
        try:
            st.image("data/feature_importance.png", use_column_width=True)
        except:
            st.info("Run train_model.py to generate this chart.")
    with i2:
        st.markdown("#### 🎯 Confusion Matrix")
        try:
            st.image("data/confusion_matrix.png", use_column_width=True)
        except:
            st.info("Run train_model.py to generate this chart.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    **AI-Driven Career Guidance & Roadmap System** predicts the best tech career
    track based on your skills and generates a personalized learning roadmap.

    #### 🧠 ML Model
    | Property | Detail |
    |----------|--------|
    | Algorithm | Random Forest (200 trees) |
    | Input | 20 skill ratings (0–10) |
    | Output | 11 career tracks |
    | Validation | 5-Fold Stratified Cross Validation |
    | Accuracy | 90%+ |

    #### ✨ Features
    - Skill-based career prediction with confidence scores
    - Resume text parsing & automatic skill scoring
    - Skill gap analysis
    - Live remote job listings (Remotive API)
    - Downloadable PDF roadmap report
    - Feature importance & confusion matrix visualizations

    #### 🏗️ Tech Stack
    | Layer | Technology |
    |-------|-----------|
    | Language | Python 3.x |
    | ML | Scikit-learn |
    | Dashboard | Streamlit |
    | PDF | ReportLab |
    | Jobs API | Remotive (free) |
    | Data | Pandas, NumPy |
    """)