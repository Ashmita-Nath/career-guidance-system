import pandas as pd
import numpy as np
import os

np.random.seed(42)

SKILLS = [
    "python", "javascript", "sql", "machine_learning", "statistics",
    "react", "nodejs", "docker", "kubernetes", "networking",
    "linux", "swift", "kotlin", "cloud", "data_analysis",
    "deep_learning", "communication", "project_management", "uiux_design", "git"
]

CAREER_PROFILES = {
    "Data Scientist": {
        "high": ["python", "sql", "machine_learning", "statistics", "data_analysis"],
        "medium": ["deep_learning", "cloud", "communication", "git"],
        "low": ["javascript", "react", "nodejs", "docker", "kubernetes",
                "networking", "linux", "swift", "kotlin", "project_management", "uiux_design"]
    },
    "Web Developer": {
        "high": ["javascript", "react", "nodejs", "git", "uiux_design"],
        "medium": ["sql", "python", "docker", "communication"],
        "low": ["machine_learning", "statistics", "kubernetes", "networking",
                "linux", "swift", "kotlin", "cloud", "data_analysis",
                "deep_learning", "project_management"]
    },
    "Mobile Developer": {
        "high": ["swift", "kotlin", "git", "uiux_design", "javascript"],
        "medium": ["react", "sql", "python", "communication"],
        "low": ["machine_learning", "statistics", "nodejs", "docker", "kubernetes",
                "networking", "linux", "cloud", "data_analysis",
                "deep_learning", "project_management"]
    },
    "DevOps Engineer": {
        "high": ["docker", "kubernetes", "linux", "cloud", "git"],
        "medium": ["python", "networking", "sql", "communication"],
        "low": ["machine_learning", "statistics", "react", "nodejs",
                "swift", "kotlin", "data_analysis", "deep_learning",
                "project_management", "uiux_design", "javascript"]
    },
    "Cybersecurity Analyst": {
        "high": ["networking", "linux", "python", "git", "cloud"],
        "medium": ["docker", "sql", "communication", "kubernetes"],
        "low": ["machine_learning", "statistics", "react", "nodejs",
                "swift", "kotlin", "data_analysis", "deep_learning",
                "project_management", "uiux_design", "javascript"]
    },
    "AI/ML Engineer": {
        "high": ["python", "machine_learning", "deep_learning", "statistics", "git"],
        "medium": ["cloud", "sql", "data_analysis", "docker", "linux"],
        "low": ["javascript", "react", "nodejs", "kubernetes", "networking",
                "swift", "kotlin", "project_management", "uiux_design", "communication"]
    },
    "Cloud Architect": {
        "high": ["cloud", "docker", "kubernetes", "linux", "networking"],
        "medium": ["python", "sql", "git", "communication", "project_management"],
        "low": ["machine_learning", "statistics", "react", "nodejs",
                "swift", "kotlin", "data_analysis", "deep_learning",
                "uiux_design", "javascript"]
    },
    "Product Manager": {
        "high": ["communication", "project_management", "uiux_design", "sql", "git"],
        "medium": ["python", "javascript", "data_analysis", "cloud"],
        "low": ["machine_learning", "statistics", "react", "nodejs", "docker",
                "kubernetes", "networking", "linux", "swift", "kotlin", "deep_learning"]
    },
    # ── 3 New Career Tracks ──────────────────────────────────────────────────
    "UI/UX Designer": {
        "high": ["uiux_design", "javascript", "react", "communication", "git"],
        "medium": ["nodejs", "python", "project_management", "sql"],
        "low": ["machine_learning", "statistics", "docker", "kubernetes",
                "networking", "linux", "swift", "kotlin", "cloud",
                "data_analysis", "deep_learning"]
    },
    "Data Engineer": {
        "high": ["python", "sql", "cloud", "data_analysis", "git"],
        "medium": ["docker", "kubernetes", "linux", "statistics", "communication"],
        "low": ["machine_learning", "deep_learning", "react", "nodejs",
                "swift", "kotlin", "uiux_design", "project_management",
                "javascript", "networking"]
    },
    "Blockchain Developer": {
        "high": ["python", "javascript", "networking", "git", "linux"],
        "medium": ["nodejs", "sql", "cloud", "docker", "communication"],
        "low": ["machine_learning", "statistics", "react", "kubernetes",
                "swift", "kotlin", "data_analysis", "deep_learning",
                "project_management", "uiux_design"]
    }
}

def generate_sample(career, n=250):
    profile = CAREER_PROFILES[career]
    rows = []
    for _ in range(n):
        row = {}
        for skill in SKILLS:
            if skill in profile["high"]:
                row[skill] = np.random.randint(6, 11)
            elif skill in profile["medium"]:
                row[skill] = np.random.randint(3, 8)
            else:
                row[skill] = np.random.randint(0, 5)
        row["career_track"] = career
        rows.append(row)
    return rows

all_rows = []
for career in CAREER_PROFILES:
    all_rows.extend(generate_sample(career, n=250))

df = pd.DataFrame(all_rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/career_dataset.csv", index=False)

print(f"✅ Dataset created: {len(df)} rows x {len(df.columns)} columns")
print(f"Career track distribution:\n{df['career_track'].value_counts()}")