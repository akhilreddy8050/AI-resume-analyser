
import streamlit as st
import PyPDF2
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer")
st.subheader("Analyze your resume with Artificial Intelligence")

st.write(
    "Upload your resume and compare it with a job description "
    "to identify matching skills and improvement areas."
)

# --------------------------------
# SKILLS DATABASE
# --------------------------------

SKILLS = [
    "python",
    "java",
    "c++",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "sql",
    "mysql",
    "mongodb",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "nlp",
    "natural language processing",
    "computer vision",
    "pandas",
    "numpy",
    "matplotlib",
    "streamlit",
    "flask",
    "django",
    "html",
    "css",
    "javascript",
    "react",
    "git",
    "github",
    "aws",
    "cloud computing",
    "statistics",
    "communication",
    "problem solving",
    "data structures",
    "algorithms"
]

# --------------------------------
# PDF TEXT EXTRACTION
# --------------------------------

def extract_resume_text(pdf_file):

    text = ""

    try:
        reader = PyPDF2.PdfReader(pdf_file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

    return text


# --------------------------------
# TEXT PREPROCESSING
# --------------------------------

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# --------------------------------
# SKILL EXTRACTION
# --------------------------------

def extract_skills(text):

    text = preprocess_text(text)

    found_skills = []

    for skill in SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills


# --------------------------------
# RESUME MATCHING
# --------------------------------

def calculate_similarity(resume_text, job_text):

    if not resume_text.strip() or not job_text.strip():
        return 0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:

        vectors = vectorizer.fit_transform(
            [resume_text, job_text]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(similarity * 100, 2)

    except ValueError:
        return 0


# --------------------------------
# RESUME SUGGESTIONS
# --------------------------------

def generate_suggestions(
    resume_text,
    job_text,
    missing_skills
):

    suggestions = []

    resume_words = len(resume_text.split())

    if resume_words < 100:

        suggestions.append(
            "Resume content is short. Add relevant "
            "projects, skills, and experience."
        )

    if not re.search(
        r"\b(project|projects)\b",
        resume_text,
        re.IGNORECASE
    ):

        suggestions.append(
            "Add a Projects section describing your "
            "academic or practical projects."
        )

    if not re.search(
        r"\b(education|degree|college|university)\b",
        resume_text,
        re.IGNORECASE
    ):

        suggestions.append(
            "Add your education details."
        )

    if missing_skills:

        suggestions.append(
            "Review the missing skills and include "
            "relevant ones only if you actually possess them."
        )

    if not suggestions:

        suggestions.append(
            "Your resume contains useful information. "
            "Continue improving it with relevant projects "
            "and measurable achievements."
        )

    return suggestions


# --------------------------------
# USER INTERFACE
# --------------------------------

col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"]
    )

with col2:

    job_description = st.text_area(
        "Paste Job Description",
        height=200,
        placeholder="Paste the job description here..."
    )


# --------------------------------
# ANALYZE BUTTON
# --------------------------------

if st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True
):

    if uploaded_file is None:

        st.warning("Please upload your resume.")

    elif not job_description.strip():

        st.warning("Please enter a job description.")

    else:

        resume_text = extract_resume_text(
            uploaded_file
        )

        if not resume_text.strip():

            st.error(
                "No readable text found in the PDF. "
                "Try a text-based PDF resume."
            )

        else:

            cleaned_resume = preprocess_text(
                resume_text
            )

            cleaned_job = preprocess_text(
                job_description
            )

            # Calculate similarity

            match_score = calculate_similarity(
                cleaned_resume,
                cleaned_job
            )

            # Extract skills

            resume_skills = extract_skills(
                cleaned_resume
            )

            job_skills = extract_skills(
                cleaned_job
            )

            matching_skills = [
                skill for skill in job_skills
                if skill in resume_skills
            ]

            missing_skills = [
                skill for skill in job_skills
                if skill not in resume_skills
            ]

            # --------------------------------
            # RESULTS
            # --------------------------------

            st.divider()

            st.header("📊 Resume Analysis Results")

            result_col1, result_col2, result_col3 = (
                st.columns(3)
            )

            with result_col1:

                st.metric(
                    "Resume Match Score",
                    f"{match_score}%"
                )

            with result_col2:

                st.metric(
                    "Matching Skills",
                    len(matching_skills)
                )

            with result_col3:

                st.metric(
                    "Missing Skills",
                    len(missing_skills)
                )

            # --------------------------------
            # PROGRESS BAR
            # --------------------------------

            st.subheader("Resume Matching Percentage")

            st.progress(
                min(match_score / 100, 1.0)
            )

            # --------------------------------
            # MATCHING SKILLS
            # --------------------------------

            st.subheader("✅ Matching Skills")

            if matching_skills:

                st.write(
                    ", ".join(matching_skills)
                )

            else:

                st.info(
                    "No matching skills detected."
                )

            # --------------------------------
            # MISSING SKILLS
            # --------------------------------

            st.subheader("⚠️ Missing Skills")

            if missing_skills:

                st.write(
                    ", ".join(missing_skills)
                )

            else:

                st.success(
                    "All detected job skills are present "
                    "in your resume."
                )

            # --------------------------------
            # SUGGESTIONS
            # --------------------------------

            st.subheader("💡 Resume Improvement Suggestions")

            suggestions = generate_suggestions(
                cleaned_resume,
                cleaned_job,
                missing_skills
            )

            for suggestion in suggestions:

                st.write(
                    "• " + suggestion
                )

            # --------------------------------
            # DOWNLOAD REPORT
            # --------------------------------

            report = f"""
AI RESUME ANALYZER REPORT

Resume Match Score: {match_score}%

Matching Skills:
{', '.join(matching_skills)}

Missing Skills:
{', '.join(missing_skills)}

Suggestions:
{chr(10).join(suggestions)}
"""

            st.download_button(
                label="📥 Download Analysis Report",
                data=report,
                file_name="resume_analysis_report.txt",
                mime="text/plain"
            )

            st.success(
                "Resume Analysis Completed Successfully!"
            )

            st.caption(
                "This is an educational resume analysis tool. "
                "The score is a text-similarity estimate, not "
                "a prediction of hiring success."
            )
