import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

DB_PATH = "dataset_cultureMonkey.xlsx"

# ========== File Text Extractors ==========
def extract_pdf_text(file):
    try:
        reader = PdfReader(file)
        return '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

def extract_word_text(file):
    try:
        doc = Document(file)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"❌ Error reading Word document: {e}")
        return ""

# ========== Resume Info Extraction ==========
def extract_resume_details(text):
    lines = text.split("\n")
    summary_sections = {
        "Skills": ["Skills", "Technical Skills", "Core Competencies"],
        "Achievements": ["Achievements", "Accomplishments", "Key Highlights"],
        "Experience": ["Experience", "Work Experience", "Professional Experience"],
        "Projects": ["Projects", "Key Projects", "Academic Projects"]
    }
    extracted_info = {key: [] for key in summary_sections}
    current_section = None
    for line in lines:
        line = line.strip()
        for section, keywords in summary_sections.items():
            if any(line.lower().startswith(keyword.lower()) for keyword in keywords):
                current_section = section
                break
        else:
            if current_section:
                extracted_info[current_section].append(line)

    formatted_output = {key: "\n".join(value) for key, value in extracted_info.items() if value}

    skills_list = extracted_info.get("Skills", [])
    skill_objects = []
    for skill in skills_list:
        if skill:
            skill_objects.append({
                "skill": skill,
                "category": "established",
                "trend_score": round(0.7 + 0.3 * (hash(skill) % 100) / 100, 2)
            })
    if skill_objects:
        formatted_output["Skills_JSON"] = skill_objects

    return formatted_output if formatted_output else "No structured data found. Please label resume sections clearly."

# ========== Resume Upload Logic ==========
def upload_data():
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader("📄 Upload a file (PDF, DOCX, or Excel)", type=["pdf", "docx", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".pdf"):
                text = extract_pdf_text(uploaded_file)
                summary = extract_resume_details(text)
                st.session_state.resume_summary = summary
                st.success("✅ Resume processed successfully!")
                st.write(summary)

            elif uploaded_file.name.endswith(".docx"):
                text = extract_word_text(uploaded_file)
                summary = extract_resume_details(text)
                st.session_state.resume_summary = summary
                st.success("✅ Resume processed successfully!")
                st.write(summary)

            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
                st.session_state.uploaded_df = df
                st.success("✅ Excel data uploaded!")
                st.write("📊 Data Preview:")
                st.dataframe(df.head())
                st.write(f"📜 Rows: {len(df)} | Columns: {', '.join(df.columns)}")

            else:
                st.error("❌ Unsupported file format!")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ========== Load Predefined Interview Questions ==========
def load_database():
    try:
        if os.path.exists(DB_PATH):
            df = pd.read_excel(DB_PATH, engine='openpyxl')
            df.columns = df.columns.str.strip()
            return df
        else:
            st.warning("⚠️ Database not found! Initializing empty one.")
            return pd.DataFrame(columns=["job_title", "job_description_text"])
    except Exception as e:
        st.error(f"❌ Error loading database: {e}")
        return pd.DataFrame()

# ========== Resume to Role Matching ==========
def match_resume_to_roles(resume_text, job_df, top_n=3):
    if job_df.empty or "job_description_text" not in job_df.columns or "job_title" not in job_df.columns:
        return []
    descriptions = job_df["job_description_text"].fillna("").tolist()
    roles = job_df["job_title"].fillna("Unknown Role").tolist()
    corpus = descriptions + [resume_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    matched_roles = [roles[i] for i in top_indices]
    return matched_roles

# ========== Visual Analysis Section ==========
def generate_visualizations(job_df):
    if job_df.empty:
        st.warning("Dataset is empty or missing")
        return

    st.subheader("📊 Visual Analysis")
    st.write(f"Dataset columns: {job_df.columns.tolist()}")
    st.write(f"Dataset shape: {job_df.shape}")

    if "company_address_region" in job_df.columns:
        location_counts = job_df['company_address_region'].dropna().value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=location_counts.values, y=location_counts.index, ax=ax)
        st.pyplot(fig)

    if "job_title" in job_df.columns:
        title_counts = job_df['job_title'].dropna().value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=title_counts.values, y=title_counts.index, ax=ax)
        st.pyplot(fig)

    if "job_posted_date" in job_df.columns:
        job_df['job_posted_date'] = pd.to_datetime(job_df['job_posted_date'], errors='coerce')
        job_df['month'] = job_df['job_posted_date'].dt.to_period('M')
        monthly_counts = job_df['month'].value_counts().sort_index()
        fig, ax = plt.subplots()
        monthly_counts.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    if "job_description_text" in job_df.columns and "seniority_level" in job_df.columns:
        seniority_clean = job_df['seniority_level'].fillna("").str.lower()

        entry_texts = job_df[seniority_clean.str.contains("entry")]['job_description_text'].dropna().str.cat(sep=' ')
        senior_texts = job_df[seniority_clean.str.contains("senior")]['job_description_text'].dropna().str.cat(sep=' ')

        entry_vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        entry_features = entry_vectorizer.fit_transform([entry_texts])
        entry_words = entry_vectorizer.get_feature_names_out()

        senior_vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        senior_features = senior_vectorizer.fit_transform([senior_texts])
        senior_words = senior_vectorizer.get_feature_names_out()

        st.markdown("#### 🔍 Skill Differences: Entry vs Senior Roles")
        st.markdown("**Top Entry-Level Skills:**")
        st.write(entry_words[:10])

        st.markdown("**Top Senior-Level Skills:**")
        st.write(senior_words[:10])

        wordcloud_entry = WordCloud(width=600, height=300, background_color='white').generate(entry_texts)
        st.image(wordcloud_entry.to_array(), caption="Entry-Level Skill Cloud")

        wordcloud_senior = WordCloud(width=600, height=300, background_color='white').generate(senior_texts)
        st.image(wordcloud_senior.to_array(), caption="Senior-Level Skill Cloud")

        all_texts = job_df['job_description_text'].dropna().str.cat(sep=' ')
        overall_vectorizer = TfidfVectorizer(stop_words='english')
        all_features = overall_vectorizer.fit_transform([all_texts])
        all_words = overall_vectorizer.get_feature_names_out()
        tfidf_scores = all_features.toarray()[0]
        top_indices = tfidf_scores.argsort()[-3:][::-1]
        st.markdown("#### 💡 Top 3 In-Demand Skills Across All Positions")
        for idx in top_indices:
            st.write(f"🔹 {all_words[idx]}")

        st.markdown("#### 📌 Interesting Pattern")
        if "employment_type" in job_df.columns:
            employment_counts = job_df['employment_type'].value_counts()
            st.write("Most common employment type:", employment_counts.idxmax())
            fig, ax = plt.subplots()
            sns.barplot(x=employment_counts.values, y=employment_counts.index, ax=ax)
            st.pyplot(fig)
        else:
            st.write("No 'employment_type' column found to analyze common job types.")

# ========== Streamlit Main UI ==========
def main():
    st.set_page_config(page_title="🤖 Job Market Analysis System ", layout="wide")
    st.title("🤖 Job Market Analysis System")
    st.markdown("Upload your resume, match to roles, and practice your interview! 🚀")
    st.sidebar.title("🛍️ Navigation")
    options = st.sidebar.radio("Choose a section:", ["🏠 Home", "📄 Resume & Interview", "⬇️ Download", "ℹ️ About"])

    if "resume_summary" not in st.session_state:
        st.session_state.resume_summary = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "role" not in st.session_state:
        st.session_state.role = None
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "transcripts" not in st.session_state:
        st.session_state.transcripts = []

    if options == "🏠 Home":
        st.header("👋 Welcome")
        st.write("""
        Welcome! This app helps you analyze your resume and prepare for interviews with ease.  
        Just upload your resume to get quick insights, personalized tips, and boost your chances of landing the job. 🌟  
        It highlights your strengths, suggests improvements, and even matches your profile to job roles.  
        Get started now and take a step closer to your dream career! 🚀
        
        **👉 Please use the side navigation bar to move to the next section.**
        """)

    elif options == "ℹ️ About":
        st.header("📚 About This App")
    
        st.markdown("""
        This app is designed to assist with **resume analysis** and **job market insights**  
        as part of a recruitment system simulation by **CultureMonkey**.
    
        Built using **Python** and **Streamlit**, the system enables users to upload resumes,  
        analyze skills, and get matched to relevant job roles. It's a smart, AI-assisted tool  
        aimed at improving candidate-job alignment. 💼
        """)
    
        st.markdown("### 🎯 Objective")
        st.markdown("""
        Design and implement a job market analysis system using the provided dataset,  
        focusing on extracting useful information from resumes and job listings.
        """)
    
        st.markdown("### 🛠️ Key Features")
        st.markdown("""
        - 📄 **Resume Parsing & Analysis**  
          Extracts skills and experience from resumes.
        
        - 🔍 **Job Role Matching**  
          Suggests job roles based on the resume content using similarity scoring.
        
        - 📊 **Skill Demand Visualization**  
          Shows trends and demand for various skills in the job market.
        
        - 🚀 **Gap Analysis**  
          Identifies missing but in-demand skills to guide candidates.
        """)
    
        st.markdown("---")
        st.markdown("Built by **Adarsh Ojaswi Singh** as part of a recruitment system simulation. ✨")


    elif options == "📄 Resume & Interview":
        col1, col2 = st.columns(2)
        with col1:
            upload_data()
        with col2:
            st.subheader("🎤 Matching Job Descriptions")
            database = st.session_state.get("uploaded_df", load_database())
            matched_roles = []
            if st.session_state.resume_summary:
                resume_text = "\n".join(
                    val if isinstance(val, str) else str(val)
                    for key, val in st.session_state.resume_summary.items()
                    if key != "Skills_JSON"
                )
                matched_roles = match_resume_to_roles(resume_text, database)
            selected_role = st.selectbox("🔍 Select matched role:", matched_roles or database["job_title"].dropna().unique().tolist())
            if st.button("▶️ Start Interview"):
                if selected_role:
                    st.session_state.role = selected_role
                    st.session_state.conversation = []
                    st.session_state.transcripts = database[database["job_title"] == selected_role]["job_description_text"].dropna().tolist()
                    if st.session_state.transcripts:
                        st.session_state.current_question = st.session_state.transcripts.pop(0)
                        st.session_state.conversation.append(("Interviewer", st.session_state.current_question))
            if st.session_state.get("current_question"):
                st.write(f"**💼 Interviewer:** {st.session_state.current_question}")
                answer = st.text_area("✍️ Your Answer:")
                if st.button("📄 Submit Response"):
                    if answer.strip():
                        st.session_state.conversation.append(("Candidate", answer))
                        if st.session_state.transcripts:
                            st.session_state.current_question = st.session_state.transcripts.pop(0)
                            st.session_state.conversation.append(("Interviewer", st.session_state.current_question))
                        else:
                            st.success("🎉 Interview complete!")
                            st.session_state.current_question = None
                    else:
                        st.warning("⚠️ Answer cannot be empty.")

            st.markdown("---")
            if st.button("📊 Visual Analysis"):
                generate_visualizations(database)

    elif options == "⬇️ Download":
        st.header("📅 Download Results")
        if st.session_state.conversation:
            transcript = "\n".join([f"{role}: {text}" for role, text in st.session_state.conversation])
            resume_summary = ""
            if st.session_state.resume_summary:
                if isinstance(st.session_state.resume_summary, dict):
                    resume_summary = "\n\n".join([f"{sec}:\n{cont}" for sec, cont in st.session_state.resume_summary.items()])
                else:
                    resume_summary = str(st.session_state.resume_summary)
            full_output = transcript + ("\n\nResume Summary:\n" + resume_summary if resume_summary else "")
            st.download_button("📥 Download Full Report", data=full_output, file_name="interview_summary.txt", mime="text/plain")
            if resume_summary:
                st.download_button("📥 Download Resume Summary", data=resume_summary, file_name="resume_summary.txt", mime="text/plain")
        else:
            st.info("ℹ️ Nothing to download yet.")

if __name__ == "__main__":
    main()
