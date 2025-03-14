import streamlit as st
import PyPDF2
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Set up Gemini API key
genai.configure(api_key="your-api-key")

# Extract text from uploaded PDF using PyPDF2
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Generate embeddings using Gemini API
def get_gemini_embedding(text):
    response = genai.embed_content(
        model="models/text-embedding-004",  # Use the correct model name
        content=text
    )
    return np.array(response["embedding"])

# Calculate similarity between job description and resumes
def calculate_similarity(job_description, resumes):
    job_embedding = get_gemini_embedding(job_description)
    resume_scores = []

    for name, resume_text in resumes:
        resume_embedding = get_gemini_embedding(resume_text)
        score = cosine_similarity(job_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0][0]
        resume_scores.append((name, score))

    return sorted(resume_scores, key=lambda x: x[1], reverse=True)

# Streamlit UI setup
st.set_page_config(page_title="AI Resume Screening & Candidate Ranking System", layout="centered")

# Header with styling
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            color: #FFFFFF;
            font-size: 40px;
            font-family: 'Times New Roman', Times, serif;

        }
        .sub-title {
            color: #00BFFF;
            font-size: 30px;
            margin-top: 20px;
            font-family: 'Times New Roman', Times, serif;
        }
    </style>
    <div class="main-title">AI Resume Screening & Candidate Ranking System</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation


# Job Description Input
st.markdown("<div class='sub-title'>Job Description</div>", unsafe_allow_html=True)
job_description = st.text_area("Enter the job description", height=150, placeholder="Enter job requirements...")

# Upload Resume Section
st.markdown("<div class='sub-title'>Upload Resumes</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# Process Resumes and Display Rankings
if uploaded_files and job_description:
    st.info("Processing resumes... This may take a moment.")

    # Extract resume text
    resumes = [(file.name, extract_text_from_pdf(file)) for file in uploaded_files]

    # Calculate similarity scores
    ranked_resumes = calculate_similarity(job_description, resumes)

    # Display rankings
    st.markdown("<div class='sub-title'>Ranking Resumes</div>", unsafe_allow_html=True)

    # Create DataFrame for better table display
    ranking_df = pd.DataFrame(ranked_resumes, columns=["Resume", "Score"])
    ranking_df["Score"] = ranking_df["Score"].round(3)

    # Styled DataFrame
    st.dataframe(ranking_df.style.format({"Score": "{:.3f}"}).set_table_styles(
        [{"selector": "thead th", "props": [("background-color", "#00BFFF"), ("color", "white")]}]
    ))

# Footer
st.markdown("---")
st.markdown("<center>Built with ❤️ using Streamlit & Gemini API</center>", unsafe_allow_html=True)
