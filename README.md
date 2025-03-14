# AI-Powered Resume Screening & Candidate Ranking System

An advanced AI-based system that automates resume screening and candidate ranking using **Google Gemini API**, **TF-IDF**, **cosine similarity**, and **Streamlit** for an interactive user interface.

## Features

- **AI-Powered Resume Matching**: Uses Google Gemini API for semantic understanding.
- **Interactive UI**: Built with **Streamlit** for a sleek, user-friendly interface.
- **Resume Parsing**: Extracts text from PDF resumes using **PyPDF2**.
- **Ranking System**: Scores resumes by matching them against job descriptions using **TF-IDF** and **cosine similarity**.
- **Dynamic Table Display**: Presents candidate rankings in a clear, sortable table.

## Project Structure

```
├── main.py            # Main application script
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MeghnaP0705/AI-Resume-Screening-and-Candidate-Matching
cd AI-Resume-Screening-and-Candidate-Matching
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Google Gemini API
1. Sign up at [Google AI Studio](https://aistudio.google.com/).
2. Generate an API key.
3. Add your key to the `main.py` code
```bash
GEMINI_API_KEY="your-api-key"  
```

### 5. Run the Application
```bash
streamlit run main.py
```

1. **Enter a job description** in the provided field.
2. **Upload resumes** (PDF format).
3. **View ranked candidates** based on relevance.

## Output
- Displays a **sortable ranking table** with resume names and similarity scores.

## Dependencies
- Python 3.10+
- Streamlit
- Google Generative AI (Gemini API)
- PyPDF2
- scikit-learn



