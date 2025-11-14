import streamlit as st
import os
import pdfplumber
import docx
import openpyxl
import json
import tempfile
from groq import Groq
import traceback
import re 
from dotenv import load_dotenv 
from datetime import date 
from streamlit.runtime.uploaded_file_manager import UploadedFile

# -------------------------
# CONFIGURATION & API SETUP (Necessary for standalone functions)
# -------------------------

GROQ_MODEL = "llama-3.1-8b-instant"
# Load environment variables (mocked if running standalone)
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq Client or Mock Client (Must be present for function definitions)
if not GROQ_API_KEY:
    class MockGroqClient:
        def chat(self):
            class Completions:
                def create(self, **kwargs):
                    raise ValueError("GROQ_API_KEY not set. AI functions disabled.")
            return Completions()
    client = MockGroqClient()
else:
    client = Groq(api_key=GROQ_API_KEY)

# --- Utility Functions (Duplicated/shared for both dashboards) ---

def go_to(page_name):
    """Changes the current page in Streamlit's session state."""
    st.session_state.page = page_name

def get_file_type(file_path):
    """Identifies the file type based on its extension."""
    ext = os.path.splitext(file_path)[1].lower().strip('.')
    if ext == 'pdf': return 'pdf'
    elif ext == 'docx': return 'docx'
    elif ext == 'xlsx': return 'xlsx'
    else: return 'txt' 

def extract_content(file_type, file_path):
    """Extracts text content from various file types (Simplified for resume parsing)."""
    text = ''
    try:
        if file_type == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        elif file_type == 'docx':
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
        # Only PDF/DOCX/TXT are critical for resumes; ignoring XLSX for this context

        if not text.strip():
            return f"Error: {file_type.upper()} content extraction failed or file is empty."
        
        return text
    
    except Exception as e:
        return f"Fatal Extraction Error: Failed to read file content ({file_type}). Error: {e}"


@st.cache_data(show_spinner="Extracting JD metadata...")
def extract_jd_metadata(jd_text):
    """Extracts structured metadata (Role, Job Type, Key Skills) from raw JD text."""
    if not GROQ_API_KEY:
        return {"role": "N/A", "job_type": "N/A", "key_skills": []}

    prompt = f"""Analyze the following Job Description and extract the key metadata.
    
    Job Description:
    {jd_text}
    
    Provide the output strictly as a JSON object with the following three keys:
    1.  **role**: The main job title (e.g., 'Data Scientist', 'Senior Software Engineer'). If not clear, default to 'General Analyst'.
    3.  **key_skills**: A list of 5 to 10 most critical hard and soft skills required.
    """
    content = ""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            parsed = json.loads(json_str)
        else:
            raise json.JSONDecodeError("Could not isolate a valid JSON structure from LLM response.", content, 0)
        
        return {
            "role": parsed.get("role", "General Analyst"),
            "key_skills": [s.strip() for s in parsed.get("key_skills", []) if isinstance(s, str)]
        }

    except Exception:
        return {"role": "General Analyst (LLM Error)", "key_skills": ["LLM Error", "Fallback"]}


@st.cache_data(show_spinner="Analyzing content with Groq LLM...")
def parse_with_llm(text):
    """Sends resume text to the LLM for structured information extraction."""
    if text.startswith("Error") or not GROQ_API_KEY:
        return {"error": "Parsing error or API key missing.", "raw_output": ""}

    prompt = f"""Extract the following information from the resume in structured JSON.
    - Name, - Email, - Phone, - Skills, - Education, 
    - Experience, - Certifications, 
    - Projects, - Strength, 
    - Personal Details, - Github, - LinkedIn
    
    Also, provide a key called **'summary'** which is a single, brief paragraph (3-4 sentences max) summarizing the candidate's career highlights and most relevant skills.
    
    Resume Text: {text}
    
    Provide the output strictly as a JSON object.
    """
    content = ""
    parsed = {}
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            parsed = json.loads(json_str)
        else:
            raise json.JSONDecodeError("Could not isolate a valid JSON structure.", content, 0)
    except Exception as e:
        parsed = {"error": f"LLM error: {e}", "raw_output": content}

    return parsed


def evaluate_jd_fit(job_description, parsed_json):
    """Evaluates how well a resume fits a given job description, including section-wise scores."""
    if not GROQ_API_KEY or "error" in parsed_json: return "AI Evaluation Disabled or resume parsing failed."
    
    relevant_resume_data = {
        'Skills': parsed_json.get('skills', 'Not found or empty'),
        'Experience': parsed_json.get('experience', 'Not found or empty'),
        'Education': parsed_json.get('education', 'Not found or empty'),
    }
    resume_summary = json.dumps(relevant_resume_data, indent=2)

    prompt = f"""Evaluate how well the following resume content matches the provided job description.
    Job Description: {job_description}
    Resume Sections for Analysis: {resume_summary}
    Provide a detailed evaluation structured as follows:
    1.  **Overall Fit Score:** A score out of 10.
    2.  **Section Match Percentages:** A percentage score for the match in the key sections (Skills, Experience, Education).
    
    Format the output strictly as follows:
    Overall Fit Score: [Score]/10
    
    --- Section Match Analysis ---
    Skills Match: [XX]%
    Experience Match: [YY]%
    Education Match: [ZZ]%
    
    Strengths/Matches:
    - Point 1
    - Point 2
    
    Areas for Improvement:
    - Point 1
    - Point 2
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL, 
        messages=[{"role": "user", "content": prompt}], 
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def parse_and_analyze_resume(file_input, jd_content, source_type='file', jd_name="Job Description"):
    """
    Handles file/text input, parsing, and analysis for the candidate dashboard.
    Returns a dictionary with results.
    """
    text = None
    file_name = f"Resume ({date.today().strftime('%Y-%m-%d')})"
    
    if source_type == 'file':
        if not isinstance(file_input, UploadedFile):
            return {"error": "Invalid file input type passed to parser."}
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file_input.name) 
        with open(temp_path, "wb") as f:
            f.write(file_input.getbuffer()) 

        file_type = get_file_type(temp_path)
        text = extract_content(file_type, temp_path)
        file_name = file_input.name.split('.')[0]
    
    elif source_type == 'text':
        text = file_input
        
    if text.startswith("Error"):
        return {"error": text}

    # 1. Parse Resume
    parsed_resume = parse_with_llm(text)
    
    if "error" in parsed_resume:
        return {"error": f"Resume Parsing Failed: {parsed_resume.get('error', 'Unknown Error')}"}
    
    # 2. Extract JD Metadata
    jd_metadata = extract_jd_metadata(jd_content)
    
    # 3. Run Match Evaluation
    match_analysis = evaluate_jd_fit(jd_content, parsed_resume)
    
    overall_score_match = re.search(r'Overall Fit Score:\s*[^\d]*(\d+)\s*/10', match_analysis, re.IGNORECASE)
    overall_score = overall_score_match.group(1) if overall_score_match else 'N/A'
    
    return {
        "name": parsed_resume.get('name', file_name),
        "date": date.today().strftime("%Y-%m-%d"),
        "jd_name": jd_name,
        "jd_role": jd_metadata.get('role', 'N/A'),
        "overall_score": overall_score,
        "match_report": match_analysis,
        "parsed_resume": parsed_resume
    }

# -------------------------
# CANDIDATE DASHBOARD FUNCTION
# -------------------------

def candidate_dashboard():
    st.title("🧑‍💻 Candidate Dashboard")
    st.caption("Analyze your resume's fit for any job instantly.")
    
    col_header, col_logout = st.columns([4, 1])
    with col_logout:
        # Note: If a user system was implemented, this would reset user-specific state too.
        if st.button("🚪 Log Out", use_container_width=True):
            # Clear candidate specific analysis data
            if 'candidate_results' in st.session_state:
                del st.session_state.candidate_results
            if 'current_resume' in st.session_state:
                del st.session_state.current_resume
            go_to("login")
            st.rerun() 
            
    st.markdown("---")

    # --- Session State Initialization for Candidate ---
    if "candidate_results" not in st.session_state:
        # This stores a history of all analyses performed by the candidate
        st.session_state.candidate_results = []
    if "current_resume" not in st.session_state:
        # This stores the currently parsed resume details
        st.session_state.current_resume = None

    # --- Main Tabs ---
    tab_analysis, tab_history = st.tabs(["🚀 New Analysis", "📝 Application History"])

    with tab_analysis:
        st.header("Resume Match Analyzer")
        st.markdown("### Step 1: Upload or Paste Your Resume")
        
        # --- Resume Input ---
        resume_method = st.radio(
            "Select Resume Input Method", 
            ["Upload File (PDF/DOCX)", "Paste Text"],
            key="candidate_resume_method"
        )
        
        uploaded_file = None
        pasted_resume_text = ""
        
        if resume_method == "Upload File (PDF/DOCX)":
            uploaded_file = st.file_uploader(
                "Upload your Resume",
                type=["pdf", "docx", "txt"],
                key="candidate_resume_upload"
            )
        else:
            pasted_resume_text = st.text_area(
                "Paste your Resume Text here (Ensure formatting is clean)",
                height=300,
                key="candidate_resume_text_area"
            )

        st.markdown("---")
        
        st.markdown("### Step 2: Provide the Job Description (JD)")
        
        # --- JD Input ---
        jd_method = st.radio(
            "Select JD Input Method", 
            ["Paste JD Text", "LinkedIn URL (Simulated)"],
            key="candidate_jd_method"
        )
        
        jd_content = ""
        jd_name = "Custom JD"
        
        if jd_method == "Paste JD Text":
            jd_content = st.text_area(
                "Paste the full Job Description Text",
                height=300,
                key="candidate_jd_text_area"
            )
            
            # Allow user to set a title for easier tracking
            if jd_content:
                 first_line = jd_content.splitlines()[0].strip()
                 jd_name = st.text_input("Job Title for Tracking", value=first_line if len(first_line) < 50 else "Pasted JD", key="jd_title_input")
            
        elif jd_method == "LinkedIn URL (Simulated)":
            st.warning("⚠️ **Note:** Due to platform restrictions, this is a **simulation**. The system extracts a sample JD based on the URL and cannot fetch the real content.")
            
            linkedin_url = st.text_input(
                "Enter LinkedIn Job URL", 
                placeholder="e.g., https://www.linkedin.com/jobs/view/...", 
                key="linkedin_url_input"
            )
            
            if linkedin_url:
                # Use a simplified mock extractor for the URL (shared utility)
                from admin_dashboard_module import extract_jd_from_linkedin_url # Assuming the shared utility is here
                jd_content = extract_jd_from_linkedin_url(linkedin_url)
                
                # Try to derive a name from the URL for tracking
                try:
                    name_match = re.search(r'/jobs/view/([^/]+)', linkedin_url)
                    jd_name = name_match.group(1).split('?')[0].replace('-', ' ').title() if name_match else "LinkedIn Job"
                except:
                    jd_name = "LinkedIn Job"
                    
                st.info(f"Using simulated JD for: **{jd_name}**")
                
        st.markdown("---")
        
        # --- Run Analysis Button ---
        if st.button("✨ Run Resume Match Analysis", key="run_analysis_btn", type="primary", use_container_width=True):
            
            # --- Validation ---
            resume_input_valid = False
            if resume_method == "Upload File (PDF/DOCX)" and uploaded_file is not None:
                resume_input_valid = True
                resume_source = uploaded_file
                source_type = 'file'
            elif resume_method == "Paste Text" and pasted_resume_text.strip():
                resume_input_valid = True
                resume_source = pasted_resume_text.strip()
                source_type = 'text'
            
            if not resume_input_valid:
                st.error("❌ Please provide your resume using the selected method.")
            elif not jd_content.strip():
                st.error("❌ Please provide the Job Description content.")
            elif not GROQ_API_KEY:
                st.error("❌ AI Analysis is disabled. Please ensure the `GROQ_API_KEY` is set.")
            else:
                # --- Execution ---
                with st.spinner(f"Running analysis against '{jd_name}'... This may take a moment."):
                    try:
                        analysis_result = parse_and_analyze_resume(
                            resume_source, 
                            jd_content.strip(), 
                            source_type=source_type, 
                            jd_name=jd_name
                        )
                        
                        if "error" in analysis_result:
                            st.error(f"Analysis Failed: {analysis_result['error']}")
                        else:
                            # Success: Store result and display
                            st.session_state.candidate_results.insert(0, analysis_result)
                            st.session_state.current_resume = analysis_result
                            st.success(f"✅ Analysis complete! Score: **{analysis_result['overall_score']}/10**")
                            st.balloons()
                            st.rerun() # Rerun to move to the display section cleanly
                            
                    except Exception as e:
                        st.error(f"An unexpected error occurred during analysis: {e}")
                        st.code(traceback.format_exc())

        st.markdown("---")
        
        # --- Display Current Analysis Result ---
        
        if st.session_state.current_resume:
            
            result = st.session_state.current_resume
            st.header(f"Results for **{result['name']}**")
            
            col_score, col_jd_info, col_date = st.columns(3)
            
            with col_score:
                score = result['overall_score']
                st.metric(
                    label="Overall Match Score", 
                    value=f"{score}/10", 
                    delta="Excellent" if int(score) >= 8 else ("Good" if int(score) >= 6 else "Needs Work"), 
                    delta_color="normal"
                )
            with col_jd_info:
                st.markdown(f"**Target Role:** `{result.get('jd_role', 'N/A')}`")
                st.markdown(f"**Job Title:** `{result.get('jd_name', 'Custom JD')}`")
            with col_date:
                st.markdown(f"**Analysis Date:** `{result['date']}`")
                st.markdown(f"**Candidate Name:** `{result['name']}`")
                
            st.markdown("---")
            
            st.subheader("Detailed Match Report")
            st.text(result['match_report'])
            
            st.markdown("---")
            
            st.subheader("Parsed Resume Data (For Review)")
            
            parsed_data = result['parsed_resume']
            
            # Display Key parsed fields
            st.markdown(f"**Summary:** *{parsed_data.get('summary', 'N/A')}*")
            st.markdown(f"**Email:** `{parsed_data.get('email', 'N/A')}` | **Phone:** `{parsed_data.get('phone', 'N/A')}`")
            
            # Display detailed sections in expanders
            if parsed_data.get('experience'):
                with st.expander("Experience Details"):
                    st.json(parsed_data['experience'])
            
            if parsed_data.get('skills'):
                with st.expander("Skills List"):
                    st.json(parsed_data['skills'])
                    
            if parsed_data.get('education'):
                with st.expander("Education Details"):
                    st.json(parsed_data['education'])
                    
            if parsed_data.get('projects') or parsed_data.get('certifications'):
                with st.expander("Projects & Certifications"):
                    st.markdown("**Projects:**")
                    st.json(parsed_data.get('projects', 'N/A'))
                    st.markdown("**Certifications:**")
                    st.json(parsed_data.get('certifications', 'N/A'))

        else:
            st.info("Run an analysis above to view your first match report here.")


    with tab_history:
        st.header("Application History")
        
        if not st.session_state.candidate_results:
            st.info("No analysis results found. Run a new analysis on the 'New Analysis' tab to build your history.")
            return

        # Prepare data for display
        history_data = []
        for res in st.session_state.candidate_results:
            # Safely handle potential score non-integer values
            try:
                numeric_score = int(res['overall_score'])
            except:
                numeric_score = 0
            
            history_data.append({
                "Date": res['date'],
                "Job Title": res['jd_name'],
                "Role Found": res['jd_role'],
                "Match Score (out of 10)": res['overall_score'],
                "Sort_Score": numeric_score, # For sorting
            })
            
        # Sort by date (most recent first) and score
        history_data.sort(key=lambda x: (x['Date'], x['Sort_Score']), reverse=True)
        
        st.markdown("### Your Past Resume Analysis Matches")
        
        # Display as a dataframe
        st.dataframe(
            history_data,
            column_order=["Date", "Job Title", "Role Found", "Match Score (out of 10)"],
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### View Detailed Reports")
        
        # Display detailed reports in expanders
        for i, res in enumerate(st.session_state.candidate_results):
            header_text = f"{res['date']} | **{res['jd_name']}** (Score: **{res['overall_score']}/10**)"
            with st.expander(header_text):
                st.markdown("#### Full Match Report")
                st.text(res['match_report'])
                
                # Optional: Show parsed resume summary
                if res.get('parsed_resume', {}).get('summary'):
                    st.markdown("---")
                    st.markdown(f"**Resume Summary:** *{res['parsed_resume']['summary']}*")


# -------------------------
# MOCK LOGIN AND MAIN APP LOGIC (For full execution)
# -------------------------

# Mock implementation of the Admin Dashboard (Required because the utility functions reference it)
def admin_dashboard():
    st.title("Admin Dashboard (Mock)")
    st.info("This is a placeholder for the Admin Dashboard. Use the Log Out button to switch.")
    if st.button("🚪 Log Out (Switch to Candidate)"):
        go_to("candidate_dashboard")
        st.session_state.user_type = "candidate"
        st.rerun()

def login_page():
    st.title("Welcome to PragyanAI")
    st.header("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username (Enter 'candidate' or 'admin')")
        password = st.text_input("Password (Any value)", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username.lower() == "candidate":
                st.session_state.logged_in = True
                st.session_state.user_type = "candidate"
                go_to("candidate_dashboard")
                st.success("Logged in as Candidate!")
                st.rerun()
            elif username.lower() == "admin":
                st.session_state.logged_in = True
                st.session_state.user_type = "admin"
                go_to("admin_dashboard")
                st.success("Logged in as Admin!")
                st.rerun()
            else:
                st.error("Invalid username. Please use 'candidate' or 'admin'.")

# --- Main App Execution ---

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="PragyanAI Candidate Dashboard")

    # Initialize state for navigation and authentication
    if 'page' not in st.session_state: st.session_state.page = "login"
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'user_type' not in st.session_state: st.session_state.user_type = None
    
    # Initialize state for candidate-specific data
    if "candidate_results" not in st.session_state: st.session_state.candidate_results = []
    if "current_resume" not in st.session_state: st.session_state.current_resume = None

    if st.session_state.logged_in:
        if st.session_state.user_type == "candidate":
            candidate_dashboard()
        elif st.session_state.user_type == "admin":
            # Running the original (or a mock) admin dashboard logic
            admin_dashboard() 
    else:
        login_page()
