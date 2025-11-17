import streamlit as st
import os
import pdfplumber
import docx
import json
import traceback
import re 
from dotenv import load_dotenv 
from datetime import datetime
from io import BytesIO 
import time
import pandas as pd

# --- CONFIGURATION & API SETUP ---

GROQ_MODEL = "llama-3.1-8b-instant"
# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# --- Define MockGroqClient globally ---

class MockGroqClient:
    """Mock client for local testing when Groq is not available or key is missing."""
    def chat(self):
        class Completions:
            def create(self, **kwargs):
                # Simple mock response
                return type('MockResponse', (object,), {'choices': [{'message': {'content': '{"name": "Mock Candidate", "summary": "Mock summary for testing.", "skills": ["Python", "Streamlit"]}'}}]})()
        return Completions()

# Initialize Groq Client or use Mock Client 
client = None
try:
    from groq import Groq
    if not GROQ_API_KEY:
        # If key is missing, treat it as a setup failure and fall back to mock
        raise ValueError("GROQ_API_KEY not set.") 
    # Attempt to initialize the real client
    client = Groq(api_key=GROQ_API_KEY)
except (ImportError, ValueError, Exception) as e:
    # Fallback to Mock Client
    st.warning(f"⚠️ Using Mock LLM Client. Groq setup failed: {e.__class__.__name__}. Set GROQ_API_KEY and install 'groq' for full functionality.")
    client = MockGroqClient()


# --- Utility Functions ---

def go_to(page_name):
    """Changes the current page in Streamlit's session state."""
    st.session_state.page = page_name

def get_file_type(file_name):
    """Identifies the file type based on its extension, handling common text formats."""
    ext = os.path.splitext(file_name)[1].lower().strip('.')
    if ext == 'pdf': return 'pdf'
    elif ext in ('docx', 'doc'): return 'docx'
    elif ext in ('txt', 'md', 'markdown', 'rtf'): return 'txt' # Treat RTF and MD as plain text for simple extraction
    elif ext == 'json': return 'json'
    elif ext in ('xlsx', 'xls', 'csv'): return 'excel' # Group CSV and XLSX for pandas handling
    else: return 'unknown' 

def extract_content(file_type, file_content_bytes, file_name):
    """Extracts text content from uploaded file content (bytes)."""
    text = ''
    excel_data = None
    try:
        if file_type == 'pdf':
            with pdfplumber.open(BytesIO(file_content_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        
        elif file_type == 'docx':
            doc = docx.Document(BytesIO(file_content_bytes))
            text = '\n'.join([para.text for para in doc.paragraphs])
        
        elif file_type == 'txt':
            try:
                # Try UTF-8 first, fallback to Latin-1
                text = file_content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                 text = file_content_bytes.decode('latin-1')
        
        elif file_type == 'json':
            try:
                data = json.loads(file_content_bytes.decode('utf-8'))
                text = "--- JSON Content Start ---\n" + json.dumps(data, indent=2) + "\n--- JSON Content End ---"
            except json.JSONDecodeError:
                return f"[Error] JSON content extraction failed: Invalid JSON format.", None
            except UnicodeDecodeError:
                return f"[Error] JSON content extraction failed: Unicode Decode Error.", None
        
        elif file_type == 'excel':
            try:
                # Use pandas to read all sheets and convert to a comprehensive text/json structure
                if file_name.endswith('.csv'):
                    df = pd.read_csv(BytesIO(file_content_bytes))
                else: # xlsx, xls
                    # Read all sheets, combine into a JSON-like string for LLM parsing
                    xls = pd.ExcelFile(BytesIO(file_content_bytes))
                    all_sheets_data = {}
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        # Convert to JSON string format for easy LLM parsing
                        all_sheets_data[sheet_name] = df.to_json(orient='records') 
                        
                    excel_data = all_sheets_data # Store structured data
                    text = json.dumps(all_sheets_data, indent=2)
                    text = f"[EXCEL_CONTENT] The following structured data was extracted:\n{text}"
                    
            except Exception as e:
                return f"[Error] Excel/CSV file parsing failed. Error: {e}", None


        if not text.strip() and file_type not in ('excel', 'json'): # Allow structured excel data without pure text
            return f"[Error] {file_type.upper()} content extraction failed or file is empty.", None
        
        return text, excel_data
    
    except Exception as e:
        return f"[Error] Fatal Extraction Error: Failed to read file content ({file_type}). Error: {e}\n{traceback.format_exc()}", None


@st.cache_data(show_spinner="Analyzing content with Groq LLM...")
def parse_resume_with_llm(text):
    """Sends resume text to the LLM for structured information extraction."""
    
    # 1. Handle Pre-flight errors (e.g., failed extraction)
    if text.startswith("[Error"):
        # Returns a dictionary with the original error message for the front end to handle
        return {"name": "Parsing Error", "error": text}

    # 2. Handle Mock Client execution (Always returns success for testing)
    if isinstance(client, MockGroqClient):
        # Mock structured data for demonstration
        return {
            "name": "Mock Candidate", 
            "email": "mock@example.com", 
            "phone": "555-1234", 
            "linkedin": "https://linkedin.com/in/mock", 
            "github": "https://github.com/mock", 
            "personal_details": "Highly motivated individual with mock experience in Python and Streamlit. This summary was parsed from a mock resume.", 
            "skills": ["Python", "Streamlit", "SQL", "AWS"], 
            "education": ["B.S. Computer Science, Mock University, 2020"], 
            "experience": ["Software Intern, Mock Solutions (2024-2025)", "Data Analyst, Test Corp (2022-2024)"], 
            "certifications": ["Mock Certification"], 
            "projects": ["Mock Project: Built a dashboard using Streamlit."], 
            "strength": ["Mock Strength"], 
            "error": None
        }

    # 3. Handle Real Groq Client execution (MOCKED HERE for a successful return structure)
    try:
        # NOTE: Implement actual Groq API call here if key is set and library is installed.
        # For this demo, we simulate a successful return for the real client.
        
        parsed_data = {
            "name": "Parsed Candidate", 
            "email": "parsed@example.com", 
            "phone": "555-9876", 
            "linkedin": "https://linkedin.com/in/parsed", 
            "github": "https://github.com/parsed", 
            "personal_details": "Actual parsed summary from LLM.", 
            "skills": ["Real", "Python", "Streamlit"], 
            "education": ["University of Code, 2021"], 
            "experience": ["Senior Developer, TechCo, 2021 - Present"], 
            "certifications": ["AWS Certified"], 
            "projects": ["Project Alpha"], 
            "strength": ["Teamwork"], 
            "error": None
        }
        return parsed_data
        
    except Exception as e:
        # Catch any exceptions during the Groq call or JSON parsing
        error_message = f"LLM Processing Error: {e.__class__.__name__} - {str(e)}"
        return {"name": "Parsing Error", "error": error_message}
    

# --- NEW HELPER FUNCTIONS FOR FILE/TEXT PROCESSING ---

def clear_interview_state():
    """Clears all session state variables related to interview/match sessions."""
    if 'interview_chat_history' in st.session_state: del st.session_state['interview_chat_history']
    if 'current_interview_jd' in st.session_state: del st.session_state['current_interview_jd']
    if 'evaluation_report' in st.session_state: del st.session_state['evaluation_report']
    if 'candidate_match_results' in st.session_state: st.session_state.candidate_match_results = []
    
    # Do NOT clear 'cv_form_data' or 'parsed' here. 

def parse_and_store_resume(content_source, file_name_key, source_type):
    """
    Handles extraction, parsing, and storage of CV data from either a file or pasted text.
    Returns a dictionary with 'parsed', 'full_text', 'excel_data', and 'name'.
    """
    extracted_text = ""
    excel_data = None
    file_name = "Pasted_Resume"

    if source_type == 'file':
        uploaded_file = content_source
        file_name = uploaded_file.name
        file_type = get_file_type(file_name)
        uploaded_file.seek(0) 
        extracted_text, excel_data = extract_content(file_type, uploaded_file.getvalue(), file_name)
    elif source_type == 'text':
        extracted_text = content_source.strip()
        file_name = "Pasted_Text"

    if extracted_text.startswith("[Error"):
        return {"error": extracted_text, "full_text": extracted_text, "excel_data": None, "name": file_name}
    
    # 2. Call LLM Parser
    parsed_data = parse_resume_with_llm(extracted_text)
    
    # 3. Handle LLM Parsing Error
    if parsed_data.get('error') is not None and parsed_data.get('error') != "":
        return {"error": parsed_data['error'], "full_text": extracted_text, "excel_data": excel_data, "name": parsed_data.get('name', file_name)}

    # 4. Create compiled text for download/Q&A
    compiled_text = ""
    for k, v in parsed_data.items():
        if v and k not in ['error']:
            compiled_text += f"{k.replace('_', ' ').title()}:\n"
            if isinstance(v, list):
                compiled_text += "\n".join([f"- {item}" for item in v]) + "\n\n"
            else:
                compiled_text += str(v) + "\n\n"

    # Attempt to set the final name based on LLM output or fallback
    final_name = parsed_data.get('name', 'Unknown_Candidate').replace(' ', '_')
    
    return {
        "parsed": parsed_data, 
        "full_text": compiled_text, 
        "excel_data": excel_data, 
        "name": final_name
    }

# --- END NEW HELPER FUNCTIONS ---


@st.cache_data(show_spinner="Analyzing JD for metadata...")
def extract_jd_metadata(jd_text):
    """Mocks the extraction of key metadata (Role, Skills, Job Type) from JD text using LLM."""
    if jd_text.startswith("[Error"):
        return {"role": "Error", "key_skills": ["Error"], "job_type": "Error"}
    
    # Simple heuristic mock
    role_match = re.search(r'(?:Role|Position|Title)[:\s]+([\w\s/-]+)', jd_text, re.IGNORECASE)
    role = role_match.group(1).strip() if role_match else "Software Engineer (Mock)"
    
    skills_match = re.findall(r'(Python|Java|SQL|AWS|Docker|Kubernetes|React|Streamlit)', jd_text, re.IGNORECASE)
    
    job_type_match = re.search(r'(Full-time|Part-time|Contract|Remote)', jd_text, re.IGNORECASE)
    job_type = job_type_match.group(1) if job_type_match else "Full-time (Mock)"
    
    return {
        "role": role, 
        "key_skills": list(set([s.lower() for s in skills_match][:5])), # Limit to 5 unique skills
        "job_type": job_type
    }

def extract_jd_from_linkedin_url(url):
    """Mocks the extraction of JD content from a LinkedIn URL."""
    if "linkedin.com/jobs" not in url:
        return f"[Error] Invalid LinkedIn Job URL: {url}"

    # Mock content based on URL structure
    role = "Data Scientist" if "data" in url.lower() else "Cloud Engineer"
    
    return f"""
    --- Simulated JD for: {role} ---
    
    Company: MockCorp
    Location: Remote
    
    Job Summary:
    We are seeking a highly skilled {role} to join our team. The ideal candidate will have expertise in Python, SQL, and AWS. Must be able to work in a fast-paced environment and deliver solutions quickly. This is a Full-time position.
    
    Responsibilities:
    * Develop and maintain data pipelines using **Python** and **SQL**.
    * Manage and deploy applications on **AWS** and **Docker**.
    * Collaborate with cross-functional teams.
    
    Qualifications:
    * 3+ years of experience.
    * Strong proficiency in **Python** and analytical tools.
    * Experience with cloud platforms (e.g., **AWS**).
    ---
    """
    
def evaluate_jd_fit(jd_content, parsed_json):
    """
    Mocks the LLM evaluation of resume fit against a JD.
    """
    
    # Mocking different scores based on JD content for demonstration
    jd_name = jd_content.splitlines()[1].strip().replace("--- Simulated JD for: ", "")
    
    if "Data Scientist" in jd_name:
        score = 8
        skills = 90
        exp = 85
        edu = 80
    elif "Cloud Engineer" in jd_name:
        score = 6
        skills = 70
        exp = 65
        edu = 75
    else:
        score = 7
        skills = 75
        exp = 70
        edu = 70
        
    time.sleep(0.5) # Simulate latency
    
    return f"""
    --- Overall Fit Score ---
    Overall Fit Score: **{score}/10**
    
    --- Section Match Analysis ---
    Skills Match: [{skills}%]
    Experience Match: [{exp}%]
    Education Match: [{edu}%]%
    
    --- Strengths/Matches ---
    The candidate shows strong proficiency in Python and Streamlit, which aligns well with the required data processing tools. Education is directly relevant.
    
    --- Weaknesses/Gaps ---
    Candidate lacks specific mention of Kubernetes or advanced Docker orchestration, which is a required skill for this role.
    
    --- Summary Recommendation ---
    A strong candidate with core skills. Recommend for interview if the experience gap in orchestration tools can be overlooked or quickly trained.
    """

def generate_cv_html(parsed_data):
    """Generates a simple, print-friendly HTML string from parsed data for PDF conversion."""
    
    # Simple CSS for a clean, print-friendly CV look
    css = """
    <style>
        @page { size: A4; margin: 1cm; }
        body { font-family: 'Arial', sans-serif; line-height: 1.5; margin: 0; padding: 0; font-size: 10pt; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
        .header h1 { margin: 0; font-size: 1.8em; }
        .contact-info { display: flex; justify-content: center; font-size: 0.8em; color: #555; }
        .contact-info span { margin: 0 8px; }
        .section { margin-bottom: 15px; page-break-inside: avoid; }
        .section h2 { border-bottom: 1px solid #999; padding-bottom: 3px; margin-bottom: 8px; font-size: 1.1em; text-transform: uppercase; color: #333; }
        .item-list ul { list-style-type: disc; margin-left: 20px; padding-left: 0; margin-top: 0; }
        .item-list ul li { margin-bottom: 3px; }
        .item-list p { margin: 3px 0 8px 0; }
        a { color: #0056b3; text-decoration: none; }
    </style>
    """
    
    # --- HTML Structure ---
    html_content = f"<html><head>{css}<title>{parsed_data.get('name', 'CV')}</title></head><body>"
    
    # 1. Header and Contact Info
    html_content += '<div class="header">'
    html_content += f"<h1>{parsed_data.get('name', 'Candidate Name')}</h1>"
    
    contact_parts = []
    if parsed_data.get('email'): contact_parts.append(f"<span>📧 {parsed_data['email']}</span>")
    if parsed_data.get('phone'): contact_parts.append(f"<span>📱 {parsed_data['phone']}</span>")
    # Clean up URL display
    linkedin_url = parsed_data.get('linkedin', '')
    github_url = parsed_data.get('github', '')
    if linkedin_url: contact_parts.append(f"<span>🔗 <a href='{linkedin_url}'>{linkedin_url.split('/')[-1] or 'LinkedIn'}</a></span>")
    if github_url: contact_parts.append(f"<span>💻 <a href='{github_url}'>{github_url.split('/')[-1] or 'GitHub'}</a></span>")
    
    html_content += f'<div class="contact-info">{" | ".join(contact_parts)}</div>'
    html_content += '</div>'
    
    # 2. Sections
    section_order = ['personal_details', 'experience', 'projects', 'education', 'certifications', 'skills', 'strength']
    
    for k in section_order:
        v = parsed_data.get(k)
        
        # Skip contact details already handled
        if k in ['name', 'email', 'phone', 'linkedin', 'github', 'error']: continue 

        if v and (isinstance(v, str) and v.strip() or isinstance(v, list) and v):
            
            # Use title() for nicer display, e.g., 'personal_details' becomes 'Personal Details'
            html_content += f'<div class="section"><h2>{k.replace("_", " ").title()}</h2>'
            html_content += '<div class="item-list">'
            
            if k == 'personal_details' and isinstance(v, str):
                html_content += f"<p>{v}</p>"
            elif isinstance(v, list):
                html_content += '<ul>'
                for item in v:
                    if item: 
                        html_content += f"<li>{item}</li>"
                html_content += '</ul>'
            else:
                # Fallback for any other string
                html_content += f"<p>{v}</p>"
                
            html_content += '</div></div>'

    html_content += '</body></html>'
    return html_content

# --- Tab Content Functions ---

def resume_parsing_tab():
    st.header("📄 Upload/Paste Resume for AI Parsing")
    
    # 1. Input Method Selection
    input_method = st.radio(
        "Select Input Method",
        ["Upload File", "Paste Text"],
        key="parsing_input_method"
    )
    
    st.markdown("---")

    # --- A. Upload File Method (UPDATED FILE TYPES HERE) ---
    if input_method == "Upload File":
        st.markdown("### 1. Upload Resume File") 
        
        # 🚨 File types expanded here
        file_types = ["pdf", "docx", "txt", "json", "md", "csv", "xlsx", "markdown", "rtf"]
        uploaded_file = st.file_uploader( 
            f"Choose {', '.join(file_types)} file", 
            type=file_types, 
            accept_multiple_files=False, 
            key='candidate_file_upload_main'
        )
        
        st.markdown(
            """
            <div style='font-size: 10px; color: grey;'>
            Supported File Types: PDF, DOCX, TXT, JSON, MARKDOWN, CSV, XLSX, RTF
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("---")

        # Initialize candidate_uploaded_resumes if not present
        if "candidate_uploaded_resumes" not in st.session_state:
            st.session_state.candidate_uploaded_resumes = []
        if "pasted_cv_text" not in st.session_state:
            st.session_state.pasted_cv_text = ""

        # --- File Management Logic ---
        if uploaded_file is not None:
            # Only store the single uploaded file if it's new
            if not st.session_state.candidate_uploaded_resumes or st.session_state.candidate_uploaded_resumes[0].name != uploaded_file.name:
                st.session_state.candidate_uploaded_resumes = [uploaded_file] 
                st.session_state.pasted_cv_text = "" # Clear pasted text
                st.toast("Resume file uploaded successfully.")
        elif st.session_state.candidate_uploaded_resumes and uploaded_file is None:
            # Case where the file is removed from the uploader
            st.session_state.candidate_uploaded_resumes = []
            st.session_state.parsed = {}
            st.session_state.full_text = ""
            st.session_state.excel_data = None
            st.toast("Upload cleared.")
            
        file_to_parse = st.session_state.candidate_uploaded_resumes[0] if st.session_state.candidate_uploaded_resumes else None
        
        st.markdown("### 2. Parse Uploaded File")
        
        if file_to_parse:
            # Check if the file is already parsed and loaded to avoid unnecessary LLM calls 
            is_already_parsed = (
                st.session_state.get('last_parsed_file_name') == file_to_parse.name and 
                st.session_state.get('parsed', {}).get('name') is not None and
                'error' not in st.session_state.get('parsed', {})
            )

            if st.button(f"Parse and Load: **{file_to_parse.name}**", use_container_width=True, disabled=is_already_parsed):
                with st.spinner(f"Parsing {file_to_parse.name}..."):
                    result = parse_and_store_resume(file_to_parse, file_name_key='single_resume_candidate', source_type='file')
                    
                    if "error" not in result:
                        st.session_state.parsed = result['parsed']
                        st.session_state.full_text = result['full_text']
                        st.session_state.excel_data = result['excel_data'] 
                        st.session_state.parsed['name'] = result['name'] 
                        st.session_state.last_parsed_file_name = file_to_parse.name # Track the parsed file
                        clear_interview_state()
                        
                        # Synchronize cv_form_data on successful parse 
                        st.session_state.cv_form_data = {**default_cv_template(), **st.session_state.parsed}
                        
                        st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                        st.info("View, edit, and download the parsed data in the **CV Management** tab.") 
                        st.rerun() # Force rerun to reflect changes in other tabs/state
                    else:
                        st.error(f"Parsing failed for {file_to_parse.name}: {result['error']}")
                        st.session_state.parsed = {"error": result['error'], "name": result['name']}
                        st.session_state.full_text = result['full_text'] or ""
                        st.session_state.excel_data = result['excel_data'] 
                        st.session_state.parsed['name'] = result['name'] 
                        
            if is_already_parsed:
                st.info(f"The file **{file_to_parse.name}** is already parsed and loaded.")

        else:
            st.info("No resume file is currently uploaded. Please upload a file above.")

    # --- B. Paste Text Method (NEW) ---
    else: # input_method == "Paste Text"
        st.markdown("### 1. Paste Your CV Text")
        
        pasted_text = st.text_area(
            "Copy and paste your entire CV or resume text here.",
            value=st.session_state.get('pasted_cv_text', ''),
            height=300,
            key='pasted_cv_text_input'
        )
        st.session_state.pasted_cv_text = pasted_text # Update session state immediately
        
        st.markdown("---")
        st.markdown("### 2. Parse Pasted Text")
        
        if pasted_text.strip():
            if st.button("Parse and Load Pasted Text", use_container_width=True):
                with st.spinner("Parsing pasted text..."):
                    # Clear file upload state
                    st.session_state.candidate_uploaded_resumes = []
                    
                    result = parse_and_store_resume(pasted_text, file_name_key='single_resume_candidate', source_type='text')
                    
                    if "error" not in result:
                        st.session_state.parsed = result['parsed']
                        st.session_state.full_text = result['full_text']
                        st.session_state.excel_data = result['excel_data'] 
                        st.session_state.parsed['name'] = result['name'] 
                        st.session_state.last_parsed_file_name = "Pasted_Text"
                        clear_interview_state()
                        
                        # Synchronize cv_form_data on successful parse
                        st.session_state.cv_form_data = {**default_cv_template(), **st.session_state.parsed}
                        
                        st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                        st.info("View, edit, and download the parsed data in the **CV Management** tab.") 
                        st.rerun() # Force rerun to reflect changes in other tabs/state
                    else:
                        st.error(f"Parsing failed: {result['error']}")
                        st.session_state.parsed = {"error": result['error'], "name": result['name']}
                        st.session_state.full_text = result['full_text'] or ""
                        st.session_state.excel_data = result['excel_data'] 
                        st.session_state.parsed['name'] = result['name'] 
        else:
            st.info("Please paste your CV text into the box above.")

def default_cv_template():
    """Returns the template for a complete CV data structure."""
    return {
        "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
        "skills": [], "experience": [], "education": [], "certifications": [], 
        "projects": [], "strength": [], "personal_details": ""
    }

# --- CV MANAGEMENT FUNCTION ---
def cv_management_tab_content():
    st.header("📝 Prepare Your CV")
    st.markdown("### 1. Form Based CV Builder")
    st.info("Fill out the details below to generate a parsed CV that can be used immediately for matching and interview prep, or start by parsing a file in the 'Resume Parsing' tab.")

    # --- CRITICAL FIX: Synchronization Logic ---
    default_parsed = default_cv_template()
    
    # 1. Initialize the form state if it doesn't exist
    if "cv_form_data" not in st.session_state:
        st.session_state.cv_form_data = default_parsed.copy()
        
    # 2. Synchronization check: If 'parsed' has data from the other tab, update the form state
    parsed_data = st.session_state.get('parsed', {})
    
    if parsed_data.get('name') and 'error' not in parsed_data:
        # Check if the form state needs updating (e.g., after a parse or if form is empty)
        # Using name and email as simple synchronization check keys
        if st.session_state.cv_form_data.get('name') != parsed_data.get('name') or \
           st.session_state.cv_form_data.get('email') != parsed_data.get('email'):
            
            # Merge parsed data into a copy of default to ensure all keys exist
            st.session_state.cv_form_data = {**default_parsed, **parsed_data}
            st.toast("CV Management form updated with data from the Resume Parsing tab.")
            # Note: No rerun needed here, as the tab is already rendering

            
    # --- END CRITICAL FIX ---
            
    
    # --- CV Builder Form ---
    with st.form("cv_builder_form"):
        st.subheader("Personal & Contact Details")
        
        # Row 1: Name, Email, Phone
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.cv_form_data['name'] = st.text_input(
                "Full Name", 
                # IMPORTANT: Must pull the initial value from st.session_state.cv_form_data to reflect parsed data
                value=st.session_state.cv_form_data.get('name', ''), 
                key="cv_name"
            )
        with col2:
            st.session_state.cv_form_data['email'] = st.text_input(
                "Email Address", 
                value=st.session_state.cv_form_data.get('email', ''), 
                key="cv_email"
            )
        with col3:
            st.session_state.cv_form_data['phone'] = st.text_input(
                "Phone Number", 
                value=st.session_state.cv_form_data.get('phone', ''), 
                key="cv_phone"
            )
        
        # Row 2: LinkedIn, GitHub
        col4, col5 = st.columns(2)
        with col4:
            st.session_state.cv_form_data['linkedin'] = st.text_input(
                "LinkedIn Profile URL", 
                value=st.session_state.cv_form_data.get('linkedin', ''), 
                key="cv_linkedin"
            )
        with col5:
            st.session_state.cv_form_data['github'] = st.text_input(
                "GitHub Profile URL", 
                value=st.session_state.cv_form_data.get('github', ''), 
                key="cv_github"
            )
        
        # Row 3: Summary/Personal Details 
        st.markdown("---")
        st.subheader("Summary / Personal Details")
        st.session_state.cv_form_data['personal_details'] = st.text_area(
            "Professional Summary or Personal Details", 
            value=st.session_state.cv_form_data.get('personal_details', ''), 
            height=100,
            key="cv_personal_details"
        )
        
        st.markdown("---")
        st.subheader("Technical Sections (One Item per Line)")

        # Skills (Handling list conversion inside the form)
        skills_text = "\n".join(st.session_state.cv_form_data.get('skills', []))
        new_skills_text = st.text_area(
            "Key Skills (Technical and Soft)", 
            value=skills_text,
            height=150,
            key="cv_skills"
        )
        
        # Experience
        experience_text = "\n".join(st.session_state.cv_form_data.get('experience', []))
        new_experience_text = st.text_area(
            "Professional Experience (Job Roles, Companies, Dates, Key Responsibilities)", 
            value=experience_text,
            height=150,
            key="cv_experience"
        )

        # Education
        education_text = "\n".join(st.session_state.cv_form_data.get('education', []))
        new_education_text = st.text_area(
            "Education (Degrees, Institutions, Dates)", 
            value=education_text,
            height=100,
            key="cv_education"
        )
        
        # Certifications
        certifications_text = "\n".join(st.session_state.cv_form_data.get('certifications', []))
        new_certifications_text = st.text_area(
            "Certifications (Name, Issuing Body, Date)", 
            value=certifications_text,
            height=100,
            key="cv_certifications"
        )
        
        # Projects
        projects_text = "\n".join(st.session_state.cv_form_data.get('projects', []))
        new_projects_text = st.text_area(
            "Projects (Name, Description, Technologies)", 
            value=projects_text,
            height=150,
            key="cv_projects"
        )
        
        # Strengths
        strength_text = "\n".join(st.session_state.cv_form_data.get('strength', []))
        new_strength_text = st.text_area(
            "Strengths / Key Personal Qualities (One per line)", 
            value=strength_text,
            height=100,
            key="cv_strength"
        )
        
        # FINAL: Update the lists in the session state data (since text_area handles string, not list)
        st.session_state.cv_form_data['skills'] = [s.strip() for s in new_skills_text.split('\n') if s.strip()]
        st.session_state.cv_form_data['experience'] = [e.strip() for e in new_experience_text.split('\n') if e.strip()]
        st.session_state.cv_form_data['education'] = [d.strip() for d in new_education_text.split('\n') if d.strip()]
        st.session_state.cv_form_data['certifications'] = [c.strip() for c in new_certifications_text.split('\n') if c.strip()]
        st.session_state.cv_form_data['projects'] = [p.strip() for p in new_projects_text.split('\n') if p.strip()]
        st.session_state.cv_form_data['strength'] = [s.strip() for s in new_strength_text.split('\n') if s.strip()]


        submit_form_button = st.form_submit_button("Generate and Load CV Data", type="primary", use_container_width=True)

    if submit_form_button:
        # 1. Basic validation
        if not st.session_state.cv_form_data.get('name') or not st.session_state.cv_form_data.get('email'):
            st.error("Please fill in at least your **Full Name** and **Email Address**.")
            return

        # 2. Update the main parsed state variable with the form data
        if 'name' not in st.session_state.cv_form_data or not st.session_state.cv_form_data.get('name'):
             st.session_state.cv_form_data['name'] = st.session_state.cv_form_data.get('email', 'Manual CV').split('@')[0]
             
        # Use a deep copy to ensure 'parsed' is the definitive source of truth 
        st.session_state.parsed = st.session_state.cv_form_data.copy()
        
        # 3. Create a compiled text representation for Q&A/Text Download
        compiled_text = ""
        for k, v in st.session_state.cv_form_data.items():
            if v and k not in ['error']:
                compiled_text += f"{k.replace('_', ' ').title()}:\n"
                if isinstance(v, list):
                    compiled_text += "\n".join([f"- {item}" for item in v]) + "\n\n"
                else:
                    compiled_text += str(v) + "\n\n"
        st.session_state.full_text = compiled_text
        st.session_state.excel_data = None # Clear excel data if we are using the form builder
        st.session_state.last_parsed_file_name = "Manual_Form_Entry" # Update source tracking
        
        # 4. Clear related states 
        clear_interview_state()

        st.success(f"✅ CV data for **{st.session_state.parsed['name']}** successfully generated and loaded! You can now use the Match tabs.")
        
        # --- CRITICAL FIX: Force rerun to display the preview immediately ---
        st.rerun() 
        
        
    st.markdown("---")
    st.subheader("2. Loaded CV Data Preview and Download")
    
    # --- TABBED VIEW SECTION (PDF/MARKDOWN/JSON) ---
    
    # Check if data is loaded and valid (i.e., not an empty dictionary or an error dictionary)
    if st.session_state.get('parsed', {}).get('name') and 'error' not in st.session_state.parsed:
        
        st.markdown(f"**Current Loaded Candidate:** **{st.session_state.parsed['name']}**")
        st.caption(f"Source: {st.session_state.get('last_parsed_file_name', 'Unknown Source')}")
        
        # Filter for non-empty/non-list fields before sending to formatter
        filled_data_for_preview = {
            k: v for k, v in st.session_state.parsed.items() 
            if v and k not in ['error'] and (isinstance(v, str) and v.strip() or isinstance(v, list) and v)
        }
        
        # Helper function for Markdown formatting
        def format_parsed_json_to_markdown(parsed_data):
            """Formats the parsed JSON data into a clean, CV-like Markdown structure."""
            md = ""
            
            # --- Personal Info (Header) ---
            if parsed_data.get('name'):
                md += f"# **{parsed_data['name']}**\n\n"
            
            contact_info = []
            if parsed_data.get('email'): contact_info.append(parsed_data['email'])
            if parsed_data.get('phone'): contact_info.append(parsed_data['phone'])
            # Ensure the links are displayed cleanly or as links
            if parsed_data.get('linkedin'): contact_info.append(f"[LinkedIn]({parsed_data['linkedin']})")
            if parsed_data.get('github'): contact_info.append(f"[GitHub]({parsed_data['github']})")
            
            if contact_info:
                # Use a table-like format for contact info
                md += f"| {' | '.join(contact_info)} |\n"
                md += "| " + " | ".join(["---"] * len(contact_info)) + " |\n\n"
            
            # --- Section Content ---
            section_order = ['personal_details', 'experience', 'projects', 'education', 'certifications', 'skills', 'strength']
            
            for k in section_order:
                v = parsed_data.get(k)
                
                # Skip contact details already handled in header
                if k in ['name', 'email', 'phone', 'linkedin', 'github', 'error']: continue 

                if v and (isinstance(v, str) and v.strip() or isinstance(v, list) and v):
                    
                    md += f"## **{k.replace('_', ' ').upper()}**\n"
                    md += "---\n"
                    
                    if k == 'personal_details' and isinstance(v, str):
                        md += f"{v}\n\n"
                    elif isinstance(v, list):
                        for item in v:
                            if item: 
                                # Use bullet points for list items (Experience, Skills, Projects, etc.)
                                md += f"- {item}\n"
                        md += "\n"
                    else:
                        # Fallback for any other string
                        md += f"{v}\n\n"
            return md


        tab_markdown, tab_json, tab_pdf = st.tabs(["📝 Markdown View", "💾 JSON View", "⬇️ PDF/HTML Download"])

        # --- Markdown View ---
        with tab_markdown:
            cv_markdown_preview = format_parsed_json_to_markdown(filled_data_for_preview)
            st.markdown(cv_markdown_preview)

            # Markdown Download Button
            st.download_button(
                label="⬇️ Download CV as Markdown (.md)",
                data=cv_markdown_preview,
                file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_CV_Document.md",
                mime="text/markdown",
                key="download_cv_markdown_final"
            )


        # --- JSON View ---
        with tab_json:
            st.json(st.session_state.parsed)
            st.info("This is the raw, structured data used by the AI tools.")

            # JSON Download Button
            json_output = json.dumps(st.session_state.parsed, indent=2)
            st.download_button(
                label="⬇️ Download CV as JSON File",
                data=json_output,
                file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_CV_Data.json",
                mime="application/json",
                key="download_cv_json_final"
            )


        # --- PDF View (Download) ---
        with tab_pdf:
            st.markdown("### Download CV as HTML (Print-to-PDF)")
            st.info("Click the button below to download an HTML file. Open the file in your browser and use the browser's **'Print'** function, selecting **'Save as PDF'** to create your final CV document.")
            st.markdown(
            """
            <div style='text-align: center; border: 1px solid #ccc; padding: 10px; margin: 10px 0;'>
                
            </div>
            """,
            unsafe_allow_html=True
            )
            
            html_output = generate_cv_html(filled_data_for_preview)

            st.download_button(
                label="⬇️ Download CV as Print-Ready HTML File (for PDF conversion)",
                data=html_output,
                file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_CV_Document.html",
                mime="text/html",
                key="download_cv_html"
            )
            
            st.markdown("---")
            st.markdown("### Raw Text Data Download (for utility)")
            st.download_button(
                label="⬇️ Download All CV Data as Raw Text (.txt)",
                data=st.session_state.full_text,
                file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_Raw_Data.txt",
                mime="text/plain",
                key="download_cv_txt_final"
            )
            
    else:
        st.info("Please fill out the form above and click 'Generate and Load CV Data' or parse a resume in the 'Resume Parsing' tab to see the preview and download options.")


def jd_management_tab_candidate():
    """JD Management Tab."""
    st.header("📚 Manage Job Descriptions for Matching")
    st.markdown("Add multiple JDs here to compare your resume against them in the next tabs.")
    
    if "candidate_jd_list" not in st.session_state: st.session_state.candidate_jd_list = []
    st.markdown("---")
    
    jd_type = st.radio("Select JD Type", ["Single JD", "Multiple JD"], key="jd_type_candidate", index=0)
    st.markdown("### Add JD by:")
    method = st.radio("Choose Method", ["Upload File", "Paste Text", "LinkedIn URL"], key="jd_add_method_candidate", index=0) 
    st.markdown("---")

    # --- LinkedIn URL Section ---
    if method == "LinkedIn URL":
        with st.form("jd_url_form_candidate", clear_on_submit=True):
            url_list = st.text_area("Enter one or more URLs (comma separated)" if jd_type == "Multiple JD" else "Enter URL", key="url_list_candidate")
            if st.form_submit_button("Add JD(s) from URL", key="add_jd_url_btn_candidate"):
                if url_list:
                    urls = [u.strip() for u in url_list.split(",")] if jd_type == "Multiple JD" else [url_list.strip()]
                    count = 0
                    for url in urls:
                        if not url: continue
                        with st.spinner(f"Attempting JD extraction and metadata analysis for: {url}"):
                            jd_text = extract_jd_from_linkedin_url(url)
                            metadata = extract_jd_metadata(jd_text)
                        
                        if jd_text.startswith("[Error"):
                            st.error(f"Failed to process {url}: {jd_text}")
                            continue
                            
                        name = f"JD from URL: {url.split('/jobs/view/')[-1].split('/')[0]}"
                        st.session_state.candidate_jd_list.append({"name": name, "content": jd_text, **metadata})
                        count += 1
                            
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully!")
                        st.rerun() 
                    else:
                        st.error("No JDs were added successfully.")

    # --- Paste Text Section ---
    elif method == "Paste Text":
        with st.form("jd_paste_form_candidate", clear_on_submit=True):
            text_list = st.text_area("Paste one or more JD texts (separate by '---')" if jd_type == "Multiple JD" else "Paste JD text here", key="text_list_candidate")
            if st.form_submit_button("Add JD(s) from Text", key="add_jd_text_btn_candidate"):
                if text_list:
                    texts = [t.strip() for t in text_list.split("---")] if jd_type == "Multiple JD" else [text_list.strip()]
                    count = 0
                    for i, text in enumerate(texts):
                        if text:
                            name_base = text.splitlines()[0].strip()[:30] if text.splitlines()[0].strip() else f"Pasted JD {len(st.session_state.candidate_jd_list) + i + 1}"
                            metadata = extract_jd_metadata(text)
                            st.session_state.candidate_jd_list.append({"name": name_base, "content": text, **metadata})
                            count += 1
                    
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully!")
                        st.rerun() 

    # --- Upload File Section ---
    elif method == "Upload File":
        # Updated to allow all supported JD file types for consistency
        jd_file_types = ["pdf", "txt", "docx", "md", "json"]
        uploaded_files = st.file_uploader(
            f"Upload JD file(s) ({', '.join(jd_file_types)})",
            type=jd_file_types,
            accept_multiple_files=(jd_type == "Multiple JD"),
            key="jd_file_uploader_candidate"
        )
        files_to_process = uploaded_files if isinstance(uploaded_files, list) else ([uploaded_files] if uploaded_files else [])
        
        with st.form("jd_upload_form_candidate", clear_on_submit=False):
            if files_to_process:
                st.markdown("##### Files Selected:")
                for file in files_to_process:
                    st.markdown(f"&emsp;📄 **{file.name}** {round(file.size / (1024*1024), 2)}MB")
                    
            if st.form_submit_button("Add JD(s) from File", key="add_jd_file_btn_candidate"):
                if not files_to_process:
                    st.warning("Please upload file(s).")
                    
                count = 0
                for file in files_to_process:
                    if file:
                        with st.spinner(f"Extracting content from {file.name}..."):
                            file_type = get_file_type(file.name)
                            file.seek(0)
                            jd_text, _ = extract_content(file_type, file.getvalue(), file.name)
                            
                        if not jd_text.startswith("[Error"):
                            metadata = extract_jd_metadata(jd_text)
                            st.session_state.candidate_jd_list.append({"name": file.name, "content": jd_text, **metadata})
                            count += 1
                        else:
                            st.error(f"Error extracting content from {file.name}: {jd_text}")
                            
                if count > 0:
                    st.success(f"✅ {count} JD(s) added successfully!")
                    st.rerun()
                elif uploaded_files:
                    st.error("No valid JD files were uploaded or content extraction failed.")

    st.markdown("---")
    # Display Added JDs
    if st.session_state.candidate_jd_list:
        
        col_display_header, col_clear_button = st.columns([3, 1])
        
        with col_display_header: st.markdown("### ✅ Current JDs Added:")
            
        with col_clear_button:
            if st.button("🗑️ Clear All JDs", key="clear_jds_candidate", use_container_width=True, help="Removes all currently loaded JDs."):
                st.session_state.candidate_jd_list = []
                if 'candidate_match_results' in st.session_state: del st.session_state['candidate_match_results']
                st.success("All JDs and associated match results have been cleared.")
                st.rerun() 

        for idx, jd_item in enumerate(st.session_state.candidate_jd_list, 1):
            title = jd_item['name']
            display_title = title.replace("--- Simulated JD for: ", "")
            with st.expander(f"**JD {idx}:** {display_title} | Role: {jd_item.get('role', 'N/A')}"):
                st.markdown(f"**Job Type:** {jd_item.get('job_type', 'N/A')} | **Key Skills:** `{', '.join(jd_item.get('key_skills', ['N/A']))}`")
                st.markdown("---")
                st.text(jd_item['content'])
    else:
        st.info("No Job Descriptions added yet.")
        
# --- Batch Match Tab Function ---

def jd_batch_match_tab():
    """The Batch JD Match tab logic."""
    st.header("🎯 Batch JD Match: Best Matches")
    st.markdown("Compare your current resume against all saved job descriptions.")

    # Determine if a resume/CV is ready
    is_resume_parsed = st.session_state.get('parsed') is not None
    
    if not is_resume_parsed or not st.session_state.parsed.get('name') or st.session_state.parsed.get('error'):
        st.warning("⚠️ Please **upload and parse your resume** in the 'Resume Parsing' tab or **build your CV** in the 'CV Management' tab first.")
        
    elif not st.session_state.candidate_jd_list:
        st.error("❌ Please **add Job Descriptions** in the 'JD Management' tab before running batch analysis.")
        
    elif isinstance(client, MockGroqClient):
        # We allow mock client to run the mock match for demonstration purposes
        st.info("ℹ️ Running in Mock LLM Mode. Match results will be simulated.")
        
    else:
        # Check if the real client has failed setup
        try:
             # Just a check to see if the client is valid and not a Mock
            if not isinstance(client, Groq):
                st.warning("⚠️ LLM client setup failed. Match analysis may not be accurate or available.")
        except NameError:
             # Handle case where Groq was never imported/initialized
             st.warning("⚠️ LLM client setup failed. Match analysis may not be accurate or available.")


    if "candidate_match_results" not in st.session_state:
        st.session_state.candidate_match_results = []

    # 1. Get all available JD names
    all_jd_names = [item['name'] for item in st.session_state.candidate_jd_list]
    
    # 2. Add multiselect widget
    selected_jd_names = st.multiselect(
        "Select Job Descriptions to Match Against",
        options=all_jd_names,
        default=all_jd_names, # Default to selecting all JDs
        key='candidate_batch_jd_select'
    )
    
    # 3. Filter the list of JDs based on selection
    jds_to_match = [
        jd_item for jd_item in st.session_state.candidate_jd_list 
        if jd_item['name'] in selected_jd_names
    ]
    
    if st.button(f"Run Match Analysis on **{len(jds_to_match)}** Selected JD(s)"):
        st.session_state.candidate_match_results = []
        
        if not jds_to_match:
            st.warning("Please select at least one Job Description to run the analysis.")
            
        elif not is_resume_parsed or not st.session_state.parsed.get('name'):
             st.warning("Please **upload and parse your resume** first.")

        else:
            resume_name = st.session_state.parsed.get('name', 'Uploaded Resume')
            parsed_json = st.session_state.parsed
            results_with_score = []

            with st.spinner(f"Matching {resume_name}'s resume against {len(jds_to_match)} selected JD(s)..."):
                
                # Loop over jds_to_match
                for jd_item in jds_to_match:
                    jd_name = jd_item['name']
                    jd_content = jd_item['content']

                    try:
                        fit_output = evaluate_jd_fit(jd_content, parsed_json) # Function used to call LLM/Mock LLM
                        
                        # --- Extract Score Data from LLM/Mock Output using Regex ---
                        overall_score_match = re.search(r'Overall Fit Score:\s*[^\d]*(\d+)\s*/10', fit_output, re.IGNORECASE)
                        section_analysis_match = re.search(
                            r'--- Section Match Analysis ---\s*(.*?)\s*--- Strengths/Matches ---', 
                            fit_output, re.DOTALL
                        )

                        skills_percent, experience_percent, education_percent = 'N/A', 'N/A', 'N/A'
                        
                        if section_analysis_match:
                            section_text = section_analysis_match.group(1)
                            skills_match = re.search(r'Skills Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                            experience_match = re.search(r'Experience Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                            education_match = re.search(r'Education Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                            
                            if skills_match: skills_percent = skills_match.group(1)
                            if experience_match: experience_percent = experience_match.group(1)
                            if education_match: education_percent = education_match.group(1)
                            
                        overall_score = overall_score_match.group(1) if overall_score_match else 'N/A'

                        results_with_score.append({
                            "jd_name": jd_name,
                            "overall_score": overall_score,
                            "numeric_score": int(overall_score) if overall_score.isdigit() else -1, # Added for sorting/ranking
                            "skills_percent": skills_percent,
                            "experience_percent": experience_percent, 
                            "education_percent": education_percent, 
                            "full_analysis": fit_output
                        })
                    except Exception as e:
                        results_with_score.append({
                            "jd_name": jd_name,
                            "overall_score": "Error",
                            "numeric_score": -1, # Set a low score for errors
                            "skills_percent": "Error",
                            "experience_percent": "Error", 
                            "education_percent": "Error", 
                            "full_analysis": f"Error running analysis for {jd_name}: {e}\n{traceback.format_exc()}"
                        })
                        
                # --- NEW RANKING LOGIC ---
                results_with_score.sort(key=lambda x: x['numeric_score'], reverse=True)
                
                current_rank = 1
                current_score = -1 
                
                for i, item in enumerate(results_with_score):
                    # Only increase rank if score changes (handles ties)
                    if item['numeric_score'] > current_score:
                        current_rank = i + 1
                        current_score = item['numeric_score']
                        
                    item['rank'] = current_rank
                    # Remove the temporary numeric_score field
                    del item['numeric_score'] 
                    
                st.session_state.candidate_match_results = results_with_score
                # --- END NEW RANKING LOGIC ---
                
                st.success("Batch analysis complete! See results below.")
                st.rerun() # Rerun to ensure the results display immediately


    # 3. Display Results (UPDATED TO INCLUDE RANK)
    if st.session_state.get('candidate_match_results'):
        st.markdown("#### Match Results for Your Resume")
        results_df = st.session_state.candidate_match_results
        
        display_data = []
        for item in results_df:
            # Find the full JD item to get the metadata
            full_jd_item = next((jd for jd in st.session_state.candidate_jd_list if jd['name'] == item['jd_name']), {})
            
            display_data.append({
                "Rank": item.get("rank", "N/A"),
                "Job Description (Ranked)": item["jd_name"].replace("--- Simulated JD for: ", ""),
                "Role": full_jd_item.get('role', 'N/A'), # Added Role
                "Job Type": full_jd_item.get('job_type', 'N/A'), # Added Job Type
                "Fit Score (out of 10)": item["overall_score"],
                "Skills (%)": item.get("skills_percent", "N/A"),
                "Experience (%)": item.get("experience_percent", "N/A"), 
                "Education (%)": item.get("education_percent", "N/A"), 
            })

        st.dataframe(display_data, use_container_width=True)

        st.markdown("##### Detailed Reports")
        for item in results_df:
            rank_display = f"Rank {item.get('rank', 'N/A')} | "
            header_text = f"{rank_display}Report for **{item['jd_name'].replace('--- Simulated JD for: ', '')}** (Score: **{item['overall_score']}/10** | S: **{item.get('skills_percent', 'N/A')}%** | E: **{item.get('experience_percent', 'N/A')}%** | Edu: **{item.get('education_percent', 'N/A')}%**)"
            with st.expander(header_text):
                # Display the full LLM analysis output
                st.code(item['full_analysis'], language='markdown')


# -------------------------
# CANDIDATE DASHBOARD FUNCTION 
# -------------------------

def candidate_dashboard():
    st.title("🧑‍💻 Candidate Dashboard")
    
    col_header, col_logout = st.columns([4, 1])
    with col_logout:
        if st.button("🚪 Log Out", use_container_width=True):
            # Clear all session states related to the dashboard but keep the page structure intact
            for key in list(st.session_state.keys()):
                if key not in ['page', 'logged_in', 'user_type']:
                    del st.session_state[key]
            go_to("login")
            st.rerun() 
            
    st.markdown("---")

    # --- Session State Initialization ---
    if "parsed" not in st.session_state: st.session_state.parsed = {} 
    if "full_text" not in st.session_state: st.session_state.full_text = ""
    # Initialize cv_form_data with a copy of the default template if it's not present (prevents key errors)
    if "cv_form_data" not in st.session_state: st.session_state.cv_form_data = default_cv_template().copy() 
    if "excel_data" not in st.session_state: st.session_state.excel_data = None
    if "candidate_uploaded_resumes" not in st.session_state: st.session_state.candidate_uploaded_resumes = []
    if "pasted_cv_text" not in st.session_state: st.session_state.pasted_cv_text = ""
    if "last_parsed_file_name" not in st.session_state: st.session_state.last_parsed_file_name = None 
    
    if "candidate_jd_list" not in st.session_state: st.session_state.candidate_jd_list = []
    if "candidate_match_results" not in st.session_state: st.session_state.candidate_match_results = []
    

    # --- Main Content with Tabs ---
    tab_parsing, tab_management, tab_jd, tab_batch_match = st.tabs(["📄 Resume Parsing", "📝 CV Management", "📚 JD Management", "🎯 Batch JD Match"])
    
    with tab_parsing:
        resume_parsing_tab()
        
    with tab_management:
        cv_management_tab_content() 
        
    with tab_jd:
        jd_management_tab_candidate()
        
    with tab_batch_match:
        jd_batch_match_tab()


# -------------------------
# MOCK LOGIN AND MAIN APP LOGIC 
# -------------------------

def login_page():
    st.set_page_config(layout="wide", page_title="PragyanAI Candidate Dashboard")
    st.title("Welcome to PragyanAI")
    st.header("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username (Enter 'candidate')")
        password = st.text_input("Password (Any value)", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username.lower() == "candidate":
                st.session_state.logged_in = True
                st.session_state.user_type = "candidate"
                go_to("candidate_dashboard")
                st.rerun()
            else:
                st.error("Invalid username. Please use 'candidate'.")

# --- Main App Execution ---

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="PragyanAI Candidate Dashboard")

    if 'page' not in st.session_state: st.session_state.page = "login"
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'user_type' not in st.session_state: st.session_state.user_type = None
    
    if st.session_state.logged_in and st.session_state.user_type == "candidate":
        candidate_dashboard()
    else:
        login_page()
