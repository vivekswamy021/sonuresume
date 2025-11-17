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
import base64 

# --- CONFIGURATION & API SETUP ---

GROQ_MODEL = "llama-3.1-8b-instant"
# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()
# Note: GROQ_API_KEY loading is kept for structure, but Mock Client is used by default if key is missing.
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# --- Define MockGroqClient globally ---

class MockGroqClient:
    """Mock client for local testing when Groq is not available or key is missing."""
    def chat(self):
        class Completions:
            def create(self, **kwargs):
                # This mock response is only used if the LLM function doesn't parse the input text first.
                return type('MockResponse', (object,), {'choices': [{'message': {'content': '{"name": "Default Mock Candidate", "email": "mock@default.com", "personal_details": "This is a default mock response, used when the input text (e.g., a simple PDF) cannot be parsed directly by the LLM function."}'}}]})()
        return Completions()

# Initialize Groq Client or use Mock Client 
client = None
try:
    # Attempt to initialize the real client
    if not GROQ_API_KEY:
        # If key is missing, treat it as a setup failure and fall back to mock
        raise ValueError("GROQ_API_KEY not set.") 
    
    # --- Simulated Groq Client with correct nested structure (Real API simulation) ---
    class Groq:
        def __init__(self, api_key): pass
        def chat(self):
            class Completions:
                def create(self, **kwargs):
                    # In a real scenario, the LLM processes the input text (kwargs['messages'][1]['content'])
                    # and returns the JSON structure. We simulate a generic successful parse here.
                    mock_llm_json = {
                        "name": "LLM Parsed Candidate", 
                        "email": "llm_parsed@example.com", 
                        "phone": "555-0000", 
                        "personal_details": "Successfully parsed by the simulated LLM service.", 
                        "skills": ["Simulated Skill", "Python", "Streamlit"], 
                        "error": None
                    }
                    
                    message_obj = type('Message', (object,), {'content': json.dumps(mock_llm_json)})()
                    choice_obj = type('Choice', (object,), {'message': message_obj})()
                    response_obj = type('MockResponse', (object,), {'choices': [choice_obj]})()
                    return response_obj
            return Completions()
        
    client = Groq(api_key=GROQ_API_KEY)
    
except (ImportError, ValueError, Exception) as e:
    # Fallback to Mock Client
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
    elif ext in ('txt', 'md', 'markdown', 'rtf'): return 'txt' 
    elif ext == 'json': return 'json'
    elif ext in ('xlsx', 'xls', 'csv'): return 'excel' 
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
                # Wrap JSON content so LLM parsing function can detect and use it
                text = file_content_bytes.decode('utf-8')
                text = "--- JSON Content Start ---\n" + text + "\n--- JSON Content End ---"
            except UnicodeDecodeError:
                return f"[Error] JSON content extraction failed: Unicode Decode Error.", None
        
        elif file_type == 'excel':
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(BytesIO(file_content_bytes))
                else: 
                    xls = pd.ExcelFile(BytesIO(file_content_bytes))
                    all_sheets_data = {}
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        all_sheets_data[sheet_name] = df.to_json(orient='records') 
                        
                    excel_data = all_sheets_data 
                    text = json.dumps(all_sheets_data, indent=2)
                    text = f"[EXCEL_CONTENT] The following structured data was extracted:\n{text}"
                    
            except Exception as e:
                return f"[Error] Excel/CSV file parsing failed. Error: {e}", None


        if not text.strip() and file_type not in ('excel', 'json'): 
            return f"[Error] {file_type.upper()} content extraction failed or file is empty.", None
        
        return text, excel_data
    
    except Exception as e:
        return f"[Error] Fatal Extraction Error: Failed to read file content ({file_type}). Error: {e}\n{traceback.format_exc()}", None


@st.cache_data(show_spinner="Analyzing content with Groq LLM...")
def parse_resume_with_llm(text):
    """Sends resume text to the LLM for structured information extraction."""
    
    # 1. Handle Pre-flight errors (e.g., failed extraction)
    if text.startswith("[Error"):
        return {"name": "Parsing Error", "error": text}

    # Helper function to get a fallback name
    def get_fallback_name():
        name = "Parsed Candidate"
        if st.session_state.get('last_parsed_file_name'):
            # Use filename, strip extension, and replace underscores/dashes with spaces
            name_part = os.path.splitext(st.session_state.last_parsed_file_name)[0]
            name = name_part.replace('_', ' ').replace('-', ' ').title().replace(' (1)', '').strip()
            if name == "Pasted Text": name = "Pasted Resume Data"
        return name if name else "Parsed Candidate"
    
    candidate_name = get_fallback_name()
    
    # 2. Check for and parse direct JSON content (for JSON file uploads)
    json_match = re.search(r'--- JSON Content Start ---\s*(.*?)\s*--- JSON Content End ---', text, re.DOTALL)
    
    if json_match:
        try:
            # If JSON is found, try to parse it directly
            json_content = json_match.group(1).strip()
            parsed_data = json.loads(json_content)
            
            # Ensure name is set, prioritizing the name from the JSON, then the filename fallback
            if not parsed_data.get('name'):
                 parsed_data['name'] = candidate_name
                 
            # Ensure error is explicitly set to None on success
            parsed_data['error'] = None
            
            # Return the actual JSON content from the uploaded file
            return parsed_data
        
        except json.JSONDecodeError:
            # If JSON parsing fails, treat it as an LLM error for the front-end to handle
            return {"name": candidate_name, "error": f"LLM Input Error: Could not decode uploaded JSON content into a valid structure."}
        
    # 3. Handle Mock Client execution (Fallback for PDF/DOCX/TXT)
    if isinstance(client, MockGroqClient):
        # Generate structured data using the dynamic name
        return {
            "name": candidate_name, 
            "email": "mock@example.com", 
            "phone": "555-1234", 
            "linkedin": "https://linkedin.com/in/mock", 
            "github": "https://github.com/mock", 
            "personal_details": f"Mock summary generated for: {candidate_name}. (Input was text/PDF, not direct JSON)", 
            "skills": ["Python", "Streamlit", "SQL", "AWS"], 
            "education": ["B.S. Computer Science, Mock University, 2020"], 
            "experience": ["Software Intern, Mock Solutions (2024-2025)", "Data Analyst, Test Corp (2022-2024)"], 
            "certifications": ["Mock Certification"], 
            "projects": ["Mock Project: Built a dashboard using Streamlit."], 
            "strength": ["Mock Strength"], 
            "error": None
        }

    # 4. Handle Real Groq Client execution (Simulated)
    try:
        # If this were a real LLM call for PDF/TXT/DOCX, it would return a parsed JSON structure.
        completion = client.chat().create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a resume parsing AI. Extract structured data from the text."},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        json_content = completion.choices[0].message.content
        parsed_data = json.loads(json_content)
        
        # Ensure name is set, using LLM's parse first, then fallback
        if not parsed_data.get('name'):
             parsed_data['name'] = candidate_name 
             
        parsed_data['error'] = None 
        return parsed_data
        
    except Exception as e:
        error_message = f"LLM Processing Error: {e.__class__.__name__} - {str(e)}"
        return {"name": candidate_name, "error": error_message} 
    

# --- HELPER FUNCTIONS FOR FILE/TEXT PROCESSING ---

def clear_interview_state():
    """Clears all session state variables related to interview/match sessions."""
    if 'interview_chat_history' in st.session_state: del st.session_state['interview_chat_history']
    if 'current_interview_jd' in st.session_state: del st.session_state['current_interview_jd']
    if 'evaluation_report' in st.session_state: del st.session_state['evaluation_report']
    if 'candidate_match_results' in st.session_state: st.session_state.candidate_match_results = []
    

def parse_and_store_resume(content_source, source_type):
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
        st.session_state.last_parsed_file_name = file_name 
        extracted_text, excel_data = extract_content(file_type, uploaded_file.getvalue(), file_name)
    elif source_type == 'text':
        extracted_text = content_source.strip()
        file_name = "Pasted_Text"
        st.session_state.last_parsed_file_name = file_name 

    if extracted_text.startswith("[Error"):
        # The extraction itself failed
        return {"error": extracted_text, "full_text": extracted_text, "excel_data": None, "name": file_name}
    
    # 2. Call LLM Parser
    parsed_data = parse_resume_with_llm(extracted_text)
    
    # 3. Handle LLM Parsing Error
    if parsed_data.get('error') is not None and parsed_data.get('error') != "":
        # The LLM parsing itself failed
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

    # The final name is now reliably set within the LLM function and is in parsed_data
    final_name = parsed_data.get('name', 'Unknown_Candidate').replace(' ', '_')
    
    return {
        "parsed": parsed_data, 
        "full_text": compiled_text, 
        "excel_data": excel_data, 
        "name": final_name
    }

# --- END HELPER FUNCTIONS ---


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

    # --- A. Upload File Method ---
    if input_method == "Upload File":
        st.markdown("### 1. Upload Resume File") 
        
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

        if "candidate_uploaded_resumes" not in st.session_state: st.session_state.candidate_uploaded_resumes = []
        if "pasted_cv_text" not in st.session_state: st.session_state.pasted_cv_text = ""
        if "last_parsed_file_name" not in st.session_state: st.session_state.last_parsed_file_name = None

        # --- File Management Logic ---
        if uploaded_file is not None:
            if not st.session_state.candidate_uploaded_resumes or st.session_state.candidate_uploaded_resumes[0].name != uploaded_file.name:
                st.session_state.candidate_uploaded_resumes = [uploaded_file] 
                st.session_state.pasted_cv_text = "" 
                st.toast("Resume file uploaded successfully.")
        elif st.session_state.candidate_uploaded_resumes and uploaded_file is None:
            st.session_state.candidate_uploaded_resumes = []
            st.session_state.parsed = {}
            st.session_state.full_text = ""
            st.session_state.excel_data = None
            st.session_state.last_parsed_file_name = None
            st.toast("Upload cleared.")
            
        file_to_parse = st.session_state.candidate_uploaded_resumes[0] if st.session_state.candidate_uploaded_resumes else None
        
        st.markdown("### 2. Parse Uploaded File")
        
        if file_to_parse:
            is_already_parsed = (
                st.session_state.get('last_parsed_file_name') == file_to_parse.name and 
                st.session_state.get('parsed', {}).get('name') is not None and
                st.session_state.get('parsed', {}).get('error') is None 
            )

            if st.button(f"Parse and Load: **{file_to_parse.name}**", use_container_width=True, disabled=is_already_parsed):
                with st.spinner(f"Parsing {file_to_parse.name}..."):
                    result = parse_and_store_resume(file_to_parse, source_type='file')
                    
                    if result.get('error') is None:
                        st.session_state.parsed = result['parsed']
                        st.session_state.full_text = result['full_text']
                        st.session_state.excel_data = result['excel_data'] 
                        clear_interview_state()
                        
                        st.success(f"✅ Successfully loaded and parsed **{st.session_state.parsed['name']}**.")
                        st.rerun() 
                    else:
                        st.error(f"Parsing failed for {file_to_parse.name}: {result['error']}")
                        st.session_state.parsed = {"error": result['error'], "name": result['name']}
                        st.session_state.full_text = result['full_text'] or ""
                        st.session_state.excel_data = result['excel_data'] 
                        
            if is_already_parsed:
                st.info(f"The file **{file_to_parse.name}** is already parsed and loaded.")

        else:
            st.info("No resume file is currently uploaded. Please upload a file above.")

    # --- B. Paste Text Method ---
    else: # input_method == "Paste Text"
        st.markdown("### 1. Paste Your CV Text")
        
        pasted_text = st.text_area(
            "Copy and paste your entire CV or resume text here.",
            value=st.session_state.get('pasted_cv_text', ''),
            height=300,
            key='pasted_cv_text_input'
        )
        st.session_state.pasted_cv_text = pasted_text 
        
        st.markdown("---")
        st.markdown("### 2. Parse Pasted Text")
        
        if pasted_text.strip():
            is_already_parsed = (
                st.session_state.get('last_parsed_file_name') == "Pasted_Text" and 
                st.session_state.get('pasted_cv_text_input', '') == pasted_text and
                st.session_state.get('parsed', {}).get('error') is None
            )

            if st.button("Parse and Load Pasted Text", use_container_width=True, disabled=is_already_parsed):
                with st.spinner("Parsing pasted text..."):
                    st.session_state.candidate_uploaded_resumes = []
                    
                    result = parse_and_store_resume(pasted_text, source_type='text')
                    
                    if result.get('error') is None:
                        st.session_state.parsed = result['parsed']
                        st.session_state.full_text = result['full_text']
                        st.session_state.excel_data = result['excel_data'] 
                        clear_interview_state()
                        
                        st.success(f"✅ Successfully loaded and parsed **{st.session_state.parsed['name']}**.")
                        st.rerun() 
                    else:
                        st.error(f"Parsing failed: {result['error']}")
                        st.session_state.parsed = {"error": result['error'], "name": result['name']}
                        st.session_state.full_text = result['full_text'] or ""
                        st.session_state.excel_data = result['excel_data'] 
        else:
            st.info("Please paste your CV text into the box above.")
            
    st.markdown("---")
    
    # --- New Section for Displaying Loaded Status ---
    st.subheader("3. Current Loaded Candidate Status")
    
    is_data_loaded_and_valid = (
        st.session_state.get('parsed', {}).get('name') is not None and 
        st.session_state.get('parsed', {}).get('error') is None
    )

    if is_data_loaded_and_valid:
        st.markdown(f"**Status:** ✅ **Loaded**")
        st.markdown(f"**Candidate:** **{st.session_state.parsed['name']}**")
        st.caption(f"Source: {st.session_state.get('last_parsed_file_name', 'Unknown Source')}")
        st.info("The parsed CV data is ready and available for matching in the **Batch JD Match** tab.")
    else:
        st.warning(f"**Status:** ❌ **Not Loaded**")
        st.info("Please parse a resume in the sections above to continue to the **JD Match** tab.")

# --- JD Management Tab Function ---
        
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
    is_resume_parsed = (
        st.session_state.get('parsed') is not None and
        st.session_state.parsed.get('name') is not None and
        st.session_state.parsed.get('error') is None
    )
    
    if not is_resume_parsed:
        st.warning("⚠️ Please **upload and parse your resume** in the 'Resume Parsing' tab first.")
        
    elif not st.session_state.candidate_jd_list:
        st.error("❌ Please **add Job Descriptions** in the 'JD Management' tab before running batch analysis.")
        
    elif isinstance(client, MockGroqClient):
        st.info("ℹ️ Running in Mock LLM Mode. Match results will be simulated.")
        
    else:
        try:
            if not isinstance(client, Groq):
                st.warning("⚠️ LLM client setup failed. Match analysis may not be accurate or available.")
        except NameError:
             st.warning("⚠️ LLM client setup failed. Match analysis may not be accurate or available.")


    if "candidate_match_results" not in st.session_state:
        st.session_state.candidate_match_results = []

    # 1. Get all available JD names
    all_jd_names = [item['name'] for item in st.session_state.candidate_jd_list]
    
    # 2. Add multiselect widget
    selected_jd_names = st.multiselect(
        "Select Job Descriptions to Match Against",
        options=all_jd_names,
        default=all_jd_names, 
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
            
        elif not is_resume_parsed:
             st.warning("Please **upload and parse your resume** first.")

        else:
            resume_name = st.session_state.parsed.get('name', 'Uploaded Resume')
            parsed_json = st.session_state.parsed
            results_with_score = []

            with st.spinner(f"Matching {resume_name}'s resume against {len(jds_to_match)} selected JD(s)..."):
                
                for jd_item in jds_to_match:
                    jd_name = jd_item['name']
                    jd_content = jd_item['content']

                    try:
                        fit_output = evaluate_jd_fit(jd_content, parsed_json) 
                        
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
                            "numeric_score": int(overall_score) if overall_score.isdigit() else -1, 
                            "skills_percent": skills_percent,
                            "experience_percent": experience_percent, 
                            "education_percent": education_percent, 
                            "full_analysis": fit_output
                        })
                    except Exception as e:
                        results_with_score.append({
                            "jd_name": jd_name,
                            "overall_score": "Error",
                            "numeric_score": -1, 
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
                    if item['numeric_score'] > current_score:
                        current_rank = i + 1
                        current_score = item['numeric_score']
                        
                    item['rank'] = current_rank
                    del item['numeric_score'] 
                    
                st.session_state.candidate_match_results = results_with_score
                # --- END NEW RANKING LOGIC ---
                
                st.success("Batch analysis complete! See results below.")
                st.rerun() 


    # 3. Display Results (UPDATED TO INCLUDE RANK)
    if st.session_state.get('candidate_match_results'):
        st.markdown("#### Match Results for Your Resume")
        results_df = st.session_state.candidate_match_results
        
        display_data = []
        for item in results_df:
            full_jd_item = next((jd for jd in st.session_state.candidate_jd_list if jd['name'] == item['jd_name']), {})
            
            display_data.append({
                "Rank": item.get("rank", "N/A"),
                "Job Description (Ranked)": item["jd_name"].replace("--- Simulated JD for: ", ""),
                "Role": full_jd_item.get('role', 'N/A'), 
                "Job Type": full_jd_item.get('job_type', 'N/A'), 
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
                st.code(item['full_analysis'], language='markdown')


# -------------------------
# CANDIDATE DASHBOARD FUNCTION 
# -------------------------

def candidate_dashboard():
    st.title("🧑‍💻 Candidate Dashboard")
    
    col_header, col_logout = st.columns([4, 1])
    with col_logout:
        if st.button("🚪 Log Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['page', 'logged_in', 'user_type']:
                    del st.session_state[key]
            go_to("login")
            st.rerun() 
            
    st.markdown("---")

    # --- Session State Initialization ---
    if "parsed" not in st.session_state: st.session_state.parsed = {} 
    if "full_text" not in st.session_state: st.session_state.full_text = ""
    if "excel_data" not in st.session_state: st.session_state.excel_data = None
    if "candidate_uploaded_resumes" not in st.session_state: st.session_state.candidate_uploaded_resumes = []
    if "pasted_cv_text" not in st.session_state: st.session_state.pasted_cv_text = ""
    if "last_parsed_file_name" not in st.session_state: st.session_state.last_parsed_file_name = None 
    
    if "candidate_jd_list" not in st.session_state: st.session_state.candidate_jd_list = []
    if "candidate_match_results" not in st.session_state: st.session_state.candidate_match_results = []
    

    # --- Main Content with Tabs ---
    tab_parsing, tab_jd, tab_batch_match = st.tabs(["📄 Resume Parsing", "📚 JD Management", "🎯 Batch JD Match"])
    
    with tab_parsing:
        resume_parsing_tab()
        
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
