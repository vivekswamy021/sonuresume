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
import tempfile
import random

# --- CONFIGURATION & API SETUP ---

GROQ_MODEL = "llama-3.1-8b-instant"
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq Client or Mock Client 
try:
    from groq import Groq
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")
    client = Groq(api_key=GROQ_API_KEY)
except (ImportError, ValueError) as e:
    # Mock Client for local testing without Groq
    class MockGroqClient:
        def chat(self):
            class Completions:
                def create(self, **kwargs):
                    return type('MockResponse', (object,), {'choices': [{'message': {'content': '{"name": "Mock Candidate", "summary": "Mock summary for testing.", "skills": ["Python", "Streamlit"]}'}}]})()
            return Completions()
    client = MockGroqClient()
    # st.info(f"AI functions are running in MOCK mode because Groq setup failed: {e}") # Commented out to reduce clutter


# --- Utility Functions ---

def go_to(page_name):
    """Changes the current page in Streamlit's session state."""
    st.session_state.page = page_name

def get_file_type(file_name):
    """Identifies the file type based on its extension."""
    ext = os.path.splitext(file_name)[1].lower().strip('.')
    if ext == 'pdf': return 'pdf'
    elif ext == 'docx' or ext == 'doc': return 'docx'
    elif ext == 'txt': return 'txt'
    else: return 'unknown' 

def extract_content(file_type, file_content_bytes, file_name):
    """Extracts text content from uploaded file content (bytes)."""
    text = ''
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
        
        elif file_type in ['txt', 'unknown']:
            try:
                text = file_content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                 text = file_content_bytes.decode('latin-1')
        
        if not text.strip():
            return f"[Error] {file_type.upper()} content extraction failed or file is empty."
        
        return text
    
    except Exception as e:
        return f"[Error] Fatal Extraction Error: Failed to read file content ({file_type}). Error: {e}\n{traceback.format_exc()}"


@st.cache_data(show_spinner="Analyzing content with Groq LLM...")
def parse_resume_with_llm(text):
    """Sends resume text to the LLM for structured information extraction."""
    # Simplified mock for demonstration
    if text.startswith("[Error") or isinstance(client, MockGroqClient):
        return {"name": "Mock Candidate", "summary": "Mock summary for testing.", "skills": ["Python", "Streamlit", "SQL", "AWS"], "experience": ["3 years"], "education": ["Master's in CS"], "error": "Mock/Parsing error."}

    # ... (Actual Groq API call logic would go here)
    # Returning a mock structure for the actual case too
    return {"name": "Parsed Candidate", "summary": "Actual parsed summary.", "skills": ["Python", "Streamlit", "SQL", "AWS"], "experience": ["3 years"], "education": ["Master's in CS"]}


@st.cache_data(show_spinner="Analyzing JD for metadata...")
def extract_jd_metadata(jd_text):
    """Mocks the extraction of key metadata (Role, Skills, Job Type) from JD text."""
    if jd_text.startswith("[Error"):
        return {"role": "Error", "key_skills": ["Error"], "job_type": "Error"}
    
    role_match = re.search(r'(?:Role|Position|Title)[:\s]+([\w\s/-]+)', jd_text, re.IGNORECASE)
    role = role_match.group(1).strip() if role_match else f"Data Scientist ({random.choice(['Senior', 'Junior'])})"
    
    skills_match = re.findall(r'(Python|Java|SQL|AWS|Docker|Kubernetes|React|Streamlit|LLMs)', jd_text, re.IGNORECASE)
    
    job_type_match = re.search(r'(Full-time|Part-time|Contract|Remote)', jd_text, re.IGNORECASE)
    job_type = job_type_match.group(1) if job_type_match else "Full-time"
    
    return {
        "role": role, 
        "key_skills": list(set([s.lower() for s in skills_match][:5])),
        "job_type": job_type
    }

def extract_jd_from_linkedin_url(url):
    """Mocks the extraction of JD content from a LinkedIn URL."""
    if "linkedin.com/jobs" not in url:
        return f"[Error] Invalid LinkedIn Job URL: {url}"

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
    
# --- NEW CORE MATCHING FUNCTION (MOCK) ---

@st.cache_data(show_spinner="Running LLM match analysis...")
def evaluate_jd_fit(jd_content, parsed_resume_json):
    """
    Mocks the LLM call to evaluate resume fit against a JD.
    In a real app, this would be a Groq call with a detailed prompt.
    """
    
    # Mock Logic: Generate scores based on random seeds for diversity
    # We use a simple hash of the JD content to create a stable but varied mock score
    score_seed = sum(ord(c) for c in jd_content) % 100
    
    overall_score = 5 + (score_seed % 6) # Score between 5 and 10
    
    skills_percent = 60 + (score_seed % 30) # 60-90%
    experience_percent = 70 + ((score_seed * 3) % 20) # 70-90%
    education_percent = 80 + ((score_seed * 5) % 15) # 80-95%

    summary = parsed_resume_json.get('summary', 'The candidate has strong technical skills.')
    
    fit_output = f"""
    Overall Fit Score: **{overall_score}** /10
    
    --- Candidate Resume Summary ---
    {summary}
    
    --- Section Match Analysis ---
    Skills Match: [{skills_percent}%] - The candidate possesses most of the key technologies (Python, SQL, AWS) but lacks proficiency in Docker mentioned in the JD.
    Experience Match: [{experience_percent}%] - The candidate's 3 years of experience align well with the mid-level requirements.
    Education Match: [{education_percent}%] - Master's degree meets/exceeds typical requirements.
    
    Strengths/Matches:
    * **Strong Technical Core:** Excellent alignment on Python and AWS.
    * **Experience Duration:** Meets the minimum bar set by the job description.
    
    Weaknesses/Gaps:
    * **Missing DevOps Tooling:** Candidate's resume lacks mention of Kubernetes or strong Docker experience, which are crucial.
    * **Soft Skills:** JD emphasizes leadership, which is not strongly highlighted in the resume.
    """
    return fit_output

# --- CV Helper Functions (Simplified for display) ---

def save_form_cv():
    """Compiles the structured CV data from form states and saves it."""
    current_form_name = st.session_state.get('form_name_value', '').strip()
    if not current_form_name:
         st.error("Please enter your **Full Name** to save the CV.") 
         return
    
    cv_key_name = f"{current_form_name.replace(' ', '_')}_Manual_CV_{datetime.now().strftime('%Y%m%d-%H%M')}"
    st.session_state.managed_cvs[cv_key_name] = {"name": current_form_name, "summary": st.session_state.get('form_summary_value', '')}
    st.session_state.show_cv_output = cv_key_name 
    # Also set the globally accessible 'parsed' key for the batch match tab
    st.session_state.parsed = st.session_state.managed_cvs[cv_key_name]
    st.session_state.is_resume_parsed = True
    st.success(f"🎉 CV for **'{current_form_name}'** saved/updated!")

def generate_and_display_cv(cv_name):
    """Generates the final structured CV data from session state and displays it."""
    if cv_name not in st.session_state.managed_cvs:
        st.error(f"Error: CV '{cv_name}' not found.")
        return
        
    cv_data = st.session_state.managed_cvs[cv_name]
    st.markdown(f"### 📄 CV View: **{cv_name}**")
    st.info(f"Summary: {cv_data.get('summary', 'No summary available.')}")
    

# --- JD Management Logic ---

def jd_management_tab_candidate():
    """JD Management Tab for Candidate."""
    st.header("📚 Manage Job Descriptions for Matching")
    st.markdown("Add multiple JDs here to compare your resume against them in the next tabs.")
    
    if "candidate_jd_list" not in st.session_state:
        st.session_state.candidate_jd_list = []
    
    st.markdown("---")
    
    jd_type = st.radio("Select JD Type", ["Single JD", "Multiple JD"], key="jd_type_candidate", index=0)
    st.markdown("### Add JD by:")
    
    method = st.radio("Choose Method", ["Upload File", "Paste Text", "LinkedIn URL"], key="jd_add_method_candidate", index=0) 
    
    st.markdown("---")

    # --- LinkedIn URL Section ---
    if method == "LinkedIn URL":
        with st.form("jd_url_form_candidate", clear_on_submit=True):
            url_list = st.text_area(
                "Enter one or more URLs (comma separated)" if jd_type == "Multiple JD" else "Enter URL", key="url_list_candidate"
            )
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
                            
                        name_base = url.split('/jobs/view/')[-1].split('/')[0] if '/jobs/view/' in url else f"URL JD"
                        name = f"JD from URL: {name_base}" 
                        if any(item['name'] == name for item in st.session_state.candidate_jd_list):
                            name = f"JD from URL: {name_base} ({len(st.session_state.candidate_jd_list) + 1})" 

                        st.session_state.candidate_jd_list.append({"name": name, "content": jd_text, **metadata})
                        count += 1
                            
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully! Check the display below for the extracted content.")
                        st.rerun()

    # --- Paste Text Section ---
    elif method == "Paste Text":
        with st.form("jd_paste_form_candidate", clear_on_submit=True):
            text_list = st.text_area(
                "Paste one or more JD texts (separate by '---')" if jd_type == "Multiple JD" else "Paste JD text here", key="text_list_candidate"
            )
            if st.form_submit_button("Add JD(s) from Text", key="add_jd_text_btn_candidate"):
                if text_list:
                    texts = [t.strip() for t in text_list.split("---")] if jd_type == "Multiple JD" else [text_list.strip()]
                    count = 0
                    for i, text in enumerate(texts):
                        if text:
                            name_base = text.splitlines()[0].strip()
                            if len(name_base) > 30: name_base = f"{name_base[:27]}..."
                            if not name_base: name_base = f"Pasted JD {len(st.session_state.candidate_jd_list) + i + 1}"
                            
                            metadata = extract_jd_metadata(text)
                            st.session_state.candidate_jd_list.append({"name": name_base, "content": text, **metadata})
                            count += 1
                    
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully!")
                        st.rerun()

    # --- Upload File Section ---
    elif method == "Upload File":
        uploaded_files = st.file_uploader(
            "Upload JD file(s)",
            type=["pdf", "txt", "docx"],
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
                            jd_text = extract_content(file_type, file.getvalue(), file.name)
                            
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
        with col_display_header:
            st.markdown("### ✅ Current JDs Added:")
            
        with col_clear_button:
            if st.button("🗑️ Clear All JDs", key="clear_jds_candidate", use_container_width=True, help="Removes all currently loaded JDs."):
                st.session_state.candidate_jd_list = []
                st.session_state.candidate_match_results = []
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
    if "managed_cvs" not in st.session_state: st.session_state.managed_cvs = {} 
    if "show_cv_output" not in st.session_state: st.session_state.show_cv_output = None 
    if "form_name_value" not in st.session_state: st.session_state.form_name_value = ""
    if "candidate_jd_list" not in st.session_state: st.session_state.candidate_jd_list = []
    if "candidate_match_results" not in st.session_state: st.session_state.candidate_match_results = []
    # Key to store the currently active parsed CV (for matching)
    if "parsed" not in st.session_state: st.session_state.parsed = None 
    # Key to check if matching is possible
    is_resume_parsed = st.session_state.parsed is not None
    
    # --- Main Content with New Tab ---
    # Tab Order: Resume Parsing, CV Management, Batch Match, JD Management
    tab_parsing, tab_management, tab_batch_match, tab_jd_mgmt = st.tabs([
        "📄 Resume Parsing", 
        "📝 CV Management (Form)", 
        "🎯 Batch JD Match", # NEW TAB
        "📚 JD Management"
    ])
    
    with tab_parsing:
        resume_parsing_tab()
        
    with tab_management:
        cv_form_content() 
        
    # --- TAB 3 (Now tab_batch_match): Batch JD Match (Candidate) ---
    with tab_batch_match:
        st.header("🎯 Batch JD Match: Best Matches")
        st.markdown("Compare your current resume against all saved job descriptions.")

        if not is_resume_parsed:
            st.warning("Please **upload and parse your resume** in the 'Resume Parsing' tab or **build your CV** in the 'CV Management' tab first.")
            
        elif not st.session_state.candidate_jd_list:
            st.error("Please **add Job Descriptions** in the 'JD Management' tab before running batch analysis.")
            
        elif isinstance(client, MockGroqClient): # Check if we are running in mock mode due to missing key
             st.error("Cannot use JD Match: GROQ_API_KEY is not configured or LLM client is in mock mode.")
             
        else:
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
            
            if st.button(f"Run Match Analysis on {len(jds_to_match)} Selected JD(s)", type="primary"):
                st.session_state.candidate_match_results = []
                
                if not jds_to_match:
                    st.warning("Please select at least one Job Description to run the analysis.")
                    
                else:
                    # Use the parsed resume data from the session state
                    resume_name = st.session_state.parsed.get('name', 'Uploaded Resume')
                    parsed_json = st.session_state.parsed
                    results_with_score = []

                    with st.spinner(f"Matching {resume_name}'s resume against {len(jds_to_match)} selected JD(s)..."):
                        
                        # Loop over jds_to_match
                        for jd_item in jds_to_match:
                            
                            jd_name = jd_item['name']
                            jd_content = jd_item['content']

                            try:
                                # Call the LLM fit evaluation function (mocked here)
                                fit_output = evaluate_jd_fit(jd_content, parsed_json)
                                
                                # --- Regex Extraction of Scores ---
                                overall_score_match = re.search(r'Overall Fit Score:\s*[^\d]*(\d+)\s*/10', fit_output, re.IGNORECASE)
                                section_analysis_match = re.search(
                                    r'--- Section Match Analysis ---\s*(.*?)\s*Strengths/Matches:', 
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
                                    "numeric_score": int(overall_score) if str(overall_score).isdigit() else -1, # Added for sorting/ranking
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
                        # 1. Sort by numeric_score (highest first)
                        results_with_score.sort(key=lambda x: x['numeric_score'], reverse=True)
                        
                        # 2. Assign Rank (handle ties)
                        current_rank = 1
                        current_score = -1 
                        
                        for i, item in enumerate(results_with_score):
                            # Assign rank only if score is valid (not -1/Error)
                            if item['numeric_score'] > current_score:
                                current_rank = i + 1
                                current_score = item['numeric_score']
                            
                            item['rank'] = current_rank if item['numeric_score'] != -1 else "Error"
                            # Remove the temporary numeric_score field
                            del item['numeric_score'] 
                            
                        st.session_state.candidate_match_results = results_with_score
                        st.success("Batch analysis complete!")


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
                        st.markdown(item['full_analysis'])

    with tab_jd_mgmt:
        jd_management_tab_candidate()


def resume_parsing_tab():
    st.header("📄 Upload/Paste Resume for AI Parsing")
    st.caption("Upload a file or paste text to extract structured data and save it as a structured CV.")
    
    with st.form("resume_parsing_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "Upload Resume File (.pdf, .docx, .txt)", 
            type=['pdf', 'docx', 'txt'], 
            accept_multiple_files=False,
            key="resume_uploader"
        )
        st.markdown("---")
        pasted_text = st.text_area("Or Paste Resume Text Here", height=200, key="resume_paster")
        st.markdown("---")

        if st.form_submit_button("✨ Parse and Structure CV", type="primary", use_container_width=True):
            extracted_text = ""
            file_name = "Pasted_Resume"
            
            if uploaded_file is not None:
                file_name = uploaded_file.name
                file_type = get_file_type(file_name)
                extracted_text = extract_content(file_type, uploaded_file.getvalue(), file_name)
            elif pasted_text.strip():
                extracted_text = pasted_text.strip()
            else:
                st.warning("Please upload a file or paste text content to proceed.")
                return

            if extracted_text.startswith("[Error"):
                st.error(f"Text Extraction Failed: {extracted_text}")
                return
                
            with st.spinner("🧠 Sending to Groq LLM for structured parsing..."):
                parsed_data = parse_resume_with_llm(extracted_text)
            
            if "error" in parsed_data:
                st.error(f"AI Parsing Failed: {parsed_data['error']}")
                return

            candidate_name = parsed_data.get('name', 'Unknown_Candidate').replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            cv_key_name = f"{candidate_name}_{timestamp}"
            
            st.session_state.managed_cvs[cv_key_name] = parsed_data
            st.session_state.show_cv_output = cv_key_name
            st.session_state.parsed = parsed_data # Set the parsed CV for batch matching
            
            st.success(f"✅ Successfully parsed and structured CV for **{candidate_name}**! Now ready for matching.")
            st.rerun()

# -------------------------
# MOCK LOGIN AND MAIN APP LOGIC 
# -------------------------

def login_page():
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
