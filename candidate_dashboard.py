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
import time

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
                    # Simple mock response
                    return type('MockResponse', (object,), {'choices': [{'message': {'content': '{"name": "Mock Candidate", "summary": "Mock summary for testing.", "skills": ["Python", "Streamlit"]}'}}]})()
            return Completions()
    client = MockGroqClient()

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
    if text.startswith("[Error") or isinstance(client, MockGroqClient):
        # Mock structured data for demonstration
        return {"name": "Mock Candidate", "email": "mock@example.com", "phone": "555-1234", "summary": "Highly motivated individual with mock experience in Python and Streamlit.", "skills": ["Python", "Streamlit", "SQL", "AWS"], "education": ["B.S. Computer Science, Mock University"], "experience": ["Software Intern, Mock Solutions (2024-2025)"], "error": "Mock/Parsing error." if isinstance(client, MockGroqClient) else text}

    # Actual Groq call logic would go here
    prompt = f"Extract structured data from resume: {text}..."
    # ... (Groq API call logic)
    return {"name": "Parsed Candidate", "summary": "Actual parsed summary.", "skills": ["Real", "Python", "Streamlit"], "education": ["University of Code"], "experience": ["Senior Developer, TechCo"]} 


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
    In a real app, this would be a Groq call with a detailed prompt.
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
    Education Match: [{edu}%]
    
    --- Strengths/Matches ---
    The candidate shows strong proficiency in Python and Streamlit, which aligns well with the required data processing tools. Education is directly relevant.
    
    --- Weaknesses/Gaps ---
    Candidate lacks specific mention of Kubernetes or advanced Docker orchestration, which is a required skill for this role.
    
    --- Summary Recommendation ---
    A strong candidate with core skills. Recommend for interview if the experience gap in orchestration tools can be overlooked or quickly trained.
    """


# --- CV Helper Functions (Simplified for display) ---

def save_form_cv():
    """Compiles the structured CV data from form states and saves it."""
    current_form_name = st.session_state.get('form_name_value', '').strip()
    if not current_form_name:
         st.error("Please enter your **Full Name** to save the CV.") 
         return
    
    cv_key_name = f"{current_form_name.replace(' ', '_')}_Manual_CV_{datetime.now().strftime('%Y%m%d-%H%M')}"

    # Simplified CV data structure - FIX: Referencing the corrected key
    final_cv_data = {
        "name": current_form_name,
        "summary": st.session_state.get('form_summary_value_input', '').strip(), # Corrected key usage
        "skills": [s.strip() for s in st.session_state.get('form_skills_value', '').split('\n') if s.strip()],
        # ... other fields
    }
    
    st.session_state.managed_cvs[cv_key_name] = final_cv_data
    # Set the current parsed resume state to the manually created one for match analysis
    st.session_state.parsed = final_cv_data 
    st.session_state.show_cv_output = cv_key_name 
    st.success(f"🎉 CV for **'{current_form_name}'** saved/updated!")

def generate_and_display_cv(cv_name):
    """Generates the final structured CV data from session state and displays it."""
    if cv_name not in st.session_state.managed_cvs: return
    cv_data = st.session_state.managed_cvs[cv_name]
    st.markdown(f"### 📄 CV View: **{cv_name}**")
    st.code(json.dumps(cv_data, indent=2))

# --- Tab Content Functions ---

def resume_parsing_tab():
    st.header("📄 Upload/Paste Resume for AI Parsing")
    
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
            cv_key_name = f"{candidate_name}_{datetime.now().strftime('%Y%m%d-%H%M')}"
            
            st.session_state.managed_cvs[cv_key_name] = parsed_data
            st.session_state.show_cv_output = cv_key_name
            st.session_state.parsed = parsed_data # CRITICAL: Store parsed data for match analysis
            
            st.success(f"✅ Successfully parsed and structured CV for **{candidate_name}**!")
            st.rerun()

def cv_form_content():
    """Manual CV Form Tab."""
    st.header("📝 CV Management (Manual Form)")
    st.info("Use this form to manually enter or edit structured CV details.")
    st.text_input("Full Name", key="form_name_value")
    
    # FIX APPLIED HERE: Removed repeated 'key' argument. Used the intended key 'form_summary_value_input'.
    st.text_area("Career Summary", key="form_summary_value_input") 
    
    st.text_area("Skills (One per line)", key="form_skills_value", help="Enter core skills, one per line.")
    st.button("💾 **Save CV Details**", type="primary", use_container_width=True, on_click=save_form_cv)
    st.markdown("---")
    if st.session_state.show_cv_output:
        generate_and_display_cv(st.session_state.show_cv_output)


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
        
# --- New Batch Match Tab Function ---

def jd_batch_match_tab():
    """The new Batch JD Match tab logic."""
    st.header("🎯 Batch JD Match: Best Matches")
    st.markdown("Compare your current resume against all saved job descriptions.")

    # Determine if a resume/CV is ready
    is_resume_parsed = st.session_state.get('parsed') is not None
    
    if not is_resume_parsed:
        st.warning("Please **upload and parse your resume** in the 'Resume Parsing' tab or **build your CV** in the 'CV Management' tab first.")
        
    elif not st.session_state.candidate_jd_list:
        st.error("Please **add Job Descriptions** in the 'JD Management' tab before running batch analysis.")
        
    elif not GROQ_API_KEY and isinstance(client, MockGroqClient):
        st.error("Cannot use JD Match: GROQ_API_KEY is not configured or LLM client failed to initialize.")
        
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
        
        if st.button(f"Run Match Analysis on **{len(jds_to_match)}** Selected JD(s)"):
            st.session_state.candidate_match_results = []
            
            if not jds_to_match:
                st.warning("Please select at least one Job Description to run the analysis.")
                
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
            for key in list(st.session_state.keys()):
                if key not in ['page', 'logged_in', 'user_type']:
                    del st.session_state[key]
            go_to("login")
            st.rerun() 
            
    st.markdown("---")

    # --- Session State Initialization ---
    if "managed_cvs" not in st.session_state: st.session_state.managed_cvs = {} 
    if "show_cv_output" not in st.session_state: st.session_state.show_cv_output = None 
    if "parsed" not in st.session_state: st.session_state.parsed = None # Stores the current structured CV/Resume data for matching
    if "candidate_jd_list" not in st.session_state: st.session_state.candidate_jd_list = []
    if "candidate_match_results" not in st.session_state: st.session_state.candidate_match_results = []
    
    # Initialize basic form states (required for save_form_cv to work)
    if "form_name_value" not in st.session_state: st.session_state.form_name_value = ""
    # Only need to check for one key, the other was the issue.
    if "form_summary_value_input" not in st.session_state: st.session_state.form_summary_value_input = "" 
    if "form_skills_value" not in st.session_state: st.session_state.form_skills_value = ""


    # --- Main Content with Tabs ---
    tab_parsing, tab_management, tab_jd, tab_batch_match = st.tabs(["📄 Resume Parsing", "📝 CV Management (Form)", "📚 JD Management", "🎯 Batch JD Match"])
    
    with tab_parsing:
        resume_parsing_tab()
        
    with tab_management:
        cv_form_content() 
        
    with tab_jd:
        jd_management_tab_candidate()
        
    with tab_batch_match:
        jd_batch_match_tab()


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
