import streamlit as st
import os
import json
import re 
from datetime import date 
import traceback
import tempfile
from groq import Groq # Assuming Groq client is initialized elsewhere
import openpyxl # For dump_to_excel
from streamlit.runtime.uploaded_file_manager import UploadedFile # For type checking

# --- CONFIGURATION (MUST BE CONSISTENT WITH MAIN APP) ---
GROQ_MODEL = "llama-3.1-8b-instant"
question_section_options = ["skills","experience", "certifications", "projects", "education"] 
DEFAULT_JOB_TYPES = ["Full-time", "Contract", "Internship", "Remote", "Part-time"]
DEFAULT_ROLES = ["Software Engineer", "Data Scientist", "Product Manager", "HR Manager", "Marketing Specialist", "Operations Analyst"]

# --- GLOBAL API/UTILITY FUNCTIONS (MOCKED/ASSUMED EXIST) ---

# Mock or assume the following global variables/functions are available 
# (You should ensure these are defined and functional in your complete script)

# 1. API Client and Key
# Assume client is initialized globally and GROQ_API_KEY is available
if 'GROQ_API_KEY' not in globals() or not GROQ_API_KEY:
    try:
        # Minimal mock if key is missing, necessary for st.cache_data to work
        class MockGroqClient:
            def chat(self):
                class Completions:
                    def create(self, **kwargs):
                        raise ValueError("GROQ_API_KEY not set. AI functions disabled.")
                return Completions()
        client = MockGroqClient()
    except:
        pass
    
# 2. Page Navigation
def go_to(page_name):
    """Changes the current page in Streamlit's session state."""
    st.session_state.page = page_name

def clear_interview_state():
    """Clears all generated questions, answers, and the evaluation report."""
    st.session_state.interview_qa = []
    st.session_state.iq_output = ""
    st.session_state.evaluation_report = ""
    st.toast("Practice answers cleared.")

# 3. File/LLM Utility Helpers (Crucial dependencies from the original script)
# NOTE: These functions must be defined exactly as they were in the full script.

def dump_to_excel(parsed_json, filename):
    """Dumps parsed JSON data to an Excel file (Simplified implementation for context)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Profile Data"
    ws.append(["Category", "Details"])
    section_order = ['name', 'email', 'phone', 'github', 'linkedin', 'experience', 'education', 'skills', 'projects', 'certifications', 'strength', 'personal_details']
    for section_key in section_order:
        if section_key in parsed_json and parsed_json[section_key]:
            content = parsed_json[section_key]
            ws.append([section_key.replace('_', ' ').title(), str(content) if not isinstance(content, list) and not isinstance(content, dict) else ""])
            if isinstance(content, list):
                for item in content:
                    if item: ws.append(["", str(item)])
    wb.save(filename)
    with open(filename, "rb") as f:
        return f.read()

def get_file_type(file_path):
    """Identifies the file type based on its extension (Placeholder)."""
    ext = os.path.splitext(file_path)[1].lower().strip('.')
    if ext == 'pdf': return 'pdf'
    elif ext == 'docx': return 'docx'
    elif ext == 'xlsx': return 'xlsx'
    else: return 'txt'

def extract_content(file_type, file_path):
    """Extracts text content from various file types (Placeholder)."""
    return f"[Content from {file_path} - type {file_type}]" if os.path.exists(file_path) else f"Error: File not found at {file_path}"
    
@st.cache_data(show_spinner="Analyzing content with Groq LLM...")
def parse_with_llm(text, return_type='json'):
    """Sends resume text to the LLM for structured information extraction (Placeholder/Mock)."""
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set. Cannot run LLM parsing.", "raw_output": ""}

    # Minimal mock for presentation purposes
    if "error" in text: return {"error": text, "raw_output": text}
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": f"Extract JSON from: {text[:100]}..."}],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        # Mock JSON creation if actual parsing is bypassed for this snippet
        mock_data = {"name": "Mock Candidate", "email": "mock@example.com", "skills": ["Python", "Streamlit"]}
        return mock_data
    except Exception as e:
        return {"error": f"LLM parsing failed: {e}", "raw_output": text}


def parse_and_store_resume(file_input, file_name_key='default', source_type='file'):
    """Handles file/text input, parsing, and stores results (Placeholder/Mock)."""
    text = f"[Text from {file_input.name}]" if source_type == 'file' else file_input
    file_name = file_input.name.split('.')[0] if source_type == 'file' else f"Pasted Text ({date.today().strftime('%Y-%m-%d')})"
    
    parsed = parse_with_llm(text, return_type='json')
    
    if "error" in parsed:
        return {"error": parsed.get('error', 'Unknown error'), "full_text": text, "name": file_name}
    
    # In a real app, dump_to_excel needs a proper path
    excel_data = b"Excel Mock Data"
    
    final_name = parsed.get('name', file_name)

    return {
        "parsed": parsed,
        "full_text": text,
        "excel_data": excel_data,
        "name": final_name
    }

@st.cache_data(show_spinner="Extracting JD metadata...")
def extract_jd_metadata(jd_text):
    """Extracts structured metadata (Role, Job Type, Key Skills) from raw JD text (Placeholder/Mock)."""
    if not GROQ_API_KEY:
        return {"role": "N/A", "job_type": "N/A", "key_skills": []}
    
    # Simple mock extraction
    role_match = re.search(r'Role:\s*([^\n]+)', jd_text, re.IGNORECASE)
    role = role_match.group(1).strip() if role_match else "General Analyst"
    
    return {"role": role, "job_type": "Full-time", "key_skills": ["SQL", "Python", "Teamwork"]}

def extract_jd_from_linkedin_url(url: str) -> str:
    """Simulates JD content extraction from a LinkedIn URL (Placeholder/Mock)."""
    job_title = "Data Scientist"
    try:
        match = re.search(r'/jobs/view/([^/]+)', url) or re.search(r'/jobs/(\w+)', url)
        if match:
            job_title = match.group(1).split('?')[0].replace('-', ' ').title()
    except:
        pass
        
    return f"--- Simulated JD for: {job_title} ---\n**Role:** {job_title}\n**Requirements:** 3+ years experience, Python, SQL.\n--- End Simulated JD ---"

def evaluate_jd_fit(job_description, parsed_json):
    """Evaluates how well a resume fits a given job description (Placeholder/Mock)."""
    if not GROQ_API_KEY: return "AI Evaluation Disabled."
    
    # Mock scores
    score = 7 
    skills = 85
    exp = 70
    edu = 90

    return f"""Overall Fit Score: [{score}]/10

--- Section Match Analysis ---
Skills Match: [{skills}]%
Experience Match: [{exp}]%
Education Match: [{edu}]%

Strengths/Matches:
- Strong overlap in Python and SQL skills.
- Relevant experience found in previous roles.

Gaps/Areas for Improvement:
- Missing certification in AWS required by the JD.

Overall Summary: Good fit, minor skill gaps.
"""

def generate_interview_questions(parsed_json, section):
    """Generates categorized interview questions using LLM (Placeholder/Mock)."""
    if not GROQ_API_KEY: return "AI Functions Disabled."
    
    return f"""[Generic]
Q1: Tell me about your {section} background.
Q2: What did you find most challenging in your {section} experience?
Q3: Why did you list X as one of your {section}?
[Basic]
Q1: Describe a time you had to quickly learn a new skill related to {section}.
Q2: How do you handle failure in a project related to your {section}?
Q3: Elaborate on your biggest achievement in {section}.
[Intermediate]
Q1: Give an example of a time your listed experience directly solved a complex business problem.
Q2: How did your education in {section} prepare you for this role?
Q3: Detail the most complex project you worked on related to {section}.
[Difficult]
Q1: Discuss a major technical decision you made using your {section} and the outcome.
Q2: Critique a current industry trend related to your {section}.
Q3: If you were to start over, what would you change about your {section} path?
"""

def evaluate_interview_answers(qa_list, parsed_json):
    """Evaluates the user's answers against the resume content (Placeholder/Mock)."""
    if not GROQ_API_KEY: return "AI Evaluation Disabled."
    
    q1 = qa_list[0]['question']
    
    return f"""## Evaluation Results

### Question 1: {q1}
Score: 8/10
Feedback:
- **Clarity & Accuracy:** The answer was clear and consistent with the resume, specifically mentioning Python.
- **Gaps & Improvements:** Could have linked the experience more explicitly to the target role's requirements.

---

## Final Assessment
Total Score: 8/10 (Based on 1 question)
Overall Summary: Good start, demonstrates solid foundational knowledge, but needs to be more specific and results-driven.
"""
# --- LLM & Extraction Functions (Cont.) ---
def qa_on_resume(question):
    """Chatbot for Resume (Q&A) using LLM (Placeholder/Mock)."""
    if not GROQ_API_KEY: return "AI Chatbot Disabled: GROQ_API_KEY not set."
    
    return f"AI Answer (Resume): Based on the skills section, the candidate appears proficient in Python, SQL, and Streamlit, as mentioned in the parsed data."

def qa_on_jd(question, selected_jd_name):
    """Chatbot for JD (Q&A) using LLM (Placeholder/Mock)."""
    if not GROQ_API_KEY: return "AI Chatbot Disabled: GROQ_API_KEY not set."

    return f"AI Answer (JD): The Job Description states the role is a 'Full-time' 'Data Scientist' position and requires 3+ years experience. This information was extracted from the JD text."


# --- UI HELPERS ---

def generate_cv_html(parsed_data):
    """Generates a simple, print-friendly HTML string from parsed data for PDF conversion (Placeholder/Mock)."""
    css = "<style>...</style>" # Simplified CSS for brevity
    html_content = f"<html><head>{css}<title>{parsed_data.get('name', 'CV')}</title></head><body>"
    html_content += f'<div class="header"><h1>{parsed_data.get("name", "Candidate Name")}</h1></div>'
    html_content += '</body></html>'
    return html_content

def format_parsed_json_to_markdown(parsed_data):
    """Formats the parsed JSON data into a clean, CV-like Markdown structure (Placeholder/Mock)."""
    md = f"# **{parsed_data.get('name', 'CANDIDATE NAME')}**\n\n"
    md += "## **SKILLS**\n---\n"
    for skill in parsed_data.get('skills', []):
        md += f"- {skill}\n"
    md += "\n(More sections would follow...)\n"
    return md


# --- TAB CONTENT FUNCTIONS (DEPENDENCIES) ---

def cv_management_tab_content():
    # ... [Code for CV Management tab content from the original script] ...
    st.header("📝 Prepare Your CV")
    st.markdown("### 1. Form Based CV Builder")
    st.info("Fill out the details below to generate a parsed CV that can be used immediately for matching and interview prep, or start by parsing a file in the 'Resume Parsing' tab.")

    default_parsed = {
        "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
        "skills": [], "experience": [], "education": [], "certifications": [], 
        "projects": [], "strength": [], "personal_details": ""
    }
    
    if "cv_form_data" not in st.session_state:
        if st.session_state.get('parsed', {}).get('name'):
            st.session_state.cv_form_data = st.session_state.parsed.copy()
        else:
            st.session_state.cv_form_data = default_parsed
    
    # --- CV Builder Form (Simplified for brevity) ---
    with st.form("cv_builder_form"):
        st.subheader("Personal & Contact Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.cv_form_data['name'] = st.text_input(
                "Full Name", value=st.session_state.cv_form_data['name'], key="cv_name"
            )
        with col2:
            st.session_state.cv_form_data['email'] = st.text_input(
                "Email Address", value=st.session_state.cv_form_data['email'], key="cv_email"
            )
        
        st.markdown("---")
        st.subheader("Technical Sections (One Item per Line)")

        skills_text = "\n".join(st.session_state.cv_form_data.get('skills', []))
        new_skills_text = st.text_area(
            "Key Skills (Technical and Soft)", value=skills_text, height=100, key="cv_skills"
        )
        st.session_state.cv_form_data['skills'] = [s.strip() for s in new_skills_text.split('\n') if s.strip()]
        
        submit_form_button = st.form_submit_button("Generate and Load CV Data", use_container_width=True)

    if submit_form_button:
        if not st.session_state.cv_form_data['name'] or not st.session_state.cv_form_data['email']:
            st.error("Please fill in at least your **Full Name** and **Email Address**.")
            return

        st.session_state.parsed = st.session_state.cv_form_data.copy()
        st.session_state.full_text = "Compiled text of all CV fields..."
        st.session_state.candidate_match_results = []
        st.session_state.interview_qa = []
        st.session_state.evaluation_report = ""

        st.success(f"✅ CV data for **{st.session_state.parsed['name']}** successfully generated and loaded!")
        
    st.markdown("---")
    st.subheader("2. Loaded CV Data Preview and Download")
    
    if st.session_state.get('parsed', {}).get('name'):
        
        filled_data_for_preview = st.session_state.parsed 
        
        tab_markdown, tab_json, tab_pdf = st.tabs(["📝 Markdown View", "💾 JSON View", "⬇️ PDF/HTML Download"])

        with tab_markdown:
            cv_markdown_preview = format_parsed_json_to_markdown(filled_data_for_preview)
            st.markdown(cv_markdown_preview)

        with tab_json:
            st.json(st.session_state.parsed)

        with tab_pdf:
            st.info("Download HTML for Print-to-PDF conversion.")
            html_output = generate_cv_html(filled_data_for_preview)
            st.download_button(
                label="⬇️ Download CV as Print-Ready HTML File",
                data=html_output,
                file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_CV_Document.html",
                mime="text/html",
                key="download_cv_html"
            )
    else:
        st.info("Please fill out the form above or parse a resume to see the preview.")


def filter_jd_tab_content():
    # ... [Code for Filter JD tab content from the original script] ...
    st.header("🔍 Filter Job Descriptions by Criteria")
    st.markdown("Use the filters below to narrow down your saved Job Descriptions.")

    if not st.session_state.candidate_jd_list:
        st.info("No Job Descriptions are currently loaded. Please add JDs in the 'JD Management' tab.")
        if 'filtered_jds_display' not in st.session_state:
            st.session_state.filtered_jds_display = []
        return
    
    # Mock/Extract available options
    unique_roles = sorted(list(set(
        [item.get('role', 'General Analyst') for item in st.session_state.candidate_jd_list] + DEFAULT_ROLES
    )))
    unique_job_types = sorted(list(set(
        [item.get('job_type', 'Full-time') for item in st.session_state.candidate_jd_list] + DEFAULT_JOB_TYPES
    )))
    unique_skills_list = ["Python", "SQL", "AWS"]

    all_jd_data = st.session_state.candidate_jd_list

    with st.form(key="jd_filter_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_skills = st.multiselect("Skills Keywords", options=unique_skills_list, default=st.session_state.get('last_selected_skills', []), key="candidate_filter_skills_multiselect")
        with col2:
            selected_job_type = st.selectbox("Job Type", options=["All Job Types"] + unique_job_types, index=0, key="filter_job_type_select")
        with col3:
            selected_role = st.selectbox("Role Title", options=["All Roles"] + unique_roles, index=0, key="filter_role_select")

        apply_filters_button = st.form_submit_button("✅ Apply Filters", type="primary", use_container_width=True)

    if apply_filters_button:
        # Simplified Filtering Logic (Always filters to JDs with 'Python' if selected_skills is not empty)
        st.session_state.last_selected_skills = selected_skills

        filtered_jds = [
            jd for jd in all_jd_data
            if (selected_role == "All Roles" or selected_role == jd.get('role', '')) and
               (selected_job_type == "All Job Types" or selected_job_type == jd.get('job_type', '')) and
               (not selected_skills or any(s.lower() in [ks.lower() for ks in jd.get('key_skills',[])] for s in selected_skills))
        ]
                
        st.session_state.filtered_jds_display = filtered_jds
        st.success(f"Filter applied! Found {len(filtered_jds)} matching Job Descriptions.")

    st.markdown("---")
    
    if 'filtered_jds_display' not in st.session_state:
        st.session_state.filtered_jds_display = []
        
    filtered_jds = st.session_state.filtered_jds_display
    
    st.subheader(f"Matching Job Descriptions ({len(filtered_jds)} found)")
    
    if filtered_jds:
        display_data = []
        for jd in filtered_jds:
            display_data.append({
                "Job Description Title": jd['name'].replace("--- Simulated JD for: ", ""),
                "Role": jd.get('role', 'N/A'),
                "Job Type": jd.get('job_type', 'N/A'),
                "Key Skills": ", ".join(jd.get('key_skills', ['N/A'])[:5]) + "...",
            })
            
        st.dataframe(display_data, use_container_width=True)

    elif st.session_state.candidate_jd_list and apply_filters_button:
        st.info("No Job Descriptions match the selected criteria.")
    elif st.session_state.candidate_jd_list and not apply_filters_button:
        st.info("Use the filters above and click **'Apply Filters'** to view matching Job Descriptions.")

# --- CANDIDATE DASHBOARD FUNCTION (THE REQUESTED CODE BLOCK) ---

def candidate_dashboard():
    st.header("👩‍🎓 Candidate Dashboard")
    st.markdown("Welcome! Use the tabs below to manage your CV and access AI preparation tools.")

    # --- Navigation ---
    nav_col, _ = st.columns([1, 1]) 

    with nav_col:
        if st.button("🚪 Log Out", key="candidate_logout_btn", use_container_width=True):
            go_to("login") 
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("Resume/CV Status")
        if st.session_state.parsed.get("name"):
            st.success(f"Currently loaded: **{st.session_state.parsed['name']}**")
        else:
            st.info("Please upload a file or use the CV builder in 'CV Management' to begin.")

    # Main Content Tabs (REARRANGED TABS HERE)
    # 1. CV Management
    # 2. Resume Parsing
    # 3. JD Management
    # 4. Batch JD Match
    # 5. Filter JD
    # 6. Resume/JD Chatbot (Q&A) <-- MOVED TO END
    # 7. Interview Prep <-- MOVED TO END
    tab_cv_mgmt, tab_parsing, tab_jd_mgmt, tab_batch_match, tab_filter_jd, tab_chatbot, tab_interview_prep = st.tabs([
        "✍️ CV Management", 
        "📄 Resume Parsing", 
        "📚 JD Management", 
        "🎯 Batch JD Match",
        "🔍 Filter JD",
        "💬 Resume/JD Chatbot (Q&A)", 
        "❓ Interview Prep"            
    ])
    
    is_resume_parsed = bool(st.session_state.get('parsed', {}).get('name')) or bool(st.session_state.get('full_text'))
    
    # --- TAB 1: CV Management ---
    with tab_cv_mgmt:
        cv_management_tab_content()

    # --- TAB 2: Resume Parsing ---
    with tab_parsing:
        st.header("Resume Upload and Parsing")
        
        input_method = st.radio(
            "Select Input Method",
            ["Upload File", "Paste Text"],
            key="parsing_input_method"
        )
        
        st.markdown("---")

        if input_method == "Upload File":
            st.markdown("### 1. Upload Resume File") 
            uploaded_file = st.file_uploader( 
                "Choose PDF, DOCX, TXT, JSON, MD, CSV, XLSX file", 
                type=["pdf", "docx", "txt", "json", "md", "csv", "xlsx", "markdown", "rtf"], 
                accept_multiple_files=False, 
                key='candidate_file_upload_main'
            )
            
            if uploaded_file is not None:
                st.session_state.candidate_uploaded_resumes = [uploaded_file] 
                st.session_state.pasted_cv_text = "" 
            
            file_to_parse = st.session_state.candidate_uploaded_resumes[0] if st.session_state.candidate_uploaded_resumes else None
            
            st.markdown("### 2. Parse Uploaded File")
            
            if file_to_parse:
                if st.button(f"Parse and Load: **{file_to_parse.name}**", use_container_width=True):
                    with st.spinner(f"Parsing {file_to_parse.name}..."):
                        result = parse_and_store_resume(file_to_parse, file_name_key='single_resume_candidate', source_type='file')
                        
                        if "error" not in result:
                            st.session_state.parsed = result['parsed']
                            st.session_state.full_text = result['full_text']
                            st.session_state.excel_data = result['excel_data'] 
                            st.session_state.parsed['name'] = result['name'] 
                            clear_interview_state()
                            st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                        else:
                            st.error(f"Parsing failed for {file_to_parse.name}: {result['error']}")
                            st.session_state.parsed = {"error": result['error'], "name": result['name']}
            else:
                st.info("No resume file is currently uploaded.")

        else: # Paste Text
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
                if st.button("Parse and Load Pasted Text", use_container_width=True):
                    with st.spinner("Parsing pasted text..."):
                        st.session_state.candidate_uploaded_resumes = []
                        
                        result = parse_and_store_resume(pasted_text, file_name_key='single_resume_candidate', source_type='text')
                        
                        if "error" not in result:
                            st.session_state.parsed = result['parsed']
                            st.session_state.full_text = result['full_text']
                            st.session_state.excel_data = result['excel_data'] 
                            st.session_state.parsed['name'] = result['name'] 
                            clear_interview_state()
                            st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                        else:
                            st.error(f"Parsing failed: {result['error']}")
                            st.session_state.parsed = {"error": result['error'], "name": result['name']}
            else:
                st.info("Please paste your CV text into the box above.")

    # --- TAB 3: JD Management ---
    with tab_jd_mgmt:
        st.header("📚 Manage Job Descriptions for Matching")
        
        if "candidate_jd_list" not in st.session_state:
             st.session_state.candidate_jd_list = []
        
        jd_type = st.radio("Select JD Type", ["Single JD", "Multiple JD"], key="jd_type_candidate")
        method = st.radio("Choose Method", ["Upload File", "Paste Text", "LinkedIn URL"], key="jd_add_method_candidate") 

        # Simplified JD Add logic (Only button for URL shown)
        if method == "LinkedIn URL":
            url_list = st.text_area("Enter URL(s)", key="url_list_candidate")
            if st.button("Add JD(s) from URL", key="add_jd_url_btn_candidate"):
                if url_list:
                    urls = [u.strip() for u in url_list.split(",")] if jd_type == "Multiple JD" else [url_list.strip()]
                    for url in urls:
                        if not url: continue
                        jd_text = extract_jd_from_linkedin_url(url)
                        metadata = extract_jd_metadata(jd_text)
                        name_base = url.split('/jobs/view/')[-1].split('/')[0] if '/jobs/view/' in url else "URL"
                        name = f"JD from URL: {name_base}" 
                        st.session_state.candidate_jd_list.append({"name": name, "content": jd_text, **metadata}) 
                    st.success(f"✅ {len(urls)} JD(s) added successfully!")
                else:
                    st.warning("Please enter at least one URL.")

        if st.session_state.candidate_jd_list:
            col_display_header, col_clear_button = st.columns([3, 1])
            with col_display_header:
                st.markdown("### ✅ Current JDs Added:")
            with col_clear_button:
                if st.button("🗑️ Clear All JDs", key="clear_jds_candidate"):
                    st.session_state.candidate_jd_list = []
                    st.session_state.candidate_match_results = [] 
                    st.session_state.filtered_jds_display = [] 
                    st.success("All JDs cleared.")
                    st.rerun() 

            for idx, jd_item in enumerate(st.session_state.candidate_jd_list, 1):
                title = jd_item['name'].replace("--- Simulated JD for: ", "")
                with st.expander(f"JD {idx}: {title} | Role: {jd_item.get('role', 'N/A')}"):
                    st.text(jd_item['content'][:200] + "...")
        else:
            st.info("No Job Descriptions added yet.")

    # --- TAB 4: Batch JD Match ---
    with tab_batch_match:
        st.header("🎯 Batch JD Match: Best Matches")
        
        if not is_resume_parsed:
            st.warning("Please **upload and parse your resume** first.")
        elif not st.session_state.candidate_jd_list:
            st.error("Please **add Job Descriptions** first.")
        elif not GROQ_API_KEY:
             st.error("Cannot use JD Match: GROQ_API_KEY is not configured.")
        else:
            if "candidate_match_results" not in st.session_state:
                st.session_state.candidate_match_results = []

            all_jd_names = [item['name'] for item in st.session_state.candidate_jd_list]
            selected_jd_names = st.multiselect(
                "Select Job Descriptions to Match Against",
                options=all_jd_names,
                default=all_jd_names, 
                key='candidate_batch_jd_select'
            )
            
            jds_to_match = [jd_item for jd_item in st.session_state.candidate_jd_list if jd_item['name'] in selected_jd_names]
            
            if st.button(f"Run Match Analysis on {len(jds_to_match)} Selected JD(s)"):
                st.session_state.candidate_match_results = []
                if not jds_to_match: st.warning("No JDs selected.")
                else:
                    with st.spinner("Matching resumes..."):
                        results_with_score = []
                        for jd_item in jds_to_match:
                            fit_output = evaluate_jd_fit(jd_item['content'], st.session_state.parsed)
                            
                            score_match = re.search(r'Overall Fit Score:\s*[^\d]*(\d+)\s*/10', fit_output, re.IGNORECASE)
                            score = score_match.group(1) if score_match else 'N/A'
                            
                            results_with_score.append({
                                "jd_name": jd_item['name'],
                                "overall_score": score,
                                "numeric_score": int(score) if score.isdigit() else -1,
                                "skills_percent": "85", "experience_percent": "70", "education_percent": "90", # Mock %
                                "full_analysis": fit_output
                            })
                            
                        results_with_score.sort(key=lambda x: x['numeric_score'], reverse=True)
                        for i, item in enumerate(results_with_score): item['rank'] = i + 1 
                        st.session_state.candidate_match_results = results_with_score
                        st.success("Batch analysis complete!")

            if st.session_state.get('candidate_match_results'):
                st.markdown("#### Match Results for Your Resume")
                display_data = []
                for item in st.session_state.candidate_match_results:
                    full_jd_item = next((jd for jd in st.session_state.candidate_jd_list if jd['name'] == item['jd_name']), {})
                    display_data.append({
                        "Rank": item.get("rank", "N/A"),
                        "Job Description (Ranked)": item["jd_name"].replace("--- Simulated JD for: ", ""),
                        "Role": full_jd_item.get('role', 'N/A'),
                        "Fit Score (out of 10)": item["overall_score"],
                    })
                st.dataframe(display_data, use_container_width=True)
                
                for item in st.session_state.candidate_match_results:
                     with st.expander(f"Rank {item.get('rank', 'N/A')} | Report for **{item['jd_name'].replace('--- Simulated JD for: ', '')}** (Score: **{item['overall_score']}/10**)"):
                        st.markdown(item['full_analysis'])

    # --- TAB 5: Filter JD ---
    with tab_filter_jd:
        filter_jd_tab_content()

    # --- TAB 6: Resume/JD Chatbot (Q&A) (MOVED TO END) ---
    with tab_chatbot:
        st.header("Resume/JD Chatbot (Q&A) 💬")
        
        sub_tab_resume, sub_tab_jd = st.tabs([
            "👤 Chat about Your Resume",
            "📄 Chat about Saved JDs"
        ])
        
        # --- RESUME CHATBOT ---
        with sub_tab_resume:
            st.markdown("### Ask any question about the currently loaded resume.")
            if not is_resume_parsed or "error" in st.session_state.parsed or not GROQ_API_KEY:
                st.warning("Prerequisite: Parse resume and ensure API key is configured.")
            else:
                if 'qa_answer_resume' not in st.session_state: st.session_state.qa_answer_resume = ""
                
                question = st.text_input("Your Question (about Resume)", key="resume_qa_question")
                
                if st.button("Get Answer (Resume)", key="qa_btn_resume"):
                    with st.spinner("Generating answer..."):
                        st.session_state.qa_answer_resume = qa_on_resume(question)

                if st.session_state.get('qa_answer_resume'):
                    st.text_area("Answer (Resume)", st.session_state.qa_answer_resume, height=150)
        
        # --- JD CHATBOT ---
        with sub_tab_jd:
            st.markdown("### Ask any question about a saved Job Description.")
            
            if not st.session_state.candidate_jd_list or not GROQ_API_KEY:
                 st.warning("Prerequisite: Add JDs in 'JD Management' and ensure API key is configured.")
            else:
                if 'qa_answer_jd' not in st.session_state: st.session_state.qa_answer_jd = ""

                jd_names = [jd['name'] for jd in st.session_state.candidate_jd_list]
                selected_jd_name = st.selectbox("Select JD to Query", options=jd_names, key="jd_qa_select")
                
                question = st.text_input("Your Question (about JD)", key="jd_qa_question")
                
                if st.button("Get Answer (JD)", key="qa_btn_jd"):
                    if selected_jd_name and question.strip():
                        with st.spinner(f"Generating answer for {selected_jd_name}..."):
                            st.session_state.qa_answer_jd = qa_on_jd(question, selected_jd_name)
                    else:
                        st.error("Please select a JD and enter a question.")

                if st.session_state.get('qa_answer_jd'):
                    st.text_area("Answer (JD)", st.session_state.qa_answer_jd, height=150)

    # --- TAB 7: Interview Prep (MOVED TO END) ---
    with tab_interview_prep:
        st.header("Interview Preparation Tools")
        if not is_resume_parsed or "error" in st.session_state.parsed or not GROQ_API_KEY:
            st.warning("Prerequisite: Upload/Parse resume and ensure API key is configured.")
        else:
            if 'iq_output' not in st.session_state: st.session_state.iq_output = ""
            if 'interview_qa' not in st.session_state: st.session_state.interview_qa = [] 
            if 'evaluation_report' not in st.session_state: st.session_state.evaluation_report = "" 
            
            st.subheader("1. Generate Interview Questions")
            
            section_choice = st.selectbox(
                "Select Section", 
                question_section_options, 
                key='iq_section_c',
                on_change=clear_interview_state 
            )
            
            if st.button("Generate Interview Questions", key='iq_btn_c'):
                with st.spinner("Generating questions..."):
                    raw_questions_response = generate_interview_questions(st.session_state.parsed, section_choice)
                    
                    st.session_state.iq_output = raw_questions_response
                    st.session_state.evaluation_report = "" 
                    
                    q_list = []
                    current_level = "Unknown"
                    for line in raw_questions_response.splitlines():
                        if line.startswith('[') and line.endswith(']'): current_level = line.strip('[]')
                        elif line.lower().startswith('q') and ':' in line:
                            question_text = line[line.find(':') + 1:].strip()
                            q_list.append({"question": f"({current_level}) {question_text}", "answer": "", "level": current_level})
                    
                    st.session_state.interview_qa = q_list
                    st.success(f"Generated {len(q_list)} questions.")

            if st.session_state.get('interview_qa'):
                st.markdown("---")
                st.subheader("2. Practice and Record Answers")
                
                with st.form("interview_practice_form"):
                    for i, qa_item in enumerate(st.session_state.interview_qa):
                        st.markdown(f"**Question {i+1}:** {qa_item['question']}")
                        answer = st.text_area(f"Your Answer for Q{i+1}", value=st.session_state.interview_qa[i]['answer'], height=100, key=f'answer_q_{i}', label_visibility='collapsed')
                        st.session_state.interview_qa[i]['answer'] = answer 
                        st.markdown("---") 
                        
                    submit_button = st.form_submit_button("Submit & Evaluate Answers", use_container_width=True)

                    if submit_button:
                        if all(item['answer'].strip() for item in st.session_state.interview_qa):
                            with st.spinner("Sending answers to AI Evaluator..."):
                                report = evaluate_interview_answers(st.session_state.interview_qa, st.session_state.parsed)
                                st.session_state.evaluation_report = report
                                st.success("Evaluation complete! See the report below.")
                        else:
                            st.error("Please answer all generated questions before submitting.")
                
                if st.session_state.get('evaluation_report'):
                    st.markdown("---")
                    st.subheader("3. AI Evaluation Report")
                    st.markdown(st.session_state.evaluation_report)
