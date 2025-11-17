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

# --- CONFIGURATION & API SETUP ---

GROQ_MODEL = "llama-3.1-8b-instant"
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
        raise ValueError("GROQ_API_KEY not set.")
    # Attempt to initialize the real client
    client = Groq(api_key=GROQ_API_KEY)
except (ImportError, ValueError, Exception) as e:
    # Fallback to Mock Client
    st.warning(f"Using Mock LLM Client. Groq setup failed: {e.__class__.__name__}. Set GROQ_API_KEY and install 'groq' for full functionality.")
    client = MockGroqClient()


# --- Utility Functions ---

def go_to(page_name):
    """Changes the current page in Streamlit's session state."""
    st.session_state.page = page_name

def get_file_type(file_name):
    """Identifies the file type based on its extension."""
    ext = os.path.splitext(file_name)[1].lower().strip('.')
    if ext == 'pdf': return 'pdf'
    elif ext in ('docx', 'doc'): return 'docx'
    elif ext == 'txt': return 'txt'
    elif ext in ('md', 'markdown'): return 'markdown'
    elif ext == 'json': return 'json'
    elif ext in ('xlsx', 'xls'): return 'xlsx'
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
        
        elif file_type in ['txt', 'markdown']:
            try:
                text = file_content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                 text = file_content_bytes.decode('latin-1')
        
        elif file_type == 'json':
            try:
                data = json.loads(file_content_bytes.decode('utf-8'))
                text = "--- JSON Content Start ---\n" + json.dumps(data, indent=2) + "\n--- JSON Content End ---"
            except json.JSONDecodeError:
                return f"[Error] JSON content extraction failed: Invalid JSON format."
            except UnicodeDecodeError:
                return f"[Error] JSON content extraction failed: Unicode Decode Error."
        
        elif file_type == 'xlsx':
            return f"[Error] XLSX/Excel file parsing is complex and requires specific libraries (pandas/openpyxl). Please copy and paste the text content from the file instead."

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
        return {"name": "Mock Candidate", "email": "mock@example.com", "phone": "555-1234", "linkedin": "linkedin.com/in/mock", "github": "github.com/mock", "personal_details": "Highly motivated individual with mock experience in Python and Streamlit.", "skills": ["Python", "Streamlit", "SQL", "AWS"], "education": ["B.S. Computer Science, Mock University"], "experience": ["Software Intern, Mock Solutions (2024-2025)"], "certifications": ["Mock Certification"], "projects": ["Mock Project"], "strength": ["Mock Strength"], "error": "Mock/Parsing error." if isinstance(client, MockGroqClient) else text}

    # Placeholder for actual Groq call
    return {"name": "Parsed Candidate", "email": "parsed@example.com", "phone": "555-9876", "linkedin": "linkedin.com/in/parsed", "github": "github.com/parsed", "personal_details": "Actual parsed summary from LLM.", "skills": ["Real", "Python", "Streamlit"], "education": ["University of Code"], "experience": ["Senior Developer, TechCo"], "certifications": ["AWS Certified"], "projects": ["Project Alpha"], "strength": ["Teamwork"], "error": None} 


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
    Education Match: [{edu}%]
    
    --- Strengths/Matches ---
    The candidate shows strong proficiency in Python and Streamlit, which aligns well with the required data processing tools. Education is directly relevant.
    
    --- Weaknesses/Gaps ---
    Candidate lacks specific mention of Kubernetes or advanced Docker orchestration, which is a required skill for this role.
    
    --- Summary Recommendation ---
    A strong candidate with core skills. Recommend for interview if the experience gap in orchestration tools can be overlooked or quickly trained.
    """

def generate_cv_html(parsed_data):
    """Generates basic HTML from the parsed CV data for print-to-PDF."""
    
    # Simple CSS for a professional look and print readiness
    css = """
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; color: #333; line-height: 1.6; }
        h1 { font-size: 24px; color: #1e3a8a; border-bottom: 3px solid #1e3a8a; padding-bottom: 5px; margin-bottom: 5px; }
        h2 { font-size: 18px; color: #1e3a8a; border-bottom: 1px solid #ddd; padding-bottom: 3px; margin-top: 20px; }
        .contact-info { font-size: 12px; margin-bottom: 15px; }
        .contact-info span { margin-right: 15px; }
        .section-content { margin-left: 20px; margin-bottom: 15px; }
        ul { list-style-type: disc; margin-left: 20px; padding-left: 0; }
        li { margin-bottom: 5px; }
        a { color: #1e3a8a; text-decoration: none; }
        @media print {
            body { margin: 0; }
            h1 { font-size: 20pt; }
            h2 { font-size: 14pt; }
        }
    </style>
    """
    
    # --- HTML Structure ---
    html = f"<!DOCTYPE html><html><head><title>{parsed_data.get('name', 'CV')}</title>{css}</head><body>"
    
    # 1. Header and Contact
    html += f"<h1>{parsed_data.get('name', 'Candidate CV')}</h1>"
    contact_parts = []
    if parsed_data.get('email'): contact_parts.append(f"<span>📧 {parsed_data['email']}</span>")
    if parsed_data.get('phone'): contact_parts.append(f"<span>📞 {parsed_data['phone']}</span>")
    if parsed_data.get('linkedin'): contact_parts.append(f"<span>🔗 <a href='{parsed_data['linkedin']}'>LinkedIn</a></span>")
    if parsed_data.get('github'): contact_parts.append(f"<span>💻 <a href='{parsed_data['github']}'>GitHub</a></span>")
    
    if contact_parts:
        html += f"<div class='contact-info'>{' | '.join(contact_parts)}</div>"

    # 2. Sections
    section_order = [
        ('personal_details', 'Professional Summary'), 
        ('experience', 'Professional Experience'), 
        ('projects', 'Key Projects'), 
        ('education', 'Education'), 
        ('certifications', 'Certifications'), 
        ('skills', 'Skills'), 
        ('strength', 'Strengths')
    ]

    for key, title in section_order:
        content = parsed_data.get(key)
        
        if content and (isinstance(content, str) and content.strip() or isinstance(content, list) and content):
            html += f"<h2>{title.upper()}</h2><div class='section-content'>"
            
            if isinstance(content, str):
                html += f"<p>{content.replace('\n', '<br>')}</p>"
            elif isinstance(content, list):
                html += "<ul>"
                for item in content:
                    if item:
                        html += f"<li>{item}</li>"
                html += "</ul>"
                
            html += "</div>"
            
    html += "</body></html>"
    return html

# --- Tab Content Functions ---

def resume_parsing_tab():
    st.header("📄 Upload/Paste Resume for AI Parsing")
    
    # File types allowed
    file_types_allowed = ['pdf', 'docx', 'txt', 'md', 'json', 'xlsx']
    
    with st.form("resume_parsing_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            f"Upload Resume File ({', '.join(file_types_allowed)})", 
            type=file_types_allowed, 
            accept_multiple_files=False,
            key="resume_uploader"
        )
        st.markdown("---")
        st.info("Limit 200MB per file. Allowed types: PDF, DOCX, TXT, MD, JSON, XLSX.")
        pasted_text = st.text_area("Or Paste Resume Text Here", height=200, key="resume_paster")
        st.markdown("---")

        if st.form_submit_button("✨ Parse and Structure CV", type="primary", use_container_width=True):
            extracted_text = ""
            file_name = "Pasted_Resume"
            
            if uploaded_file is not None:
                file_name = uploaded_file.name
                file_type = get_file_type(file_name)
                uploaded_file.seek(0) 
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
            
            if "error" in parsed_data and parsed_data.get('error') != "Mock/Parsing error.":
                st.error(f"AI Parsing Failed: {parsed_data['error']}")
                return

            candidate_name = parsed_data.get('name', 'Unknown_Candidate').replace(' ', '_')
            
            # CRITICAL: Store parsed data for match analysis and form initialization
            st.session_state.parsed = parsed_data 
            
            # Also update the form data state so the form reflects the newly parsed data
            st.session_state.cv_form_data = parsed_data.copy()
            
            # Create a compiled text representation for Q&A/Text Download (from new parsed data)
            compiled_text = ""
            for k, v in parsed_data.items():
                if v and k not in ['error']:
                    compiled_text += f"{k.replace('_', ' ').title()}:\n"
                    if isinstance(v, list):
                        compiled_text += "\n".join([f"- {item}" for item in v]) + "\n\n"
                    else:
                        compiled_text += str(v) + "\n\n"
            st.session_state.full_text = compiled_text
            
            st.success(f"✅ Successfully parsed and loaded CV for **{candidate_name}**! Check the 'CV Management' tab to review/edit.")
            st.rerun()

# --- FIXED CV MANAGEMENT FUNCTION ---
def cv_management_tab_content():
    st.header("📝 Prepare Your CV")
    st.markdown("### 1. Form Based CV Builder")
    st.info("Fill out the details below to generate a parsed CV that can be used immediately for matching and interview prep, or start by parsing a file in the 'Resume Parsing' tab.")

    # Define the template for a complete CV data structure
    default_parsed = {
        "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
        "skills": [], "experience": [], "education": [], "certifications": [], 
        "projects": [], "strength": [], "personal_details": ""
    }
    
    # Use a specific session state key for form data, initializing from parsed if available
    if "cv_form_data" not in st.session_state:
        # If parsed data exists and has a name, merge it into the default structure
        if st.session_state.get('parsed', {}).get('name'):
            # Merge parsed data into a copy of default to ensure all keys exist
            st.session_state.cv_form_data = {**default_parsed, **st.session_state.parsed}
        else:
            # Otherwise, use the clean default structure
            st.session_state.cv_form_data = default_parsed.copy()
            
    # CRITICAL FIX: Ensure form data keys are accessed safely, which the initialization above now guarantees, 
    # but defensive coding for list-to-string conversion is still necessary.
    
    # --- CV Builder Form ---
    with st.form("cv_builder_form"):
        st.subheader("Personal & Contact Details")
        
        # Row 1: Name, Email, Phone
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.cv_form_data['name'] = st.text_input(
                "Full Name", 
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
            "Professional Summary or Personal Details (e.g., date of birth, address, nationality)", 
            value=st.session_state.cv_form_data.get('personal_details', ''), 
            height=100,
            key="cv_personal_details"
        )
        
        st.markdown("---")
        st.subheader("Technical Sections (One Item per Line)")

        # Skills
        # Ensure conversion from list to string for display, handling None/empty gracefully
        skills_text = "\n".join(st.session_state.cv_form_data.get('skills', []))
        new_skills_text = st.text_area(
            "Key Skills (Technical and Soft)", 
            value=skills_text,
            height=150,
            key="cv_skills"
        )
        st.session_state.cv_form_data['skills'] = [s.strip() for s in new_skills_text.split('\n') if s.strip()]
        
        # Experience
        experience_text = "\n".join(st.session_state.cv_form_data.get('experience', []))
        new_experience_text = st.text_area(
            "Professional Experience (Job Roles, Companies, Dates, Key Responsibilities)", 
            value=experience_text,
            height=150,
            key="cv_experience"
        )
        st.session_state.cv_form_data['experience'] = [e.strip() for e in new_experience_text.split('\n') if e.strip()]

        # Education
        education_text = "\n".join(st.session_state.cv_form_data.get('education', []))
        new_education_text = st.text_area(
            "Education (Degrees, Institutions, Dates)", 
            value=education_text,
            height=100,
            key="cv_education"
        )
        st.session_state.cv_form_data['education'] = [d.strip() for d in new_education_text.split('\n') if d.strip()]
        
        # Certifications
        certifications_text = "\n".join(st.session_state.cv_form_data.get('certifications', []))
        new_certifications_text = st.text_area(
            "Certifications (Name, Issuing Body, Date)", 
            value=certifications_text,
            height=100,
            key="cv_certifications"
        )
        st.session_state.cv_form_data['certifications'] = [c.strip() for c in new_certifications_text.split('\n') if c.strip()]
        
        # Projects
        projects_text = "\n".join(st.session_state.cv_form_data.get('projects', []))
        new_projects_text = st.text_area(
            "Projects (Name, Description, Technologies)", 
            value=projects_text,
            height=150,
            key="cv_projects"
        )
        st.session_state.cv_form_data['projects'] = [p.strip() for p in new_projects_text.split('\n') if p.strip()]
        
        # Strengths
        strength_text = "\n".join(st.session_state.cv_form_data.get('strength', []))
        new_strength_text = st.text_area(
            "Strengths / Key Personal Qualities (One per line)", 
            value=strength_text,
            height=100,
            key="cv_strength"
        )
        st.session_state.cv_form_data['strength'] = [s.strip() for s in new_strength_text.split('\n') if s.strip()]


        submit_form_button = st.form_submit_button("Generate and Load CV Data", type="primary", use_container_width=True)

    if submit_form_button:
        # 1. Basic validation
        if not st.session_state.cv_form_data.get('name') or not st.session_state.cv_form_data.get('email'):
            st.error("Please fill in at least your **Full Name** and **Email Address**.")
            return

        # 2. Update the main session state variables (as if a file was parsed)
        st.session_state.parsed = st.session_state.cv_form_data.copy()
        
        # 3. Create a compiled text representation for Q&A/Text Download
        compiled_text = ""
        for k, v in st.session_state.cv_form_data.items():
            if v:
                compiled_text += f"{k.replace('_', ' ').title()}:\n"
                if isinstance(v, list):
                    compiled_text += "\n".join([f"- {item}" for item in v]) + "\n\n"
                else:
                    compiled_text += str(v) + "\n\n"
        st.session_state.full_text = compiled_text
        
        # 4. Clear related states (since this is a new resume)
        if 'candidate_match_results' in st.session_state: st.session_state.candidate_match_results = []
        if 'interview_qa' in st.session_state: del st.session_state.interview_qa
        if 'evaluation_report' in st.session_state: del st.session_state.evaluation_report

        st.success(f"✅ CV data for **{st.session_state.parsed['name']}** successfully generated and loaded! You can now use the Match tabs.")
        st.rerun() 
        
    st.markdown("---")
    st.subheader("2. Loaded CV Data Preview and Download")
    
    # --- TABBED VIEW SECTION (PDF/MARKDOWN/JSON) ---
    if st.session_state.get('parsed', {}).get('name'):
        
        # Filter for non-empty/non-list fields before sending to formatter
        filled_data_for_preview = {
            k: v for k, v in st.session_state.parsed.items() 
            if v and (isinstance(v, str) and v.strip() or isinstance(v, list) and v)
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
            if parsed_data.get('linkedin'): contact_info.append(f"[LinkedIn]({parsed_data['linkedin']})")
            if parsed_data.get('github'): contact_info.append(f"[GitHub]({parsed_data['github']})")
            
            if contact_info:
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
            st.info("Click the button below to download an HTML file. Open the file in your browser and use the browser's **'Print'** function, selecting **'Save as PDF'** to create your final CV document. ")
            
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
        
# --- Batch Match Tab Function ---

def jd_batch_match_tab():
    """The Batch JD Match tab logic."""
    st.header("🎯 Batch JD Match: Best Matches")
    st.markdown("Compare your current resume against all saved job descriptions.")

    # Determine if a resume/CV is ready
    is_resume_parsed = st.session_state.get('parsed') is not None
    
    if not is_resume_parsed or not st.session_state.parsed.get('name'):
        st.warning("Please **upload and parse your resume** in the 'Resume Parsing' tab or **build your CV** in the 'CV Management' tab first.")
        
    elif not st.session_state.candidate_jd_list:
        st.error("Please **add Job Descriptions** in the 'JD Management' tab before running batch analysis.")
        
    elif isinstance(client, MockGroqClient):
        st.error("Cannot run LLM match analysis: The LLM client is a **Mock Client**. Please configure `GROQ_API_KEY` and install 'groq' for full functionality.")
        
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

    # --- Session State Initialization (CRITICAL FIX: Initialize parsed as empty dict) ---
    if "parsed" not in st.session_state: st.session_state.parsed = {} # Initialized to {} instead of None
    if "full_text" not in st.session_state: st.session_state.full_text = ""
    if "cv_form_data" not in st.session_state: st.session_state.cv_form_data = {} 
    
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
