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
from datetime import datetime
import random 
import pandas as pd

# -------------------------
# CONFIGURATION & API SETUP 
# -------------------------

GROQ_MODEL = "llama-3.1-8b-instant"
# Load environment variables (ensure .env file is present with GROQ_API_KEY)
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq Client or Mock Client 
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

# --- Utility Functions ---

def go_to(page_name):
    """Changes the current page in Streamlit's session state."""
    st.session_state.page = page_name

def get_file_type(file_name):
    """Identifies the file type based on its extension."""
    ext = os.path.splitext(file_name)[1].lower().strip('.')
    # Handling specific non-traditional resume file types
    if ext == 'pdf': return 'pdf'
    elif ext == 'docx': return 'docx'
    elif ext == 'json': return 'json'
    elif ext == 'csv': return 'csv'
    elif ext == 'md': return 'markdown'
    elif ext == 'txt': return 'txt'
    else: return 'unknown' 

def extract_content(file_type, file_content, file_name):
    """Extracts text content from uploaded file content (bytes) or pasted text."""
    text = ''
    try:
        if file_type == 'pdf':
            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
                with pdfplumber.open(tmp_file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
        
        elif file_type == 'docx':
            with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
                doc = docx.Document(tmp_file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
        
        elif file_type in ['json', 'csv', 'markdown', 'txt', 'unknown']:
            # For text-based files, decode the bytes
            try:
                text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                 try:
                    text = file_content.decode('latin-1')
                 except Exception:
                     return f"Extraction Error: Could not decode text file {file_name}."
        
        if not text.strip():
            return f"Error: {file_type.upper()} content extraction failed or file is empty."
        
        return text
    
    except Exception as e:
        return f"Fatal Extraction Error: Failed to read file content ({file_type}). Error: {e}\n{traceback.format_exc()}"


@st.cache_data(show_spinner="Analyzing CV content with Groq LLM...")
def parse_cv_with_llm(text):
    """Sends resume text to the LLM for structured information extraction."""
    if text.startswith("Error") or not GROQ_API_KEY:
        # Returning a dictionary with an error key is safer for subsequent checks
        return {"error": "Parsing error or API key missing or file content extraction failed.", "raw_output": text}

    prompt = f"""Extract the following information from the resume in structured JSON.
    - Name, - Email, - Phone, - Skills (as a list), - Education (list of degrees/schools/dates), 
    - Experience (list of jobs/roles/dates/companies), - Certifications (list), 
    - Projects (list), - Strength (list), 
    - Github (link), - LinkedIn (link)
    
    For all lists (Skills, Education, Experience, Certifications, Projects, Strength), provide them as a Python list of strings or dictionaries as appropriate.
    
    Also, provide a key called **'summary'** which is a single, brief paragraph (3-4 sentences max) summarizing the candidate's career highlights and most relevant skills.
    
    Resume Text: {text}
    
    Provide the output strictly as a JSON object, without any surrounding markdown or commentary.
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
        
        # Robustly isolate JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            # Clean up potential markdown wrappers
            json_str = json_str.replace('```json', '').replace('```', '').strip() 
            parsed = json.loads(json_str)
        else:
            raise json.JSONDecodeError("Could not isolate a valid JSON structure.", content, 0)

        # --- FIX: Robustly ensure 'skills' is a clean list of strings ---
        if 'skills' in parsed and isinstance(parsed['skills'], list):
            # First, cast everything to a string
            parsed['skills'] = [str(s) for s in parsed['skills'] if s and isinstance(s, (str, list, dict))]
            # Flatten skills that might be nested lists/dicts from LLM output
            flat_skills = []
            for s in parsed['skills']:
                if isinstance(s, str):
                    flat_skills.extend([skill.strip() for skill in s.split(',')])
                elif isinstance(s, (list, dict)):
                    # Handle complex LLM output like [{'name': 'Python'}, {'name': 'SQL'}]
                    flat_skills.extend([str(item) for item in s])

            # Final clean list of non-empty strings
            parsed['skills'] = [s.strip() for s in flat_skills if s.strip()]
        # --- END FIX ---

    except Exception as e:
        parsed = {"error": f"LLM parsing error: {e}", "raw_output": content}

    return parsed

@st.cache_data(show_spinner="Analyzing JD content with Groq LLM...")
def parse_jd_with_llm(text, jd_title="Job Description"):
    """Sends JD text to the LLM for structured information extraction."""
    if text.startswith("Error") or not GROQ_API_KEY:
        return {"error": "Parsing error or API key missing or file content extraction failed.", "raw_output": text}

    prompt = f"""Extract the following key information from the Job Description in structured JSON for a matching algorithm:
    - **title**: The primary job title.
    - **required_skills**: A list of essential technical and soft skills.
    - **responsibilities**: A list of key duties/responsibilities.
    - **qualifications**: A list of required educational background or certifications (e.g., 'B.Tech in CS', 'PMP certification').
    - **experience_level**: The required seniority (e.g., 'Senior', 'Mid-level', 'Entry').
    
    JD Text (Title: {jd_title}): {text}
    
    Provide the output strictly as a JSON object, without any surrounding markdown or commentary.
    """
    content = ""
    parsed = {}
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1 # Lower temperature for better structure
        )
        content = response.choices[0].message.content.strip()
        
        # Robustly isolate JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            json_str = json_str.replace('```json', '').replace('```', '').strip() 
            parsed = json.loads(json_str)
        else:
            raise json.JSONDecodeError("Could not isolate a valid JSON structure.", content, 0)
    except Exception as e:
        parsed = {"error": f"LLM parsing error: {e}", "raw_output": content}

    return parsed

# --- CORE MATCHING LOGIC (ENHANCED MOCK) ---
def mock_jd_match(cv_data, jd_data):
    """
    Compares CV data against JD data using a detailed, weighted mock calculation
    to derive Skills, Experience, and Education match percentages.
    """
    
    # 1. Prepare Data (with enhanced cleaning)
    
    # Robustly process required JD skills
    jd_skills_raw = jd_data.get('required_skills', [])
    if not isinstance(jd_skills_raw, list): jd_skills_raw = []
    # Flatten and clean the JD skills list
    jd_skills = set()
    for s in jd_skills_raw:
        if isinstance(s, str):
            jd_skills.update(skill.lower().strip() for skill in s.split(',') if skill.strip())
        elif isinstance(s, list):
            jd_skills.update(skill.lower().strip() for item in s for skill in str(item).split(',') if skill.strip())
            
    # Robustly process CV skills (already cleaned in parse_cv_with_llm, but reinforce here)
    cv_skills_raw = cv_data.get('skills', [])
    if not isinstance(cv_skills_raw, list): cv_skills_raw = []
    # Flatten and clean the CV skills list
    cv_skills = set()
    for s in cv_skills_raw:
        if isinstance(s, str):
            cv_skills.update(skill.lower().strip() for skill in s.split(',') if skill.strip())
        elif isinstance(s, list):
             cv_skills.update(skill.lower().strip() for item in s for skill in str(item).split(',') if skill.strip())

    # Education/Qualification Preparation
    jd_qualifications = {q.lower() for q in jd_data.get('qualifications', []) if isinstance(q, str)}
    cv_education = {e.get('degree', '').lower() for e in cv_data.get('education', []) if isinstance(e, dict)}
    cv_certifications = {c.get('name', '').lower() for c in cv_data.get('certifications', []) if isinstance(c, dict)}
    
    # Experience Preparation
    cv_experience_roles = {e.get('role', '').lower() for e in cv_data.get('experience', []) if isinstance(e, dict)}
    jd_title_lower = jd_data.get('title', '').lower()
    
    # --- 2. Skills Match Calculation ---
    
    jd_skills_count = len(jd_skills)
    
    if jd_skills_count == 0:
        skills_percent = 75 # Default high score if no required skills are listed
        common_skills = set()
    else:
        common_skills = jd_skills.intersection(cv_skills)
        overlap_ratio = len(common_skills) / jd_skills_count
        skills_percent = int(overlap_ratio * 100)
    
    skills_percent = max(0, min(100, skills_percent))


    # --- 3. Education/Qualification Match Calculation ---
    
    if not jd_qualifications:
        education_percent = 70 
    else:
        required_count = len(jd_qualifications)
        match_count = 0
        
        for jd_qual in jd_qualifications:
            # Check for substring match in education or certification
            if any(jd_qual in edu for edu in cv_education):
                 match_count += 1
            elif any(jd_qual in cert for cert in cv_certifications):
                 match_count += 1

        if required_count > 0:
            education_percent = int((match_count / required_count) * 100)
        else:
             education_percent = 0 
        
    education_percent = max(0, min(100, education_percent))


    # --- 4. Experience Match Calculation ---
    
    experience_level_jd = jd_data.get('experience_level', 'mid-level').lower()
    
    # Check if any CV role title contains JD title (or vice versa, for broader match)
    cv_has_relevant_role = any(
        (jd_title_lower in role or role in jd_title_lower)
        and (len(role) > 5 or len(jd_title_lower) > 5) # Avoid matching short, generic words
        for role in cv_experience_roles
    )
    
    cv_years_mock = len(cv_data.get('experience', []))
    
    exp_score = 0
    
    if cv_years_mock > 0:
         exp_score += 10
         
    if cv_has_relevant_role:
        exp_score += 40 
        
    if experience_level_jd == 'senior' and cv_years_mock >= 5:
        exp_score += 40
    elif experience_level_jd == 'mid-level' and cv_years_mock >= 2:
        exp_score += 40
    elif experience_level_jd == 'entry' and cv_years_mock >= 0:
         exp_score += 40
    
    exp_score += min(10, cv_years_mock * 2) 

    experience_percent = max(0, min(100, exp_score))
    
    # --- 5. Final Fit Score (Weighted Average) ---
    
    # Weighting: Skills (50%), Experience (35%), Education (15%)
    weighted_score = (skills_percent * 0.50) + (experience_percent * 0.35) + (education_percent * 0.15)
    
    final_score_100 = int(weighted_score)
    final_score_10 = round(final_score_100 / 10, 1)

    # --- 6. Summary Generation ---
    summary = f"Match based on **{len(common_skills)}/{jd_skills_count}** required skills found. "
    
    if final_score_100 > 90:
        summary += "Excellent match! Candidate is highly qualified and experienced."
    elif final_score_100 > 70:
        summary += "Strong match. Minor gaps in qualifications/experience noted but core skills align."
    elif final_score_100 > 50:
        summary += "Fair match. Several core requirements are met, but skill/experience gaps exist."
    else:
        summary += "Low match. Significant gaps in core requirements observed."
        
    return {
        "score_100": final_score_100,
        "score_10": final_score_10,
        "summary": summary,
        "skills_percent": skills_percent,
        "experience_percent": experience_percent,
        "education_percent": education_percent,
        "common_skills": list(common_skills) # Added for detailed report
    }


# --- PDF/HTML Generation & CV Display Functions ---

def generate_pdf_mock(cv_data, cv_name):
    """Mocks the generation of a PDF file and returns its path/bytes."""
    
    warning_message = f"🚨 PDF generation is disabled! Use the 'Download CV as HTML (Print-to-PDF)' button instead. The actual library (fpdf) is not installed."
    
    return warning_message.encode('utf-8') 

def format_cv_to_html(cv_data, cv_name):
    """Formats the structured CV data into a clean HTML string for printing."""
    
    def list_to_html(items, tag='li'):
        if not items:
            return ""
        string_items = [str(item) for item in items]
        list_items_html = ''.join(f'<{tag}>{item}</{tag}>' for item in string_items)
        return f"<ul>{list_items_html}</ul>"

    def format_section(title, items, format_func):
        html = f'<h2>{title}</h2>'
        if not items:
            return html + '<p>No entries found.</p>'
        
        for item in items:
            if isinstance(item, dict):
                html += format_func(item)
            else:
                html += f'<div class="entry"><p><strong>Error:</strong> Corrupted entry: <code>{str(item)}</code></p></div>'
        return html

    def format_experience(exp):
        return f"""
        <div class="entry">
            <h3>{exp.get('role', 'N/A')}</h3>
            <p><strong>Company:</strong> {exp.get('company', 'N/A')}</p>
            <p><strong>Dates:</strong> {exp.get('dates', 'N/A')}</p>
            <p><strong>Focus:</strong> {exp.get('project', 'General Duties')}</p>
        </div>
        """
    
    def format_education(edu):
        return f"""
        <div class="entry">
            <h3>{edu.get('degree', 'N/A')}</h3>
            <p><strong>Institution:</strong> {edu.get('college', 'N/A')} ({edu.get('university', 'N/A')})</p>
            <p><strong>Dates:</strong> {edu.get('dates', 'N/A')}</p>
        </div>
        """

    def format_certifications(cert):
        return f"""
        <div class="entry">
            <p><strong>{cert.get('name', 'N/A')}</strong> - {cert.get('title', 'N/A')}</p>
            <p><em>Issued by:</em> {cert.get('given_by', 'N/A')}</p>
            <p><em>Date:</em> {cert.get('date_received', 'N/A')}</p>
        </div>
        """

    def format_projects(proj):
        tech_str = ', '.join([str(t) for t in proj.get('technologies', [])])
        link = f' | <a href="{proj["app_link"]}">{proj["app_link"]}</a>' if proj.get("app_link") and proj.get("app_link") != "N/A" else ""
        return f"""
        <div class="entry">
            <h3>{proj.get('name', 'N/A')}</h3>
            <p><strong>Description:</strong> {proj.get('description', 'N/A')}</p>
            <p><strong>Technologies:</strong> {tech_str} {link}</p>
        </div>
        """
        
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>CV: {cv_data.get('name', cv_name)}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            h1 {{ border-bottom: 2px solid #555; padding-bottom: 5px; }}
            h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-top: 20px; }}
            .header, .contact, .section {{ margin-bottom: 15px; }}
            .contact p {{ margin: 0; }}
            .entry {{ margin-bottom: 10px; padding-left: 10px; border-left: 3px solid #eee; }}
            .summary {{ font-style: italic; background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
            ul {{ list-style-type: disc; margin-left: 20px; padding-left: 0; }}
            /* Print-specific styles */
            @media print {{
                body {{ margin: 0; padding: 0; }}
                h1 {{ border-bottom: 2px solid black; }}
                h2 {{ border-bottom: 1px solid black; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{cv_data.get('name', cv_name)}</h1>
        </div>
        
        <div class="contact">
            <p><strong>Email:</strong> {cv_data.get('email', 'N/A')}</p>
            <p><strong>Phone:</strong> {cv_data.get('phone', 'N/A')}</p>
            <p><strong>LinkedIn:</strong> <a href="{cv_data.get('linkedin', '#')}">{cv_data.get('linkedin', 'N/A')}</a></p>
            <p><strong>GitHub:</strong> <a href="{cv_data.get('github', '#')}">{cv_data.get('github', 'N/A')}</a></p>
        </div>

        <div class="section summary">
            <h2>Summary</h2>
            <p>{cv_data.get('summary', 'N/A')}</p>
        </div>

        <div class="section">
            <h2>Skills</h2>
            {list_to_html(cv_data.get('skills', []))}
        </div>

        <div class="section">
            {format_section('Experience', cv_data.get('experience', []), format_experience)}
        </div>

        <div class="section">
            {format_section('Education', cv_data.get('education', []), format_education)}
        </div>

        <div class="section">
            {format_section('Certifications', cv_data.get('certifications', []), format_certifications)}
        </div>

        <div class="section">
            {format_section('Projects', cv_data.get('projects', []), format_projects)}
        </div>
        
        <div class="section">
            <h2>Strengths</h2>
            {list_to_html(cv_data.get('strength', []))}
        </div>

    </body>
    </html>
    """
    return html_content.strip()

def format_cv_to_markdown(cv_data, cv_name):
    """Formats the structured CV data into a viewable Markdown string."""
    md = f"""
# {cv_data.get('name', cv_name)}
### Contact & Links
* **Email:** {cv_data.get('email', 'N/A')}
* **Phone:** {cv_data.get('phone', 'N/A')}
* **LinkedIn:** [{cv_data.get('linkedin', 'N/A')}]({cv_data.get('linkedin', '#')})
* **GitHub:** [{cv_data.get('github', 'N/A')}]({cv_data.get('github', '#')})

---
## Summary
> {cv_data.get('summary', 'N/A')}

---
## Skills
* {', '.join([str(s) for s in cv_data.get('skills', ['N/A'])])}

---
## Experience
"""
    if cv_data.get('experience'):
        for exp in cv_data['experience']:
            if isinstance(exp, dict):
                md += f"""
### **{exp.get('role', 'N/A')}**
* **Company:** {exp.get('company', 'N/A')}
* **Dates:** {exp.get('dates', 'N/A')}
* **Key Project/Focus:** {exp.get('project', 'General Duties')}
"""
            else:
                md += f"* **Error:** Corrupted entry: `{str(exp)}`\n"
    else:
        md += "* No experience entries found."

    md += """
---
## Education
"""
    if cv_data.get('education'):
        for edu in cv_data['education']:
            if isinstance(edu, dict):
                md += f"""
### **{edu.get('degree', 'N/A')}**
* **Institution:** {edu.get('college', 'N/A')} ({edu.get('university', 'N/A')})
* **Dates:** {edu.get('dates', 'N/A')}
"""
            else:
                md += f"* **Error:** Corrupted entry: `{str(edu)}`\n"
    else:
        md += "* No education entries found."
    
    md += """
---
## Certifications
"""
    if cv_data.get('certifications'):
        for cert in cv_data['certifications']:
            if isinstance(cert, dict):
                md += f"""
* **{cert.get('name', 'N/A')}** - {cert.get('title', 'N/A')}
    * *Issued by:* {cert.get('given_by', 'N/A')}
    * *Date:* {cert.get('date_received', 'N/A')}
"""
            else:
                 md += f"* **Error:** Corrupted entry: `{str(cert)}`\n"
    else:
        md += "* No certification entries found."
        
    md += """
---
## Projects
"""
    if cv_data.get('projects'):
        for proj in cv_data['projects']:
            if isinstance(proj, dict):
                tech_str = ', '.join([str(t) for t in proj.get('technologies', [])])
                app_link = proj.get('app_link', 'N/A')
                
                link_md = ""
                if app_link and app_link != 'N/A':
                    link_md = f"\n* **App/Repo Link:** [{app_link}]({app_link})"
                    
                md += f"""
### **{proj.get('name', 'N/A')}**
* *Description:* {proj.get('description', 'N/A')}
* *Technologies:* {tech_str}
{link_md}
"""
            else:
                md += f"* **Error:** Corrupted entry: `{str(proj)}`\n"
    else:
        md += "* No project entries found."
        
    md += """
---
## Strengths
"""
    if cv_data.get('strength'):
        md += "* " + "\n* ".join([str(s) for s in cv_data.get('strength', ['N/A'])])
    else:
        md += "* No strengths listed."

    return md

def generate_and_display_cv(cv_name):
    """Generates the final structured CV data from session state and displays it."""
    
    if cv_name not in st.session_state.managed_cvs:
        st.error(f"Error: CV '{cv_name}' not found in managed CVs.")
        return
        
    cv_data = st.session_state.managed_cvs[cv_name]
    
    if not isinstance(cv_data, dict):
        st.error(f"Error: Stored data for CV '{cv_name}' is corrupted or contains an error string. Please re-parse or re-save the CV.")
        st.code(str(cv_data))
        return
        
    st.markdown(f"### 📄 CV View: **{cv_data.get('name', cv_name)}**")
    
    tab_md, tab_json, tab_pdf = st.tabs(["Markdown View", "JSON Data", "HTML/PDF Download"])

    with tab_md:
        md_output = format_cv_to_markdown(cv_data, cv_name)
        st.markdown(md_output)
        
        st.download_button(
            label="Download Markdown (.md)",
            data=md_output.encode('utf-8'),
            file_name=f"{cv_name}_cv.md",
            mime="text/markdown",
            key=f"download_md_btn_{cv_name}" 
        )

    with tab_json:
        json_output = json.dumps(cv_data, indent=4)
        st.code(json_output, language="json")
        st.download_button(
            label="Download JSON (.json)",
            data=json_output,
            file_name=f"{cv_name}_data.json",
            mime="application/json",
            key=f"download_json_btn_{cv_name}" 
        )
        
    with tab_pdf:
        html_output = format_cv_to_html(cv_data, cv_name)
        
        st.info("To get a PDF, download the HTML file, open it in your browser, and use the browser's 'Print' function (Ctrl+P or Cmd+P), selecting 'Save as PDF' as the destination.")
        
        st.download_button(
            label="Download CV as HTML (Print-to-PDF)",
            data=html_output.encode('utf-8'),
            file_name=f"{cv_name}.html",
            mime="text/html",
            key=f"download_html_btn_{cv_name}"
        )

# --- Shared Manual Input Logic (CV Form) ---

def save_form_cv():
    """
    Callback function to compile the structured CV data from form states and save it.
    """
    current_form_name = st.session_state.get('form_name_value', '').strip()
    
    if not current_form_name:
         st.error("Please enter your **Full Name** to save the CV.") 
         return
    
    cv_key_name = st.session_state.get('current_resume_name')
    if not cv_key_name or (cv_key_name not in st.session_state.managed_cvs):
         timestamp = datetime.now().strftime("%Y%m%d-%H%M")
         cv_key_name = f"{current_form_name.replace(' ', '_')}_Manual_CV_{timestamp}"

    final_cv_data = {
        "name": current_form_name,
        "email": st.session_state.get('form_email_value', '').strip(),
        "phone": st.session_state.get('form_phone_value', '').strip(),
        "linkedin": st.session_state.get('form_linkedin_value', '').strip(),
        "github": st.session_state.get('form_github_value', '').strip(),
        "summary": st.session_state.get('form_summary_value', '').strip(),
        "skills": [s.strip() for s in st.session_state.get('form_skills_value', '').split('\n') if s.strip()],
        "education": st.session_state.get('form_education', []), 
        "experience": st.session_state.get('form_experience', []), 
        "certifications": st.session_state.get('form_certifications', []), 
        "projects": st.session_state.get('form_projects', []),
        "strength": [s.strip() for s in st.session_state.get('form_strengths_input', '').split('\n') if s.strip()] 
    }
    
    st.session_state.managed_cvs[cv_key_name] = final_cv_data
    st.session_state.current_resume_name = cv_key_name
    st.session_state.show_cv_output = cv_key_name 
    
    st.success(f"🎉 CV for **'{current_form_name}'** saved/updated as **'{cv_key_name}'**!")

def add_education_entry(degree, college, university, date_from, date_to, state_key='form_education'):
    if not degree or not college or not university:
        st.error("Please fill in **Degree**, **College**, and **University**.")
        return
        
    entry = {
        "degree": degree,
        "college": college,
        "university": university,
        "dates": f"{date_from.year} - {date_to.year}"
    }
    
    if state_key not in st.session_state: st.session_state[state_key] = []
    st.session_state[state_key].append(entry)
    st.toast(f"Added Education: {degree}")

def add_experience_entry(company, role, ctc, project, date_from, date_to, state_key='form_experience'):
    if not company or not role or not date_from or not date_to:
        st.error("Please fill in **Company Name**, **Role**, and **Dates**.")
        return
        
    entry = {
        "company": company,
        "role": role,
        "ctc": ctc if ctc else "N/A",
        "project": project if project else "General Duties",
        "dates": f"{date_from.year} - {date_to.year}"
    }
    
    if state_key not in st.session_state: st.session_state[state_key] = []
    st.session_state[state_key].append(entry)
    st.toast(f"Added Experience: {role} at {company}")

def add_certification_entry(name, title, given_by, received_by, course, date_val, state_key='form_certifications'):
    if not name or not title or not given_by or not course:
        st.error("Please fill in **Name**, **Title**, **Given By**, and **Course**.")
        return
        
    entry = {
        "name": name,
        "title": title,
        "given_by": given_by,
        "received_by_name": received_by if received_by else "N/A",
        "course": course,
        "date_received": date_val.strftime("%Y-%m-%d")
    }
    
    if state_key not in st.session_state: st.session_state[state_key] = []
    st.session_state[state_key].append(entry)
    st.toast(f"Added Certification: {name} ({title})")

def add_project_entry(name, description, technologies, app_link, state_key='form_projects'):
    if not name or not description or not technologies:
        st.error("Please fill in **Project Name**, **Description**, and **Technologies Used**.")
        return
        
    entry = {
        "name": name,
        "description": description,
        "technologies": [t.strip() for t in technologies.split(',') if t.strip()], 
        "app_link": app_link if app_link else "N/A"
    }
    
    if state_key not in st.session_state: st.session_state[state_key] = []
    st.session_state[state_key].append(entry)
    st.toast(f"Added Project: {name}")

def remove_entry(index, state_key, entry_type='Item'):
    if 0 <= index < len(st.session_state.get(state_key, [])):
        entry_data = st.session_state[state_key][index]
        if state_key == 'form_education':
            removed_name = entry_data.get('degree', entry_type) if isinstance(entry_data, dict) else entry_type
        elif state_key == 'form_experience':
            removed_name = entry_data.get('role', entry_type) if isinstance(entry_data, dict) else entry_type
        elif state_key == 'form_certifications':
            removed_name = entry_data.get('name', entry_type) if isinstance(entry_data, dict) else entry_type
        elif state_key == 'form_projects':
            removed_name = entry_data.get('name', entry_type) if isinstance(entry_data, dict) else entry_type
        else:
            removed_name = entry_type
            
        del st.session_state[state_key][index]
        st.toast(f"Removed {entry_type}: {removed_name}")

# --- Resume Parsing Tab ---
def resume_parsing_tab():
    st.header("Upload/Paste Resume for AI Parsing")
    st.caption("Upload a file or paste text to extract structured data and save it as a structured CV.")
    
    uploaded_file = st.file_uploader(
        "Upload Resume File", 
        type=['pdf', 'docx', 'txt', 'json', 'csv', 'md'], 
        accept_multiple_files=False,
        key="resume_uploader"
    )

    st.markdown("---")
    
    pasted_text = st.text_area(
        "Or Paste Resume Text Here",
        height=300,
        key="resume_paster"
    )
    
    st.markdown("---")
    
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. AI Parsing features are disabled.")
        process_button = st.button("✨ Parse and Load Uploaded File", type="primary", use_container_width=True, disabled=True)
    else:
        process_button = st.button("✨ Parse and Load Uploaded File", type="primary", use_container_width=True)


    if process_button:
        extracted_text = ""
        file_name = "Pasted_Resume"
        
        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
            file_type = get_file_type(file_name)
            
            with st.spinner(f"Extracting text from {file_name}..."):
                extracted_text = extract_content(file_type, file_bytes, file_name)
                
        elif pasted_text.strip():
            extracted_text = pasted_text.strip()
            
        else:
            st.warning("Please upload a file or paste text content to proceed with parsing.")
            return

        if extracted_text.startswith("Error") or not extracted_text:
            st.error(f"Text Extraction Failed: {extracted_text}")
            error_key = f"ERROR_{file_name}_{datetime.now().strftime('%H%M')}"
            st.session_state.managed_cvs[error_key] = extracted_text
            st.session_state.current_resume_name = error_key
            st.session_state.show_cv_output = error_key
            return
            
        with st.spinner("🧠 Sending to Groq LLM for structured parsing..."):
            parsed_data = parse_cv_with_llm(extracted_text)
        
        if "error" in parsed_data:
            st.error(f"AI Parsing Failed: {parsed_data['error']}")
            st.code(parsed_data.get('raw_output', 'No raw output available.'), language='text')
            error_key = f"ERROR_{file_name}_{datetime.now().strftime('%H%M')}"
            st.session_state.managed_cvs[error_key] = "AI Parsing Error: " + parsed_data['error']
            st.session_state.current_resume_name = error_key
            st.session_state.show_cv_output = error_key
            return

        candidate_name = parsed_data.get('name', 'Unknown_Candidate').replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        cv_key_name = f"{candidate_name}_{timestamp}"
        
        st.session_state.managed_cvs[cv_key_name] = parsed_data
        st.session_state.current_resume_name = cv_key_name
        
        if isinstance(parsed_data, dict):
            st.session_state.form_name_value = parsed_data.get('name', '')
            st.session_state.form_email_value = parsed_data.get('email', '')
            st.session_state.form_phone_value = parsed_data.get('phone', '')
            st.session_state.form_linkedin_value = parsed_data.get('linkedin', '')
            st.session_state.form_github_value = parsed_data.get('github', '')
            st.session_state.form_summary_value = parsed_data.get('summary', '')
            
            if isinstance(parsed_data.get('skills'), list):
                # Use the cleaned list of skills from the parser
                st.session_state.form_skills_value = "\n".join([str(s) for s in parsed_data['skills']])
            else:
                 st.session_state.form_skills_value = ""

            if isinstance(parsed_data.get('strength'), list):
                st.session_state.form_strengths_input = "\n".join([str(s) for s in parsed_data['strength']])
            else:
                 st.session_state.form_strengths_input = ""
                
            st.session_state.form_education = parsed_data.get('education', [])
            st.session_state.form_experience = parsed_data.get('experience', [])
            st.session_state.form_certifications = parsed_data.get('certifications', [])
            st.session_state.form_projects = parsed_data.get('projects', [])
        
            st.success(f"✅ Successfully parsed and structured CV for **{candidate_name}**! Data loaded for editing.")
        else:
            st.error(f"❌ Internal Error: Parsed data for **{candidate_name}** is not a dictionary. Please check LLM output logic.")

        st.session_state.show_cv_output = cv_key_name
        st.rerun() 

# --- CORE CV FORM FUNCTION ---

def cv_form_content():
    """Contains the logic for the manual CV form entry."""
    st.markdown("### Prepare your CV (Form-Based)")
    st.caption("Manually enter your CV details. Click **'Save Final CV Details'** at the bottom to save/update your structured CV.")
    
    # --- 1. Personal Details Form ---
    st.markdown("#### 1. Personal & Summary Details")
    
    col_name, col_email = st.columns(2)
    with col_name:
        st.text_input("Full Name", key="form_name_value")
    with col_email:
        st.text_input("Email", key="form_email_value")
        
    col_phone, col_linkedin, col_github = st.columns(3)
    with col_phone:
        st.text_input("Phone Number", key="form_phone_value")
    with col_linkedin:
        st.text_input("LinkedIn Link", key="form_linkedin_value")
    with col_github:
        st.text_input("GitHub Link", key="form_github_value")
        
    st.text_area("Career Summary / Objective (3-4 sentences)", height=100, key="form_summary_value")
    
    st.markdown("---")

    # --- 2. Skills ---
    st.markdown("#### 2. Skills")
    st.text_area("Skills (Enter one skill per line)", height=100, key="form_skills_value")
    
    # --- 3. Experience ---
    st.markdown("#### 3. Experience")
    
    with st.form("form_experience_entry", clear_on_submit=True):
        col_comp, col_role = st.columns(2)
        with col_comp:
            new_company = st.text_input("Company Name", key="form_new_company")
        with col_role:
            new_role = st.text_input("Role / Designation", key="form_new_role")
        
        col_ctc, col_proj = st.columns(2)
        with col_ctc:
            new_ctc = st.text_input("CTC (Optional)", key="form_new_ctc")
        with col_proj:
            new_project = st.text_input("Key Project / Main Focus", key="form_new_project")

        col_from, col_to = st.columns(2)
        with col_from:
            default_date_from = date(2020, 1, 1)
            try:
                if st.session_state.form_experience and isinstance(st.session_state.form_experience[-1], dict):
                    latest_exp_date_str = st.session_state.form_experience[-1]['dates'].split(' - ')[0]
                    if len(latest_exp_date_str) == 4 and latest_exp_date_str.isdigit():
                        default_date_from = datetime.strptime(latest_exp_date_str, "%Y").date()
                    elif '-' in latest_exp_date_str: 
                        # Attempt to parse a date part like 'YYYY' or a full date
                        parts = latest_exp_date_str.split('-')
                        if len(parts[0].strip()) == 4 and parts[0].strip().isdigit():
                            default_date_from = datetime.strptime(parts[0].strip(), "%Y").date()
            except Exception:
                 pass
            
            new_exp_date_from = st.date_input("Date From (Start)", value=default_date_from, key="form_new_exp_date_from")
        with col_to:
            new_exp_date_to = st.date_input("Date To (End/Present)", value=date.today(), key="form_new_exp_date_to")

        if st.form_submit_button("Add Experience and Save CV"):
            add_experience_entry(
                new_company.strip(), 
                new_role.strip(), 
                new_ctc.strip(),
                new_project.strip(),
                new_exp_date_from, 
                new_exp_date_to,
                state_key='form_experience'
            )
            save_form_cv() 

    if st.session_state.form_experience:
        st.markdown("##### Current Experience Entries:")
        experience_list = st.session_state.form_experience
        for i, entry in enumerate(experience_list):
            col_exp, col_rem = st.columns([0.8, 0.2])
            with col_exp:
                if isinstance(entry, dict):
                    st.code(f"{entry.get('role', 'N/A')} at {entry.get('company', 'N/A')} ({entry.get('dates', 'N/A')})", language="text")
                else:
                    st.code(f"Corrupted Entry: {str(entry)}", language="text")
                    
            with col_rem:
                st.button(
                    "Remove", 
                    key=f"remove_exp_{i}", 
                    on_click=remove_entry, 
                    args=(i, 'form_experience', 'Experience'),
                    type="secondary", 
                    use_container_width=True
                )
    
    # --- 4. Education ---
    st.markdown("#### 4. Education")

    with st.form("form_education_entry", clear_on_submit=True):
        col_degree, col_college = st.columns(2)
        with col_degree:
            new_degree = st.text_input("Degree/Qualification", key="form_new_degree")
        with col_college:
            new_college = st.text_input("College/Institution Name", key="form_new_college")
        
        new_university = st.text_input("Affiliating University Name", key="form_new_university")

        col_from, col_to = st.columns(2)
        with col_from:
            new_date_from = st.date_input("Date From (Start)", value=date(2018, 1, 1), key="form_new_date_from")
        with col_to:
            new_date_to = st.date_input("Date To (End/Expected)", value=date.today(), key="form_new_date_to")

        if st.form_submit_button("Add Education and Save CV"):
            add_education_entry(
                new_degree.strip(), 
                new_college.strip(), 
                new_university.strip(), 
                new_date_from, 
                new_date_to,
                state_key='form_education'
            )
            save_form_cv() 

    if st.session_state.form_education:
        st.markdown("##### Current Education Entries:")
        for i, entry in enumerate(st.session_state.form_education):
            col_edu, col_rem = st.columns([0.8, 0.2])
            with col_edu:
                if isinstance(entry, dict):
                    st.code(f"{entry.get('degree', 'N/A')} at {entry.get('college', 'N/A')} ({entry.get('university', 'N/A')}) ({entry.get('dates', 'N/A')})", language="text")
                else:
                    st.code(f"Corrupted Entry: {str(entry)}", language="text")

            with col_rem:
                st.button(
                    "Remove", 
                    key=f"remove_edu_{i}", 
                    on_click=remove_entry, 
                    args=(i, 'form_education', 'Education'),
                    type="secondary",
                    use_container_width=True
                )
    
    # -----------------------------
    # 5. CERTIFICATIONS SECTION
    # -----------------------------
    st.markdown("#### 5. Certifications")
    
    with st.form("form_certification_entry", clear_on_submit=True):
        col_cert_name, col_cert_title = st.columns(2)
        with col_cert_name:
            new_cert_name = st.text_input("Certification Name (e.g., AWS Certified)", key="form_new_cert_name")
        with col_cert_title:
            new_cert_title = st.text_input("Title (e.g., Solutions Architect - Associate)", key="form_new_cert_title")
            
        col_given, col_received = st.columns(2)
        with col_given:
            new_given_by = st.text_input("Given By (Issuing Authority)", key="form_new_given_by")
        with col_received:
            new_received_by = st.text_input("Received By Name (Optional)", key="form_new_received_by")
            
        new_course = st.text_input("Related Course/Training (Optional)", key="form_new_course")

        new_date_received = st.date_input("Date Received", value=date.today(), key="form_new_date_received")

        if st.form_submit_button("Add Certification and Save CV"):
            add_certification_entry(
                new_cert_name.strip(), 
                new_cert_title.strip(), 
                new_given_by.strip(), 
                new_received_by.strip(),
                new_course.strip(),
                new_date_received,
                state_key='form_certifications'
            )
            save_form_cv() 

    if st.session_state.form_certifications:
        st.markdown("##### Current Certification Entries:")
        for i, entry in enumerate(st.session_state.form_certifications):
            col_cert, col_rem = st.columns([0.8, 0.2])
            with col_cert:
                if isinstance(entry, dict):
                    st.code(f"{entry.get('name', 'N/A')} - {entry.get('title', 'N/A')} (Issued: {entry.get('date_received', 'N/A')})", language="text")
                else:
                    st.code(f"Corrupted Entry: {str(entry)}")
            with col_rem:
                st.button(
                    "Remove", 
                    key=f"remove_cert_{i}", 
                    on_click=remove_entry, 
                    args=(i, 'form_certifications', 'Certification'),
                    type="secondary",
                    use_container_width=True
                )
    
    # -----------------------------
    # 6. PROJECTS SECTION
    # -----------------------------
    st.markdown("#### 6. Projects")
    
    with st.form("form_project_entry", clear_on_submit=True):
        new_project_name = st.text_input("Project Name", key="form_new_project_name")
        new_project_description = st.text_area("Description of Project", height=100, key="form_new_project_description")
            
        col_tech, col_link = st.columns(2)
        with col_tech:
            new_technologies = st.text_input("Technologies Used (Comma separated list, e.g., Python, SQL, Streamlit)", key="form_new_technologies")
        with col_link:
            new_app_link = st.text_input("App Link / Repository URL (Optional)", key="form_new_app_link")

        if st.form_submit_button("Add Project and Save CV"):
            add_project_entry(
                new_project_name.strip(), 
                new_project_description.strip(), 
                new_technologies.strip(), 
                new_app_link.strip(),
                state_key='form_projects'
            )
            save_form_cv() 

    if st.session_state.form_projects:
        st.markdown("##### Current Project Entries:")
        
        for i, entry in enumerate(st.session_state.form_projects):
            with st.container(border=True):
                if isinstance(entry, dict):
                    st.markdown(f"**{i+1}. {entry.get('name', 'N/A')}**")
                    st.caption(f"Technologies: {', '.join([str(t) for t in entry.get('technologies', [])])}")
                    st.markdown(f"Description: *{entry.get('description', 'N/A')}*")
                    if entry.get('app_link') and entry['app_link'] != "N/A":
                        st.markdown(f"Link: [{entry['app_link']}]({entry['app_link']})")
                else:
                    st.error(f"Corrupted Entry: {str(entry)}")
                
                st.button(
                    "Remove Project", 
                    key=f"remove_project_{i}", 
                    on_click=remove_entry, 
                    args=(i, 'form_projects', 'Project'),
                    type="secondary"
                )
    
    # -----------------------------
    # 7. STRENGTHS SECTION
    # -----------------------------
    st.markdown("#### 7. Strengths")
    st.text_area(
        "Your Key Strengths (Enter one strength or attribute per line)", 
        height=100, 
        key="form_strengths_input",
        help="E.g., Problem-Solving, Team Leadership, Adaptability, Communication"
    )
    
    # --- Final Save Button ---
    st.markdown("---")
    st.button("💾 **Save Final CV Details**", key="final_save_button", type="primary", use_container_width=True, on_click=save_form_cv)
    
    st.markdown("---")
    
    # --- CV Output Display Section ---
    if st.session_state.show_cv_output:
        generate_and_display_cv(st.session_state.show_cv_output)
        st.markdown("---")

def tab_cv_management():
    # Initialization for list-based state
    if "form_education" not in st.session_state: st.session_state.form_education = []
    if "form_experience" not in st.session_state: st.session_state.form_experience = []
    if "form_certifications" not in st.session_state: st.session_state.form_certifications = []
    if "form_projects" not in st.session_state: st.session_state.form_projects = []

    # Reset form state if the current CV is an error string
    current_cv_name = st.session_state.get('current_resume_name')
    if current_cv_name and current_cv_name in st.session_state.managed_cvs:
        cv_data = st.session_state.managed_cvs[current_cv_name]
        
        if isinstance(cv_data, str):
            st.warning(f"⚠️ Corrupted CV data found under key '{current_cv_name}'. Resetting form fields to empty.")
            
            st.session_state.form_name_value = ""
            st.session_state.form_email_value = ""
            st.session_state.form_phone_value = ""
            st.session_state.form_linkedin_value = ""
            st.session_state.form_github_value = ""
            st.session_state.form_summary_value = ""
            st.session_state.form_skills_value = ""
            st.session_state.form_strengths_input = ""
            st.session_state.form_education = []
            st.session_state.form_experience = []
            st.session_state.form_certifications = []
            st.session_state.form_projects = []
            st.session_state.current_resume_name = None 
            st.session_state.show_cv_output = None
            
            st.error(f"Corrupted Data: {cv_data}")
            
    cv_form_content()

# -------------------------
# JD MANAGEMENT TAB CONTENT
# -------------------------

def process_jd_file(file, jd_type):
    """Handles processing a single JD file."""
    file_name = file.name
    file_bytes = file.getvalue()
    file_type = get_file_type(file_name)
    jd_key = file_name.replace('.', '_').replace(' ', '_').replace('-', '_') + "_" + datetime.now().strftime("%H%M")
    
    extracted_text = extract_content(file_type, file_bytes, file_name)
    
    if extracted_text.startswith("Error"):
        st.session_state.managed_jds[jd_key] = f"Extraction Error: Failed to read file content ({file_type})."
        return False, f"Extraction Failed for {file_name}: {extracted_text}"
        
    parsed_data = parse_jd_with_llm(extracted_text, jd_title=file_name)
    
    if "error" in parsed_data:
        st.session_state.managed_jds[jd_key] = f"AI Parsing Error: {parsed_data['error']}"
        return False, f"AI Parsing Failed for {file_name}: {parsed_data['error']}"
    
    st.session_state.managed_jds[jd_key] = parsed_data
    st.session_state.managed_jds[jd_key]['raw_text'] = extracted_text
    return True, f"Successfully parsed and saved JD **{jd_key}** (Title: {parsed_data.get('title', 'N/A')})"

def process_jd_text(text):
    """Handles processing pasted JD text."""
    jd_key = "Pasted_JD_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parsed_data = parse_jd_with_llm(text, jd_title="Pasted Text JD")
    
    if "error" in parsed_data:
        st.session_state.managed_jds[jd_key] = f"AI Parsing Error: {parsed_data['error']}"
        return False, f"AI Parsing Failed: {parsed_data['error']}"
        
    st.session_state.managed_jds[jd_key] = parsed_data
    st.session_state.managed_jds[jd_key]['raw_text'] = text
    return True, f"Successfully parsed and saved JD **{jd_key}** (Title: {parsed_data.get('title', 'N/A')})"

def clear_all_jds():
    """Callback to clear all JDs."""
    st.session_state.managed_jds = {}
    st.session_state.selected_jd_key = None
    st.toast("All saved JDs cleared!")

# Modified function: REMOVED RAW TEXT TAB/DISPLAY
def display_jd_details(key):
    """Displays the details of the selected JD (Structured Summary ONLY)."""
    jd_data = st.session_state.managed_jds.get(key)
    
    if not jd_data or isinstance(jd_data, str):
        st.error(f"Error: JD '{key}' data is corrupted or missing.")
        return

    st.markdown(f"### JD Details: **{jd_data.get('title', 'N/A')}**")
    
    st.markdown("#### Structured Summary")
    st.markdown(f"**Job Title:** {jd_data.get('title', 'N/A')}")
    st.markdown(f"**Experience Level:** {jd_data.get('experience_level', 'N/A')}")
    
    st.markdown("#### Required Skills")
    st.markdown("* " + "\n* ".join(jd_data.get('required_skills', ['N/A'])))

    st.markdown("#### Qualifications")
    st.markdown("* " + "\n* ".join(jd_data.get('qualifications', ['N/A'])))
    
    st.markdown("#### Responsibilities")
    st.markdown("* " + "\n* ".join(jd_data.get('responsibilities', ['N/A'])))
    
    if st.button("⬅️ Hide Details", key="hide_jd_details"):
        st.session_state.selected_jd_key = None
        st.rerun()

def jd_management_tab():
    st.header("Job Description (JD) Management")
    st.caption("Upload or paste job descriptions. They will be parsed and saved for matching against your CV.")
    
    st.markdown("#### 1. Select JD Type")
    jd_type = st.radio(
        "Choose JD scope:",
        ["Single JD", "Multiple JD"],
        index=0,
        horizontal=True,
        key="jd_type_select"
    )
    
    st.markdown("---")
    
    st.markdown("#### 2. Add JD by:")
    
    jd_method = st.radio(
        "Choose Method:",
        ["Upload File", "Paste Text", "LinkedIn URL"],
        index=0,
        horizontal=True,
        key="jd_method_select"
    )

    st.markdown("---")
    
    if jd_method == "Upload File":
        st.markdown("##### Upload JD File(s)")
        
        uploaded_jds = st.file_uploader(
            "Drag and drop file(s) here",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=(jd_type == "Multiple JD"),
            key="jd_uploader"
        )
        st.caption("Limit 200MB per file • PDF, TXT, DOCX")
        
        if st.button("Add JD(s)", type="primary", use_container_width=True, key="upload_jd_button"):
            if uploaded_jds:
                files_to_process = uploaded_jds if isinstance(uploaded_jds, list) else [uploaded_jds]
                
                with st.spinner(f"Processing {len(files_to_process)} JD file(s)..."):
                    results = [process_jd_file(f, jd_type) for f in files_to_process]
                
                success_count = sum(r[0] for r in results)
                st.success(f"✅ Finished processing: {success_count} success(es).")
                for success, message in results:
                    if success:
                        st.text(message)
                    else:
                        st.error(message)

            else:
                st.warning("Please upload at least one JD file.")
        
    elif jd_method == "Paste Text":
        st.markdown("##### Paste JD Text")
        
        pasted_jd_text = st.text_area(
            "Paste the job description text here:",
            height=300,
            key="jd_paster"
        )
        
        if st.button("Add JD", type="primary", use_container_width=True, key="paste_jd_button"):
            if pasted_jd_text.strip():
                with st.spinner("Processing pasted JD text..."):
                    success, message = process_jd_text(pasted_jd_text.strip())
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please paste the JD text.")

    elif jd_method == "LinkedIn URL":
        st.markdown("##### Enter LinkedIn URL (Requires Web Scraping)")
        
        linkedin_url = st.text_input(
            "Enter the full LinkedIn Job URL:",
            key="jd_linkedin_url",
            placeholder="https://www.linkedin.com/jobs/view/..."
        )
        
        if st.button("Fetch and Add JD (Mock)", type="primary", use_container_width=True, key="url_jd_button"):
            if linkedin_url.strip():
                if "linkedin.com/jobs/view" in linkedin_url:
                    st.info(f"Web scraping is not implemented in this demo. Please use the **Upload File** or **Paste Text** methods for a functional test.")
                else:
                    st.error("Please enter a valid LinkedIn Job URL.")
            else:
                st.warning("Please enter a LinkedIn Job URL.")

    st.markdown("---")
    st.markdown("#### 3. Saved Job Descriptions")
    
    if st.session_state.managed_jds:
        jd_keys = [k for k, v in st.session_state.managed_jds.items() if isinstance(v, dict)]
        error_keys = [k for k, v in st.session_state.managed_jds.items() if isinstance(v, str)]
        
        st.button("🗑️ Clear All JDs", key="clear_all_jds", on_click=clear_all_jds)

        if st.session_state.get('selected_jd_key'):
            display_jd_details(st.session_state.selected_jd_key)
        else:
            if jd_keys:
                st.markdown("##### Select a JD to View Details:")
                
                max_cols = 3 
                cols = st.columns(max_cols) 

                for i, key in enumerate(jd_keys):
                    jd_data = st.session_state.managed_jds[key]
                    title = jd_data.get('title', 'N/A')
                    
                    with cols[i % max_cols]:
                        with st.container(border=True):
                            st.markdown(f"**{i+1}. {title}**")
                            skills_preview = ', '.join(jd_data.get('required_skills', ['No skills listed'])[:2])
                            if len(jd_data.get('required_skills', [])) > 2:
                                skills_preview += '...'
                                
                            st.caption(f"Key: `{key}` | Skills: {skills_preview}")

                            if st.button("👁️ View Details", key=f"view_jd_btn_{key}", use_container_width=True):
                                st.session_state.selected_jd_key = key
                                st.rerun()
            
            if error_keys:
                 st.error("⚠️ The following keys contain corrupted or failed parsing data and cannot be displayed structured details:")
                 st.code("\n".join(error_keys), language='text')

            if not jd_keys and not error_keys:
                st.info("No JDs saved yet. Add one above to enable batch matching.")
                
    else:
        st.info("No JDs saved yet. Add one above to enable batch matching.")


# -------------------------
# BATCH JD MATCH TAB CONTENT (FIXED)
# -------------------------

def batch_jd_match_tab():
    st.header("🏆 Batch JD Match")
    st.caption("Select your CV and the JDs you want to match against to generate a consolidated fitment report.")

    cv_keys = [k for k, v in st.session_state.managed_cvs.items() if isinstance(v, dict) and v.get('name')]
    jd_keys_valid = [k for k, v in st.session_state.managed_jds.items() if isinstance(v, dict) and v.get('title')]
    
    st.markdown("#### 1. Select CV and JDs")

    # --- CV Selection ---
    if not cv_keys:
        st.warning("⚠️ **No CVs available.** Please upload or create a CV in the 'Resume Parsing' or 'CV Management' tabs.")
        selected_cv = None
        cv_data = None
    else:
        # Use the currently selected resume name if available, otherwise default to the first
        default_cv_index = cv_keys.index(st.session_state.current_resume_name) if st.session_state.current_resume_name in cv_keys else 0
        
        cv_options = {k: st.session_state.managed_cvs[k].get('name', k) for k in cv_keys}
        
        selected_cv_key = st.selectbox(
            "Select Candidate CV",
            options=list(cv_options.keys()),
            format_func=lambda k: cv_options[k],
            index=default_cv_index,
            key="batch_cv_selector"
        )
        selected_cv = selected_cv_key
        cv_data = st.session_state.managed_cvs.get(selected_cv_key)
        st.info(f"CV Selected: **{cv_data.get('name', 'N/A')}**")
        
    st.markdown("---")

    # --- JD Selection ---
    if not jd_keys_valid:
        st.warning("⚠️ **No valid JDs available.** Please add JDs in the 'JD Management' tab.")
        selected_jds = []
    else:
        jd_options = {k: st.session_state.managed_jds[k].get('title', k) for k in jd_keys_valid}
        
        selected_jds = st.multiselect(
            "Select Job Descriptions to Match Against",
            options=list(jd_options.keys()),
            format_func=lambda k: jd_options[k],
            default=st.session_state.get('selected_jds_for_match', []),
            key="batch_jd_multiselect"
        )
        # Persist the selection
        st.session_state.selected_jds_for_match = selected_jds

    st.markdown("---")

    # --- Match Button ---
    match_button = st.button(
        "🚀 Run Match Analysis", 
        type="primary", 
        use_container_width=True, 
        disabled=not (selected_cv and selected_jds)
    )

    # --- Core Logic Execution ---
    if match_button:
        if not cv_data or isinstance(cv_data, str):
            st.error("Cannot run match: The selected CV data is corrupted or missing.")
            return

        match_results = []
        with st.spinner(f"Analyzing {len(selected_jds)} JD(s) against {cv_data.get('name', 'CV')}..."):
            for jd_key in selected_jds:
                jd_data = st.session_state.managed_jds.get(jd_key)
                if jd_data and isinstance(jd_data, dict):
                    result = mock_jd_match(cv_data, jd_data)
                    result['jd_key'] = jd_key
                    result['jd_title'] = jd_data.get('title', jd_key)
                    match_results.append(result)
                else:
                    st.warning(f"Skipped JD **{jd_key}** due to corrupted data.")
        
        # Sort results by score (highest first)
        match_results.sort(key=lambda x: x['score_100'], reverse=True)
        st.session_state.candidate_results = match_results
        st.success("Analysis Complete! Check the report below.")
    
    st.markdown("---")

    # --- 2. Results Display ---
    st.markdown("#### 2. Match Report")
    
    if st.session_state.get('candidate_results'):
        results = st.session_state.candidate_results
        
        # Create DataFrame for the main table display
        df = pd.DataFrame([
            {
                "Job Title": r['jd_title'],
                "Overall Fit (%)": r['score_100'],
                "Skills Match (%)": r['skills_percent'],
                "Experience Match (%)": r['experience_percent'],
                "Education Match (%)": r['education_percent']
            }
            for r in results
        ]).set_index("Job Title")
        
        # Use standard Streamlit Dataframe display (FIXED from Matplotlib Error)
        st.dataframe(
            df, 
            use_container_width=True,
            column_config={
                "Overall Fit (%)": st.column_config.ProgressColumn(
                    "Overall Fit (%)",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )

        st.markdown("##### Detailed Reports")
        for i, result in enumerate(results):
            jd_key = result['jd_key']
            jd_data = st.session_state.managed_jds.get(jd_key, {})

            with st.expander(f"Report {i+1}: **{result['jd_title']}** (Score: {result['score_100']}%)"):
                st.markdown(f"**Match Summary:** {result['summary']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Skills Match", f"{result['skills_percent']}%", help="Weighted 50% of the overall score.")
                col2.metric("Experience Match", f"{result['experience_percent']}%", help="Weighted 35% of the overall score.")
                col3.metric("Education Match", f"{result['education_percent']}%", help="Weighted 15% of the overall score.")
                
                st.markdown("###### Skill Overlap Details")
                
                # Fetch clean skills list/set from the matching function output
                common_skills = set(result.get('common_skills', []))
                
                # Fetch clean required skills set from the JD data used in matching
                jd_skills_raw = jd_data.get('required_skills', [])
                if not isinstance(jd_skills_raw, list): jd_skills_raw = []
                required_skills = set()
                for s in jd_skills_raw:
                    if isinstance(s, str):
                        required_skills.update(skill.lower().strip() for skill in s.split(',') if skill.strip())
                    elif isinstance(s, list):
                        required_skills.update(skill.lower().strip() for item in s for skill in str(item).split(',') if skill.strip())


                missing_skills = required_skills - common_skills

                st.markdown(f"**Total Required Skills:** {len(required_skills)}")
                st.markdown(f"**Found Skills:** {len(common_skills)}")
                
                col_found, col_missing = st.columns(2)
                
                with col_found:
                    with st.container(border=True):
                        st.markdown("**✅ Found Skills (Matching Requirements):**")
                        if common_skills:
                            st.markdown("* " + "\n* ".join(sorted(list(common_skills))))
                        else:
                            st.info("No matching skills found in this CV for this JD.")

                with col_missing:
                    with st.container(border=True):
                        st.markdown("**❌ Missing Skills (Required by JD):**")
                        if missing_skills:
                            st.markdown("* " + "\n* ".join(sorted(list(missing_skills))))
                        else:
                            st.success("All required skills found!")

    else:
        st.info("Run the Match Analysis above to generate the report.")
        
# -------------------------
# CANDIDATE DASHBOARD FUNCTION
# -------------------------

def candidate_dashboard():
    st.title("🧑‍💻 Candidate Dashboard")
    
    col_header, col_logout = st.columns([4, 1])
    with col_logout:
        if st.button("🚪 Log Out", use_container_width=True):
            keys_to_delete = ['candidate_results', 'current_resume', 'manual_education', 'managed_cvs', 'current_resume_name', 'form_education', 'form_experience', 'form_certifications', 'form_projects', 'show_cv_output', 'form_name_value', 'form_email_value', 'form_phone_value', 'form_linkedin_value', 'form_github_value', 'form_summary_value', 'form_skills_value', 'form_strengths_input', 'form_cv_key_name', 'resume_uploader', 'resume_paster', 'jd_type_select', 'jd_method_select', 'jd_uploader', 'jd_paster', 'jd_linkedin_url', 'managed_jds', 'selected_jds_for_match', 'selected_jd_key']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            go_to("login")
            st.rerun() 
            
    st.markdown("---")

    # --- Session State Initialization for Candidate ---
    if "managed_cvs" not in st.session_state: st.session_state.managed_cvs = {} 
    if "managed_jds" not in st.session_state: st.session_state.managed_jds = {} 
    if "current_resume_name" not in st.session_state: st.session_state.current_resume_name = None 
    if "show_cv_output" not in st.session_state: st.session_state.show_cv_output = None 
    if "selected_jd_key" not in st.session_state: st.session_state.selected_jd_key = None
    if "selected_jds_for_match" not in st.session_state: st.session_state.selected_jds_for_match = [] # For batch matching UI persistence
    if "candidate_results" not in st.session_state: st.session_state.candidate_results = None # To store the match results
    
    # Initialize keys for personal details to ensure stability
    if "form_name_value" not in st.session_state: st.session_state.form_name_value = ""
    if "form_email_value" not in st.session_state: st.session_state.form_email_value = ""
    if "form_phone_value" not in st.session_state: st.session_state.form_phone_value = ""
    if "form_linkedin_value" not in st.session_state: st.session_state.form_linkedin_value = ""
    if "form_github_value" not in st.session_state: st.session_state.form_github_value = ""
    if "form_summary_value" not in st.session_state: st.session_state.form_summary_value = ""
    if "form_skills_value" not in st.session_state: st.session_state.form_skills_value = ""
    if "form_strengths_input" not in st.session_state: st.session_state.form_strengths_input = ""

    # --- Main Content with Tabs ---
    tab_parsing, tab_management, tab_jd, tab_match = st.tabs(["📄 Resume Parsing", "📝 CV Management (Form)", "💼 JD Management", "🏆 Batch JD Match"])
    
    with tab_parsing:
        resume_parsing_tab()
        
    with tab_management:
        tab_cv_management()
        
    with tab_jd:
        jd_management_tab()

    with tab_match:
        batch_jd_match_tab()


# -------------------------
# MOCK LOGIN AND MAIN APP LOGIC 
# -------------------------

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
    
    if st.session_state.logged_in:
        if st.session_state.user_type == "candidate":
            candidate_dashboard()
        elif st.session_state.user_type == "admin":
            admin_dashboard() 
    else:
        login_page()
