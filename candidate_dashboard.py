import streamlit as st
import os
import tempfile
import json
import re
import traceback
from datetime import date
from typing import Dict, Any, List

# ==============================================================================
# 1. DEPENDENCIES & HELPER FUNCTIONS (Stubs for demonstration)
# ==============================================================================

# --- Utility Functions ---

def go_to(page_name):
    """Changes the current page in Streamlit's session state."""
    st.session_state.page = page_name

def clear_interview_state():
    """Clears all generated questions, answers, and the evaluation report."""
    st.session_state.interview_qa = []
    st.session_state.iq_output = ""
    st.session_state.evaluation_report = ""
    st.toast("Practice answers cleared.")
    
# --- External LLM/File Logic (Simplified or Stubbed for standalone copy) ---
question_section_options = ["skills","experience", "certifications", "projects", "education"]
DEFAULT_JOB_TYPES = ["Full-time", "Contract", "Internship", "Remote", "Part-time"]
DEFAULT_ROLES = ["Software Engineer", "Data Scientist", "Product Manager", "HR Manager", "Marketing Specialist", "Operations Analyst"]

# STUBS for functions that require the actual full application code or external APIs
def extract_jd_from_linkedin_url(url: str) -> str:
    """Stub: Simulates JD content extraction."""
    return f"--- Simulated JD for: {url}\n\nJob Description content extracted from LinkedIn URL. This includes role details, requirements, and company information."

def extract_jd_metadata(jd_text):
    """Stub: Simulates extraction of structured metadata."""
    # Simple logic to return different data based on content to make the dashboard dynamic
    if "Software Engineer" in jd_text:
        return {"role": "Software Engineer", "job_type": "Full-time", "key_skills": ["Python", "Flask", "AWS", "SQL", "CI/CD"]}
    elif "Data Scientist" in jd_text:
        return {"role": "Data Scientist", "job_type": "Contract", "key_skills": ["Python", "Machine Learning", "TensorFlow", "Pandas", "Statistics"]}
    return {"role": "General Analyst", "job_type": "Full-time", "key_skills": ["Python", "SQL", "Cloud"]}

def parse_and_store_resume(file_input, file_name_key='default', source_type='file'):
    """Stub: Simulates parsing and stores results into a structure.
    
    Updated default structured experience and education.
    """
    if st.session_state.get('parsed', {}).get('name') and st.session_state.parsed.get('name') != "":
         return {"parsed": st.session_state.parsed, "full_text": st.session_state.full_text, "excel_data": None, "name": st.session_state.parsed['name']}
    
    # Placeholder data for a fresh parse
    if source_type == 'file':
        name_from_file = getattr(file_input, 'name', 'Uploaded_Resume').split('.')[0].replace('_', ' ')
    else:
        name_from_file = "Parsed Text CV"
        
    # Example structured experience for default parsing
    default_structured_experience = [
        {
            "company": "Prgayan AI", 
            "role": "AIML Engineer", 
            "from_year": "2025", 
            "to_year": "Present", 
            "ctc": "Negotiable", 
            "responsibilities": "Developing and deploying AI/ML models for NLP and Computer Vision projects."
        },
        {
            "company": "DataStart Innovations", 
            "role": "Junior Developer", 
            "from_year": "2022", 
            "to_year": "2024", 
            "ctc": "$60k", 
            "responsibilities": "Developed ETL pipelines using Python and SQL."
        }
    ]
    
    # Example structured education for default parsing (NEW STRUCTURE)
    default_structured_education = [
        {
            "degree": "M.Sc. Computer Science", 
            "college": "University of Excellence", 
            "university": "City University",
            "from_year": "2020",
            "to_year": "2022",
            "score": "8.5",
            "type": "CGPA"
        },
        {
            "degree": "B.Tech. Information Technology", 
            "college": "College of Engineering", 
            "university": "State University",
            "from_year": "2016",
            "to_year": "2020",
            "score": "75",
            "type": "Percentage"
        }
    ]
    
    # Example structured certifications for default parsing
    default_structured_certifications = [
        {
            "title": "AWS Certified Cloud Practitioner", 
            "given_by": "Amazon Web Services", 
            "issue_date": "2023-10-01"
        }
    ]
    
    # In the new structure, 'experience', 'certifications', and 'education' will hold the structured data
    parsed_data = {
        "name": name_from_file, 
        "email": "candidate@example.com", 
        "phone": "555-123-4567",
        "linkedin": "linkedin.com/in/candidate", 
        "github": "github.com/candidate",
        "skills": ["Python", "Machine Learning", "Streamlit", "Data Analysis", "TensorFlow"], 
        "experience": default_structured_experience, 
        "structured_experience": default_structured_experience, # Structured list for form
        
        "education": default_structured_education, # Storing structured data here
        "structured_education": default_structured_education, # New structured list for form
        
        "certifications": default_structured_certifications, # Storing structured data here
        "structured_certifications": default_structured_certifications, # New structured list for form
        
        "projects": ["Built this Streamlit Dashboard"], 
        "strength": ["Problem Solver", "Quick Learner"], 
        "personal_details": "Highly motivated and results-oriented professional with 3+ years experience in AIML."
    }
    
    # Create a placeholder full_text 
    compiled_text = ""
    for k, v in parsed_data.items():
        # Exclude structured lists from raw text generation, except for the main keys
        if v and k not in ["structured_experience", "structured_certifications", "structured_education"]: 
            compiled_text += f"{k.replace('_', ' ').title()}:\n"
            if isinstance(v, list):
                 # For raw text, we will flatten the structured data into simple strings (JSON format)
                if all(isinstance(item, dict) for item in v):
                     compiled_text += "\n".join([json.dumps(item) for item in v]) + "\n\n"
                else:
                    compiled_text += "\n".join([f"- {item}" for item in v if isinstance(item, str)]) + "\n\n"
            else:
                compiled_text += str(v) + "\n\n"

    return {"parsed": parsed_data, "full_text": compiled_text, "excel_data": None, "name": parsed_data['name']}

def qa_on_resume(question):
    """Stub: Simulates Q&A on resume."""
    if "skills" in question.lower():
        return f"Based on the resume, the key skills are: {', '.join(st.session_state.parsed.get('skills', ['No skills found']))}. The candidate has a strong background in data tools."
    return f"Based on the resume, the answer to '{question}' is: [Simulated Answer - Check experience/projects section for details. All data is stored as structured data.]"

def qa_on_jd(question, selected_jd_name):
    """Stub: Simulates Q&A on JD."""
    return f"Based on the JD '{selected_jd_name}', the answer to '{question}' is: [Simulated Answer - The JD content specifies a 5+ years experience requirement and mandatory Python/SQL skills]."

def evaluate_jd_fit(job_description, parsed_json):
    """Stub: Simulates JD fit evaluation."""
    # Use random score for variation
    import random
    score = random.randint(5, 9)
    skills = random.randint(60, 95)
    experience = random.randint(50, 90)
    
    return f"""Overall Fit Score: {score}/10
--- Section Match Analysis ---
Skills Match: {skills}%
Experience Match: {experience}%
Education Match: 90% (Based on B.Tech and M.Sc. degrees)

Strengths/Matches:
- Candidate's Python and ML skills ({skills}%) are an excellent match for this JD.
- Experience ({experience}%) is relevant, particularly the **AIML Engineer at Prgayan AI** role.
- Education is a **Strong** match with advanced degrees listed.

Gaps/Areas for Improvement:
- Needs more specific experience in the [Niche Technology] mentioned in the JD.
- The resume summary could be tailored more closely to the [Specific Industry] focus of this role.

Overall Summary: This is a **Strong** fit. Focus on experience and advanced education in the interview.
"""

def generate_interview_questions(parsed_json, section):
    """Stub: Simulates interview question generation."""
    
    # Customize question based on the new structured education data
    education_data = parsed_json.get('education', [])
    first_degree = {}
    if education_data and isinstance(education_data[0], dict):
        first_degree = education_data[0]
        score_display = f"{first_degree.get('score', 'N/A')} {first_degree.get('type', 'Score')}"
    else:
        score_display = "N/A"

    if section == "education":
        return f"""[Technical/Academic]
Q1: Tell me about your **{first_degree.get('degree', 'highest degree')}** and how it prepared you for this role.
Q2: What was your favorite technical project or thesis during your time at **{first_degree.get('university', 'university')}**?
Q3: How do you think your academic performance (**{score_display}**) reflects your work ethic?
[Behavioral]
Q4: Describe a time you struggled academically and how you overcame it.
Q5: How do you keep your technical skills updated now that you've finished your formal education?
"""
    
    return f"""[Behavioral]
Q1: Tell me about a time you applied your strongest skill, **{parsed_json.get('skills', ['No skill'])[0]}**, to solve a major problem.
Q2: Describe a project where your work in the **{section}** section directly led to a quantifiable business outcome.
[Technical]
Q3: How do you handle a scenario where a tool in your **{section}** section fails in production?
Q4: What is the most challenging concept you learned in your **{section}** area?
[General]
Q5: Why are you looking to move from your current role/studies?
"""

def evaluate_interview_answers(qa_list, parsed_json):
    """Stub: Simulates interview answer evaluation."""
    
    total_score = len(qa_list) * 7 # Average score for simulation
    
    feedback_parts = ["## Evaluation Results"]
    for i, qa_item in enumerate(qa_list):
        feedback_parts.append(f"""
### Question {i+1}: {qa_item['question']}
Score: 7/10
Feedback:
- **Clarity & Accuracy:** The answer for this question was good, addressing the core topic.
- **Improvements:** Try to use the **STAR** (Situation, Task, Action, Result) method, especially for behavioral questions. Quantify your results.
""")
        
    feedback_parts.append(f"""
---
## Final Assessment
Total Score: {total_score}/{len(qa_list) * 10}
Overall Summary: The candidate shows **Good** fundamental knowledge. To score higher, better integrate answers with accomplishments listed in the resume (e.g., mention specific projects from the Prgayan AI role).
""")
    
    return "\n".join(feedback_parts)

# --- Simplified HTML/Markdown output for Structured Data ---
def generate_cv_html(parsed_data):
    """Generates CV HTML with simplified plain text output for structured sections."""
    skills_list = "".join([f"<li>{s}</li>" for s in parsed_data.get('skills', []) if isinstance(s, str)])
    
    # Education HTML
    education_list = ""
    for edu in parsed_data.get('education', []):
        if isinstance(edu, dict):
            # Format: Degree (Score) | College, University (Start Year - End Year)
            score_display = f"{edu.get('score', 'N/A')} {edu.get('type', '')}".strip()
            education_list += f"""
            <li>
                **{edu.get('degree', 'N/A')}** ({score_display}) | {edu.get('college', 'N/A')}, {edu.get('university', 'N/A')} 
                <br>({edu.get('from_year', '')} - {edu.get('to_year', '')})
            </li>
            """

    # Experience HTML
    experience_list = ""
    for exp in parsed_data.get('experience', []):
        if isinstance(exp, dict):
            # Format: Role at Company (Start Year - End Year). Responsibilities: <text>
            experience_list += f"""
            <li>
                **{exp.get('role', 'N/A')}** at {exp.get('company', 'N/A')} ({exp.get('from_year', '')} - {exp.get('to_year', '')}).
                <br>Responsibilities: {exp.get('responsibilities', 'N/A')}
            </li>
            """

    # Certifications HTML
    certifications_list = ""
    for cert in parsed_data.get('certifications', []):
        if isinstance(cert, dict):
            # Format: Title - Issued by: Organization, Date: <date>
            certifications_list += f"""
            <li>
                {cert.get('title', 'N/A')} - Issued by: {cert.get('given_by', 'N/A')}, Date: {cert.get('issue_date', 'N/A')}
            </li>
            """

    
    return f"""
    <html>
    <head>
        <title>{parsed_data.get('name', 'CV Preview')}</title>
        <style>body{{font-family: Arial, sans-serif; margin: 40px;}} h1{{color: #2e6c80; border-bottom: 2px solid #2e6c80;}} h2{{color: #3d99b1;}} ul{{list-style-type: none; padding: 0;}} li{{margin-bottom: 10px;}}</style>
    </head>
    <body>
        <h1>{parsed_data.get('name', 'CV Preview')}</h1>
        <p>Email: {parsed_data.get('email', 'N/A')} | Phone: {parsed_data.get('phone', 'N/A')}</p>
        <p>LinkedIn: <a href="{parsed_data.get('linkedin', '#')}">{parsed_data.get('linkedin', 'N/A')}</a></p>
        
        <h2>Key Skills</h2>
        <ul>{skills_list}</ul>
        
        <h2>Experience</h2>
        <ul>{experience_list}</ul>
        
        <h2>Education</h2>
        <ul>{education_list}</ul>
        
        <h2>Certifications</h2>
        <ul>{certifications_list}</ul>
        
        <p>Generated by AI Dashboard on {date.today()}</p>
    </body>
    </html>
    """

def format_parsed_json_to_markdown(parsed_data):
    """Generates CV Markdown with simplified plain text output for structured sections."""
    md = f"# **{parsed_data.get('name', 'CV Preview').upper()}**\n"
    md += f"**Contact:** {parsed_data.get('email', 'N/A')} | {parsed_data.get('phone', 'N/A')} | [LinkedIn]({parsed_data.get('linkedin', '#')})\n"
    md += "\n"
    md += f"## **SUMMARY**\n---\n"
    md += parsed_data.get('personal_details', 'No summary provided.') + "\n\n"
    
    md += "\n\n## **EXPERIENCE**\n---\n"
    experience_md = []
    for exp in parsed_data.get('experience', []):
        if isinstance(exp, dict):
            # Format: Role at Company (Start Year - End Year). Responsibilities: <text>
            experience_md.append(
                f"**{exp.get('role', 'N/A')}** at {exp.get('company', 'N/A')} ({exp.get('from_year', '')} - {exp.get('to_year', '')}).\n"
                f"Responsibilities: {exp.get('responsibilities', 'N/A')}" 
            )
    md += "\n\n".join(experience_md)

    md += "\n\n## **EDUCATION**\n---\n"
    education_md = []
    for edu in parsed_data.get('education', []):
        if isinstance(edu, dict):
            # Format: Degree (Score Type) | College, University (Start Year - End Year)
            score_display = f"{edu.get('score', 'N/A')} {edu.get('type', '')}".strip()
            education_md.append(
                f"**{edu.get('degree', 'N/A')}** ({score_display}) at {edu.get('college', 'N/A')} / {edu.get('university', 'N/A')}\n"
                f"Duration: {edu.get('from_year', '')} - {edu.get('to_year', '')}"
            )
    md += "\n\n".join(education_md)
    
    md += "\n\n## **CERTIFICATIONS**\n---\n"
    certifications_md = []
    for cert in parsed_data.get('certifications', []):
        if isinstance(cert, dict):
            # Format: Title - Issued by: Organization, Date: <date>
            certifications_md.append(
                f"{cert.get('title', 'N/A')} - Issued by: {cert.get('given_by', 'N/A')}, Date: {cert.get('issue_date', 'N/A')}"
            )
    md += "- " + "\n- ".join(certifications_md)

    md += "\n\n## **SKILLS**\n---\n"
    md += "- " + "\n- ".join(parsed_data.get('skills', ['No skills listed']) if all(isinstance(s, str) for s in parsed_data.get('skills', [])) else ["Skills list structure mismatch"])

    md += "\n\n## **PROJECTS**\n---\n"
    md += "- " + "\n- ".join(parsed_data.get('projects', ['No projects listed']) if all(isinstance(p, str) for p in parsed_data.get('projects', [])) else ["Projects list structure mismatch"])

    md += "\n\n## **STRENGTHS**\n---\n"
    md += "- " + "\n- ".join(parsed_data.get('strength', ['No strengths listed']) if all(isinstance(s, str) for s in parsed_data.get('strength', [])) else ["Strengths list structure mismatch"])
    
    return md

# ==============================================================================
# 2. TAB CONTENT FUNCTIONS (Updated CV Management)
# ==============================================================================

def cv_management_tab_content():
    st.header("📝 Prepare Your CV")
    st.markdown("### 1. Form Based CV Builder")
    st.info("Fill out the details below to generate a parsed CV. **NOTE: The dynamic lists (Education, Certifications, Experience) are managed *outside* this main form to comply with Streamlit's technical rules. Please use the Add/Remove sections below.**")

    # --- Session State Initialization for CV Builder ---
    default_parsed = {
        "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
        "skills": [], "experience": [], "education": [], "certifications": [], 
        "projects": [], "strength": [], "personal_details": "",
        "structured_experience": [],
        "structured_certifications": [],
        "structured_education": [] # New structured education key
    }
    
    if "cv_form_data" not in st.session_state:
        if st.session_state.get('parsed', {}).get('name') and st.session_state.parsed.get('name') != "":
            st.session_state.cv_form_data = st.session_state.parsed.copy()
            # Ensure the structured lists are present, falling back to the main list if needed
            if 'structured_experience' not in st.session_state.cv_form_data:
                st.session_state.cv_form_data['structured_experience'] = st.session_state.cv_form_data.get('experience', []) if all(isinstance(i, dict) for i in st.session_state.cv_form_data.get('experience', [])) else [] 
            if 'structured_certifications' not in st.session_state.cv_form_data:
                st.session_state.cv_form_data['structured_certifications'] = st.session_state.cv_form_data.get('certifications', []) if all(isinstance(i, dict) for i in st.session_state.cv_form_data.get('certifications', [])) else []
            # NEW: Initialize structured_education
            if 'structured_education' not in st.session_state.cv_form_data:
                st.session_state.cv_form_data['structured_education'] = st.session_state.cv_form_data.get('education', []) if all(isinstance(i, dict) for i in st.session_state.cv_form_data.get('education', [])) else []
        else:
            st.session_state.cv_form_data = default_parsed
            
    # CRITICAL: Ensure lists are initialized correctly
    if 'structured_experience' not in st.session_state.cv_form_data:
         st.session_state.cv_form_data['structured_experience'] = []
    if 'structured_certifications' not in st.session_state.cv_form_data:
         st.session_state.cv_form_data['structured_certifications'] = []
    if 'structured_education' not in st.session_state.cv_form_data: # Ensure it exists
         st.session_state.cv_form_data['structured_education'] = []
    if not isinstance(st.session_state.cv_form_data.get('skills'), list):
         st.session_state.cv_form_data['skills'] = []
    if not isinstance(st.session_state.cv_form_data.get('projects'), list):
         st.session_state.cv_form_data['projects'] = []

    
    # Initialize/reset temp data structures 
    current_year = date.today().year
    
    # --- CV Builder Form (SINGLE BLOCK) ---
    # This form collects all static data and triggers the submission logic
    with st.form("cv_builder_form", clear_on_submit=False):
        
        # --- 1. PERSONAL & CONTACT DETAILS ---
        st.subheader("1. Personal, Contact, and Summary Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.cv_form_data['name'] = st.text_input(
                "Full Name", 
                value=st.session_state.cv_form_data['name'], 
                key="cv_name_input"
            ).strip() 
        with col2:
            st.session_state.cv_form_data['email'] = st.text_input(
                "Email Address", 
                value=st.session_state.cv_form_data['email'], 
                key="cv_email_input"
            ).strip() 
        with col3:
            st.session_state.cv_form_data['phone'] = st.text_input(
                "Phone Number", 
                value=st.session_state.cv_form_data['phone'], 
                key="cv_phone_input"
            ).strip() 
        
        col4, col5 = st.columns(2)
        with col4:
            st.session_state.cv_form_data['linkedin'] = st.text_input(
                "LinkedIn Profile URL", 
                value=st.session_state.cv_form_data.get('linkedin', ''), 
                key="cv_linkedin_input"
            ).strip() 
        with col5:
            st.session_state.cv_form_data['github'] = st.text_input(
                "GitHub Profile URL", 
                value=st.session_state.cv_form_data.get('github', ''), 
                key="cv_github_input"
            ).strip() 
        
        st.session_state.cv_form_data['personal_details'] = st.text_area(
            "Professional Summary (A brief pitch about yourself)", 
            value=st.session_state.cv_form_data.get('personal_details', ''), 
            height=100,
            key="cv_personal_details_input"
        ).strip() 
        
        # --- 2. SKILLS (Now inside the single form) ---
        st.markdown("---")
        st.subheader("2. Key Skills (One Item per Line)")

        skills_text = "\n".join(st.session_state.cv_form_data.get('skills', []) if all(isinstance(s, str) for s in st.session_state.cv_form_data.get('skills', [])) else [])
        new_skills_text = st.text_area(
            "Technical and Soft Skills", 
            value=skills_text,
            height=100,
            key="cv_skills_input_form" 
        )
        # Update session state on submit
        st.session_state.cv_form_data['skills'] = [s.strip() for s in new_skills_text.split('\n') if s.strip()]
        
        # --- 3. EDUCATION PLACEHOLDER (Instruct user to use external forms) ---
        st.markdown("---")
        st.subheader("3. Education Details")
        st.info("⚠️ **Education** is managed using the dynamic 'Add Entry' form **outside** this main form. Please scroll down to manage entries.")
        
        # --- 4. CERTIFICATIONS PLACEHOLDER (Instruct user to use external forms) ---
        st.markdown("---")
        st.subheader("4. Certifications")
        st.info("⚠️ **Certifications** are managed using the dynamic 'Add Certificate' form **outside** this main form. Please scroll down to manage entries.")
        
        # --- 5. PROJECTS (Now inside the single form) ---
        st.markdown("---")
        st.subheader("5. Projects (One Item per Line)")
        projects_text = "\n".join(st.session_state.cv_form_data.get('projects', []) if all(isinstance(p, str) for p in st.session_state.cv_form_data.get('projects', [])) else [])
        new_projects_text = st.text_area(
            "Projects (Name, Description, Technologies)", 
            value=projects_text,
            height=100,
            key="cv_projects_input_form"
        )
        st.session_state.cv_form_data['projects'] = [p.strip() for p in new_projects_text.split('\n') if p.strip()]

        # --- 6. STRENGTHS (Now inside the single form) ---
        st.markdown("---")
        st.subheader("6. Strengths (One Item per Line)")
        strength_text = "\n".join(st.session_state.cv_form_data.get('strength', []) if all(isinstance(s, str) for s in st.session_state.cv_form_data.get('strength', [])) else [])
        new_strength_text = st.text_area(
            "Key Personal Qualities", 
            value=strength_text,
            height=70,
            key="cv_strength_input_form"
        )
        st.session_state.cv_form_data['strength'] = [s.strip() for s in new_strength_text.split('\n') if s.strip()]

        # --- 7. EXPERIENCE PLACEHOLDER (Instruct user to use external forms) ---
        st.markdown("---")
        st.subheader("7. Professional Experience")
        st.info("⚠️ **Experience** is managed using the dynamic 'Add Experience' form **outside** this main form. Please scroll down to manage entries.")
        
        # CRITICAL: The submit button is ONLY placed here, inside the one form block.
        st.markdown("---")
        st.subheader("8. Generate or Load ALL CV Data")
        st.warning("Ensure you have added your **Education, Certifications, and Experience** entries using the dynamic forms below this main form, then click below to finalize.")
        submit_form_button = st.form_submit_button("Generate and Load ALL CV Data", type="primary", use_container_width=True)

    
    # --- FINAL SUBMISSION LOGIC (for the main form) ---
    if submit_form_button:
        if not st.session_state.cv_form_data['name'] or not st.session_state.cv_form_data['email']:
            st.error("Please fill in at least your **Full Name** and **Email Address**.")
            return

        # 1. Synchronize the structured lists into the main keys for AI consumption
        # NOTE: This pulls the latest data from the lists managed OUTSIDE the form.
        st.session_state.cv_form_data['experience'] = st.session_state.cv_form_data.get('structured_experience', [])
        st.session_state.cv_form_data['certifications'] = st.session_state.cv_form_data.get('structured_certifications', [])
        st.session_state.cv_form_data['education'] = st.session_state.cv_form_data.get('structured_education', [])
        
        # 2. Update the main parsed state
        st.session_state.parsed = st.session_state.cv_form_data.copy()
        
        # 3. Create a placeholder full_text for the AI tools
        compiled_text = ""
        EXCLUDE_KEYS = ["structured_experience", "structured_certifications", "structured_education"] 
        
        for k, v in st.session_state.cv_form_data.items():
            if k in EXCLUDE_KEYS:
                continue
            
            if v and (isinstance(v, str) and v.strip() or isinstance(v, list) and v):
                compiled_text += f"{k.replace('_', ' ').title()}:\n"
                if isinstance(v, list):
                    if all(isinstance(item, dict) for item in v):
                         compiled_text += "\n".join([json.dumps(item) for item in v]) + "\n\n"
                    elif all(isinstance(item, str) for item in v):
                        compiled_text += "\n".join([f"- {item}" for item in v]) + "\n\n"
                else:
                    compiled_text += str(v) + "\n\n"
                    
        st.session_state.full_text = compiled_text
        
        # 4. Reset matching/interview state
        st.session_state.candidate_match_results = []
        st.session_state.interview_qa = []
        st.session_state.evaluation_report = ""

        st.success(f"✅ CV data for **{st.session_state.parsed['name']}** successfully generated and loaded! All major sections are stored as **structured data**.")
        
    
    # -------------------------------------------------------------------------
    # --- DYNAMIC EDUCATION SECTION (OUTSIDE the main form - Corresponds to Section 3) ---
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🎓 **Dynamic Education Management**")
    st.markdown("Use the fields below to add structured education entries one by one.")
    
    # Function to handle adding the education entry
    def add_education_entry():
        current_year = date.today().year
        
        degree_val = st.session_state.get("temp_edu_degree_key", "").strip() 
        college_val = st.session_state.get("temp_edu_college_key", "").strip() 
        university_val = st.session_state.get("temp_edu_university_key", "").strip() 
        from_year_val = st.session_state.get("temp_edu_from_year_key", "").strip()
        to_year_val = st.session_state.get("temp_edu_to_year_key", str(current_year)).strip()
        score_val = st.session_state.get("temp_edu_score_key", "").strip()
        score_type_val = st.session_state.get("temp_edu_type_key", "CGPA").strip()
        
        if not degree_val or not college_val or not from_year_val:
            st.error("Please fill in **Degree**, **College**, and **From Year**.")
            return

        new_entry = {
            "degree": degree_val,
            "college": college_val,
            "university": university_val,
            "from_year": from_year_val,
            "to_year": to_year_val,
            "score": score_val,
            "type": score_type_val
        }
        
        st.session_state.cv_form_data['structured_education'].append(new_entry)
        
        # Clear temp state/widget values to refresh the input fields
        st.session_state["temp_edu_degree_key"] = ""
        st.session_state["temp_edu_college_key"] = ""
        st.session_state["temp_edu_university_key"] = ""
        st.session_state["temp_edu_from_year_key"] = str(current_year) 
        st.session_state["temp_edu_to_year_key"] = "Present" 
        st.session_state["temp_edu_score_key"] = ""
        st.session_state["temp_edu_type_key"] = "CGPA"
        
        st.toast(f"Education: {new_entry['degree']} added.")
        
    def remove_education_entry(index):
        if 0 <= index < len(st.session_state.cv_form_data['structured_education']):
            removed_degree = st.session_state.cv_form_data['structured_education'][index]['degree']
            del st.session_state.cv_form_data['structured_education'][index]
            st.toast(f"Education '{removed_degree}' removed.")
            st.rerun() 

    # Input fields for a single education entry
    with st.container(border=True):
        st.markdown("##### Add New Education Entry")
        current_year = date.today().year
        year_options = [str(y) for y in range(current_year, 1950, -1)]
        
        # Input fields
        col_d, col_c = st.columns(2)
        with col_d:
            st.text_input(
                "Degree/Qualification", 
                value=st.session_state.get("temp_edu_degree_key", ""),
                key="temp_edu_degree_key", 
                placeholder="e.g., M.Sc. Computer Science"
            )
            
        with col_c:
            st.text_input(
                "College Name", 
                value=st.session_state.get("temp_edu_college_key", ""),
                key="temp_edu_college_key", 
                placeholder="e.g., MIT, Chennai"
            )
            
        st.text_input(
            "University Name", 
            value=st.session_state.get("temp_edu_university_key", ""),
            key="temp_edu_university_key", 
            placeholder="e.g., Anna University"
        )
        
        # Years
        col_fy, col_ty = st.columns(2)
        
        with col_fy:
            current_from_year = st.session_state.get("temp_edu_from_year_key", str(current_year))
            from_year_index = year_options.index(current_from_year) if current_from_year in year_options else 0
            
            st.selectbox(
                "From Year", 
                options=year_options, 
                index=from_year_index,
                key="temp_edu_from_year_key"
            )
            
        with col_ty:
            to_year_options = ["Present"] + year_options
            current_to_year = st.session_state.get("temp_edu_to_year_key", "Present")
            to_year_index = to_year_options.index(current_to_year) if current_to_year in to_year_options else 0
            
            st.selectbox(
                "To Year", 
                options=to_year_options, 
                index=to_year_index,
                key="temp_edu_to_year_key"
            )
            
        # Score
        col_s, col_st = st.columns([2, 1])
        with col_s:
            st.text_input(
                "CGPA or Score Value", 
                value=st.session_state.get("temp_edu_score_key", ""),
                key="temp_edu_score_key", 
                placeholder="e.g., 8.5 or 90"
            )
        with col_st:
            st.selectbox(
                "Type",
                options=["CGPA", "Percentage", "Grade"],
                index=["CGPA", "Percentage", "Grade"].index(st.session_state.get("temp_edu_type_key", "CGPA")),
                key="temp_edu_type_key",
                label_visibility='collapsed'
            )
            
        st.button("➕ Add Education Entry", on_click=add_education_entry, use_container_width=True, type="secondary")

    # Display current education entries
    st.markdown("##### Current Education Entries")
    if st.session_state.cv_form_data['structured_education']:
        for i, entry in enumerate(st.session_state.cv_form_data['structured_education']):
            score_display = f"{entry.get('score', 'N/A')} {entry.get('type', '')}".strip()
            expander_title = f"{entry['degree']} - {score_display} ({entry['from_year']} - {entry['to_year']})"
            
            with st.expander(expander_title, expanded=False):
                st.markdown(f"**Degree:** {entry['degree']}")
                st.markdown(f"**College:** {entry['college']}")
                st.markdown(f"**University:** {entry['university']}")
                st.markdown(f"**Duration:** {entry['from_year']} - {entry['to_year']}")
                st.markdown(f"**Score:** {score_display}")
                
                st.button("❌ Remove", key=f"remove_edu_{i}", on_click=remove_education_entry, args=(i,), type="secondary") 
    else:
        st.info("No education entries added yet.")


    
    # -------------------------------------------------------------------------
    # --- DYNAMIC CERTIFICATION SECTION (OUTSIDE the main form - Corresponds to Section 4) ---
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🏅 **Dynamic Certifications Management**")
    st.markdown("Use the fields below to add structured certification entries one by one.")
    
    # Function to handle adding the certification entry
    def add_certification_entry():
        title_val = st.session_state.get("temp_cert_title_key", "").strip()
        given_by_val = st.session_state.get("temp_cert_given_by_key", "").strip()
        issue_date_val = st.session_state.get("temp_cert_issue_date_key", str(date.today().year)).strip()
        
        if not title_val or not given_by_val:
            st.error("Please fill in **Certification Title** and **Issuing Organization**.")
            return

        new_entry = {
            "title": title_val,
            "given_by": given_by_val,
            "issue_date": issue_date_val
        }
        
        st.session_state.cv_form_data['structured_certifications'].append(new_entry)
        
        st.session_state["temp_cert_title_key"] = ""
        st.session_state["temp_cert_given_by_key"] = ""
        st.session_state["temp_cert_issue_date_key"] = str(date.today().year)
        
        st.toast(f"Certificate: {new_entry['title']} added.")
        
    def remove_certification_entry(index):
        if 0 <= index < len(st.session_state.cv_form_data['structured_certifications']):
            removed_title = st.session_state.cv_form_data['structured_certifications'][index]['title']
            del st.session_state.cv_form_data['structured_certifications'][index]
            st.toast(f"Certificate '{removed_title}' removed.")
            st.rerun()

    # Input fields for a single certification entry
    with st.container(border=True):
        st.markdown("##### Add New Certificate")
        
        col_t, col_g = st.columns(2)
        with col_t:
            st.text_input(
                "Certification Title", 
                value=st.session_state.get("temp_cert_title_key", ""),
                key="temp_cert_title_key", 
                placeholder="e.g., Google Cloud Architect"
            )
            
        with col_g:
            st.text_input(
                "Issuing Organization", 
                value=st.session_state.get("temp_cert_given_by_key", ""),
                key="temp_cert_given_by_key", 
                placeholder="e.g., Coursera, AWS, PMI"
            )
            
        col_d, _ = st.columns(2)
        with col_d:
            st.text_input(
                "Issue Date (YYYY-MM-DD or Year)", 
                value=st.session_state.get("temp_cert_issue_date_key", str(date.today().year)),
                key="temp_cert_issue_date_key", 
                placeholder="e.g., 2024-05-15 or 2023"
            )
            
        st.button("➕ Add Certificate", on_click=add_certification_entry, use_container_width=True, type="secondary")

    # Display current certification entries
    st.markdown("##### Current Certifications")
    if st.session_state.cv_form_data['structured_certifications']:
        for i, entry in enumerate(st.session_state.cv_form_data['structured_certifications']):
            
            expander_title = f"{entry['title']} by {entry['given_by']} (Issued: {entry['issue_date']})"
            
            with st.expander(expander_title, expanded=False):
                st.markdown(f"**Title:** {entry['title']}")
                st.markdown(f"**Issued By:** {entry['given_by']}")
                st.markdown(f"**Issue Date:** {entry['issue_date']}")
                
                st.button("❌ Remove", key=f"remove_cert_{i}", on_click=remove_certification_entry, args=(i,), type="secondary") 
    else:
        st.info("No certifications added yet.")

    
    # -------------------------------------------------------------------------
    # --- DYNAMIC EXPERIENCE SECTION (OUTSIDE the main form - Corresponds to Section 7) ---
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💼 **Dynamic Professional Experience Management**")
    st.markdown("Use the fields below to add structured experience entries one by one.")
    
    # Function to handle adding the experience entry
    def add_experience_entry():
        current_year = date.today().year
        
        company_val = st.session_state.get("temp_exp_company_key", "").strip()
        role_val = st.session_state.get("temp_exp_role_key", "").strip()
        from_year_val = st.session_state.get("temp_exp_from_year_key", "").strip()
        to_year_val = st.session_state.get("temp_exp_to_year_key", "Present").strip()
        ctc_val = st.session_state.get("temp_exp_ctc_key", "").strip()
        responsibilities_val = st.session_state.get("temp_exp_responsibilities_key", "").strip()
        
        if not company_val or not role_val or not from_year_val:
            st.error("Please fill in **Company**, **Role**, and **From Year**.")
            return

        new_entry = {
            "company": company_val,
            "role": role_val,
            "from_year": from_year_val,
            "to_year": to_year_val,
            "ctc": ctc_val,
            "responsibilities": responsibilities_val
        }
        
        st.session_state.cv_form_data['structured_experience'].append(new_entry)
        
        # Clear temp state/widget values to refresh the input fields
        st.session_state["temp_exp_company_key"] = ""
        st.session_state["temp_exp_role_key"] = ""
        st.session_state["temp_exp_from_year_key"] = str(current_year)
        st.session_state["temp_exp_to_year_key"] = "Present"
        st.session_state["temp_exp_ctc_key"] = ""
        st.session_state["temp_exp_responsibilities_key"] = ""
        
        st.toast(f"Experience at {new_entry['company']} added.")
        
    def remove_experience_entry(index):
        if 0 <= index < len(st.session_state.cv_form_data['structured_experience']):
            removed_company = st.session_state.cv_form_data['structured_experience'][index]['company']
            del st.session_state.cv_form_data['structured_experience'][index]
            st.toast(f"Experience at {removed_company} removed.")
            st.rerun()

    # Input fields for a single experience entry
    with st.container(border=True):
        st.markdown("##### Add New Experience Entry")
        
        col_c, col_r = st.columns(2)
        
        current_year = date.today().year
        year_options = [str(y) for y in range(current_year, 1950, -1)]
        
        with col_c:
            company_val = st.text_input(
                "Company Name", 
                value=st.session_state.get("temp_exp_company_key", ""),
                key="temp_exp_company_key", # The key holds the current widget value
                placeholder="e.g., Google"
            )
            
        with col_r:
            role_val = st.text_input(
                "Role/Title", 
                value=st.session_state.get("temp_exp_role_key", ""),
                key="temp_exp_role_key", 
                placeholder="e.g., Data Scientist"
            )

        col_fy, col_ty, col_c3 = st.columns(3)
        
        current_from_year = st.session_state.get("temp_exp_from_year_key", str(current_year))
        current_to_year = st.session_state.get("temp_exp_to_year_key", "Present")
        
        with col_fy:
            from_year_options = year_options
            try:
                from_year_index = from_year_options.index(current_from_year) 
            except ValueError:
                from_year_index = 0
            
            st.selectbox(
                "From Year", 
                options=from_year_options, 
                index=from_year_index, 
                key="temp_exp_from_year_key"
            )
            
        with col_ty:
            to_year_options = ["Present"] + year_options
            try:
                to_year_index = to_year_options.index(current_to_year)
            except ValueError:
                to_year_index = 0
            
            st.selectbox(
                "To Year", 
                options=to_year_options, 
                index=to_year_index,
                key="temp_exp_to_year_key"
            )
            
        with col_c3:
            ctc_val = st.text_input(
                "CTC (Annual)", 
                value=st.session_state.get("temp_exp_ctc_key", ""),
                key="temp_exp_ctc_key", 
                placeholder="e.g., $150k / 20L INR"
            )

        responsibilities_val = st.text_area(
            "Key Responsibilities/Achievements (Brief summary)", 
            value=st.session_state.get("temp_exp_responsibilities_key", ""),
            height=70, 
            key="temp_exp_responsibilities_key"
        )
        
        st.button("➕ Add This Experience", on_click=add_experience_entry, use_container_width=True, type="secondary")

    # Display current experience entries
    st.markdown("##### Current Experience Entries")
    if st.session_state.cv_form_data['structured_experience']:
        for i, entry in enumerate(st.session_state.cv_form_data['structured_experience']):
            
            expander_title = f"{entry['role']} at {entry['company']} ({entry['from_year']} - {entry['to_year']})"
            
            with st.expander(expander_title, expanded=False):
                col_disp_1, col_disp_2, col_disp_3 = st.columns([1, 1, 0.5])
                col_disp_1.markdown(f"**Role:** {entry['role']}")
                col_disp_2.markdown(f"**Duration:** {entry['from_year']} - {entry['to_year']}")
                col_disp_3.markdown(f"**CTC:** {entry['ctc']}")
                st.markdown(f"**Responsibilities:** {entry['responsibilities']}")
                
                st.button("❌ Remove", key=f"remove_exp_{i}", on_click=remove_experience_entry, args=(i,), type="secondary") 
    else:
        st.info("No experience entries added yet.")
    
    # --- CV Preview and Download ---
    st.markdown("---")
    st.subheader("9. Loaded CV Data Preview and Download")
    
    if st.session_state.get('parsed', {}).get('name') and st.session_state.parsed.get('name') != "":
        
        EXCLUDE_KEYS_PREVIEW = ["structured_experience", "structured_certifications", "structured_education"]
        filled_data_for_preview = {
            k: v for k, v in st.session_state.parsed.items() 
            if v and k not in EXCLUDE_KEYS_PREVIEW and (isinstance(v, str) and v.strip() or isinstance(v, list) and v)
        }
        
        tab_markdown, tab_json, tab_pdf = st.tabs(["📝 Markdown View", "💾 JSON View", "⬇️ PDF/HTML Download"])

        with tab_markdown:
            cv_markdown_preview = format_parsed_json_to_markdown(filled_data_for_preview)
            st.markdown(cv_markdown_preview)

            st.download_button(
                label="⬇️ Download CV as Markdown (.md)",
                data=cv_markdown_preview,
                file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_CV_Document.md",
                mime="text/markdown",
                key="download_cv_markdown_final"
            )

        with tab_json:
            st.json(filled_data_for_preview)
            st.info("This is the raw, structured data used by the AI tools.")

            json_output = json.dumps(filled_data_for_preview, indent=2)
            st.download_button(
                label="⬇️ Download CV as JSON File",
                data=json_output,
                file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_CV_Data.json",
                mime="application/json",
                key="download_cv_json_final"
            )

        with tab_pdf:
            st.markdown("### Download CV as HTML (Print-to-PDF)")
            st.info("Click the button below to download an HTML file. Open the file in your browser and use the browser's **'Print'** function, selecting **'Save as PDF'** to create your final CV document.")
            
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
        st.info("Please fill out the form above and click 'Generate and Load ALL CV Data' or parse a resume in the 'Resume Parsing' tab to see the preview and download options.")


def filter_jd_tab_content():
    # Content identical to the previous response's filter_jd_tab_content
    st.header("🔍 Filter Job Descriptions by Criteria")
    st.markdown("Use the filters below to narrow down your saved Job Descriptions.")

    if not st.session_state.candidate_jd_list:
        st.info("No Job Descriptions are currently loaded. Please add JDs in the 'JD Management' tab.")
        if 'filtered_jds_display' not in st.session_state:
            st.session_state.filtered_jds_display = []
        return
    
    unique_roles = sorted(list(set(
        [item.get('role', 'General Analyst') for item in st.session_state.candidate_jd_list] + DEFAULT_ROLES
    )))
    unique_job_types = sorted(list(set(
        [item.get('job_type', 'Full-time') for item in st.session_state.candidate_jd_list] + DEFAULT_JOB_TYPES
    )))
    
    STARTER_KEYWORDS = {
        "Python", "MySQL", "GCP", "cloud computing", "ML", 
        "API services", "LLM integration", "JavaScript", "SQL", "AWS" 
    }
    
    all_unique_skills = set(STARTER_KEYWORDS)
    for jd in st.session_state.candidate_jd_list:
        valid_skills = [
            skill.strip() for skill in jd.get('key_skills', []) 
            if isinstance(skill, str) and skill.strip()
        ]
        all_unique_skills.update(valid_skills)
    
    unique_skills_list = sorted(list(all_unique_skills))
    
    if not unique_skills_list:
        unique_skills_list = ["No skills extracted from current JDs"]

    all_jd_data = st.session_state.candidate_jd_list

    with st.form(key="jd_filter_form"):
        st.markdown("### Select Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_skills = st.multiselect(
                "Skills Keywords (Select multiple)",
                options=unique_skills_list,
                default=st.session_state.get('last_selected_skills', []),
                key="candidate_filter_skills_multiselect", 
                help="Select one or more skills. JDs containing ANY of the selected skills will be shown."
            )
            
        with col2:
            selected_job_type = st.selectbox(
                "Job Type",
                options=["All Job Types"] + unique_job_types,
                index=0, 
                key="filter_job_type_select"
            )
            
        with col3:
            selected_role = st.selectbox(
                "Role Title",
                options=["All Roles"] + unique_roles,
                index=0, 
                key="filter_role_select"
            )

        apply_filters_button = st.form_submit_button("✅ Apply Filters", type="primary", use_container_width=True)

    if apply_filters_button:
        st.session_state.last_selected_skills = selected_skills

        filtered_jds = []
        selected_skills_lower = [k.strip().lower() for k in selected_skills]
        
        for jd in all_jd_data:
            jd_role = jd.get('role', 'General Analyst')
            jd_job_type = jd.get('job_type', 'Full-time')
            jd_key_skills = [
                s.lower() for s in jd.get('key_skills', []) 
                if isinstance(s, str) and s.strip()
            ]
            
            role_match = (selected_role == "All Roles") or (selected_role == jd_role)
            job_type_match = (selected_job_type == "All Job Types") or (selected_job_type == jd_job_type)
            
            skill_match = True
            if selected_skills_lower:
                if not any(k in jd_key_skills for k in selected_skills_lower):
                    skill_match = False
            
            if role_match and job_type_match and skill_match:
                filtered_jds.append(jd)
                
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

        st.markdown("##### Detailed View")
        for idx, jd in enumerate(filtered_jds, 1):
            with st.expander(f"JD {idx}: {jd['name'].replace('--- Simulated JD for: ', '')} - ({jd.get('role', 'N/A')})"):
                st.markdown(f"**Job Type:** {jd.get('job_type', 'N/A')}")
                st.markdown(f"**Extracted Skills:** {', '.join(jd.get('key_skills', ['N/A']))}")
                st.markdown("---")
                st.text(jd['content'])
    elif st.session_state.candidate_jd_list and apply_filters_button:
        st.info("No Job Descriptions match the selected criteria. Try broadening your filter selections.")
    elif st.session_state.candidate_jd_list and not apply_filters_button:
        st.info("Use the filters above and click **'Apply Filters'** to view matching Job Descriptions.")


# ==============================================================================
# 3. MAIN CANDIDATE DASHBOARD FUNCTION
# ==============================================================================

def candidate_dashboard():
    st.set_page_config(
        page_title="Candidate AI Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.header("👩‍🎓 Candidate AI Dashboard")
    st.markdown("Welcome! Use the tabs below to manage your CV and access AI preparation tools.")

    # --- Session State Initialization (CRITICAL BLOCK) ---
    if 'page' not in st.session_state: st.session_state.page = "login"
    if 'parsed' not in st.session_state: st.session_state.parsed = {}
    if 'full_text' not in st.session_state: st.session_state.full_text = ""
    if 'excel_data' not in st.session_state: st.session_state.excel_data = None
    if 'qa_answer_resume' not in st.session_state: st.session_state.qa_answer_resume = ""
    if 'qa_answer_jd' not in st.session_state: st.session_state.qa_answer_jd = ""
    if 'iq_output' not in st.session_state: st.session_state.iq_output = ""
    if 'jd_fit_output' not in st.session_state: st.session_state.jd_fit_output = ""
    if 'candidate_jd_list' not in st.session_state: st.session_state.candidate_jd_list = []
    if 'candidate_match_results' not in st.session_state: st.session_state.candidate_match_results = []
    if 'candidate_uploaded_resumes' not in st.session_state: st.session_state.candidate_uploaded_resumes = []
    if 'pasted_cv_text' not in st.session_state: st.session_state.pasted_cv_text = "" 
    if 'interview_qa' not in st.session_state: st.session_state.interview_qa = [] 
    if 'evaluation_report' not in st.session_state: st.session_state.evaluation_report = ""
    
    # Initialize main cv_form_data structure
    if "cv_form_data" not in st.session_state: 
        st.session_state.cv_form_data = {
            "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
            "skills": [], "experience": [], "education": [], "certifications": [], 
            "projects": [], "strength": [], "personal_details": "",
            "structured_experience": [], 
            "structured_certifications": [],
            "structured_education": [] 
        }
    
    # Initialize temp data structures 
    current_year = date.today().year
    
    # Initialize widget keys for the "Add New Education Entry" form (NEW)
    if "temp_edu_degree_key" not in st.session_state: st.session_state["temp_edu_degree_key"] = ""
    if "temp_edu_college_key" not in st.session_state: st.session_state["temp_edu_college_key"] = ""
    if "temp_edu_university_key" not in st.session_state: st.session_state["temp_edu_university_key"] = ""
    if "temp_edu_from_year_key" not in st.session_state: st.session_state["temp_edu_from_year_key"] = str(current_year)
    if "temp_edu_to_year_key" not in st.session_state: st.session_state["temp_edu_to_year_key"] = "Present"
    if "temp_edu_score_key" not in st.session_state: st.session_state["temp_edu_score_key"] = ""
    if "temp_edu_type_key" not in st.session_state: st.session_state["temp_edu_type_key"] = "CGPA"
    
    # Initialize widget keys for the "Add New Experience Entry" form
    if "temp_exp_company_key" not in st.session_state: st.session_state["temp_exp_company_key"] = ""
    if "temp_exp_role_key" not in st.session_state: st.session_state["temp_exp_role_key"] = ""
    if "temp_exp_from_year_key" not in st.session_state: st.session_state["temp_exp_from_year_key"] = str(current_year)
    if "temp_exp_to_year_key" not in st.session_state: st.session_state["temp_exp_to_year_key"] = "Present"
    if "temp_exp_ctc_key" not in st.session_state: st.session_state["temp_exp_ctc_key"] = ""
    if "temp_exp_responsibilities_key" not in st.session_state: st.session_state["temp_exp_responsibilities_key"] = ""

    # Initialize widget keys for the "Add New Certification Entry" form
    if "temp_cert_title_key" not in st.session_state: st.session_state["temp_cert_title_key"] = ""
    if "temp_cert_given_by_key" not in st.session_state: st.session_state["temp_cert_given_by_key"] = ""
    if "temp_cert_issue_date_key" not in st.session_state: st.session_state["temp_cert_issue_date_key"] = str(date.today().year)
        
    if "candidate_filter_skills_multiselect" not in st.session_state:
        st.session_state.candidate_filter_skills_multiselect = []
    if "filtered_jds_display" not in st.session_state:
        st.session_state.filtered_jds_display = []
    if "last_selected_skills" not in st.session_state:
        st.session_state.last_selected_skills = []
        
    # --- END Session State Initialization ---

    # --- NAVIGATION BLOCK (Sidebar) ---
    with st.sidebar:
        st.header("Resume/CV Status")
        
        if st.session_state.parsed.get("name") and st.session_state.parsed.get('name') != "":
            st.success(f"Currently loaded: **{st.session_state.parsed['name']}**")
        elif st.session_state.full_text:
            st.warning("Resume content is loaded (raw text).")
        else:
            st.info("Please upload a file or use the CV builder.")
            
        st.markdown("---")
        if st.button("🚪 Log Out", key="candidate_logout_btn", use_container_width=True, type="secondary"): 
            go_to("login") 
    # --- END NAVIGATION BLOCK ---
    
    # Main Content Tabs (REARRANGED TABS)
    # ---------------------------------------------------------------------------------
    tab_cv_mgmt, tab_parsing, tab_jd_mgmt, tab_batch_match, tab_filter_jd, tab_chatbot, tab_interview_prep = st.tabs([
        "✍️ CV Management",          
        "📄 Resume Parsing",         
        "📚 JD Management",          
        "🎯 Batch JD Match",         
        "🔍 Filter JD",              
        "💬 Resume/JD Chatbot (Q&A)",
        "❓ Interview Prep"           
    ])
    
    is_resume_parsed = bool(st.session_state.get('parsed', {}).get('name')) and st.session_state.parsed.get('name') != ""
    
    # --- TAB 0: CV Management ---
    with tab_cv_mgmt:
        cv_management_tab_content()

    # --- TAB 1: Resume Parsing ---
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
            
            st.markdown(
                """
                <div style='font-size: 10px; color: grey;'>
                Supported File Types: PDF, DOCX, TXT, JSON, MARKDOWN, CSV, XLSX, RTF
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.markdown("---")

            if uploaded_file is not None:
                if not st.session_state.candidate_uploaded_resumes or st.session_state.candidate_uploaded_resumes[0].name != uploaded_file.name:
                    st.session_state.candidate_uploaded_resumes = [uploaded_file] 
                    st.session_state.pasted_cv_text = ""
                    st.toast("Resume file uploaded successfully.")
            elif st.session_state.candidate_uploaded_resumes and uploaded_file is None:
                st.session_state.candidate_uploaded_resumes = []
                st.session_state.parsed = {}
                st.session_state.full_text = ""
                st.toast("Upload cleared.")
            
            file_to_parse = st.session_state.candidate_uploaded_resumes[0] if st.session_state.candidate_uploaded_resumes else None
            
            st.markdown("### 2. Parse Uploaded File")
            
            if file_to_parse:
                if st.button(f"Parse and Load: **{file_to_parse.name}**", key="parse_file_btn", use_container_width=True): 
                    with st.spinner(f"Parsing {file_to_parse.name}..."):
                        result = parse_and_store_resume(file_to_parse, file_name_key='single_resume_candidate', source_type='file')
                        
                        if "error" not in result:
                            st.session_state.parsed = result['parsed']
                            st.session_state.full_text = result['full_text']
                            st.session_state.excel_data = result['excel_data'] 
                            st.session_state.parsed['name'] = result['name'] 
                            st.session_state.cv_form_data = st.session_state.parsed.copy() 
                            clear_interview_state()
                            st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                            st.info("View, edit, and download the parsed data in the **CV Management** tab.") 
                        else:
                            st.error(f"Parsing failed for {file_to_parse.name}: {result['error']}")
                            st.session_state.parsed = {"error": result['error'], "name": result['name']}
                            st.session_state.full_text = result['full_text'] or ""
            else:
                st.info("No resume file is currently uploaded. Please upload a file above.")

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
                if st.button("Parse and Load Pasted Text", key="parse_text_btn", use_container_width=True): 
                    with st.spinner("Parsing pasted text..."):
                        st.session_state.candidate_uploaded_resumes = []
                        
                        result = parse_and_store_resume(pasted_text, file_name_key='single_resume_candidate', source_type='text')
                        
                        if "error" not in result:
                            st.session_state.parsed = result['parsed']
                            st.session_state.full_text = result['full_text']
                            st.session_state.excel_data = result['excel_data'] 
                            st.session_state.parsed['name'] = result['name'] 
                            st.session_state.cv_form_data = st.session_state.parsed.copy() 
                            clear_interview_state()
                            st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                            st.info("View, edit, and download the parsed data in the **CV Management** tab.") 
                        else:
                            st.error(f"Parsing failed: {result['error']}")
                            st.session_state.parsed = {"error": result['error'], "name": result['name']}
                            st.session_state.full_text = result['full_text'] or ""
            else:
                st.info("Please paste your CV text into the box above.")

    # --- TAB 2: JD Management ---
    with tab_jd_mgmt:
        st.header("📚 Manage Job Descriptions for Matching")
        st.markdown("Add multiple JDs here to compare your resume against them in the next tabs.")
        
        jd_type = st.radio("Select JD Type", ["Single JD", "Multiple JD"], key="jd_type_candidate")
        st.markdown("### Add JD by:")
        
        method = st.radio("Choose Method", ["Upload File", "Paste Text", "LinkedIn URL"], key="jd_add_method_candidate") 

        # URL
        if method == "LinkedIn URL":
            url_list = st.text_area(
                "Enter one or more URLs (comma separated)" if jd_type == "Multiple JD" else "Enter URL", key="url_list_candidate"
            )
            if st.button("Add JD(s) from URL", key="add_jd_url_btn_candidate"):
                if url_list:
                    urls = [u.strip() for u in url_list.split(",")] if jd_type == "Multiple JD" else [url_list.strip()]
                    
                    count = 0
                    for url in urls:
                        if not url: continue
                        
                        with st.spinner(f"Attempting JD extraction and metadata analysis for: {url}"):
                            jd_text = extract_jd_from_linkedin_url(url)
                            metadata = extract_jd_metadata(jd_text)
                        
                        name_base = url.split('/jobs/view/')[-1].split('/')[0] if '/jobs/view/' in url else f"URL {count+1}"
                        name = f"JD from URL: {name_base}" 
                        if name in [item['name'] for item in st.session_state.candidate_jd_list]:
                            name = f"JD from URL: {name_base} ({len(st.session_state.candidate_jd_list) + 1})" 

                        st.session_state.candidate_jd_list.append({"name": name, "content": jd_text, **metadata})
                        
                        if not jd_text.startswith("[Error"):
                            count += 1
                                
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully! Check the display below for the extracted content.")
                    else:
                        st.error("No JDs were added successfully.")


        # Paste Text
        elif method == "Paste Text":
            text_list = st.text_area(
                "Paste one or more JD texts (separate by '---')" if jd_type == "Multiple JD" else "Paste JD text here", key="text_list_candidate"
            )
            if st.button("Add JD(s) from Text", key="add_jd_text_btn_candidate"):
                if text_list:
                    texts = [t.strip() for t in text_list.split("---")] if jd_type == "Multiple JD" else [text_list.strip()]
                    for i, text in enumerate(texts):
                         if text:
                            name_base = text.splitlines()[0].strip()
                            if len(name_base) > 30: name_base = f"{name_base[:27]}..."
                            if not name_base: name_base = f"Pasted JD {len(st.session_state.candidate_jd_list) + i + 1}"
                            
                            metadata = extract_jd_metadata(text)
                            st.session_state.candidate_jd_list.append({"name": name_base, "content": text, **metadata})
                    st.success(f"✅ {len(texts)} JD(s) added successfully!")

        # Upload File
        elif method == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload JD file(s)",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=(jd_type == "Multiple JD"), 
                key="jd_file_uploader_candidate"
            )
            if st.button("Add JD(s) from File", key="add_jd_file_btn_candidate"):
                files_to_process = uploaded_files if isinstance(uploaded_files, list) else ([uploaded_files] if uploaded_files else [])
                if files_to_process:
                    st.session_state.candidate_jd_list.append({"name": files_to_process[0].name, "content": "Simulated JD Text", "role": "Simulated Role", "job_type": "Full-time", "key_skills": ["Stub"]})
                    st.success(f"Simulated addition of {len(files_to_process)} JD(s).")
                else:
                    st.warning("Please upload file(s).")


        # Display Added JDs
        if st.session_state.candidate_jd_list:
            
            col_display_header, col_clear_button = st.columns([3, 1])
            
            with col_display_header:
                st.markdown("### ✅ Current JDs Added:")
                
            with col_clear_button:
                if st.button("🗑️ Clear All JDs", key="clear_jds_candidate", use_container_width=True, help="Removes all currently loaded JDs.", type="secondary"): 
                    st.session_state.candidate_jd_list = []
                    st.session_state.candidate_match_results = [] 
                    st.session_state.filtered_jds_display = [] 
                    st.success("All JDs and associated match results have been cleared.")
                    st.rerun() 

            for idx, jd_item in enumerate(st.session_state.candidate_jd_list, 1):
                title = jd_item['name']
                display_title = title.replace("--- Simulated JD for: ", "")
                with st.expander(f"JD {idx}: {display_title} | Role: {jd_item.get('role', 'N/A')}"):
                    st.markdown(f"**Job Type:** {jd_item.get('job_type', 'N/A')} | **Key Skills:** {', '.join(jd_item.get('key_skills', ['N/A']))}")
                    st.markdown("---")
                    st.text(jd_item['content'])
        else:
            st.info("No Job Descriptions added yet.")

    # --- TAB 4: Batch JD Match ---
    with tab_batch_match:
        st.header("🎯 Batch JD Match: Best Matches")
        st.markdown("Compare your current resume against all saved job descriptions.")

        if not is_resume_parsed:
            st.warning("Please upload and parse your resume in the 'Resume Parsing' tab or build your CV in the 'CV Management' tab first.")
        
        elif not st.session_state.candidate_jd_list:
            st.error("Please add Job Descriptions in the 'JD Management' tab before running batch analysis.")
            
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
            
            jds_to_match = [
                jd_item for jd_item in st.session_state.candidate_jd_list 
                if jd_item['name'] in selected_jd_names
            ]
            
            if st.button(f"Run Match Analysis on {len(jds_to_match)} Selected JD(s)", key='run_match_analysis_btn', type="primary"): 
                st.session_state.candidate_match_results = []
                
                if not jds_to_match:
                    st.warning("Please select at least one Job Description to run the analysis.")
                    
                else:
                    results_with_score = []
                    
                    with st.spinner(f"Matching resume against {len(jds_to_match)} selected JD(s)..."):
                        
                        for jd_item in jds_to_match:
                            jd_name = jd_item['name']
                            jd_content = jd_item['content']

                            try:
                                fit_output = evaluate_jd_fit(jd_content, st.session_state.parsed)
                                
                                score_match = re.search(r'Overall Fit Score:\s*(\d+)/10', fit_output)
                                skills_match = re.search(r'Skills Match:\s*(\d+)%', fit_output)
                                experience_match = re.search(r'Experience Match:\s*(\d+)%', fit_output)
                                
                                overall_score = score_match.group(1) if score_match else 'N/A'
                                
                                results_with_score.append({
                                    "jd_name": jd_name,
                                    "overall_score": overall_score,
                                    "numeric_score": int(overall_score) if overall_score.isdigit() else -1,
                                    "skills_percent": skills_match.group(1) if skills_match else 'N/A',
                                    "experience_percent": experience_match.group(1) if experience_match else 'N/A', 
                                    "education_percent": '90' if overall_score.isdigit() else 'N/A',   
                                    "full_analysis": fit_output
                                })
                            except Exception as e:
                                results_with_score.append({"jd_name": jd_name, "overall_score": "Error", "numeric_score": -1, "skills_percent": "Error", "experience_percent": "Error", "education_percent": "Error", "full_analysis": f"Error: {e}"})
                                
                        # Ranking Logic
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
                        st.success("Batch analysis complete!")


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
                        st.markdown(item['full_analysis'])

    # --- TAB 5: Filter JD ---
    with tab_filter_jd:
        filter_jd_tab_content()

    # --- TAB 6: Resume/JD Chatbot (Q&A) ---
    with tab_chatbot:
        st.header("Resume/JD Chatbot (Q&A) 💬")
        
        sub_tab_resume, sub_tab_jd = st.tabs([
            "👤 Chat about Your Resume",
            "📄 Chat about Saved JDs"
        ])
        
        # --- RESUME CHATBOT CONTENT ---
        with sub_tab_resume:
            st.markdown("### Ask any question about the currently loaded resume.")
            if not is_resume_parsed:
                st.warning("Please upload and parse a resume in the 'Resume Parsing' tab or use the 'CV Management' tab first.")
            else: 
                
                if 'qa_answer_resume' not in st.session_state: st.session_state.qa_answer_resume = ""
                
                question = st.text_input(
                    "Your Question (about Resume)", 
                    placeholder="e.g., What are the candidate's key skills?",
                    key="resume_qa_question"
                )
                
                if st.button("Get Answer (Resume)", key="qa_btn_resume"):
                    with st.spinner("Generating answer..."):
                        try:
                            answer = qa_on_resume(question) 
                            st.session_state.qa_answer_resume = answer
                        except Exception as e:
                            st.error(f"Error during Resume Q&A: {e}")
                            st.session_state.qa_answer_resume = "Could not generate an answer."

                if st.session_state.get('qa_answer_resume'):
                    st.text_area("Answer (Resume)", st.session_state.qa_answer_resume, height=150, key='resume_qa_answer_display')
        
        # --- JD CHATBOT CONTENT ---
        with sub_tab_jd:
            st.markdown("### Ask any question about a saved Job Description.")
            
            if not st.session_state.candidate_jd_list:
                st.warning("Please add Job Descriptions in the 'JD Management' tab first.")
            else: 
                if 'qa_answer_jd' not in st.session_state: st.session_state.qa_answer_jd = ""

                jd_names = [jd['name'] for jd in st.session_state.candidate_jd_list]
                selected_jd_name = st.selectbox(
                    "Select Job Description to Query",
                    options=jd_names,
                    key="jd_qa_select"
                )
                
                question = st.text_input(
                    "Your Question (about JD)", 
                    placeholder="e.g., What is the minimum experience required for this role?",
                    key="jd_qa_question"
                )
                
                if st.button("Get Answer (JD)", key="qa_btn_jd"):
                    if selected_jd_name and question.strip():
                        with st.spinner(f"Generating answer for {selected_jd_name}..."):
                            try:
                                answer = qa_on_jd(question, selected_jd_name) 
                                st.session_state.qa_answer_jd = answer
                            except Exception as e:
                                st.error(f"Error during JD Q&A: {e}")
                                st.session_state.qa_answer_jd = "Could not generate an answer."
                    else:
                        st.error("Please select a JD and enter a question.")

                if st.session_state.get('qa_answer_jd'):
                    st.text_area("Answer (JD)", st.session_state.qa_answer_jd, height=150, key='jd_qa_answer_display')


    # --- TAB 7: Interview Prep ---
    with tab_interview_prep:
        st.header("Interview Preparation Tools")
        if not is_resume_parsed or "error" in st.session_state.parsed:
            st.warning("Please upload and successfully parse a resume first.")
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
                    try:
                        raw_questions_response = generate_interview_questions(st.session_state.parsed, section_choice) 
                        st.session_state.iq_output = raw_questions_response
                        
                        st.session_state.interview_qa = [] 
                        st.session_state.evaluation_report = "" 
                        
                        # Parsing logic for generated questions (assuming the LLM format is followed)
                        q_list = []
                        current_level = "Generic"
                        for line in raw_questions_response.splitlines():
                            line = line.strip()
                            if line.startswith('[') and line.endswith(']'):
                                current_level = line.strip('[]')
                            elif line.lower().startswith('q') and ':' in line:
                                question_text = line[line.find(':') + 1:].strip()
                                q_list.append({"question": f"({current_level}) {question_text}", "answer": "", "level": current_level})
                                
                        st.session_state.interview_qa = q_list
                        st.success(f"Generated {len(q_list)} questions based on your **{section_choice}** section.")
                        
                    except Exception as e:
                        st.error(f"Error generating questions: {e}")
                        st.session_state.iq_output = "Error generating questions."
                        st.session_state.interview_qa = []

            if st.session_state.get('interview_qa'):
                st.markdown("---")
                st.subheader("2. Practice and Record Answers")
                
                with st.form("interview_practice_form"):
                    
                    for i, qa_item in enumerate(st.session_state.interview_qa):
                        st.markdown(f"**Question {i+1}:** {qa_item['question']}")
                        
                        answer = st.text_area(
                            f"Your Answer for Q{i+1}", 
                            value=st.session_state.interview_qa[i]['answer'], 
                            height=100,
                            key=f'answer_q_{i}',
                            label_visibility='collapsed'
                        )
                        st.session_state.interview_qa[i]['answer'] = answer 
                        st.markdown("---") 
                        
                    submit_button = st.form_submit_button("Submit & Evaluate Answers", use_container_width=True, type="secondary")

                    if submit_button:
                        if all(item['answer'].strip() for item in st.session_state.interview_qa):
                            with st.spinner("Sending answers to AI Evaluator..."):
                                try:
                                    report = evaluate_interview_answers(
                                        st.session_state.interview_qa,
                                        st.session_state.parsed
                                    ) 
                                    st.session_state.evaluation_report = report
                                    st.success("Evaluation complete! See the report below.")
                                except Exception as e:
                                    st.error(f"Evaluation failed: {e}")
                                    st.session_state.evaluation_report = f"Evaluation failed: {e}\n{traceback.format_exc()}"
                        else:
                            st.error("Please answer all generated questions before submitting.")
                
                if st.session_state.get('evaluation_report'):
                    st.markdown("---")
                    st.subheader("3. AI Evaluation Report")
                    st.markdown(st.session_state.evaluation_report)

# ==============================================================================
# 4. MAIN EXECUTION BLOCK (CRITICAL FOR STREAMLIT)
# ==============================================================================

if __name__ == '__main__':
    if 'page' not in st.session_state:
        st.session_state.page = "dashboard"
    
    if st.session_state.page == "dashboard":
        candidate_dashboard()
    else:
        st.title("Welcome to the Candidate Dashboard")
        if st.button("Start Dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
