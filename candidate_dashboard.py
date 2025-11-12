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
    
def generate_education_string(entry: Dict[str, str]) -> str:
    """Formats a structured education entry into a single string for storage."""
    degree = entry.get('degree', 'N/A')
    college = entry.get('college', 'N/A')
    university = entry.get('university', 'N/A')
    from_year = entry.get('from_year', 'N/A')
    to_year = entry.get('to_year', 'Present')
    
    if from_year == to_year:
        duration = f"({from_year})"
    elif from_year != 'N/A' and to_year != 'N/A':
        duration = f"({from_year} - {to_year})"
    else:
        duration = ""

    # Example format: "M.Sc. Computer Science from University of Excellence, 2018"
    # Example format: "B.Tech. in Electrical Engg. (2014-2018) | College of Technology | University of Example"
    return f"{degree} {duration} | {college} | {university}"


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
    """Stub: Simulates parsing and stores results into a structure."""
    if st.session_state.get('parsed', {}).get('name') and st.session_state.parsed.get('name') != "":
         return {"parsed": st.session_state.parsed, "full_text": st.session_state.full_text, "excel_data": None, "name": st.session_state.parsed['name']}
    
    # Placeholder data for a fresh parse
    if source_type == 'file':
        name_from_file = getattr(file_input, 'name', 'Uploaded_Resume').split('.')[0].replace('_', ' ')
    else:
        name_from_file = "Parsed Text CV"

    parsed_data = {
        "name": name_from_file, 
        "email": "candidate@example.com", 
        "phone": "555-123-4567",
        "linkedin": "linkedin.com/in/candidate", 
        "github": "github.com/candidate",
        "skills": ["Python", "SQL", "Streamlit", "Data Analysis", "Git"], 
        "experience": ["5 years at TechCorp as a Data Analyst, focusing on ETL and reporting."], 
        "education": [
            "M.Sc. Computer Science (2016 - 2018) | University of Excellence | City University",
            "B.Tech. Information Technology (2012 - 2016) | College of Engineering | State University"
        ], 
        "certifications": ["AWS Certified Cloud Practitioner"], 
        "projects": ["Built this Streamlit Dashboard"], 
        "strength": ["Problem Solver", "Quick Learner"], 
        "personal_details": "Highly motivated and results-oriented professional."
    }
    
    # Create a placeholder full_text 
    compiled_text = ""
    for k, v in parsed_data.items():
        if v:
            compiled_text += f"{k.replace('_', ' ').title()}:\n"
            if isinstance(v, list):
                compiled_text += "\n".join([f"- {item}" for item in v]) + "\n\n"
            else:
                compiled_text += str(v) + "\n\n"

    return {"parsed": parsed_data, "full_text": compiled_text, "excel_data": None, "name": parsed_data['name']}

def qa_on_resume(question):
    """Stub: Simulates Q&A on resume."""
    if "skills" in question.lower():
        return f"Based on the resume, the key skills are: {', '.join(st.session_state.parsed.get('skills', ['No skills found']))}. The candidate has a strong background in data tools."
    return f"Based on the resume, the answer to '{question}' is: [Simulated Answer - Check experience/projects section for details]."

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
Education Match: 80%

Strengths/Matches:
- Candidate's Python and SQL skills ({skills}%) are an excellent match for this JD.
- Experience ({experience}%) is relevant, though perhaps slightly under the ideal level.

Gaps/Areas for Improvement:
- Needs more specific experience in the [Niche Technology] mentioned in the JD.
- The resume summary could be tailored more closely to the [Specific Industry] focus of this role.

Overall Summary: This is a **Strong** fit. Focus on experience in the interview.
"""

def generate_interview_questions(parsed_json, section):
    """Stub: Simulates interview question generation."""
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
Overall Summary: The candidate shows **Good** fundamental knowledge. To score higher, better integrate answers with accomplishments listed in the resume (e.g., mention specific projects).
""")
    
    return "\n".join(feedback_parts)

def generate_cv_html(parsed_data):
    """Stub: Simulates CV HTML generation."""
    skills_list = "".join([f"<li>{s}</li>" for s in parsed_data.get('skills', [])])
    education_list = "".join([f"<li>{e}</li>" for e in parsed_data.get('education', [])])
    return f"""
    <html>
    <head>
        <title>{parsed_data.get('name', 'CV Preview')}</title>
        <style>body{{font-family: Arial, sans-serif; margin: 40px;}} h1{{color: #2e6c80; border-bottom: 2px solid #2e6c80;}} h2{{color: #3d99b1;}} ul{{list-style-type: none; padding: 0;}}</style>
    </head>
    <body>
        <h1>{parsed_data.get('name', 'CV Preview')}</h1>
        <p>Email: {parsed_data.get('email', 'N/A')} | Phone: {parsed_data.get('phone', 'N/A')}</p>
        <p>LinkedIn: <a href="{parsed_data.get('linkedin', '#')}">{parsed_data.get('linkedin', 'N/A')}</a></p>
        
        <h2>Key Skills</h2>
        <ul>{skills_list}</ul>
        
        <h2>Experience</h2>
        <p>{' | '.join(parsed_data.get('experience', ['No experience listed']))}</p>
        
        <h2>Education</h2>
        <ul>{education_list}</ul>
        
        <p>Generated by AI Dashboard on {date.today()}</p>
    </body>
    </html>
    """

def format_parsed_json_to_markdown(parsed_data):
    """Stub: Simulates CV Markdown generation."""
    md = f"# **{parsed_data.get('name', 'CV Preview').upper()}**\n"
    md += f"**Contact:** {parsed_data.get('email', 'N/A')} | {parsed_data.get('phone', 'N/A')} | [LinkedIn]({parsed_data.get('linkedin', '#')})\n"
    md += "\n"
    md += f"## **SUMMARY**\n---\n"
    md += parsed_data.get('personal_details', 'No summary provided.') + "\n\n"
    md += "## **SKILLS**\n---\n"
    md += "- " + "\n- ".join(parsed_data.get('skills', ['No skills listed']))
    md += "\n\n## **EXPERIENCE**\n---\n"
    md += "- " + "\n- ".join(parsed_data.get('experience', ['No experience listed']))
    md += "\n\n## **EDUCATION**\n---\n"
    md += "- " + "\n- ".join(parsed_data.get('education', ['No education listed']))
    return md

# ==============================================================================
# 2. TAB CONTENT FUNCTIONS
# ==============================================================================

def cv_management_tab_content():
    st.header("📝 Prepare Your CV")
    st.markdown("### 1. Form Based CV Builder")
    st.info("Fill out the details below to generate a parsed CV that can be used immediately for matching and interview prep, or start by parsing a file in the 'Resume Parsing' tab.")

    # Initialize the parsed data if not already existing
    default_parsed = {
        "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
        "skills": [], "experience": [], "education": [], "certifications": [], 
        "projects": [], "strength": [], "personal_details": ""
    }
    
    if "cv_form_data" not in st.session_state:
        # Load from parsed if it exists
        if st.session_state.get('parsed', {}).get('name') and st.session_state.parsed.get('name') != "":
            st.session_state.cv_form_data = st.session_state.parsed.copy()
        else:
            st.session_state.cv_form_data = default_parsed
            
    # CRITICAL: Ensure education is a list of strings for compatibility
    if not isinstance(st.session_state.cv_form_data.get('education'), list):
         st.session_state.cv_form_data['education'] = []

    
    # --- CV Builder Form (Main Sections) ---
    with st.form("cv_builder_form"):
        st.subheader("Personal & Contact Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.cv_form_data['name'] = st.text_input(
                "Full Name", 
                value=st.session_state.cv_form_data['name'], 
                key="cv_name"
            )
        with col2:
            st.session_state.cv_form_data['email'] = st.text_input(
                "Email Address", 
                value=st.session_state.cv_form_data['email'], 
                key="cv_email"
            )
        with col3:
            st.session_state.cv_form_data['phone'] = st.text_input(
                "Phone Number", 
                value=st.session_state.cv_form_data['phone'], 
                key="cv_phone"
            )
        
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

        skills_text = "\n".join(st.session_state.cv_form_data.get('skills', []))
        new_skills_text = st.text_area(
            "Key Skills (Technical and Soft)", 
            value=skills_text,
            height=150,
            key="cv_skills"
        )
        st.session_state.cv_form_data['skills'] = [s.strip() for s in new_skills_text.split('\n') if s.strip()]
        
        experience_text = "\n".join(st.session_state.cv_form_data.get('experience', []))
        new_experience_text = st.text_area(
            "Professional Experience (Job Roles, Companies, Dates, Key Responsibilities)", 
            value=experience_text,
            height=150,
            key="cv_experience"
        )
        st.session_state.cv_form_data['experience'] = [e.strip() for e in new_experience_text.split('\n') if e.strip()]
        
        certifications_text = "\n".join(st.session_state.cv_form_data.get('certifications', []))
        new_certifications_text = st.text_area(
            "Certifications (Name, Issuing Body, Date)", 
            value=certifications_text,
            height=100,
            key="cv_certifications"
        )
        st.session_state.cv_form_data['certifications'] = [c.strip() for c in new_certifications_text.split('\n') if c.strip()]
        
        projects_text = "\n".join(st.session_state.cv_form_data.get('projects', []))
        new_projects_text = st.text_area(
            "Projects (Name, Description, Technologies)", 
            value=projects_text,
            height=150,
            key="cv_projects"
        )
        st.session_state.cv_form_data['projects'] = [p.strip() for p in new_projects_text.split('\n') if p.strip()]
        
        strength_text = "\n".join(st.session_state.cv_form_data.get('strength', []))
        new_strength_text = st.text_area(
            "Strengths / Key Personal Qualities (One per line)", 
            value=strength_text,
            height=100,
            key="cv_strength"
        )
        st.session_state.cv_form_data['strength'] = [s.strip() for s in new_strength_text.split('\n') if s.strip()]


        submit_form_button = st.form_submit_button("Generate and Load ALL CV Data", type="primary", use_container_width=True)

    # --- Education Input Form (New Structured Input) ---
    st.markdown("---")
    st.subheader("Education (Structured Input)")
    
    # Display existing education entries
    if st.session_state.cv_form_data.get('education'):
        st.markdown("##### Current Education Entries:")
        for i, entry in enumerate(st.session_state.cv_form_data['education']):
            col_entry, col_delete = st.columns([10, 1])
            with col_entry:
                st.write(f"**{i+1}.** {entry}")
            with col_delete:
                if st.button("🗑️", key=f"delete_edu_{i}"):
                    st.session_state.cv_form_data['education'].pop(i)
                    st.success("Entry deleted. Click 'Generate and Load ALL CV Data' to save changes.")
                    st.rerun() 
        st.markdown("---")

    
    with st.form("education_add_form"):
        col_d, col_c = st.columns(2)
        with col_d:
            degree = st.text_input("Degree / Qualification", key="edu_degree")
        with col_c:
            college = st.text_input("College / Institution Name", key="edu_college")
            
        col_u, col_f, col_t = st.columns(3)
        with col_u:
            university = st.text_input("University / Board", key="edu_university")
        with col_f:
            from_year = st.text_input("From Year (e.g., 2014)", key="edu_from_year", max_chars=4)
        with col_t:
            to_year = st.text_input("To Year (e.g., 2018 or Present)", key="edu_to_year", max_chars=7)
            
        # Button logic executed AFTER the form submission
        add_edu_button = st.form_submit_button("➕ Add Education Entry", use_container_width=True)
        
        if add_edu_button:
            if degree and college and from_year:
                new_entry_dict = {
                    'degree': degree,
                    'college': college,
                    'university': university if university else "N/A",
                    'from_year': from_year,
                    'to_year': to_year if to_year else "Present"
                }
                
                # Format the entry into the required string format
                new_entry_string = generate_education_string(new_entry_dict)
                
                # Append to the list and clear inputs by rerunning
                st.session_state.cv_form_data['education'].append(new_entry_string)
                st.success(f"Education entry added: {new_entry_string}. Click 'Generate and Load ALL CV Data' above to use it for AI tools.")
                
                # Clear the education form fields by using a temporary state update and rerun
                st.session_state.edu_degree = ""
                st.session_state.edu_college = ""
                st.session_state.edu_university = ""
                st.session_state.edu_from_year = ""
                st.session_state.edu_to_year = ""
                st.rerun() 
            else:
                st.error("Please fill in at least Degree, College/Institution, and From Year for the education entry.")

    # --- FINAL SUBMISSION LOGIC (for the main form) ---
    if submit_form_button:
        if not st.session_state.cv_form_data['name'] or not st.session_state.cv_form_data['email']:
            st.error("Please fill in at least your **Full Name** and **Email Address**.")
            return

        st.session_state.parsed = st.session_state.cv_form_data.copy()
        
        # Create a placeholder full_text for the AI tools
        compiled_text = ""
        for k, v in st.session_state.cv_form_data.items():
            if v:
                compiled_text += f"{k.replace('_', ' ').title()}:\n"
                if isinstance(v, list):
                    compiled_text += "\n".join([f"- {item}" for item in v]) + "\n\n"
                else:
                    compiled_text += str(v) + "\n\n"
        st.session_state.full_text = compiled_text
        
        st.session_state.candidate_match_results = []
        st.session_state.interview_qa = []
        st.session_state.evaluation_report = ""

        st.success(f"✅ CV data for **{st.session_state.parsed['name']}** successfully generated and loaded! You can now use the Chatbot, Match, and Interview Prep tabs.")
        
    st.markdown("---")
    st.subheader("2. Loaded CV Data Preview and Download")
    
    if st.session_state.get('parsed', {}).get('name') and st.session_state.parsed.get('name') != "":
        
        filled_data_for_preview = {
            k: v for k, v in st.session_state.parsed.items() 
            if v and (isinstance(v, str) and v.strip() or isinstance(v, list) and v)
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
            st.json(st.session_state.parsed)
            st.info("This is the raw, structured data used by the AI tools.")

            json_output = json.dumps(st.session_state.parsed, indent=2)
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
    if "cv_form_data" not in st.session_state: 
        st.session_state.cv_form_data = {
            "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
            "skills": [], "experience": [], "education": [], "certifications": [], 
            "projects": [], "strength": [], "personal_details": ""
        }
    if "candidate_filter_skills_multiselect" not in st.session_state:
        st.session_state.candidate_filter_skills_multiselect = []
    if "filtered_jds_display" not in st.session_state:
        st.session_state.filtered_jds_display = []
    if "last_selected_skills" not in st.session_state:
        st.session_state.last_selected_skills = []
        
    # Initialize fields for the dynamic education form to prevent Rerun errors
    if "edu_degree" not in st.session_state: st.session_state.edu_degree = ""
    if "edu_college" not in st.session_state: st.session_state.edu_college = ""
    if "edu_university" not in st.session_state: st.session_state.edu_university = ""
    if "edu_from_year" not in st.session_state: st.session_state.edu_from_year = ""
    if "edu_to_year" not in st.session_state: st.session_state.edu_to_year = ""
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
        if st.button("🚪 Log Out", key="candidate_logout_btn", use_container_width=True):
            go_to("login") 
    # --- END NAVIGATION BLOCK ---
    
    # Main Content Tabs (REARRANGED TABS)
    # ---------------------------------------------------------------------------------
    tab_cv_mgmt, tab_parsing, tab_jd_mgmt, tab_batch_match, tab_filter_jd, tab_chatbot, tab_interview_prep = st.tabs([
        "✍️ CV Management",          # 1. CV Builder/Preview
        "📄 Resume Parsing",         # 2. File/Text Upload
        "📚 JD Management",          # 3. Add JDs
        "🎯 Batch JD Match",         # 4. Compare CV to JDs
        "🔍 Filter JD",              # 5. Filter Saved JDs
        "💬 Resume/JD Chatbot (Q&A)",# 6. Chatbot
        "❓ Interview Prep"           # 7. Q&A Practice
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
                # Logic to handle new upload vs. existing one
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
                if st.button("🗑️ Clear All JDs", key="clear_jds_candidate", use_container_width=True, help="Removes all currently loaded JDs."):
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

    # --- TAB 3: Batch JD Match ---
    with tab_batch_match:
        st.header("🎯 Batch JD Match: Best Matches")
        st.markdown("Compare your current resume against all saved job descriptions.")

        if not is_resume_parsed:
            st.warning("Please **upload and parse your resume** in the 'Resume Parsing' tab or **build your CV** in the 'CV Management' tab first.")
        
        elif not st.session_state.candidate_jd_list:
            st.error("Please **add Job Descriptions** in the 'JD Management' tab before running batch analysis.")
            
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
            
            if st.button(f"Run Match Analysis on {len(jds_to_match)} Selected JD(s)"):
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
                                    "education_percent": '80' if overall_score.isdigit() else 'N/A',   
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

    # --- TAB 4: Filter JD ---
    with tab_filter_jd:
        filter_jd_tab_content()

    # --- TAB 5: Resume/JD Chatbot (Q&A) ---
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
                    st.text_area("Answer (Resume)", st.session_state.qa_answer_resume, height=150)
        
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
                    st.text_area("Answer (JD)", st.session_state.qa_answer_jd, height=150)


    # --- TAB 6: Interview Prep ---
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
                        
                    submit_button = st.form_submit_button("Submit & Evaluate Answers", use_container_width=True)

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
    # Add a login/landing page if needed, otherwise, run the dashboard directly
    # A simple example:
    if 'page' not in st.session_state:
        st.session_state.page = "dashboard"
    
    if st.session_state.page == "dashboard":
        candidate_dashboard()
    else:
        # Simple placeholder for a 'login' or landing page
        st.title("Welcome to the Candidate Dashboard")
        if st.button("Start Dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
