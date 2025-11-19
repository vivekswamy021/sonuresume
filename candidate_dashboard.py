import streamlit as st
import os
import pdfplumber
import docx
import json
import traceback
import re 
from dotenv import load_dotenv 
from io import BytesIO 
import pandas as pd
import base64 

# --- CONFIGURATION & API SETUP ---

GROQ_MODEL = "llama-3.1-8b-instant"
# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# --- Default/Mock Data for Filtering ---
DEFAULT_ROLES = ["Data Scientist", "Cloud Engineer", "Software Engineer", "AI/ML Engineer"]
DEFAULT_JOB_TYPES = ["Full-time", "Contract", "Remote"]
STARTER_KEYWORDS = {
    "Python", "MySQL", "GCP", "cloud computing", "ML", 
    "API services", "LLM integration", "JavaScript", "SQL", "AWS", "MLOps", "Data Visualization"
}
# --- End Default/Mock Data ---


# --- Define MockGroqClient globally (Necessary for testing without API Key) ---

class MockGroqClient:
    """Mock client for local testing when Groq is not available or key is missing."""
    def chat(self):
        class Completions:
            def create(self, **kwargs):
                prompt_content = kwargs.get('messages', [{}])[0].get('content', '')
                
                # --- Specific Mock Logic for Interview Prep ---
                if "Generate a list of interview questions" in prompt_content:
                    
                    if "targeting the **JD**" in prompt_content:
                        # JD Based Mock
                        section = "Cloud Engineer"
                        mock_questions_raw = f"""
                        [Basic/Screening]
                        Q1: Describe the core differences between AWS and GCP services related to the JD.
                        
                        [Intermediate/Technical]
                        Q2: Explain how you would implement CI/CD for a project involving Docker and Kubernetes.
                        Q3: What are the key security considerations for the **{section}** role?
                        
                        [Advanced/Behavioral]
                        Q4: Discuss a time you had to troubleshoot a production issue related to infrastructure automation.
                        Q5: How do you prioritize optimizing cloud costs?
                        """
                    else:
                        # Resume Section Based Mock
                        section_match = re.search(r'targeting the \*\*(.+?)\*\* section', prompt_content)
                        section = section_match.group(1).strip() if section_match else "General"
                        
                        mock_questions_raw = f"""
                        [Basic/Screening]
                        Q1: Tell me about your most recent project related to **{section}**.
                        
                        [Intermediate/Technical]
                        Q2: Describe a complex technical challenge you overcame in the **{section}** area.
                        Q3: How do you measure the success of your work in **{section}**?
                        
                        [Advanced/Behavioral]
                        Q4: Give an example of technical debt you encountered related to **{section}** and how you resolved it.
                        Q5: How do you keep up to date with the latest trends in **{section}**?
                        """
                    # Return the raw text as expected by the new parser logic
                    return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': mock_questions_raw})})()]})

                elif "Evaluate the candidate's answers to the following questions" in prompt_content:
                    # Simple mock evaluation logic
                    if "Q2" in prompt_content and "complex technical challenge" in prompt_content:
                        score = 8
                        feedback = "Excellent structure using the STAR method (simulated). You clearly articulated the situation and your actions. **Focus on quantifying the results.**"
                    else:
                        score = 6
                        feedback = "Good technical detail, but the answers were a bit generic (simulated). Try to connect your skills directly to the business impact."

                    mock_evaluation = f"""
                    --- AI Evaluation Report ---
                    
                    **Overall Score:** {score}/10
                    **Summary:** The candidate provided decent technical background but lacked deep, quantifiable examples for most questions. The answer to Q2 was strong.
                    
                    **Q1 Feedback:** {feedback}
                    
                    **Q2 Feedback:** Strong response. Excellent use of technical terms and process.
                    
                    **Q3 Feedback:** Answer was too theoretical. Need a real-world project example.
                    
                    **Next Steps:** Review the job description and prepare more quantifiable achievements related to this area.
                    """
                    return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': mock_evaluation})})()]})

                # --- Gap Analysis Mock Logic ---
                elif "Generate a detailed course plan and suggest relevant certifications" in prompt_content:
                    gap_match = re.search(r'Gaps Identified:\s*(.*)', prompt_content, re.DOTALL)
                    gap_summary = gap_match.group(1).strip() if gap_match else "Missing key skills in Cloud and CI/CD."
                    
                    mock_plan = f"""
                    ## 💡 Detailed Course Plan: Addressing Gaps in Cloud/CI/CD (Simulated)
                    
                    The goal is to cover the identified gaps: **{gap_summary}**.
                    
                    ### Phase 1: Foundational Cloud Skills (4 Weeks)
                    * **Module 1 (AWS/GCP):** Core services (EC2, S3, IAM, VPC). Focus on security best practices.
                    * **Module 2 (IaC):** Introduction to **Terraform** or CloudFormation/Deployment Manager. Hands-on simple infrastructure provisioning.
                    
                    ### Phase 2: Automation & DevOps (6 Weeks)
                    * **Module 3 (CI/CD Principles):** Theory and practice of continuous integration/delivery using **GitLab CI** or Jenkins.
                    * **Module 4 (Containerization):** Advanced Dockerfile creation and multi-container application deployment with Docker Compose.
                    * **Module 5 (Kubernetes Basics):** Deploying and scaling applications using basic K8s objects (Pods, Deployments, Services).
                    
                    ### Phase 3: Project and Certification Prep (4 Weeks)
                    * **Project:** Build a fully automated CI/CD pipeline deploying a microservice to a managed Kubernetes cluster (EKS/GKE).
                    
                    ---
                    
                    ## 🏅 Suggested Certifications
                    
                    * **For AWS Focus:** **AWS Certified Solutions Architect – Associate** (Covers broad cloud knowledge).
                    * **For GCP Focus:** **Google Cloud Professional Cloud Architect** (A high-value certification).
                    * **For DevOps/CI/CD:** **Certified Kubernetes Administrator (CKA)** or **HashiCorp Certified Terraform Associate**.
                    
                    ---
                    **Next Step:** Focus on the **AWS Certified Solutions Architect** path first, as it provides the quickest return on investment for entry to mid-level cloud roles.
                    """
                    return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': mock_plan})})()]})


                # --- Existing Mock Logic (JD Q&A, Resume Q&A, Cover Letter) ---
                elif "Answer the following question about the Job Description concisely and directly." in prompt_content:
                    question_match = re.search(r'Question:\s*(.*)', prompt_content)
                    question = question_match.group(1).strip() if question_match else "a question"
                    
                    if 'role' in question.lower():
                        return type('MockResponse', (object,), {'choices': [type('Message', (object,), {'content': 'The required role in this Job Description is Cloud Engineer.'})()]})
                    elif 'experience' in question.lower():
                        return type('MockResponse', (object,), {'choices': [type('Message', (object,), {'content': 'The job requires 3+ years of experience in AWS/GCP and infrastructure automation.'})()]})
                    else:
                        return type('MockResponse', (object,), {'choices': [type('Message', (object,), {'content': 'Mock answer for JD question: The JD mentions Python and Docker as key skills.'})()]})

                elif "Answer the following question about the resume concisely and directly." in prompt_content:
                    question_match = re.search(r'Question:\s*(.*)', prompt_content)
                    question = question_match.group(1).strip() if question_match else "a question"
                    
                    if 'name' in question.lower():
                        return type('MockResponse', (object,), {'choices': [type('Message', (object,), {'content': 'The candidate\'s name is Vivek Swamy.'})()]})
                    elif 'skills' in question.lower():
                        return type('MockResponse', (object,), {'choices': [type('Message', (object,), {'content': 'Key skills include Python, SQL, AWS, and MLOps.'})()]})
                    else:
                        return type('MockResponse', (object,), {'choices': [type('Message', (object,), {'content': f'Based on the mock resume data, I can provide a simulated answer to your question about {question}.'})()]})

                elif "You are an expert cover letter generator" in prompt_content:
                    role_match = re.search(r'Job Description Role: (.*?)[\.\n]', prompt_content)
                    role = role_match.group(1).strip() if role_match else "Software Engineer"
                    
                    mock_cover_letter = f"""
                    [Date]
                    
                    [Hiring Manager Name/Title, if known]
                    [Company Name]
                    
                    **Subject: Application for {role} Position - Vivek Swamy**
                    
                    Dear Hiring Manager,
                    
                    I am writing to express my enthusiastic interest in the **{role}** position at MockCorp, as detailed in the attached job description. My background, highlighted by strong skills in Python, AWS, and MLOps, aligns perfectly with your requirements for [Key Requirement from JD - e.g., cloud infrastructure management].
                    
                    During my time at Test Corp (simulated experience), I was responsible for [specific achievement related to JD]. My resume further details my proficiency in [Skill 1] and [Skill 2], which I believe would make me an immediate asset to your team.
                    
                    I am confident in my ability to contribute to your company's goals and I look forward to the opportunity to discuss my application further.
                    
                    Sincerely,
                    
                    Vivek Swamy
                    [vivek.swamy@example.com]
                    """
                    return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': mock_cover_letter})})()]})
                
                # Mock candidate data (Vivek Swamy) for parsing
                mock_llm_json = {
                    "name": "Vivek Swamy", 
                    "email": "vivek.swamy@example.com", 
                    "phone": "555-1234", 
                    "linkedin": "https://linkedin.com/in/vivek-swamy-mock", 
                    "github": "https://github.com/vivek-mock", 
                    "personal_details": "Mock summary generated for: Vivek Swamy.", 
                    "skills": [
                        "Python", "SQL", "AWS", "Streamlit", 
                        "LLM Integration", "MLOps", "Data Visualization", 
                        "Docker", "Kubernetes", "Java", "API Services" 
                    ], 
                    "education": ["B.S. Computer Science, Mock University, 2020"], 
                    "experience": ["Software Intern, Mock Solutions (2024-2025)", "Data Analyst, Test Corp (2022-2024)"], 
                    "certifications": ["Mock Certification in AWS Cloud"], 
                    "projects": ["Mock Project: Built an MLOps pipeline using Docker and Kubernetes."], 
                    "strength": ["Mock Strength"], 
                }
                
                # Mock response content for GroqClient initialization check (for parsing)
                message_obj = type('Message', (object,), {'content': json.dumps(mock_llm_json)})()
                choice_obj = type('Choice', (object,), {'message': message_obj})()
                response_obj = type('MockResponse', (object,), {'choices': [choice_obj]})()
                return response_obj
        
        # Add a placeholder for the completions object if we need a mock response for fit evaluation
        class FitCompletions(Completions):
            def create(self, **kwargs):
                prompt_content = kwargs.get('messages', [{}])[0].get('content', '')
                
                if "Evaluate how well the following resume content matches the provided job description" in prompt_content:
                    # SIMULATED FIT LOGIC (Fallback for when the LLM-dependent function tries to run without a key)
                    
                    # Simple heuristic mock score based on role title in the prompt
                    jd_role_match = re.search(r'(?:Role|Engineer|Scientist)[:\s]+([\w\s/-]+)', prompt_content)
                    jd_role = jd_role_match.group(1).lower().strip() if jd_role_match else "default"
                    
                    if 'ai/ml' in jd_role or 'mlops' in jd_role:
                        score = 8
                    elif 'data scientist' in jd_role:
                        score = 7
                    elif 'cloud engineer' in jd_role:
                        score = 6
                    else:
                        score = 5
                        
                    # Calculate percentages based on the score to differentiate the rows
                    skills_p = 50 + (score * 5)
                    exp_p = 60 + (score * 3)
                    edu_p = 70 + (score * 1)
                    
                    # NOTE: This mock output uses the strict format expected by the regex parser below.
                    mock_fit_output = f"""
                    Overall Fit Score: {score}/10
                    
                    --- Section Match Analysis ---
                    Skills Match: {skills_p}%
                    Experience Match: {exp_p}%
                    Education Match: {edu_p}%
                    
                    Strengths/Matches:
                    - Mock Match Point 1 (Role: {jd_role})
                    - Mock Match Point 2
                    
                    Gaps/Areas for Improvement:
                    - Missing hands-on experience in **Terraform**.
                    - Lack of project experience deploying applications to **GCP/EKS**.
                    - Weak documentation skills in CI/CD pipeline development.
                    
                    Overall Summary: Mock summary for score {score}.
                    """
                    message_obj = type('Message', (object,), {'content': mock_fit_output})()
                    choice_obj = type('Choice', (object,), {'message': message_obj})()
                    response_obj = type('MockResponse', (object,), {'choices': [choice_obj]})()
                    return response_obj
                
                # If it's not a fit evaluation, run standard Completions logic
                return super().create(**kwargs)

        return FitCompletions()

# Initialize the Groq client or the Mock client based on the environment variable
try:
    from groq import Groq
    
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        # Custom flag to indicate a successful connection attempt to the real client
        class GroqPlaceholder(Groq): 
             def __init__(self, api_key): 
                 super().__init__(api_key=api_key)
                 self.client_ready = True
        client = GroqPlaceholder(api_key=GROQ_API_KEY)
    else:
        # Fallback if key is missing but Groq is installed
        raise ValueError("GROQ_API_KEY not set. Using Mock Client.")
        
except (ImportError, ValueError, NameError) as e:
    # Fallback to Mock Client if import fails or key is missing
    client = MockGroqClient()
    
# --- END API SETUP ---


# --- Utility Functions ---

def clear_interview_state(mode):
    """Clears all session state variables related to interview preparation for a specific mode."""
    if mode == 'resume':
        if 'iq_output_resume' in st.session_state: del st.session_state['iq_output_resume']
        if 'interview_qa_resume' in st.session_state: del st.session_state['interview_qa_resume']
        if 'evaluation_report_resume' in st.session_state: del st.session_state['evaluation_report_resume']
    elif mode == 'jd':
        if 'iq_output_jd' in st.session_state: del st.session_state['iq_output_jd']
        if 'interview_qa_jd' in st.session_state: del st.session_state['interview_qa_jd']
        if 'evaluation_report_jd' in st.session_state: del st.session_state['evaluation_report_jd']
    
    # Also clear the gap analysis plan when interview state is cleared (as it's derived from the match)
    if 'gap_analysis_plan' in st.session_state: del st.session_state['gap_analysis_plan']


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
                        # Store as JSON strings for LLM input
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
    """
    Sends resume text to the LLM for structured information extraction.
    """
    
    def get_fallback_name():
        return "Vivek Swamy" 

    if text.startswith("[Error"):
        return {"name": "Parsing Error", "error": text}

    json_match_external = re.search(r'--- JSON Content Start ---\s*(.*?)\s*--- JSON Content End ---', text, re.DOTALL)
    
    if json_match_external:
        try:
            json_content = json_match_external.group(1).strip()
            parsed_data = json.loads(json_content)
            
            if not parsed_data.get('name'):
                 parsed_data['name'] = get_fallback_name()
                 
            parsed_data['error'] = None 
            
            return parsed_data
        
        except json.JSONDecodeError:
            return {"name": get_fallback_name(), "error": f"LLM Input Error: Could not decode uploaded JSON content into a valid structure."}
            
    if isinstance(client, MockGroqClient) or not GROQ_API_KEY:
        try:
            completion = client.chat().create(model=GROQ_MODEL, messages=[{}])
            content = completion.choices[0].message.content.strip()
            parsed_data = json.loads(content)
            
            if not parsed_data.get('name'):
                 parsed_data['name'] = get_fallback_name()
            
            parsed_data['error'] = None 
            return parsed_data
            
        except Exception as e:
            return {"name": get_fallback_name(), "error": f"Mock Client Error: {e}"}

    
    prompt = f"""Extract the following information from the resume in structured JSON.
    Ensure all relevant details for each category are captured.
    - Name, - Email, - - Phone, - Skills (list), - Education (list of degrees/institutions/dates), 
    - Experience (list of job roles/companies/dates/responsibilities), - Certifications (list), 
    - Projects (list of project names/descriptions/technologies), - Strength (list of personal strengths/qualities), 
    - Personal Details (e.g., address, date of birth, nationality), - Github (URL), - LinkedIn (URL)
    
    Resume Text:
    {text}
    
    Provide the output strictly as a JSON object.
    """
    content = ""
    parsed = {}
    json_str = ""
    
    try:
        response = client.chat.completions.create( 
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0).strip()
            
            if json_str.startswith('```json'):
                json_str = json_str[len('```json'):]
            if json_str.endswith('```'):
                json_str = json_str[:-len('```')]
            
            json_str = json_str.strip()
            
            parsed = json.loads(json_str)
        else:
            raise json.JSONDecodeError("Could not isolate a valid JSON structure from LLM response.", content, 0)
        
        if not parsed.get('name'):
            parsed['name'] = get_fallback_name()
            
        parsed['error'] = None 
        return parsed

    except json.JSONDecodeError as e:
        error_msg = f"JSON decoding error from LLM. LLM returned malformed JSON. Error: {e} | Malformed string segment:\n---\n{json_str[:200]}..."
        return {"name": get_fallback_name(), "error": error_msg}
        
    except Exception as e:
        error_msg = f"LLM API interaction error: {e}"
        return {"name": get_fallback_name(), "error": error_msg}

# Updated signature to match the request
def parse_and_store_resume(content_source, file_name_key, source_type):
    """Handles extraction, parsing, and storage of CV data from either a file, pasted text, or generated data."""
    extracted_text = ""
    excel_data = None
    file_name = "Pasted_Resume"

    if source_type == 'file':
        uploaded_file = content_source
        file_name = uploaded_file.name
        file_type = get_file_type(file_name)
        uploaded_file.seek(0) 
        st.session_state.current_parsing_source_name = file_name 
        extracted_text, excel_data = extract_content(file_type, uploaded_file.getvalue(), file_name)
    elif source_type == 'text':
        extracted_text = content_source.strip()
        file_name = "Pasted_Text"
        st.session_state.current_parsing_source_name = file_name 
    elif source_type == 'generated':
        # If the source is already a structured dictionary (from CV Management), bypass LLM parsing
        parsed_data = content_source
        file_name = "Form_Generated_CV"
        st.session_state.current_parsing_source_name = file_name
        
        # We need to construct the compiled_text for the 'full_text' and Q&A features
        compiled_text = generate_markdown_from_parsed(parsed_data)
        
        return {
            "parsed": parsed_data, 
            "full_text": compiled_text, 
            "excel_data": None, 
            "name": parsed_data.get('name', 'Generated_Candidate').replace(' ', '_')
        }


    if extracted_text.startswith("[Error"):
        return {"error": extracted_text, "full_text": extracted_text, "excel_data": None, "name": file_name}
    
    # 2. Call LLM Parser
    parsed_data = parse_resume_with_llm(extracted_text)
    
    # 3. Handle LLM Parsing Error
    if parsed_data.get('error') is not None: 
        error_name = parsed_data.get('name', file_name) 
        return {"error": parsed_data['error'], "full_text": extracted_text, "excel_data": excel_data, "name": error_name}

    # 4. Create compiled text for download/Q&A
    compiled_text = generate_markdown_from_parsed(parsed_data)

    # Ensure final_name uses the parsed name
    final_name = parsed_data.get('name', 'Unknown_Candidate').replace(' ', '_') 
    
    return {
        "parsed": parsed_data, 
        "full_text": compiled_text, 
        "excel_data": excel_data, 
        "name": final_name
    }

def generate_markdown_from_parsed(parsed_data):
    """Generates the Markdown text representation from the structured dictionary."""
    compiled_text = ""
    for k, v in parsed_data.items():
        if v and k not in ['error']:
            compiled_text += f"## {k.replace('_', ' ').title()}\n\n"
            if isinstance(v, list):
                # Ensure all list items are strings for clean display
                compiled_text += "\n".join([f"* {str(item)}" for item in v]) + "\n\n"
            else:
                compiled_text += str(v) + "\n\n"
    return compiled_text

def get_download_link(data, filename, file_format, title="Parsed Data"):
    """
    Generates a base64 encoded download link for the given data and format.
    """
    mime_type = "application/octet-stream"
    
    if file_format in ('json', 'markdown', 'text'):
        data_bytes = data.encode('utf-8')
        if file_format == 'json':
            mime_type = "application/json"
        elif file_format == 'markdown':
            mime_type = "text/markdown"
        else: # text
            mime_type = "text/plain"
            
    elif file_format == 'html':
        # Convert markdown-like text to basic HTML for a clean printable document
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{filename}</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 40px; line-height: 1.6; max-width: 800px; margin: auto; }}
                h1 {{ color: #1E90FF; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                h2 {{ color: #333; margin-top: 20px; }}
                pre, .cover-letter {{ white-space: pre-wrap; word-wrap: break-word; background: #f4f4f4; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                p {{ margin-bottom: 15px; }}
            </style>
        </head>
        <body>
        <h1>{title}: {filename.replace('.html', '')}</h1>
        <hr/>
        <div class="cover-letter">
        {data.replace('\n', '<br>')}
        </div>
        <p style="margin-top: 30px; font-size: 10px; color: grey;">Generated by PragyanAI</p>
        </body>
        </html>
        """
        data_bytes = html_content.encode('utf-8')
        mime_type = "text/html"
    else:
        return "" 

    b64 = base64.b64encode(data_bytes).decode()
    
    # Return the full data URI
    return f"data:{mime_type};base64,{b64}"

def render_download_button(data_uri, filename, label, color):
    """Renders an HTML button that triggers a file download."""
    
    if color == 'json':
        bg_color = "#4CAF50" # Green
        icon = "💾"
    elif color == 'markdown':
        bg_color = "#008CBA" # Blue
        icon = "⬇️"
    elif color == 'html':
        bg_color = "#f44336" # Red
        icon = "📄"
    elif color == 'cover':
        bg_color = "#FFC300" # Yellow/Orange
        icon = "✉️"
    else:
        bg_color = "#555555"
        icon = ""
        
    st.markdown(
        f"""
        <a href="{data_uri}" download="{filename}" style="text-decoration: none;">
            <button style="
                background-color: {bg_color}; 
                color: white; 
                border: none; 
                padding: 10px 10px; 
                text-align: center; 
                text-decoration: none; 
                display: inline-block; 
                font-size: 14px; 
                margin: 4px 0; 
                cursor: pointer; 
                border-radius: 4px;
                width: 100%;">
                {icon} {label}
            </button>
        </a>
        """, 
        unsafe_allow_html=True
    )
    
# --- END HELPER FUNCTIONS ---


# --- LLM Functions (Used across tabs) ---

@st.cache_data(show_spinner="Analyzing JD for metadata...")
def extract_jd_metadata(jd_text):
    """Mocks the extraction of key metadata (Role, Skills, Job Type) from JD text using LLM."""
    
    # Check if the input is an error string from extraction
    if isinstance(jd_text, str) and jd_text.startswith("[Error"):
        return {"role": "Extraction Error", "key_skills": ["Error"], "job_type": "Error"}
    
    # Ensure jd_text is a string before proceeding
    if not isinstance(jd_text, str):
        jd_text = str(jd_text) # Force string conversion if it somehow wasn't
    
    role_match = re.search(r'(?:Role|Position|Title|Engineer|Scientist)[:\s\n]+([\w\s/-]+)', jd_text, re.IGNORECASE)
    role = role_match.group(1).strip() if role_match else "Software Engineer (Mock)"
    
    # Simple regex to extract common keywords
    skills_match = re.findall(r'(Python|Java|SQL|AWS|Docker|Kubernetes|React|Streamlit|Cloud|Data|ML|LLM|MLOps|Visualization|Deep Learning|TensorFlow|Pytorch|Terraform|GCP|EKS)', jd_text, re.IGNORECASE)
    
    if 'data scientist' in jd_text.lower() or 'machine learning' in jd_text.lower():
         role = "Data Scientist/ML Engineer"
    elif 'cloud engineer' in jd_text.lower() or 'aws' in jd_text.lower() or 'gcp' in jd_text.lower():
         role = "Cloud Engineer"
    
    job_type_match = re.search(r'(Full-time|Part-time|Contract|Remote|Hybrid)', jd_text, re.IGNORECASE)
    job_type = job_type_match.group(1) if job_type_match else "Full-time (Mock)"
    
    return {
        "role": role, 
        "key_skills": list(set([s.lower() for s in skills_match])), 
        "job_type": job_type
    }

def evaluate_jd_fit(job_description, parsed_json):
    """
    Evaluates how well a resume fits a given job description, 
    including section-wise scores, by calling the Groq LLM API.
    """
    global client, GROQ_MODEL, GROQ_API_KEY
    
    if parsed_json.get('error') is not None: 
         return f"Cannot evaluate due to resume parsing errors: {parsed_json['error']}"

    if isinstance(client, MockGroqClient) or not GROQ_API_KEY:
         # Mock Client is hardcoded to return a structured output including Gaps.
         response = client.chat().create(model=GROQ_MODEL, messages=[{"role": "user", "content": f"Evaluate how well the following resume content matches the provided job description: {job_description}"}])
         return response.choices[0].message.content.strip()

    if not job_description.strip(): return "Please paste a job description."

    relevant_resume_data = {
        'Skills': parsed_json.get('skills', 'Not found or empty'),
        'Experience': parsed_json.get('experience', 'Not found or empty'),
        'Education': parsed_json.get('education', 'Not found or empty'),
    }
    resume_summary = json.dumps(relevant_resume_data, indent=2)

    prompt = f"""Evaluate how well the following resume content matches the provided job description.
    
    Job Description: {job_description}
    
    Resume Sections for Analysis:
    {resume_summary}
    
    Provide a detailed evaluation structured as follows:
    1.  **Overall Fit Score:** A score out of 10.
    2.  **Section Match Percentages:** A percentage score for the match in the key sections (Skills, Experience, Education).
    3.  **Strengths/Matches:** Key points where the resume aligns well with the JD.
    4.  **Gaps/Areas for Improvement:** Key requirements in the JD that are missing or weak in the resume. Focus on specific technical skills or experience areas.
    5.  **Overall Summary:** A concise summary of the fit.
    
    **Format the output strictly as follows, ensuring the scores are easily parsable (use brackets or no brackets around scores, but they must be present):**
    Overall Fit Score: [Score]/10
    
    --- Section Match Analysis ---
    Skills Match: [XX]%
    Experience Match: [YY]%
    Education Match: [ZZ]%
    
    Strengths/Matches:
    - Point 1
    - Point 2
    
    Gaps/Areas for Improvement:
    - Point 1 (Specific Skill/Experience Gap)
    - Point 2 (Specific Skill/Experience Gap)
    
    Overall Summary: [Concise summary]
    """

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_output = f"AI Evaluation Error: Failed to connect or receive response from LLM. Error: {e}\n{traceback.format_exc()}"
        return error_output

def generate_cover_letter_llm(jd_content, parsed_json, preferred_style="Standard"):
    """
    Generates a cover letter based on JD and parsed resume data.
    """
    global client, GROQ_MODEL, GROQ_API_KEY
    
    if parsed_json.get('error') is not None: 
         return f"Cannot generate cover letter due to resume parsing errors: {parsed_json['error']}"

    if not jd_content.strip(): return "Please provide a Job Description to generate the letter."

    candidate_name = parsed_json.get('name', 'The Candidate')
    candidate_email = parsed_json.get('email', '[Candidate Email]')
    # Ensure list items are strings before joining
    candidate_skills = ", ".join([str(s) for s in parsed_json.get('skills', [])])
    candidate_experience = "\n".join([str(e) for e in parsed_json.get('experience', [])])
    
    jd_metadata = extract_jd_metadata(jd_content)
    # Safely get the role from metadata (which is now guaranteed to be a dict)
    jd_role = jd_metadata.get('role', 'the position')

    prompt = f"""
    You are an expert cover letter generator. Your task is to write a highly professional, engaging, and concise cover letter 
    that highlights the candidate's fit for the specific job description provided.
    
    **Instructions:**
    1.  **Style:** Adopt a **{preferred_style}** tone.
    2.  **Structure:** Use standard cover letter format.
    3.  **Customization:** Directly reference the skills and experience listed in the candidate's resume that match the job description's requirements.
    4.  **Length:** Keep it brief, no more than four paragraphs.
    5.  **Output Format:** Output the letter text only, using double newlines for paragraph separation. Include placeholders like [Date], [Hiring Manager Name/Title, if known], and [Company Name] where necessary. Use bold formatting for the job title.
    
    --- Candidate Information ---
    Candidate Name: {candidate_name}
    Candidate Contact: {candidate_email}
    Key Skills: {candidate_skills}
    Relevant Experience: {candidate_experience}
    
    --- Job Description Information ---
    Job Description Role: {jd_role}
    Job Description Content:
    {jd_content}
    """
    
    if isinstance(client, MockGroqClient) or not GROQ_API_KEY:
         response = client.chat().create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}])
         return response.choices[0].message.content.strip()

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.7 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_output = f"AI Generation Error: Failed to connect or receive response from LLM. Error: {e}\n{traceback.format_exc()}"
        return error_output

def generate_gap_course_plan(gap_analysis_text, jd_role, candidate_skills):
    """
    Generates a detailed course plan and certification suggestions to fill identified gaps.
    """
    global client, GROQ_MODEL, GROQ_API_KEY
    
    if not gap_analysis_text.strip() or "No significant gaps" in gap_analysis_text:
        return "No specific gaps were identified in the match analysis. Focus on advanced skills in your core area."
        
    if isinstance(client, MockGroqClient) or not GROQ_API_KEY:
         # Mock client returns a hardcoded, structured plan (see MockGroqClient)
         response = client.chat().create(model=GROQ_MODEL, messages=[{"role": "user", "content": f"Generate a detailed course plan and suggest relevant certifications for Gaps Identified: {gap_analysis_text}"}])
         return response.choices[0].message.content.strip()

    prompt = f"""
    You are an expert career consultant. Based on the candidate's profile and the identified skill gaps for the role of **{jd_role}**, 
    generate a detailed course plan and suggest relevant certifications.
    
    **Context:**
    - Target Role: {jd_role}
    - Candidate's Current Key Skills: {', '.join(candidate_skills)}
    
    **Gaps Identified:**
    {gap_analysis_text}
    
    **Instructions:**
    1.  **Course Plan:** Structure the plan into 2-3 chronological phases (e.g., Foundational, Intermediate, Advanced/Project). Include specific topics (e.g., Python Basics, Docker Networking, Terraform Modules). Suggest a rough time estimate (e.g., weeks) for each phase.
    2.  **Certifications:** Suggest 2-3 industry-recognized certifications that directly address the identified gaps and enhance the resume for the target role.
    3.  **Output Format:** Use Markdown. Use the headings '## Detailed Course Plan' and '## Suggested Certifications'.
    """

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.6 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_output = f"AI Generation Error: Failed to connect or receive response from LLM for course plan. Error: {e}\n{traceback.format_exc()}"
        return error_output

# --- ADAPTED LLM Functions for Interview Preparation (Modified) ---

def generate_interview_questions(source_data, source_type, identifier):
    """
    Generates interview questions based on either a resume section or a full JD.
    source_type can be 'resume' (source_data is parsed_json) or 'jd' (source_data is jd_content string).
    identifier is the section name (e.g., 'Skills') or JD name.
    """
    global client, GROQ_MODEL
    
    if source_type == 'resume':
        target_section_display = identifier
        target_section_key = identifier.lower().replace(' ', '_')
        resume_content = source_data.get(target_section_key, "Content not found in this section.")
        
        # Ensure resume_content is a string
        if isinstance(resume_content, list):
            content_str = "\n".join([str(item) for item in resume_content])
        else:
            content_str = str(resume_content)
        
        if "Content not found" in content_str or not content_str.strip():
            return f"Error: Content for resume section '{target_section_display}' is empty or invalid."
            
        context_block = f"""
    --- Candidate Resume Content for Section: {target_section_display} ---
    {content_str}
    
    Generate a list of interview questions specifically targeting the **{target_section_display}** section of the candidate's resume.
    """
        
    elif source_type == 'jd':
        jd_content = identifier
        
        if not jd_content.strip():
            return "Error: Job Description content is empty."
            
        context_block = f"""
    --- Job Description (JD) Content for Role: {source_data} ---
    {jd_content}
    
    Generate a list of interview questions specifically targeting the **JD** requirements and the stated role, to assess candidate fit.
    """
        
    else:
        return "Error: Invalid question source type."


    prompt = f"""
    You are an expert technical interviewer. Based ONLY on the following information, 
    generate a list of interview questions.
    
    **Instructions:**
    1. Generate 5-7 questions across **3 difficulty levels**: [Basic/Screening], [Intermediate/Technical], and [Advanced/Behavioral].
    2. The output must be a raw string. Start a new line for each question.
    3. Use the following strict format for your output:
    [Level Name]
    Q1: Question text...
    Q2: Question text...
    ...
    
    {context_block}
    
    ---
    Output:
    """

    try:
        if isinstance(client, MockGroqClient) or not GROQ_API_KEY:
             response = client.chat().create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}])
        else:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
        return response.choices[0].message.content.strip()
            
    except Exception as e:
        error_msg = f"AI Question Generation Error: {e}\nTrace: {traceback.format_exc()}"
        st.error(error_msg)
        return f"Error generating questions: {error_msg}"


def evaluate_interview_answers(qa_list, resume_context):
    """
    Evaluates a list of candidate's recorded answers based on the questions and resume context.
    The output is a full markdown report.
    """
    global client, GROQ_MODEL
    
    # Format Q&A for LLM
    qa_exchange = "\n\n--- Candidate Answers ---\n\n"
    for i, item in enumerate(qa_list):
        # Ensure question and answer are strings
        question = str(item['question'].replace(f"({item['level']})", '').strip())
        answer = str(item['answer'])
        qa_exchange += f"Q{i+1} ({item['level']}): {question}\n"
        qa_exchange += f"Answer {i+1}: {answer}\n"
        qa_exchange += "---"

    prompt = f"""
    You are an expert interviewer evaluating a candidate's recorded answers.
    
    **Evaluation Task:**
    Evaluate the candidate's answers based on the provided questions and their resume/JD context.
    
    **Instructions for Report:**
    1.  Provide an **Overall Score (X/10)** at the beginning of the report.
    2.  Give a **Summary** of the candidate's performance (e.g., strength in technical depth, weakness in behavioral structure).
    3.  For **each question** answered, provide specific, actionable, constructive feedback. Use markdown headings (e.g., **Q1 Feedback**).
    4.  Ensure the report is professional and directly addresses consistency with the context.
    
    --- Context Used for Interview ---
    {resume_context}
    
    --- Interview Exchange ---
    {qa_exchange}
    
    ---
    **Output the evaluation report clearly using markdown.**
    """

    try:
        if isinstance(client, MockGroqClient) or not GROQ_API_KEY:
             response = client.chat().create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}])
        else:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Evaluation Error: Failed to connect to LLM for scoring. Error: {e}"

# --- END ADAPTED LLM Functions ---

# --- Tab Content Functions ---
    
def resume_parsing_tab():
    # --- TAB 1: Resume Parsing ---
    st.header("📄 Resume Upload and Parsing")
    
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

        if "candidate_uploaded_resumes" not in st.session_state: st.session_state.candidate_uploaded_resumes = []
        if "pasted_cv_text" not in st.session_state: st.session_state.pasted_cv_text = ""
        
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
            st.toast("Upload cleared.")
        
        file_to_parse = st.session_state.candidate_uploaded_resumes[0] if st.session_state.candidate_uploaded_resumes else None
        
        st.markdown("### 2. Parse Uploaded File")
        
        if file_to_parse:
            if st.button(f"Parse and Load: **{file_to_parse.name}**", use_container_width=True):
                with st.spinner(f"Parsing {file_to_parse.name}..."):
                    # Use source_type='file'
                    result = parse_and_store_resume(file_to_parse, file_name_key='single_resume_candidate', source_type='file')
                    
                    if result.get('error') is None:
                        st.session_state.parsed = result['parsed']
                        st.session_state.full_text = result['full_text']
                        st.session_state.excel_data = result['excel_data'] 
                        st.session_state.parsed['name'] = result['name'] 
                        clear_interview_state('resume')
                        clear_interview_state('jd')
                        if 'gap_analysis_plan' in st.session_state: del st.session_state['gap_analysis_plan']
                        st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                        st.info("The parsed data is ready for matching.")
                        st.rerun() 
                    else:
                        st.error(f"Parsing failed for {file_to_parse.name}: {result['error']}")
                        st.session_state.parsed = {"error": result['error'], "name": result['name']}
                        st.session_state.full_text = result['full_text'] or ""
                        st.session_state.excel_data = result['excel_data'] 
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
                    
                    # Use source_type='text'
                    result = parse_and_store_resume(pasted_text, file_name_key='single_resume_candidate', source_type='text')
                    
                    if result.get('error') is None:
                        st.session_state.parsed = result['parsed']
                        st.session_state.full_text = result['full_text']
                        st.session_state.excel_data = result['excel_data'] 
                        st.session_state.parsed['name'] = result['name'] 
                        clear_interview_state('resume')
                        clear_interview_state('jd')
                        if 'gap_analysis_plan' in st.session_state: del st.session_state['gap_analysis_plan']
                        st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                        st.info("The parsed data is ready for matching.") 
                        st.rerun()
                    else:
                        st.error(f"Parsing failed: {result['error']}")
                        st.session_state.parsed = {"error": result['error'], "name": result['name']}
                        st.session_state.full_text = result['full_text'] or ""
                        st.session_state.excel_data = result['excel_data'] 
        else:
            st.info("Please paste your CV text into the box above.")
            
    st.markdown("---")
        
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

    def extract_jd_from_linkedin_url(url):
        if "linkedin.com/jobs" not in url:
            return f"[Error] Invalid LinkedIn Job URL: {url}"

        url_lower = url.lower()
        
        if "data-scientist" in url_lower:
            role = "Data Scientist"
            skills = ["Python", "SQL", "ML", "Data Analysis", "Pytorch", "Visualization"]
            focus = "machine learning and statistical modeling"
            
        elif "cloud-engineer" in url_lower or "aws" in url_lower:
            role = "Cloud Engineer"
            skills = ["AWS", "Docker", "Kubernetes", "Cloud Services", "GCP", "Terraform"]
            focus = "infrastructure as code and cloud deployment"
            
        elif "ml-engineer" in url_lower or "ai-engineer" in url_lower:
            role = "AI/ML Engineer"
            skills = ["MLOps", "LLM", "Deep Learning", "Python", "TensorFlow", "API Services"]
            focus = "production-level AI/ML model development and deployment"
            
        else:
            role = "Software Engineer"
            skills = ["Java", "API", "SQL", "React", "JavaScript"]
            focus = "full-stack application development"
        
        skills_str = ", ".join(skills)

        return f"""
        --- Simulated JD for: {role} ---
        
        Company: MockCorp
        Location: Remote
        
        Job Summary:
        We are seeking a highly skilled **{role}** to join our team. The ideal candidate will have expertise in {skills_str}. Must be focused on **{focus}**. This is a Full-time position.
        
        Responsibilities:
        * Develop and maintain systems using **{skills[0]}** and **{skills[1]}** in a collaborative environment.
        * Manage and deploy applications on **{skills[2]}** platforms.
        * Collaborate with cross-functional teams.
        
        Qualifications:
        * 3+ years of experience.
        * Strong proficiency in **{skills[0]}** and analytical tools.
        * Experience with cloud platforms (e.g., AWS).
        ---
        """

""" # This is the closing of the multiline string above
    
    # This 'elif' must align with the outer conditional statement that controls the input method
    elif method == "LinkedIn URL":
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
                        
                        if metadata.get('role') == 'Extraction Error': # Check for error from metadata
                            st.error(f"Failed to process {url}: {jd_text}")
                            continue
                            
                        name = f"JD for {metadata.get('role', 'Unknown Role')}"
                        # Store metadata directly into the list item
                        st.session_state.candidate_jd_list.append({"name": name, "content": jd_text, **metadata})
                        count += 1
                            
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully!")
                        st.rerun() 
                    else:
                        st.error("No JDs were added successfully.")

    elif method == "Paste Text":
        with st.form("jd_paste_form_candidate", clear_on_submit=True):
            text_list = st.text_area("Paste one or more JD texts (separate by '---')" if jd_type == "Multiple JD" else "Paste JD text here", key="text_list_candidate")
            if st.form_submit_button("Add JD(s) from Text", key="add_jd_text_btn_candidate"):
                if text_list:
                    texts = [t.strip() for t in text_list.split("---")] if jd_type == "Multiple JD" else [text_list.strip()]
                    count = 0
                    for i, text in enumerate(texts):
                        if text:
                            metadata = extract_jd_metadata(text)
                            
                            if metadata.get('role') == 'Extraction Error': # Check for error from metadata
                                st.error(f"Failed to extract metadata for pasted text {i+1}.")
                                continue
                                
                            name_base = metadata.get('role', f"Pasted JD {len(st.session_state.candidate_jd_list) + i + 1}")
                            # Store metadata directly into the list item
                            st.session_state.candidate_jd_list.append({"name": name_base, "content": text, **metadata})
                            count += 1
                    
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully!")
                        st.rerun() 

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
                            
                            # **CRITICAL FIX LOCATION:** Ensure jd_item is a dictionary.
                            # `extract_jd_metadata` is now guaranteed to return a dictionary
                            # or a dictionary containing "Extraction Error". We check for that.
                            if metadata.get('role') == 'Extraction Error': 
                                st.error(f"Failed to extract metadata for {file.name}.")
                                continue
                                
                            # Store metadata directly into the list item
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
    if st.session_state.candidate_jd_list:
        
        col_display_header, col_clear_button = st.columns([3, 1])
        
        with col_display_header: st.markdown("### ✅ Current JDs Added:")
            
        with col_clear_button:
            if st.button("🗑️ Clear All JDs", key="clear_jds_candidate", use_container_width=True, help="Removes all currently loaded JDs."):
                st.session_state.candidate_jd_list = []
                if 'candidate_match_results' in st.session_state: del st.session_state['candidate_match_results']
                if 'jd_chatbot_history' in st.session_state: del st.session_state['jd_chatbot_history']
                if 'gap_analysis_plan' in st.session_state: del st.session_state['gap_analysis_plan']
                clear_interview_state('jd')
                st.success("All JDs and associated data have been cleared.")
                st.rerun() 

        for idx, jd_item in enumerate(st.session_state.candidate_jd_list, 1):
            # Access metadata using .get() for safety, though it should be a dict now
            title = jd_item.get('name', f'JD {idx}')
            role = jd_item.get('role', 'N/A')
            job_type = jd_item.get('job_type', 'N/A')
            key_skills = jd_item.get('key_skills', ['N/A'])
            content = jd_item.get('content', 'No content extracted.')
            
            display_title = title.replace("--- Simulated JD for: ", "")
            with st.expander(f"**JD {idx}:** {display_title} | Role: {role}"):
                st.markdown(f"**Job Type:** {job_type} | **Key Skills:** `{', '.join(key_skills)}`")
                st.markdown("---")
                st.text(content)
    else:
        st.info("No Job Descriptions added yet.")
        
# --- Batch Match Tab Function (UPDATED) ---

def jd_batch_match_tab():
    """The Batch JD Match tab logic, enhanced to display rankings and section scores."""
    st.header("🎯 Batch JD Match: Best Matches")
    st.markdown("Compare your current resume against all saved job descriptions.")
    
    # Determine if a resume/CV is ready
    is_resume_parsed = (
        st.session_state.get('parsed') is not None and
        st.session_state.parsed.get('name') is not None and
        st.session_state.parsed.get('error') is None
    )
    
    # Check if we are running in Mock Mode
    is_mock_mode = isinstance(client, MockGroqClient) and not GROQ_API_KEY
    
    if not is_resume_parsed:
        st.warning("⚠️ Please **upload and parse your resume** in the 'Resume Parsing' tab first.")
        
        if st.session_state.get('parsed', {}).get('error') is not None:
             st.error(f"Resume Parsing Error: {st.session_state.parsed.get('error')}")

    elif not st.session_state.candidate_jd_list:
        st.error("❌ Please **add Job Descriptions** in the 'JD Management' tab before running batch analysis.")
        
    elif not GROQ_API_KEY and not is_mock_mode:
        st.error("Cannot use JD Match: GROQ_API_KEY is not configured.")
        
    elif is_mock_mode:
        st.info("ℹ️ Running in **Mock LLM Mode** for fit evaluation. Results are simulated for consistency, but a valid GROQ_API_KEY is recommended for real AI analysis.")
        
    # else:
    #     if not hasattr(client, 'client_ready') or not client.client_ready:
    #         st.warning("⚠️ LLM client setup failed or key is missing. Match analysis may not be accurate or available.")


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
    
    if st.button(f"Run Match Analysis on **{len(jds_to_match)}** Selected JD(s)", type="primary", use_container_width=True):
        st.session_state.candidate_match_results = []
        if 'gap_analysis_plan' in st.session_state: del st.session_state['gap_analysis_plan']
        
        if not jds_to_match:
            st.warning("Please select at least one Job Description to run the analysis.")
            
        elif not is_resume_parsed:
             st.warning("Please **upload and parse your resume** successfully first.")

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
                        
                        # --- Parsing Logic for Structured Output ---
                        overall_score_match = re.search(r'Overall Fit Score:\s*\[?\s*(\d+)\s*\]?\s*/10', fit_output, re.IGNORECASE)
                        
                        section_analysis_match = re.search(
                            r'--- Section Match Analysis ---\s*(.*?)\s*(?:Strengths/Matches|Overall Summary):', 
                            fit_output, re.DOTALL | re.IGNORECASE
                        )
                        
                        # Extract Gaps/Areas for Improvement
                        gaps_match = re.search(
                            r'Gaps/Areas for Improvement:\s*(.*?)\s*(?:Overall Summary|---|$)', 
                            fit_output, re.DOTALL | re.IGNORECASE
                        )
                        raw_gaps = gaps_match.group(1).strip() if gaps_match else "No significant gaps identified in the LLM analysis."
                        
                        skills_percent, experience_percent, education_percent = 'N/A', 'N/A', 'N/A'
                        
                        if section_analysis_match:
                            section_text = section_analysis_match.group(1)
                            
                            skills_match = re.search(r'Skills\s*Match:\s*\[?\s*(\d+)%\s*\]?', section_text, re.IGNORECASE)
                            experience_match = re.search(r'Experience\s*Match:\s*\[?\s*(\d+)%\s*\]?', section_text, re.IGNORECASE)
                            education_match = re.search(r'Education\s*Match:\s*\[?\s*(\d+)%\s*\]?', section_text, re.IGNORECASE)
                            
                            if skills_match: skills_percent = skills_match.group(1)
                            if experience_match: experience_percent = experience_match.group(1)
                            if education_match: education_percent = education_match.group(1)
                            
                        overall_score = overall_score_match.group(1) if overall_score_match else 'N/A'
                        
                        if "AI Evaluation Error" in fit_output:
                            overall_score = "Error (API)"
                        elif "Cannot evaluate" in fit_output:
                            overall_score = "Error (Parse)"
                            

                        results_with_score.append({
                            "jd_name": jd_name,
                            "overall_score": overall_score,
                            "numeric_score": int(overall_score) if overall_score.isdigit() else -1, 
                            "skills_percent": skills_percent,
                            "experience_percent": experience_percent, 
                            "education_percent": education_percent, 
                            "full_analysis": fit_output,
                            "gaps": raw_gaps
                        })
                    except Exception as e:
                        results_with_score.append({
                            "jd_name": jd_name,
                            "overall_score": "Error (Extract)",
                            "numeric_score": -1, 
                            "skills_percent": "Error",
                            "experience_percent": "Error", 
                            "education_percent": "Error", 
                            "full_analysis": f"Error parsing LLM analysis for {jd_name}: {e}\nFull LLM Output:\n---\n{fit_output}\n---",
                            "gaps": "Extraction failed due to internal error."
                        })
                        
                # --- Ranking Logic ---
                results_with_score.sort(key=lambda x: x['numeric_score'], reverse=True)
                
                current_rank = 1
                current_score = -1 
                
                for i, item in enumerate(results_with_score):
                    if item['numeric_score'] < current_score:
                        current_rank = i + 1
                        current_score = item['numeric_score']
                    elif i == 0:
                        current_score = item['numeric_score']
                        
                    item['rank'] = current_rank
                    
                    if 'numeric_score' in item:
                         del item['numeric_score'] 
                    
                st.session_state.candidate_match_results = results_with_score
                st.success("Batch analysis complete!")
                st.rerun() 


    if st.session_state.get('candidate_match_results'):
        st.markdown("---")
        st.subheader("Results Overview: Ranked Matches")

        # Prepare data for display dataframe
        display_df_data = []
        for item in st.session_state.candidate_match_results:
            display_df_data.append({
                "Rank": f"🥇 {item['rank']}" if item['rank'] == 1 else item['rank'],
                "Job Description": item['jd_name'].replace("--- Simulated JD for: ", ""),
                "Overall Score": f"{item['overall_score']}/10",
                "Skills Match (%)": item['skills_percent'],
                "Experience Match (%)": item['experience_percent'],
                "Education Match (%)": item['education_percent'],
            })

        df = pd.DataFrame(display_df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("Detailed Match Analysis")
        
        for item in st.session_state.candidate_match_results:
            with st.expander(f"Rank {item['rank']} | {item['jd_name']} ({item['overall_score']}/10)"):
                st.markdown(item['full_analysis'])

    else:
         st.markdown("---")
         st.info("Run the match analysis above to evaluate your resume against the selected Job Descriptions.")
        
# --- Filter JD Tab Function (unchanged) ---

def filter_jd_tab_content():
    """Filter JD Tab."""
    st.header("🔍 Filter Job Descriptions by Criteria")
    st.markdown("Use the filters below to narrow down your saved Job Descriptions.")

    if not st.session_state.candidate_jd_list:
        st.info("No Job Descriptions are currently loaded. Please add JDs in the 'JD Management' tab.")
        if 'filtered_jds_display' not in st.session_state:
            st.session_state.filtered_jds_display = []
        return
    
    global DEFAULT_ROLES, DEFAULT_JOB_TYPES, STARTER_KEYWORDS
    
    # Safely extract roles, types, and skills from loaded JDs
    unique_roles = sorted(list(set(
        [item.get('role', 'General Analyst') for item in st.session_state.candidate_jd_list] + DEFAULT_ROLES
    )))
    # Note: Using DEFAULT_JOB_TYPES as a base, ensuring all loaded types are included.
    unique_job_types = sorted(list(set(
        [item.get('job_type', 'Full-time') for item in st.session_state.candidate_jd_list] + DEFAULT_JOB_TYPES
    )))
    
    all_unique_skills = set(STARTER_KEYWORDS)
    for jd in st.session_state.candidate_jd_list:
        # jd_item['key_skills'] is guaranteed to be a list due to fix in JD Management
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
            # Use .get() for safe access
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
                # Use .get() for safe access
                "Job Description Title": jd.get('name', 'N/A').replace("--- Simulated JD for: ", ""),
                "Role": jd.get('role', 'N/A'),
                "Job Type": jd.get('job_type', 'N/A'),
                "Key Skills": ", ".join(jd.get('key_skills', ['N/A'])[:5]) + "...",
            })
            
        st.dataframe(display_data, use_container_width=True)

        st.markdown("##### Detailed View")
        for idx, jd in enumerate(filtered_jds, 1):
            with st.expander(f"JD {idx}: {jd.get('name', 'N/A').replace('--- Simulated JD for: ', '')} - ({jd.get('role', 'N/A')})"):
                st.markdown(f"**Job Type:** {jd.get('job_type', 'N/A')}")
                st.markdown(f"**Extracted Skills:** {', '.join(jd.get('key_skills', ['N/A']))}")
                st.markdown("---")
                st.text(jd.get('content', 'Content not available'))
    elif st.session_state.candidate_jd_list and apply_filters_button:
        st.info("No Job Descriptions match the selected criteria. Try broadening your filter selections.")
    elif st.session_state.candidate_jd_list and not apply_filters_button:
        st.info("Use the filters above and click **'Apply Filters'** to view matching Job Descriptions.")

# --- Parsed Data Tab (unchanged) ---

def parsed_data_tab():
    """Parsed Data View Tab."""
    st.header("✨ Parsed Resume Data View")
    st.markdown("This tab displays the loaded candidate data and provides download options.")
    st.markdown("---")

    is_data_loaded_and_valid = (
        st.session_state.get('parsed', {}).get('name') is not None and 
        st.session_state.get('parsed', {}).get('error') is None
    )

    if is_data_loaded_and_valid:
        
        candidate_name = st.session_state.parsed['name']
        
        source_key = st.session_state.get('current_parsing_source_name', 'Unknown Source')
        if source_key == "Pasted_Text":
            source_display = "Pasted CV Data"
        elif source_key == "Form_Generated_CV":
            source_display = "Form Generated CV"
        else:
            source_display = source_key.replace('_', ' ').replace('-', ' ') 

        base_filename = f"{candidate_name.replace(' ', '_')}_Parsed_Resume"
        parsed_json_data = json.dumps(st.session_state.parsed, indent=4)
        parsed_markdown_data = st.session_state.full_text
        
        json_filename = f"{base_filename}.json"
        md_filename = f"{base_filename}.md"
        html_filename = f"{base_filename}.html"
        
        json_data_uri = get_download_link(parsed_json_data, json_filename, 'json', title="Parsed Resume Data")
        md_data_uri = get_download_link(parsed_markdown_data, md_filename, 'markdown', title="Parsed Resume Data")
        html_data_uri = get_download_link(parsed_markdown_data.replace('\n', '<br>').replace('##', '<h2>'), html_filename, 'html', title="Parsed Resume Data") 
        
        
        tab_markdown, tab_json, tab_download = st.tabs([
            "📄 Markdown View", 
            "💾 JSON View", 
            "⬇️ PDF/HTML Download"
        ])

        with tab_markdown:
            st.markdown(f"**Candidate:** **{candidate_name}**")
            st.caption(f"Source: {source_display}")
            st.markdown("---")
            st.markdown("### Resume Content in Markdown Format")
            st.markdown(st.session_state.full_text)
            
            if st.session_state.excel_data:
                 st.markdown("### Extracted Spreadsheet Data (if applicable)")
                 st.json(st.session_state.excel_data)
                 
            st.markdown("---")
            st.markdown("##### Download Markdown Data")
            render_download_button(
                md_data_uri, 
                md_filename, 
                f"⬇️ Download Markdown (.md)", 
                'markdown'
            )


        with tab_json:
            st.markdown(f"**Candidate:** **{candidate_name}**")
            st.caption(f"Source: {source_display}")
            st.markdown("---")
            st.markdown("### Structured Data in JSON Format")
            st.json(st.session_state.parsed)

            st.markdown("---")
            st.markdown("##### Download JSON Data")
            render_download_button(
                json_data_uri, 
                json_filename, 
                f"💾 Download JSON (.json)", 
                'json'
            )

        with tab_download:
            
            st.markdown("### Download Viewable Document")
            st.info("This download provides the data in an HTML file that can be easily viewed or printed/saved as a PDF.")
            
            col_html = st.columns(1)[0]

            with col_html:
                st.markdown(f"**{html_filename.replace('.html', '.pdf/html')}**", help="Viewable document format.")
                render_download_button(
                    html_data_uri, 
                    html_filename, 
                    f"📄 Download HTML (PDF Sim.)", 
                    'html'
                )
                
            st.markdown("---")
            st.info("For structured data (JSON) or raw text (Markdown), please check their respective viewing tabs.")

    else:
        st.warning(f"**Status:** ❌ **No Valid Resume Data Loaded**")
        if st.session_state.get('parsed', {}).get('error') is not None:
             st.error(f"Last Parsing Error: {st.session_state.parsed['error']}")
        st.info("Please successfully parse a resume in the **Resume Parsing** tab.")

# --- Cover Letter Generation Tab (unchanged) ---

def generate_cover_letter_tab():
    """Cover Letter Tab."""
    st.header("✉️ Generate Cover Letter")
    st.markdown("Create a customized cover letter for a specific Job Description using your parsed resume data.")
    st.markdown("---")

    is_resume_parsed = (
        st.session_state.get('parsed', {}).get('name') is not None and 
        st.session_state.get('parsed', {}).get('error') is None
    )
    
    if not is_resume_parsed:
        st.warning("⚠️ **Cover Letter Disabled:** Please parse a valid resume in the 'Resume Parsing' tab or **generate one in 'CV Management'** first.")
        return
        
    if not st.session_state.get('candidate_jd_list'):
        st.error("❌ Please **add Job Descriptions** in the 'JD Management' tab first.")
        return
        
    jd_names = [jd.get('name') for jd in st.session_state.candidate_jd_list if jd.get('name')]
    selected_jd_name = st.selectbox(
        "Select Job Description for Cover Letter",
        options=jd_names,
        key="selected_jd_for_cl"
    )

    selected_jd = next((jd for jd in st.session_state.candidate_jd_list if jd.get('name') == selected_jd_name), None)
    
    if not selected_jd:
        st.error("Selected JD not found.")
        return
    
    st.markdown("---")
    col_style, col_gen = st.columns([1, 1])
    
    with col_style:
        style = st.selectbox(
            "Select Letter Style/Tone",
            options=["Standard", "Enthusiastic", "Professional", "Concise"],
            key="cl_style"
        )
        
    with col_gen:
        st.write("") 
        st.write("") 
        if st.button("✨ Generate Cover Letter", use_container_width=True, type="primary"):
            st.session_state.generated_cover_letter = ""
            with st.spinner(f"Generating personalized cover letter for **{selected_jd['name']}**..."):
                letter_text = generate_cover_letter_llm(
                    jd_content=selected_jd.get('content', ''), 
                    parsed_json=st.session_state.parsed,
                    preferred_style=style
                )
                st.session_state.generated_cover_letter = letter_text
                st.session_state.cl_jd_name = selected_jd_name 
                st.rerun()
                
    st.markdown("---")
    
    if "generated_cover_letter" in st.session_state and st.session_state.generated_cover_letter:
        
        st.subheader(f"✅ Generated Cover Letter for: {st.session_state.cl_jd_name}")
        
        if st.session_state.generated_cover_letter.startswith("Cannot generate") or st.session_state.generated_cover_letter.startswith("AI Generation Error"):
            st.error(st.session_state.generated_cover_letter)
            return

        final_letter_text = st.text_area(
            "Review and edit the generated cover letter:",
            value=st.session_state.generated_cover_letter,
            height=400,
            key="final_cover_letter_edit"
        )
        
        st.markdown("##### Download Options")
        
        candidate_name = st.session_state.parsed.get('name', 'Candidate').replace(' ', '_')
        jd_role = selected_jd.get('role', 'Job').replace('/', '_').replace(' ', '_')
        base_filename = f"{candidate_name}_CoverLetter_{jd_role}"
        
        html_filename = f"{base_filename}.html"
        txt_filename = f"{base_filename}.txt"
        
        html_data_uri = get_download_link(
            final_letter_text, 
            html_filename, 
            'html',
            title=f"Cover Letter for {jd_role}"
        )
        txt_data_uri = get_download_link(
            final_letter_text, 
            txt_filename, 
            'text',
            title=f"Cover Letter for {jd_role}"
        )
        
        col_html_dl, col_txt_dl = st.columns(2)
        
        with col_html_dl:
            render_download_button(
                html_data_uri, 
                html_filename, 
                f"📄 Download as HTML (Print to PDF)", 
                'cover'
            )
            
        with col_txt_dl:
            render_download_button(
                txt_data_uri, 
                txt_filename, 
                f"⬇️ Download as Plain Text (.txt)", 
                'markdown'
            )
            
    elif "generated_cover_letter" not in st.session_state or not st.session_state.generated_cover_letter:
        st.info("Select a Job Description and click 'Generate Cover Letter' to begin.")
        
# --- Interview Preparation Tab (unchanged) ---

def parse_questions_from_raw(raw_questions_response):
    """Parses the structured raw LLM output into a list of Q&A dictionaries."""
    q_list = []
    current_level = "General"
    
    for line in raw_questions_response.splitlines():
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            current_level = line.strip('[]')
        elif line.lower().startswith('q') and ':' in line:
            question_text = line[line.find(':') + 1:].strip()
            q_list.append({
                "question": f"({current_level}) {question_text}",
                "answer": "", 
                "level": current_level
            })
    return q_list


def display_evaluation_form(mode, qa_data_list, context_for_eval):
    """Handles the display of the Q&A form and evaluation logic for a given mode."""

    current_qa_key = f'interview_qa_{mode}'
    current_report_key = f'evaluation_report_{mode}'
    
    if qa_data_list:
        st.markdown("---")
        st.subheader("2. Practice and Record Answers")
        
        with st.form(f"interview_practice_form_{mode}"):
            
            # Use the actual list from session state for mutation
            current_qa_list = st.session_state[current_qa_key]
            
            for i, qa_item in enumerate(current_qa_list):
                st.markdown(f"**Question {i+1}:** {qa_item['question']}")
                
                # Use the unique key for persistent state
                answer_key = f'answer_q_{mode}_{i}'
                
                # Set initial value from session state, ensuring persistence
                answer = st.text_area(
                    f"Your Answer for Q{i+1}", 
                    value=current_qa_list[i]['answer'], 
                    height=100,
                    key=answer_key,
                    label_visibility='collapsed'
                )
                
                # Update session state with the latest value from the text area
                current_qa_list[i]['answer'] = answer 
                st.markdown("---") 
                
            submit_button = st.form_submit_button("Submit & Evaluate Answers", use_container_width=True, type="primary")

            if submit_button:
                
                if all(item['answer'].strip() for item in current_qa_list):
                    with st.spinner("Sending answers to AI Evaluator..."):
                        try:
                            # Use the adapted evaluation function
                            report = evaluate_interview_answers(
                                current_qa_list,
                                context_for_eval # Full resume text or JD content
                            )
                            st.session_state[current_report_key] = report
                            st.success("Evaluation complete! See the report below.")
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
                            st.session_state[current_report_key] = f"Evaluation failed: {e}\n{traceback.format_exc()}"
                else:
                    st.error("Please answer all generated questions before submitting.")
        
        if st.session_state.get(current_report_key):
            st.markdown("---")
            st.subheader("3. AI Evaluation Report")
            st.markdown(st.session_state[current_report_key])
            
        st.markdown("---")
        if st.button(f"🗑️ Clear {mode.upper()} Practice Session (Questions and Answers)", key=f"clear_interview_prep_session_{mode}"):
            clear_interview_state(mode)
            st.success("Practice session cleared.")
            st.rerun()

def interview_preparation_tab():
    """
    Interview Preparation Tab Logic with two sub-tabs: Resume Based and JD Based.
    """
    st.header("🎤 Interview Preparation Tools")

    # Determine if a resume/CV is ready
    is_resume_parsed = (
        st.session_state.get('parsed') is not None and
        st.session_state.parsed.get('name') is not None and
        st.session_state.parsed.get('error') is None
    )
    
    is_jd_loaded = bool(st.session_state.get('candidate_jd_list'))

    # Check if we are running in Mock Mode
    is_mock_mode = isinstance(client, MockGroqClient) and not GROQ_API_KEY
    
    if not GROQ_API_KEY and not is_mock_mode:
        st.error("Cannot use Interview Prep: GROQ_API_KEY is not configured.")
        return

    # Initialize Interview Prep States for both modes and the mode tracker
    if 'iq_mode' not in st.session_state: st.session_state.iq_mode = 'resume' 
    if 'iq_output_resume' not in st.session_state: st.session_state.iq_output_resume = ""
    if 'interview_qa_resume' not in st.session_state: st.session_state.interview_qa_resume = [] 
    if 'evaluation_report_resume' not in st.session_state: st.session_state.evaluation_report_resume = "" 
    
    if 'iq_output_jd' not in st.session_state: st.session_state.iq_output_jd = ""
    if 'interview_qa_jd' not in st.session_state: st.session_state.interview_qa_jd = [] 
    if 'evaluation_report_jd' not in st.session_state: st.session_state.evaluation_report_jd = "" 
    
    st.markdown("---")

    tab_resume, tab_jd = st.tabs(["👤 Resume Based Q&A", "💼 JD Based Q&A"])

    with tab_resume:
        st.session_state.iq_mode = 'resume'
        
        if not is_resume_parsed:
            st.warning("Please upload and successfully parse a resume (or generate one in CV Management) first.")
            return

        # Generate section options dynamically
        parsed_keys = st.session_state.parsed.keys()
        question_section_options = [k.replace('_', ' ').title() for k in parsed_keys if k not in ['name', 'email', 'phone', 'error', 'linkedin', 'github', 'personal_details']]
        # Only sections with valid content
        question_section_options = sorted([o for o in question_section_options if o and st.session_state.parsed.get(o.lower().replace(' ', '_')) and str(st.session_state.parsed.get(o.lower().replace(' ', '_'))).strip()])

        if not question_section_options:
            st.error("No relevant sections (Experience, Skills, Projects) found in the parsed resume for question generation.")
            return
            
        st.subheader("1. Generate Interview Questions (Resume)")
        
        section_choice = st.selectbox(
            "Select Resume Section to Focus On", 
            question_section_options, 
            key='iq_section_resume_c',
            on_change=lambda: clear_interview_state('resume')
        )
        
        if st.button("Generate Resume Questions", key='iq_btn_resume_c', use_container_width=True):
            with st.spinner("Generating questions based on resume section..."):
                try:
                    # Clear current mode state first
                    clear_interview_state('resume')

                    # Call the unified generation function (Mode: resume)
                    raw_questions_response = generate_interview_questions(
                        source_data=st.session_state.parsed, 
                        source_type='resume', 
                        identifier=section_choice
                    )
                    
                    if raw_questions_response.startswith("Error:"):
                         st.error(raw_questions_response)
                         st.session_state.iq_output_resume = raw_questions_response
                         return

                    st.session_state.iq_output_resume = raw_questions_response
                    q_list = parse_questions_from_raw(raw_questions_response)
                        
                    st.session_state.interview_qa_resume = q_list
                    
                    if q_list:
                        st.success(f"Generated {len(q_list)} questions based on your **{section_choice}** section.")
                    else:
                        st.warning(f"Could not parse any questions from the LLM response.")
                    
                except Exception as e:
                    st.error(f"Error generating questions: {e}\nTrace: {traceback.format_exc()}")
                    st.session_state.iq_output_resume = "Error generating questions."
                    st.session_state.interview_qa_resume = []
        
        # Display/Evaluation Logic for Resume Mode
        display_evaluation_form('resume', st.session_state.interview_qa_resume, st.session_state.full_text)


    with tab_jd:
        st.session_state.iq_mode = 'jd'

        if not is_jd_loaded:
            st.warning("Please load Job Descriptions in the 'JD Management' tab first.")
            return
            
        st.subheader("1. Generate Interview Questions (JD)")
        
        jd_names = [jd.get('name') for jd in st.session_state.candidate_jd_list if jd.get('name')]
        selected_jd_name = st.selectbox(
            "Select Job Description",
            options=jd_names,
            key='iq_jd_name_c',
            on_change=lambda: clear_interview_state('jd')
        )

        selected_jd = next((jd for jd in st.session_state.candidate_jd_list if jd.get('name') == selected_jd_name), None)
        
        if st.button("Generate JD Questions", key='iq_btn_jd_c', use_container_width=True):
            if not selected_jd:
                st.error("Please select a Job Description.")
                return

            with st.spinner(f"Generating questions based on JD: {selected_jd_name}..."):
                try:
                    # Clear current mode state first
                    clear_interview_state('jd')
                    
                    # Call the unified generation function (Mode: jd)
                    raw_questions_response = generate_interview_questions(
                        source_data=selected_jd.get('name', 'N/A'), 
                        source_type='jd', 
                        identifier=selected_jd.get('content', '')
                    )
                    
                    if raw_questions_response.startswith("Error:"):
                         st.error(raw_questions_response)
                         st.session_state.iq_output_jd = raw_questions_response
                         return

                    st.session_state.iq_output_jd = raw_questions_response
                    q_list = parse_questions_from_raw(raw_questions_response)
                        
                    st.session_state.interview_qa_jd = q_list
                    
                    if q_list:
                        st.success(f"Generated {len(q_list)} questions based on **{selected_jd_name}**.")
                    else:
                        st.warning(f"Could not parse any questions from the LLM response.")
                    
                except Exception as e:
                    st.error(f"Error generating questions: {e}\nTrace: {traceback.format_exc()}")
                    st.session_state.iq_output_jd = "Error generating questions."
                    st.session_state.interview_qa_jd = []

        # Display/Evaluation Logic for JD Mode
        display_evaluation_form('jd', st.session_state.interview_qa_jd, selected_jd.get('content', '') if selected_jd else "")


# --------------------------------------------------------------------------------------
# NEW TAB: GAP ANALYSIS & COURSE PLAN
# --------------------------------------------------------------------------------------

def gap_analysis_tab():
    """
    Tab to analyze gaps from the top matched JD and generate a course plan.
    """
    st.header("💡 Gap Analysis & Course Plan")
    st.markdown("This tool analyzes your biggest skill gaps from your best-matched Job Description and suggests a course plan and certifications to close the gap.")
    st.markdown("---")

    is_resume_parsed = (
        st.session_state.get('parsed', {}).get('name') is not None and 
        st.session_state.parsed.get('error') is None
    )

    if not is_resume_parsed:
        st.warning("⚠️ **Course Plan Disabled:** Please upload and successfully parse a resume (or generate one in CV Management) first.")
        return
        
    if not st.session_state.get('candidate_match_results'):
        st.error("❌ **Course Plan Disabled:** Please run the **Batch JD Match** analysis first to identify your best fit JD.")
        return

    # 1. Identify the Top Matched JD
    # Results are stored in sorted order (highest score first)
    top_match = st.session_state.candidate_match_results[0]
    top_jd_name = top_match['jd_name']
    
    # Extract the full JD content for context
    top_jd_item = next((jd for jd in st.session_state.candidate_jd_list if jd.get('name') == top_jd_name), None)
    
    if not top_jd_item:
        st.error("Could not find the full JD content for the top match. Please re-run the Batch Match.")
        return

    # Extract the Gaps/Areas for Improvement section from the full analysis output
    gaps_content = top_match.get('gaps', 'Error: Gaps analysis not found.')
    
    if 'gap_analysis_plan' not in st.session_state:
        st.session_state.gap_analysis_plan = ""

    st.subheader(f"1. Top Match Analysis")
    st.info(f"The analysis focuses on your best-matching JD: **{top_jd_name.replace('--- Simulated JD for: ', '')}** (Score: **{top_match['overall_score']}/10**)")
    
    st.markdown("##### Identified Skill Gaps from AI Match Report:")
    if "No significant gaps identified" in gaps_content or gaps_content.startswith("Error"):
        st.warning(gaps_content)
        gap_summary = "No immediate, specific technical gaps found. Focus on general upskilling for the target role."
    else:
        # Simple formatting cleanup for bulleted list
        st.markdown(gaps_content)
        gap_summary = gaps_content.replace('\n', ' ').strip()
        
    st.markdown("---")

    st.subheader(f"2. Generate Detailed Course Plan")
    
    if st.button("🚀 Generate Course Plan & Certifications", use_container_width=True, type="primary"):
        with st.spinner(f"Generating comprehensive course plan for **{top_jd_name}**..."):
            
            # Fetch candidate skills for better plan generation context
            candidate_skills = st.session_state.parsed.get('skills', [])
            
            plan = generate_gap_course_plan(
                gap_analysis_text=gap_summary,
                jd_role=top_jd_item.get('role', 'Target Role'),
                candidate_skills=candidate_skills
            )
            st.session_state.gap_analysis_plan = plan
            st.rerun()

    st.markdown("---")
    
    if st.session_state.gap_analysis_plan:
        st.subheader("3. AI-Generated 'How to Fill the Gap' Plan")
        
        if st.session_state.gap_analysis_plan.startswith("AI Generation Error") or st.session_state.gap_analysis_plan.startswith("No specific gaps"):
            st.error(st.session_state.gap_analysis_plan)
        else:
            st.markdown(st.session_state.gap_analysis_plan)
            
        st.markdown("---")
        
        # Download button for the plan
        plan_filename = f"{st.session_state.parsed['name'].replace(' ', '_')}_GapPlan_{top_jd_item.get('role', 'Job').replace('/', '_').replace(' ', '_')}.md"
        plan_data_uri = get_download_link(st.session_state.gap_analysis_plan, plan_filename, 'markdown', title="Gap Analysis Course Plan")

        col_dl, _ = st.columns([1, 3])
        with col_dl:
            render_download_button(
                plan_data_uri, 
                plan_filename, 
                f"⬇️ Download Course Plan (.md)", 
                'markdown'
            )
    else:
        st.info("Click the 'Generate Course Plan & Certifications' button above to get your personalized study roadmap.")


# --------------------------------------------------------------------------------------
# END GAP ANALYSIS TAB
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# CHATBOT FUNCTIONALITY (unchanged)
# --------------------------------------------------------------------------------------

def qa_on_resume(question):
    """Chatbot for Resume (Q&A) using LLM."""
    global client, GROQ_MODEL, GROQ_API_KEY
    
    if not GROQ_API_KEY and not isinstance(client, MockGroqClient):
        return "AI Chatbot Disabled: GROQ_API_KEY not set."
        
    parsed_json = st.session_state.parsed
    full_text = st.session_state.full_text
    
    if not parsed_json or parsed_json.get('error') is not None:
         return "Please parse a valid resume first to enable the Q&A feature."

    prompt = f"""Given the following resume information:
    Resume Text: {full_text}
    Parsed Resume Data (JSON): {json.dumps(parsed_json, indent=2)}
    Answer the following question about the resume concisely and directly.
    If the information is not present, state that clearly and briefly (e.g., 'Information not found on the resume.').
    Question: {question}
    """
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Chatbot Error: Failed to get response from LLM. Error: {e}"


def qa_on_jd(question, jd_content):
    """Chatbot for Job Description (Q&A) using LLM."""
    global client, GROQ_MODEL, GROQ_API_KEY
    
    if not GROQ_API_KEY and not isinstance(client, MockGroqClient):
        return "AI Chatbot Disabled: GROQ_API_KEY not set."

    if not jd_content or not jd_content.strip():
        return "Please select a valid Job Description to chat about."

    prompt = f"""Given the following Job Description (JD) text:
    Job Description Text: {jd_content}
    Answer the following question about the Job Description concisely and directly.
    If the information is not present, state that clearly and briefly (e.g., 'The JD does not specify that information.').
    Question: {question}
    """
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Chatbot Error: Failed to get response from LLM. Error: {e}"

def resume_qa_content():
    """Content for the Resume Q&A sub-tab."""
    st.subheader("👤 Resume Q&A Chatbot")
    st.markdown("Ask specific questions about the currently loaded resume.")

    is_data_loaded_and_valid = (
        st.session_state.get('parsed', {}).get('name') is not None and 
        st.session_state.get('parsed', {}).get('error') is None
    )
    
    if not is_data_loaded_and_valid:
        st.warning("⚠️ **Q&A Disabled:** Please parse a valid resume (or generate one in CV Management) first.")
        return
    
    if "resume_chatbot_history" not in st.session_state:
        st.session_state.resume_chatbot_history = []

    st.info(f"Chatting about: **{st.session_state.parsed['name']}**")
    st.markdown("---")
    
    for message in st.session_state.resume_chatbot_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the resume...", key="resume_qa_input"):
        st.session_state.resume_chatbot_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            ai_response = qa_on_resume(prompt)

        with st.chat_message("assistant"):
            st.markdown(ai_response)
            
        st.session_state.resume_chatbot_history.append({"role": "assistant", "content": ai_response})
        st.rerun()

    if st.session_state.resume_chatbot_history:
        st.markdown("---")
        if st.button("🗑️ Clear Resume Chat History", key="clear_resume_chatbot_history"):
            st.session_state.resume_chatbot_history = []
            st.rerun()

def jd_qa_content():
    """Content for the JD Q&A sub-tab."""
    st.subheader("💼 JD Q&A Chatbot")
    st.markdown("Select a Job Description and ask questions about its requirements.")

    if not st.session_state.get('candidate_jd_list'):
        st.warning("⚠️ **Q&A Disabled:** Please load Job Descriptions in the 'JD Management' tab first.")
        return

    jd_names = [jd.get('name') for jd in st.session_state.candidate_jd_list if jd.get('name')]
    selected_jd_name = st.selectbox(
        "Select Job Description",
        options=jd_names,
        key="selected_jd_for_qa"
    )

    if "jd_chatbot_history" not in st.session_state:
        st.session_state.jd_chatbot_history = {} 

    selected_jd = next((jd for jd in st.session_state.candidate_jd_list if jd.get('name') == selected_jd_name), None)
    jd_content = selected_jd.get('content', '') if selected_jd else ""

    current_jd_history = st.session_state.jd_chatbot_history.setdefault(selected_jd_name, [])

    st.info(f"Chatting about: **{selected_jd_name}** (Role: {selected_jd.get('role', 'N/A')})")
    st.markdown("---")

    for message in current_jd_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask about the requirements of: {selected_jd_name}...", key="jd_qa_input"):
        current_jd_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            ai_response = qa_on_jd(prompt, jd_content)

        with st.chat_message("assistant"):
            st.markdown(ai_response)
            
        current_jd_history.append({"role": "assistant", "content": ai_response})
        st.rerun()

    if current_jd_history:
        st.markdown("---")
        if st.button(f"🗑️ Clear Chat History for {selected_jd_name}", key="clear_jd_chatbot_history"):
            st.session_state.jd_chatbot_history[selected_jd_name] = []
            st.rerun()

def chatbot_tab_content():
    """Main Content for the Chatbot Tab with sub-tabs."""
    st.header("🤖 AI Chatbot Assistant")
    
    tab_resume, tab_jd = st.tabs(["👤 Resume Q&A", "💼 JD Q&A"])
    
    with tab_resume:
        resume_qa_content()
        
    with tab_jd:
        jd_qa_content()

# --------------------------------------------------------------------------------------
# END CHATBOT FUNCTIONALITY
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# NEW TAB: CV MANAGEMENT (Form-Based Resume Generation)
# --------------------------------------------------------------------------------------

def cv_management_tab():
    """
    Tab for generating a CV using a structured form, replacing the need for file parsing.
    """
    st.header("📝 CV Management: Generate Form-Based CV")
    st.markdown("Manually enter your details to generate a structured CV/Resume for analysis.")

    if 'generated_cv_data' not in st.session_state:
        st.session_state.generated_cv_data = {}
        
    if 'generated_cv_education' not in st.session_state:
        st.session_state.generated_cv_education = []
        
    if 'generated_cv_projects' not in st.session_state:
        st.session_state.generated_cv_projects = []

    if 'generated_cv_experience' not in st.session_state:
        st.session_state.generated_cv_experience = []
        
    if 'generated_cv_certifications' not in st.session_state:
        st.session_state.generated_cv_certifications = []

    
    st.markdown("---")
    
    with st.form("cv_generation_form"):
        st.subheader("Personal Details & Skills")
        
        col_name, col_contact = st.columns(2)
        with col_name:
            name = st.text_input("Full Name", value=st.session_state.generated_cv_data.get('name', ''))
        with col_contact:
            email = st.text_input("Email", value=st.session_state.generated_cv_data.get('email', ''))
            phone = st.text_input("Phone", value=st.session_state.generated_cv_data.get('phone', ''))
        
        st.text_area("Key Skills (Comma Separated)", 
            value=", ".join(st.session_state.generated_cv_data.get('skills', [])), 
            key="cv_skills_input"
        )
        
        st.markdown("---")
        st.subheader("Education")
        
        # Education Entry Mini-Form
        with st.expander("Add/View Education Entries", expanded=False):
            if st.button("➕ Add New Education Entry", key="add_edu_btn"):
                st.session_state.generated_cv_education.append({"degree": "", "university": "", "year_from": "", "year_to": ""})
            
            for i, edu in enumerate(st.session_state.generated_cv_education):
                st.markdown(f"**Entry {i+1}**")
                col_deg, col_uni = st.columns(2)
                with col_deg:
                    edu['degree'] = st.text_input("Degree", value=edu['degree'], key=f"edu_degree_{i}")
                with col_uni:
                    edu['university'] = st.text_input("University", value=edu['university'], key=f"edu_uni_{i}")
                col_from, col_to, col_del = st.columns([1, 1, 0.5])
                with col_from:
                    edu['year_from'] = st.text_input("Year From", value=edu['year_from'], key=f"edu_from_{i}")
                with col_to:
                    edu['year_to'] = st.text_input("Year To/Present", value=edu['year_to'], key=f"edu_to_{i}")
                with col_del:
                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️", key=f"delete_edu_{i}"):
                        st.session_state.generated_cv_education.pop(i)
                        st.experimental_rerun()
                        
        st.markdown("---")
        st.subheader("Projects")

        # Projects Entry Mini-Form
        with st.expander("Add/View Project Entries", expanded=False):
            if st.button("➕ Add New Project Entry", key="add_project_btn"):
                st.session_state.generated_cv_projects.append({"project_name": "", "description": "", "applink": "", "tools_used": ""})
            
            for i, proj in enumerate(st.session_state.generated_cv_projects):
                st.markdown(f"**Project {i+1}**")
                proj['project_name'] = st.text_input("Project Name", value=proj['project_name'], key=f"proj_name_{i}")
                proj['description'] = st.text_area("Description", value=proj['description'], key=f"proj_desc_{i}")
                col_link, col_tools, col_del = st.columns([1, 1, 0.5])
                with col_link:
                    proj['applink'] = st.text_input("App/Demo Link", value=proj['applink'], key=f"proj_link_{i}")
                with col_tools:
                    proj['tools_used'] = st.text_input("Tools Used (Comma Separated)", value=proj['tools_used'], key=f"proj_tools_{i}")
                with col_del:
                    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️", key=f"delete_proj_{i}"):
                        st.session_state.generated_cv_projects.pop(i)
                        st.experimental_rerun()

        st.markdown("---")
        st.subheader("Experience")

        # Experience Entry Mini-Form
        with st.expander("Add/View Experience Entries", expanded=False):
            if st.button("➕ Add New Experience Entry", key="add_exp_btn"):
                st.session_state.generated_cv_experience.append({"company_name": "", "role": "", "ctc": "", "year_from": "", "year_to": ""})
            
            for i, exp in enumerate(st.session_state.generated_cv_experience):
                st.markdown(f"**Job {i+1}**")
                exp['company_name'] = st.text_input("Company Name", value=exp['company_name'], key=f"exp_company_{i}")
                exp['role'] = st.text_input("Role", value=exp['role'], key=f"exp_role_{i}")
                exp['ctc'] = st.text_input("CTC (Optional)", value=exp['ctc'], key=f"exp_ctc_{i}")
                col_from, col_to, col_del = st.columns([1, 1, 0.5])
                with col_from:
                    exp['year_from'] = st.text_input("Year From", value=exp['year_from'], key=f"exp_from_{i}")
                with col_to:
                    exp['year_to'] = st.text_input("Year To/Present", value=exp['year_to'], key=f"exp_to_{i}")
                with col_del:
                    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️", key=f"delete_exp_{i}"):
                        st.session_state.generated_cv_experience.pop(i)
                        st.experimental_rerun()

        st.markdown("---")
        st.subheader("Certifications")

        # Certification Entry Mini-Form
        with st.expander("Add/View Certifications", expanded=False):
            if st.button("➕ Add New Certification", key="add_cert_btn"):
                st.session_state.generated_cv_certifications.append({"title": "", "given_by": "", "received_date": ""})
            
            for i, cert in enumerate(st.session_state.generated_cv_certifications):
                st.markdown(f"**Certification {i+1}**")
                cert['title'] = st.text_input("Title", value=cert['title'], key=f"cert_title_{i}")
                col_by, col_date, col_del = st.columns([1, 1, 0.5])
                with col_by:
                    cert['given_by'] = st.text_input("Given By (Issuing Body)", value=cert['given_by'], key=f"cert_by_{i}")
                with col_date:
                    cert['received_date'] = st.text_input("Date Received", value=cert['received_date'], help="e.g., Nov 2024", key=f"cert_date_{i}")
                with col_del:
                    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️", key=f"delete_cert_{i}"):
                        st.session_state.generated_cv_certifications.pop(i)
                        st.experimental_rerun()


        st.markdown("---")
        submitted = st.form_submit_button("Generate & Load CV for Analysis", type="primary", use_container_width=True)

    if submitted:
        if not name.strip() or not email.strip():
            st.error("Please enter at least your Full Name and Email.")
            return

        # 1. Compile Structured Data
        compiled_education = [
            f"{edu['degree']} in {edu['university']} ({edu['year_from']} - {edu['year_to']})" 
            for edu in st.session_state.generated_cv_education if edu['degree'] and edu['university']
        ]
        
        compiled_projects = [
            f"{proj['project_name']} ({proj['tools_used']}): {proj['description']} [Link: {proj['applink']}]" 
            for proj in st.session_state.generated_cv_projects if proj['project_name'] and proj['description']
        ]
        
        compiled_experience = []
        for exp in st.session_state.generated_cv_experience:
            if exp['company_name'] and exp['role']:
                ctc_display = f" (CTC: {exp['ctc']})" if exp['ctc'].strip() else ""
                compiled_experience.append(
                    f"{exp['role']} at {exp['company_name']} ({exp['year_from']} - {exp['year_to']}){ctc_display}"
                )
                
        compiled_certifications = [
            f"{cert['title']} (Issued by: {cert['given_by']}, Date: {cert['received_date']})" 
            for cert in st.session_state.generated_cv_certifications if cert['title'] and cert['given_by']
        ]
        
        skills_list = [s.strip() for s in st.session_state.cv_skills_input.split(',') if s.strip()]

        generated_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "personal_details": f"Candidate: {name}. Contact: {email}",
            "skills": skills_list,
            "education": compiled_education,
            "experience": compiled_experience,
            "projects": compiled_projects,
            "certifications": compiled_certifications,
            "strength": [],
            "linkedin": "",
            "github": "",
            "error": None
        }

        # 2. Store original form data (for persistence)
        st.session_state.generated_cv_data = generated_data

        # 3. Load into main parsing state (source_type='generated')
        result = parse_and_store_resume(generated_data, file_name_key='generated_cv_candidate', source_type='generated')
        
        if result.get('error') is None:
            st.session_state.parsed = result['parsed']
            st.session_state.full_text = result['full_text']
            st.session_state.excel_data = result['excel_data'] 
            st.session_state.parsed['name'] = result['name'] 
            clear_interview_state('resume')
            clear_interview_state('jd')
            if 'gap_analysis_plan' in st.session_state: del st.session_state['gap_analysis_plan']
            st.success(f"✅ Successfully generated and loaded CV for **{result['name']}**.")
            st.info("The generated CV data is now the active resume for all matching and AI analysis features.")
            st.rerun()
        else:
            st.error(f"Generation failed internally: {result['error']}")
            st.session_state.parsed = {"error": result['error'], "name": result['name']}
            st.session_state.full_text = result['full_text'] or ""

# --------------------------------------------------------------------------------------
# END CV MANAGEMENT TAB
# --------------------------------------------------------------------------------------


# -------------------------
# CANDIDATE DASHBOARD FUNCTION 
# -------------------------

def candidate_dashboard():
    # Set page config once at the start
    st.set_page_config(layout="wide", page_title="PragyanAI Candidate Dashboard")
    
    st.title("🧑‍💻 Candidate Dashboard")
            
    st.markdown("---")

    # --- Session State Initialization ---
    if "parsed" not in st.session_state: st.session_state.parsed = {} 
    if "full_text" not in st.session_state: st.session_state.full_text = ""
    if "excel_data" not in st.session_state: st.session_state.excel_data = None
    if "candidate_uploaded_resumes" not in st.session_state: st.session_state.candidate_uploaded_resumes = []
    if "pasted_cv_text" not in st.session_state: st.session_state.pasted_cv_text = ""
    if "current_parsing_source_name" not in st.session_state: st.session_state.current_parsing_source_name = None 
    
    if "candidate_jd_list" not in st.session_state: st.session_state.candidate_jd_list = []
    if "candidate_match_results" not in st.session_state: st.session_state.candidate_match_results = []
    if 'filtered_jds_display' not in st.session_state: st.session_state.filtered_jds_display = []
    if 'last_selected_skills' not in st.session_state: st.session_state.last_selected_skills = []
    if 'generated_cover_letter' not in st.session_state: st.session_state.generated_cover_letter = "" 
    if 'cl_jd_name' not in st.session_state: st.session_state.cl_jd_name = "" 
    
    # --- NEW CV Management States (for form persistence) ---
    if 'generated_cv_data' not in st.session_state: st.session_state.generated_cv_data = {}
    if 'generated_cv_education' not in st.session_state: st.session_state.generated_cv_education = []
    if 'generated_cv_projects' not in st.session_state: st.session_state.generated_cv_projects = []
    if 'generated_cv_experience' not in st.session_state: st.session_state.generated_cv_experience = []
    if 'generated_cv_certifications' not in st.session_state: st.session_state.generated_cv_certifications = []
    
    # --- INTERVIEW Preparation States ---
    if 'iq_mode' not in st.session_state: st.session_state.iq_mode = 'resume' 
    if 'iq_output_resume' not in st.session_state: st.session_state.iq_output_resume = ""
    if 'interview_qa_resume' not in st.session_state: st.session_state.interview_qa_resume = [] 
    if 'evaluation_report_resume' not in st.session_state: st.session_state.evaluation_report_resume = "" 
    
    if 'iq_output_jd' not in st.session_state: st.session_state.iq_output_jd = ""
    if 'interview_qa_jd' not in st.session_state: st.session_state.interview_qa_jd = [] 
    if 'evaluation_report_jd' not in st.session_state: st.session_state.evaluation_report_jd = "" 
    
    # --- GAP ANALYSIS STATE ---
    if 'gap_analysis_plan' not in st.session_state: st.session_state.gap_analysis_plan = ""
    
    if "resume_chatbot_history" not in st.session_state: st.session_state.resume_chatbot_history = []
    if "jd_chatbot_history" not in st.session_state: st.session_state.jd_chatbot_history = {} 
    
    if 'candidate_job_types' not in st.session_state: 
        st.session_state.candidate_job_types = DEFAULT_JOB_TYPES 

    # --- Main Content with Tabs (New tab added) ---
    tab_cv_manage, tab_parsing, tab_data_view, tab_jd, tab_batch_match, tab_filter_jd, tab_chatbot, tab_cover_letter, tab_interview_prep, tab_gap_analysis = st.tabs(
        [
            "📝 CV Management", # NEW TAB
            "📄 Resume Parsing", 
            "✨ Parsed Data View", 
            "📚 JD Management", 
            "🎯 Batch JD Match", 
            "🔍 Filter JD", 
            "🤖 Chatbot", 
            "✉️ Generate Cover Letter", 
            "🎤 Interview Preparation",
            "💡 Gap Analysis & Course Plan"
        ]
    )
    
    with tab_cv_manage:
        cv_management_tab()
        
    with tab_parsing:
        resume_parsing_tab()
        
    with tab_data_view:
        parsed_data_tab()
        
    with tab_jd:
        jd_management_tab_candidate()
        
    with tab_batch_match:
        jd_batch_match_tab()
        
    with tab_filter_jd:
        filter_jd_tab_content()
        
    with tab_chatbot:
        chatbot_tab_content()
        
    with tab_cover_letter:
        generate_cover_letter_tab() 
        
    with tab_interview_prep:
        interview_preparation_tab() 
        
    with tab_gap_analysis:
        gap_analysis_tab()


# -------------------------
# MAIN APP EXECUTION
# -------------------------

if __name__ == '__main__':
    candidate_dashboard()
