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
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# --- Default/Mock Data for Filtering ---
DEFAULT_ROLES = ["Data Scientist", "Cloud Engineer", "Software Engineer", "AI/ML Engineer"]
DEFAULT_JOB_TYPES = ["Full-time", "Contract", "Remote"]
STARTER_KEYWORDS = {
    "Python", "MySQL", "GCP", "cloud computing", "ML", 
    "API services", "LLM integration", "JavaScript", "SQL", "AWS", "MLOps", "Data Visualization"
}
# --- End Default/Mock Data ---


# --- Define MockGroqClient globally ---

class MockGroqClient:
    """Mock client for local testing when Groq is not available or key is missing."""
    # The structure must mimic the actual Groq client for client = Groq(...) to work.
    def chat(self):
        class Completions:
            def create(self, **kwargs):
                prompt_content = kwargs.get('messages', [{}])[0].get('content', '')
                
                # Check if it's a JD Q&A call
                if "Answer the following question about the Job Description concisely and directly." in prompt_content:
                    question_match = re.search(r'Question:\s*(.*)', prompt_content)
                    question = question_match.group(1).strip() if question_match else "a question"
                    
                    if 'role' in question.lower():
                        return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': 'The required role in this Job Description is Cloud Engineer.'})})()]})
                    elif 'experience' in question.lower():
                        return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': 'The job requires 3+ years of experience in AWS/GCP and infrastructure automation.'})})()]})
                    else:
                        return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': f'Mock answer for JD question: The JD mentions Python and Docker as key skills.'})})()]})

                # Check if it's a Resume Q&A call
                elif "Answer the following question about the resume concisely and directly." in prompt_content:
                    question_match = re.search(r'Question:\s*(.*)', prompt_content)
                    question = question_match.group(1).strip() if question_match else "a question"
                    
                    if 'name' in question.lower():
                        return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': 'The candidate\'s name is Vivek Swamy.'})})()]})
                    elif 'skills' in question.lower():
                        return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': 'Key skills include Python, SQL, AWS, and MLOps.'})})()]})
                    else:
                        return type('MockResponse', (object,), {'choices': [type('Choice', (object,), {'message': type('Message', (object,), {'content': f'Based on the mock resume data, I can provide a simulated answer to your question about {question}.'})})()]})


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
                    "experience": [
                        "Software Intern at Mock Solutions (2024-2025). Built deployment pipelines.", 
                        "Data Analyst at Test Corp (2022-2024). Managed SQL databases."
                    ], 
                    "certifications": ["Mock Certification in AWS Cloud"], 
                    "projects": ["Mock Project: Built an MLOps pipeline using Docker and Kubernetes."], 
                    "strength": ["Mock Strength"], 
                }
                
                # Mock response content for GroqClient initialization check
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
                    - Mock Gap 1
                    
                    Overall Summary: Mock summary for score {score}.
                    """
                    message_obj = type('Message', (object,), {'content': mock_fit_output})()
                    choice_obj = type('Choice', (object,), {'message': message_obj})()
                    response_obj = type('MockResponse', (object,), {'choices': [choice_obj]})()
                    return response_obj
                
                # Return standard parsing mock or Q&A mock
                return super().create(**kwargs)

        return FitCompletions()

try:
    # Attempt to import the real Groq client
    from groq import Groq
    
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        # Check if the client is really ready or just a placeholder
        if client:
             class GroqPlaceholder(Groq): 
                 def __init__(self, api_key): 
                     super().__init__(api_key=api_key)
                     self.client_ready = True
             client = GroqPlaceholder(api_key=GROQ_API_KEY)
        else:
            raise ValueError("Groq client not initialized successfully, falling back to Mock.")

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Using Mock Client.")
        
except (ImportError, ValueError, NameError) as e:
    # Fallback to Mock Client
    client = MockGroqClient()
    
# --- END API SETUP ---


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
    """
    Sends resume text to the LLM for structured information extraction.
    """
    
    def get_fallback_name():
        # Using a consistent fallback name for error handling
        return "Vivek Swamy" 

    # 1. Handle Pre-flight errors (e.g., failed extraction)
    if text.startswith("[Error"):
        return {"name": "Parsing Error", "error": text}

    # 2. Check for and parse direct JSON content (for JSON file uploads or generated data)
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
            
    # 3. Handle Mock Client execution (Fallback for PDF/DOCX/TXT)
    if isinstance(client, MockGroqClient) or not GROQ_API_KEY:
        try:
            # Use the mock client's default mock response for parsing
            completion = client.chat().create(model=GROQ_MODEL, messages=[{}])
            content = completion.choices[0].message.content.strip()
            parsed_data = json.loads(content)
            
            if not parsed_data.get('name'):
                 parsed_data['name'] = get_fallback_name()
            
            parsed_data['error'] = None 
            return parsed_data
            
        except Exception as e:
            return {"name": get_fallback_name(), "error": f"Mock Client Error: {e}"}

    # 4. Handle Real Groq Client execution 
    
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

        # --- CRITICAL FIX: AGGRESSIVE JSON ISOLATION USING REGEX ---
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
        
        # --- END CRITICAL FIX ---
        
        # Final cleanup for the app structure
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

# --- HELPER FUNCTIONS FOR FILE/TEXT PROCESSING ---

def clear_interview_state():
    """Clears all session state variables related to interview/match sessions."""
    # Note: st.session_state items are deleted if they exist, which is cleaner than setting to empty
    if 'interview_chat_history' in st.session_state: del st.session_state['interview_chat_history']
    if 'current_interview_jd' in st.session_state: del st.session_state['current_interview_jd']
    if 'evaluation_report' in st.session_state: del st.session_state['evaluation_report']
    # Match results are set to empty list rather than deleting, as they are part of the Batch Match state structure
    if 'candidate_match_results' in st.session_state: st.session_state.candidate_match_results = []
    
# Updated signature to match the request
def parse_and_store_resume(content_source, file_name_key, source_type):
    """Handles extraction, parsing, and storage of CV data from either a file or pasted text."""
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
    elif source_type == 'json_data':
        # Used for generated CV data which is already a parsed JSON dictionary
        parsed_data = content_source
        file_name = "Generated_CV"
        st.session_state.current_parsing_source_name = file_name 
        
        # Ensure 'experience' is compiled into a list of strings for compatibility with LLM output
        compiled_experience_list = []
        for exp in parsed_data.get('experience', []):
            if isinstance(exp, dict):
                # Format structured experience back into a standard string format for LLM input/full_text viewing
                # NOTE: Using achievements as a raw string from the structure
                exp_str = f"Job Title: {exp.get('role', 'N/A')}, Company: {exp.get('company', 'N/A')}, Dates: {exp.get('dates', 'N/A')}. CTC: {exp.get('ctc', 'N/A')}. Key Achievements: {exp.get('achievements', 'N/A')}"
                compiled_experience_list.append(exp_str)
            else:
                compiled_experience_list.append(str(exp))
        
        # Replace the structured data with the compiled string list for 'full_text' generation
        data_for_text = parsed_data.copy()
        data_for_text['experience'] = compiled_experience_list

        compiled_text = ""
        for k, v in data_for_text.items():
            if v and k not in ['error']:
                compiled_text += f"## {k.replace('_', ' ').title()}\n\n"
                if isinstance(v, list):
                    # Ensure all items in the list are strings before joining
                    compiled_text += "\n".join([str(item) for item in v if item is not None]) + "\n\n"
                else:
                    compiled_text += str(v) + "\n\n"
        
        final_name = parsed_data.get('name', 'Unknown_Candidate').replace(' ', '_') 
        
        return {
            "parsed": parsed_data, 
            "full_text": compiled_text, 
            "excel_data": None, 
            "name": final_name
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
    compiled_text = ""
    for k, v in parsed_data.items():
        if v and k not in ['error']:
            compiled_text += f"## {k.replace('_', ' ').title()}\n\n"
            if isinstance(v, list):
                # Ensure all items in the list are strings before joining (Fix for TypeError)
                compiled_text += "\n".join([str(item) for item in v if item is not None]) + "\n\n"
            else:
                compiled_text += str(v) + "\n\n"

    # Ensure final_name uses the parsed name
    final_name = parsed_data.get('name', 'Unknown_Candidate').replace(' ', '_') 
    
    return {
        "parsed": parsed_data, 
        "full_text": compiled_text, 
        "excel_data": excel_data, 
        "name": final_name
    }


def get_download_link(data, filename, file_format):
    """
    Generates a base64 encoded download link for the given data and format.
    """
    mime_type = "application/octet-stream"
    
    if file_format == 'json':
        data_bytes = data.encode('utf-8')
        mime_type = "application/json"
    elif file_format == 'markdown':
        data_bytes = data.encode('utf-8')
        mime_type = "text/markdown"
    elif file_format == 'html':
        # Create a simple HTML document for rendering
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>{filename}</title></head>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
        <h1 style="color: #4CAF50;">Parsed Resume Data: {filename.replace('.html', '')}</h1>
        <hr/>
        <pre style="white-space: pre-wrap; word-wrap: break-word; background: #f4f4f4; padding: 10px; border: 1px solid #ddd;">
        {data}
        </pre>
        <p>Generated by PragyanAI</p>
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


@st.cache_data(show_spinner="Analyzing JD for metadata...")
def extract_jd_metadata(jd_text):
    """Mocks the extraction of key metadata (Role, Skills, Job Type) from JD text using LLM."""
    if jd_text.startswith("[Error"):
        return {"role": "Error", "key_skills": ["Error"], "job_type": "Error"}
    
    # Simple heuristic mock
    role_match = re.search(r'(?:Role|Position|Title|Engineer|Scientist)[:\s\n]+([\w\s/-]+)', jd_text, re.IGNORECASE)
    role = role_match.group(1).strip() if role_match else "Software Engineer (Mock)"
    
    # Extract Skills from JD content - ENHANCED SKILL LIST
    skills_match = re.findall(r'(Python|Java|SQL|AWS|Docker|Kubernetes|React|Streamlit|Cloud|Data|ML|LLM|MLOps|Visualization|Deep Learning|TensorFlow|Pytorch)', jd_text, re.IGNORECASE)
    
    # Simple heuristic to improve role names if generic title is found
    if 'data scientist' in jd_text.lower() or 'machine learning' in jd_text.lower():
         role = "Data Scientist/ML Engineer"
    elif 'cloud engineer' in jd_text.lower() or 'aws' in jd_text.lower() or 'gcp' in jd_text.lower():
         role = "Cloud Engineer"
    
    job_type_match = re.search(r'(Full-time|Part-time|Contract|Remote|Hybrid)', jd_text, re.IGNORECASE)
    job_type = job_type_match.group(1) if job_type_match else "Full-time (Mock)"
    
    return {
        "role": role, 
        "key_skills": list(set([s.lower() for s in skills_match])), # Keep all unique skills found
        "job_type": job_type
    }

def extract_jd_from_linkedin_url(url):
    """Mocks the extraction of JD content from a LinkedIn URL."""
    if "linkedin.com/jobs" not in url:
        return f"[Error] Invalid LinkedIn Job URL: {url}"

    # Mock content based on URL structure
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
    
# --- EVALUATE JD FIT FUNCTION (LLM-DEPENDENT) ---
def evaluate_jd_fit(job_description, parsed_json):
    """
    Evaluates how well a resume fits a given job description, 
    including section-wise scores, by calling the Groq LLM API.
    """
    # Use the client object, which can be the real Groq client or the MockGroqClient
    global client, GROQ_MODEL, GROQ_API_KEY
    
    if parsed_json.get('error') is not None: 
         # This message is now explicitly caught in the match_batch_tab to set the score to 'Cannot evaluate'
         return f"Cannot evaluate due to resume parsing errors: {parsed_json['error']}"


    if isinstance(client, MockGroqClient) and not GROQ_API_KEY:
         # In mock mode, use the special mock implementation for fit evaluation
         response = client.chat().create(model=GROQ_MODEL, messages=[{"role": "user", "content": f"Evaluate how well the following resume content matches the provided job description: {job_description}"}])
         return response.choices[0].message.content.strip()


    if not job_description.strip(): return "Please paste a job description."

    # Prepare relevant resume data for the LLM
    # NOTE: The LLM expects a list of strings for 'Experience', so we convert the structured data back.
    
    # 1. Format structured experience list back into a string list
    experience_list_str = []
    for exp in parsed_json.get('experience', []):
        if isinstance(exp, dict):
            # Format structured experience back into a standard string format for LLM input
            exp_str = f"Role: {exp.get('role', 'N/A')}, Company: {exp.get('company', 'N/A')}, Dates: {exp.get('dates', 'N/A')}. CTC: {exp.get('ctc', 'N/A')}. Achievements: {exp.get('achievements', 'N/A')}"
            experience_list_str.append(exp_str)
        else:
            experience_list_str.append(str(exp))
            
    relevant_resume_data = {
        'Skills': parsed_json.get('skills', 'Not found or empty'),
        'Experience': experience_list_str,
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
    4.  **Gaps/Areas for Improvement:** Key requirements in the JD that are missing or weak in the resume.
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
    - Point 1
    - Point 2
    
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
# --- END EVALUATE JD FIT FUNCTION ---

# --- Tab Content Functions ---
    
def resume_parsing_tab():
    # --- TAB 1 (Now tab_parsing): Resume Parsing (MODIFIED: Added Paste Your CV option) ---
    st.header("📄 Resume Upload and Parsing")
    
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
            if st.button(f"Parse and Load: **{file_to_parse.name}**", use_container_width=True):
                with st.spinner(f"Parsing {file_to_parse.name}..."):
                    result = parse_and_store_resume(file_to_parse, file_name_key='single_resume_candidate', source_type='file')
                    
                    if result.get('error') is None:
                        st.session_state.parsed = result['parsed']
                        st.session_state.full_text = result['full_text']
                        st.session_state.excel_data = result['excel_data'] 
                        st.session_state.parsed['name'] = result['name'] 
                        clear_interview_state()
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
                    
                    if result.get('error') is None:
                        st.session_state.parsed = result['parsed']
                        st.session_state.full_text = result['full_text']
                        st.session_state.excel_data = result['excel_data'] 
                        st.session_state.parsed['name'] = result['name'] 
                        clear_interview_state()
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

# --- CV Management Tab Functions ---

# Helper function to convert simple string lists for Education/Skills/Projects back to multiline text
def list_to_text(data_list):
    """Converts a list (potentially containing non-string items) into a multiline string."""
    if not data_list:
        return ""
    if isinstance(data_list, list):
        # Crucial: Ensure every item in the list is converted to a string before joining
        return "\n".join([str(item).strip() for item in data_list if item is not None])
    # Fallback for unexpected non-list, non-empty data
    return str(data_list)
    
# Helper function to format multiline text back into a list of strings
def format_to_list(text_input):
    """Formats multiline text input (newline or comma separated) back into a clean list of strings."""
    if not text_input or not text_input.strip():
        return []
    
    # First split by newline, then strip and filter empty strings
    lines = [item.strip() for item in text_input.split('\n') if item.strip()]
    
    # If the input was a single line of comma-separated items, handle that too
    if len(lines) == 1 and ',' in lines[0]:
        return [item.strip() for item in lines[0].split(',') if item.strip()]
    
    return lines

# Helper function to convert unstructured LLM-parsed experience list (strings) 
# into the new structured list of dicts format, falling back to string if needed.
def initialize_experience_data(initial_data):
    """Initializes the session state structured experience list."""
    
    # If the session state already has the structured list, use it
    if st.session_state.get('structured_experience') is not None:
         return st.session_state.structured_experience
         
    # Otherwise, try to convert the LLM-parsed list of strings into a structured format
    exp_list = initial_data.get("experience", [])
    
    # If the LLM returned structured data (e.g., from a JSON file), use it directly
    if exp_list and isinstance(exp_list, list) and exp_list and isinstance(exp_list[0], dict) and 'role' in exp_list[0]:
        return exp_list
        
    # Attempt to parse unstructured strings. This is a best-effort approach.
    structured_list = []
    
    # Simple regex to try and find Role/Company/Dates (best effort)
    # Pattern looks for 'Role at Company (Date-Date)'
    pattern = re.compile(r"(.+?)\s+at\s+(.+?)\s+\((.+?)\)")
    
    for item in exp_list:
        if not isinstance(item, str): 
            item = str(item)
            
        match = pattern.search(item)
        
        if match:
            role, company, dates = match.groups()
            achievements = item.replace(match.group(0), "").strip(". ").strip() # everything else is achievements
        else:
            # Fallback to simple structure
            role = item.split(" at ")[0].strip() if " at " in item else item
            company = item.split(" at ")[1].split(" (")[0].strip() if " at " in item and " (" in item else "N/A"
            dates = "N/A"
            achievements = "N/A"

        structured_list.append({
            "role": role,
            "company": company,
            "dates": dates,
            "ctc": "N/A",  # Cannot reliably parse CTC from unstructured text
            "achievements": achievements
        })
        
    return structured_list


def generate_cv_form():
    """Allows candidates to enter details via a form to generate a structured CV."""
    
    st.header("📝 Generate Your CV (Form Based)")
    st.markdown("Fill out the form below to create a structured CV/Resume, which will be loaded into the dashboard for analysis.")

    # Initialize the form state with current parsed data if available, or empty defaults
    if 'parsed' in st.session_state and st.session_state.parsed.get('name') and st.session_state.parsed.get('error') is None:
        initial_data = st.session_state.parsed
    else:
        # Default empty structure matching the LLM output format
        initial_data = {
            "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
            "personal_details": "", "skills": [], "education": [], "experience": [], # Experience is an empty list here
            "certifications": [], "projects": [], "strength": []
        }
        
    # --- CRITICAL: Initialize or load structured experience data ---
    if 'structured_experience' not in st.session_state:
        st.session_state.structured_experience = initialize_experience_data(initial_data)

    
    # --- DYNAMIC EXPERIENCE SECTION (OUTSIDE MAIN FORM) ---
    st.markdown("### 2. Work Experience")
    st.markdown("Add your work history entries one by one. Use the 'Remove' buttons below the entries to delete existing ones.")

    # Display existing entries with removal button (MUST be outside st.form for st.button to work)
    if st.session_state.structured_experience:
        for i, exp in enumerate(st.session_state.structured_experience):
            # Use a container or columns to group the display and the remove button
            col_display, col_remove = st.columns([4, 1])
            with col_display:
                st.markdown(f"**{i+1}. {exp['role']}** at **{exp['company']}** ({exp['dates']})")
                st.caption(f"CTC: {exp['ctc']} | Achievements: {exp['achievements'][:80]}...")
            with col_remove:
                # The st.button is now correctly outside the main form context.
                if st.button("🗑️ Remove", key=f"remove_exp_{i}"):
                    st.session_state.structured_experience.pop(i)
                    st.toast(f"Removed experience entry {i+1}.")
                    st.rerun() 
        st.markdown("---")

    # --- MINI-FORM FOR ADDING NEW EXPERIENCE (MUST BE A SEPARATE FORM) ---
    st.markdown("##### Add New Experience Entry")
    with st.form("add_experience_mini_form", clear_on_submit=True):
        col_new1, col_new2, col_new3 = st.columns([2, 1, 2])
        
        with col_new1:
            new_role = st.text_input("Job Role", key="new_role")
            new_dates = st.text_input("Dates (e.g., 2022 - Present)", key="new_dates")
        with col_new2:
             new_ctc = st.text_input("CTC (optional)", key="new_ctc")
        with col_new3:
            new_company = st.text_input("Company Name", key="new_company")
            st.write("") # Spacer

        new_achievements = st.text_area(
            "Key Achievements/Responsibilities (Bullet points/separate by newline)", 
            key="new_achievements", 
            height=80
        )
        
        # Use st.form_submit_button for the button inside this mini-form
        add_submitted = st.form_submit_button("➕ Add Experience Entry", type="secondary", use_container_width=True)

        if add_submitted:
            if new_role and new_company and new_dates:
                # Format achievements into a clean list of strings
                achievement_list = format_to_list(new_achievements)
                
                new_entry = {
                    "role": new_role.strip(),
                    "company": new_company.strip(),
                    "dates": new_dates.strip(),
                    "ctc": new_ctc.strip() or "N/A",
                    "achievements": "\n".join(achievement_list) # Store as multi-line string
                }
                st.session_state.structured_experience.append(new_entry)
                st.success(f"Added: {new_role} at {new_company}")
                # st.rerun() is not needed here as st.form_submit_button causes a rerun and clears the input

            else:
                st.error("Please fill in Job Role, Company Name, and Dates.")
        
    st.markdown("---")
    # --- End Dynamic Experience Section ---

    
    # --- MAIN CV GENERATION FORM ---
    with st.form("cv_generation_form"):
        st.markdown("### 1. Personal Details")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", value=initial_data.get("name", ""), key="cv_name")
            email = st.text_input("Email", value=initial_data.get("email", ""), key="cv_email")
            linkedin = st.text_input("LinkedIn URL", value=initial_data.get("linkedin", ""), key="cv_linkedin")
        with col2:
            phone = st.text_input("Phone Number", value=initial_data.get("phone", ""), key="cv_phone")
            github = st.text_input("GitHub URL", value=initial_data.get("github", ""), key="cv_github")
            personal_details = st.text_area("Personal Summary/Objective", value=initial_data.get("personal_details", ""), key="cv_personal_details", height=100)
            
        st.markdown("---")
        st.markdown("### 3. Core Skills and Education") # Re-numbered

        skills = st.text_area(
            "Skills (e.g., Python, AWS, SQL, Docker - one skill or tool per line or comma separated)",
            value=list_to_text(initial_data.get("skills", [])), 
            height=150,
            key="cv_skills"
        )
        
        col_edu, col_cert = st.columns(2)
        with col_edu:
            education = st.text_area(
                "Education (Degree, Institution, Year - one entry per line)",
                value=list_to_text(initial_data.get("education", [])), 
                height=150,
                key="cv_education"
            )
        with col_cert:
            certifications = st.text_area(
                "Certifications (Certification Name, Provider, Year - one entry per line)",
                value=list_to_text(initial_data.get("certifications", [])), 
                height=150,
                key="cv_certifications"
            )
        
        projects = st.text_area(
            "Projects (Project Name, Description, Technologies Used - one project per line)",
            value=list_to_text(initial_data.get("projects", [])), 
            height=150,
            key="cv_projects"
        )

        # The main submit button for the entire CV data
        submitted = st.form_submit_button("💾 Generate and Load CV Data", type="primary", use_container_width=True)

    if submitted:
        if not name or not email:
            st.error("Please enter at least your Full Name and Email address.")
            return

        # 1. Construct the final structured JSON object
        generated_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "github": github,
            "personal_details": personal_details,
            "skills": format_to_list(skills),
            "education": format_to_list(education),
            "experience": st.session_state.structured_experience, # Use the structured list
            "certifications": format_to_list(certifications),
            "projects": format_to_list(projects),
            # 'strength' field is often implied/part of summary, but we include it for structure consistency
            "strength": format_to_list(initial_data.get("strength", []))
        }

        # 2. Load the generated data into the dashboard state
        with st.spinner(f"Loading generated data for {name}..."):
            # Use the dedicated handling function with source_type='json_data'
            result = parse_and_store_resume(generated_data, file_name_key='generated_cv', source_type='json_data')
            
            # Update session state with the new parsed data and clear matching state
            st.session_state.parsed = result['parsed']
            st.session_state.full_text = result['full_text']
            st.session_state.excel_data = result['excel_data'] 
            st.session_state.parsed['name'] = result['name'] 
            
            # Re-initialize structured_experience based on the newly loaded data
            # NOTE: This step is crucial to ensure the display syncs after load
            st.session_state.structured_experience = initialize_experience_data(st.session_state.parsed)
            
            clear_interview_state()
            
            st.success(f"✅ CV data successfully generated and loaded for **{name}**.")
            st.info("The generated CV data is now available in the 'Parsed Data View' tab and ready for matching.")
            st.rerun()

def cv_management_tab():
    """Main function for the new CV Management Tab."""
    st.header("🛠️ CV Management Tools")
    
    tab_generate_cv = st.tabs(["📝 Generate Your CV (Form Based)"])[0]
    
    with tab_generate_cv:
        generate_cv_form()


# --- JD Management Tab Function ---
        
def jd_management_tab_candidate():
    """JD Management Tab."""
    st.header("📚 Manage Job Descriptions for Matching")
    st.markdown("Add multiple JDs here to compare your resume against them in the next tabs.")
    
    if "candidate_jd_list" not in st.session_state: st.session_state.candidate_jd_list = []
    st.markdown("---")
    
    # JD Type Radio Buttons
    jd_type = st.radio("Select JD Type", ["Single JD", "Multiple JD"], key="jd_type_candidate", index=0)
    st.markdown("### Add JD by:")
    # Method Radio Buttons
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
                            
                        # Use role for a better name display
                        name = f"JD for {metadata.get('role', 'Unknown Role')}"
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
                            metadata = extract_jd_metadata(text)
                            # Use extracted role for better naming
                            name_base = metadata.get('role', f"Pasted JD {len(st.session_state.candidate_jd_list) + i + 1}")
                            st.session_state.candidate_jd_list.append({"name": name_base, "content": text, **metadata})
                            count += 1
                    
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully!")
                        st.rerun() 

    # --- Upload File Section ---
    elif method == "Upload File":
        jd_file_types = ["pdf", "txt", "docx", "md", "json"]
        # File uploader component
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
                # Clear JD Q&A history if it exists
                if 'jd_chatbot_history' in st.session_state: del st.session_state['jd_chatbot_history']
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
    
    # Check if we are running in Mock Mode
    is_mock_mode = isinstance(client, MockGroqClient) and not GROQ_API_KEY
    
    if not is_resume_parsed:
        st.warning("⚠️ Please **upload and parse your resume** in the 'Resume Parsing' tab or **generate a CV** in the 'CV Management' tab first.")
        
        # Display the specific parsing error if it exists
        if st.session_state.get('parsed', {}).get('error') is not None:
             st.error(f"Resume Parsing Error: {st.session_state.parsed.get('error')}")

    elif not st.session_state.candidate_jd_list:
        st.error("❌ Please **add Job Descriptions** in the 'JD Management' tab before running batch analysis.")
        
    elif not GROQ_API_KEY and not is_mock_mode:
        st.error("Cannot use JD Match: GROQ_API_KEY is not configured.")
        
    elif is_mock_mode:
        st.info("ℹ️ Running in **Mock LLM Mode** for fit evaluation. Results are simulated for consistency, but a valid GROQ_API_KEY is recommended for real AI analysis.")
        
    else:
        # Check if the client is not the mock client and the key is set (i.e., we are using the real LLM)
        if not hasattr(client, 'client_ready') or not client.client_ready:
            st.warning("⚠️ LLM client setup failed or key is missing. Match analysis may not be accurate or available.")


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
             # The warning above will already be shown, but we prevent the loop from running
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
                        # Call the LLM-dependent evaluation function
                        fit_output = evaluate_jd_fit(jd_content, parsed_json) 
                        
                        # --- FIX: ROBUST REGEX EXTRACTION FOR SCORE AND PERCENTAGES ---
                        overall_score_match = re.search(r'Overall Fit Score:\s*\[?\s*(\d+)\s*\]?\s*/10', fit_output, re.IGNORECASE)
                        
                        section_analysis_match = re.search(
                            r'--- Section Match Analysis ---\s*(.*?)\s*(?:Strengths/Matches|Overall Summary):', 
                            fit_output, re.DOTALL | re.IGNORECASE
                        )
                        
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
                        
                        # 4. Check for API/Mock/Parsing errors
                        if "AI Evaluation Error" in fit_output:
                            overall_score = "Error (API)"
                        elif "Cannot evaluate" in fit_output:
                            overall_score = "Error (Parse)"
                            
                        # --- END FIX ---

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
                        # This catches parsing errors if the LLM output is malformed even if the API call succeeded.
                        results_with_score.append({
                            "jd_name": jd_name,
                            "overall_score": "Error (Extract)",
                            "numeric_score": -1, 
                            "skills_percent": "Error",
                            "experience_percent": "Error", 
                            "education_percent": "Error", 
                            "full_analysis": f"Error parsing LLM analysis for {jd_name}: {e}\nFull LLM Output:\n---\n{fit_output}\n---"
                        })
                        
                # --- NEW RANKING LOGIC (Handles ties correctly) ---
                results_with_score.sort(key=lambda x: x['numeric_score'], reverse=True)
                
                current_rank = 1
                current_score = -1 
                
                for i, item in enumerate(results_with_score):
                    # Check for ties in score
                    if item['numeric_score'] < current_score:
                        current_rank = i + 1
                        current_score = item['numeric_score']
                    elif i == 0:
                        current_score = item['numeric_score']
                        
                    item['rank'] = current_rank
                    
                    # Clean up the temp score field
                    if 'numeric_score' in item:
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
            
            # Simple fix to make the role name more readable for display if it's the mock-extracted role
            role_display = full_jd_item.get('role', 'N/A').replace("/ML Engineer", " Engineer")
            
            # Clean up score display based on error type
            overall_score_display = item["overall_score"]
            if overall_score_display.startswith("Error"):
                overall_score_display = "Error" 

            display_data.append({
                "Rank": item.get("rank", "N/A"),
                "Job Description (Ranked)": item["jd_name"].replace("--- Simulated JD for: ", ""),
                "Role": role_display, 
                "Job Type": full_jd_item.get('job_type', 'N/A'), 
                "Fit Score (out of 10)": overall_score_display,
                "Skills (%)": item.get("skills_percent", "N/A"),
                "Experience (%)": item.get("experience_percent", "N/A"), 
                "Education (%)": item.get("education_percent", "N/A"), 
            })

        st.dataframe(display_data, use_container_width=True)

        st.markdown("##### Detailed Reports")
        for item in results_df:
            rank_display = f"Rank {item.get('rank', 'N/A')} | "
            
            header_score = item['overall_score']
            if header_score.startswith("Error"):
                 header_score = "Evaluation Error"
                 
            # Ensure the full analysis is displayed with markdown formatting
            header_text = f"{rank_display}Report for **{item['jd_name'].replace('--- Simulated JD for: ', '')}** (Score: **{header_score}** | S: **{item.get('skills_percent', 'N/A')}%** | E: **{item.get('experience_percent', 'N/A')}%** | Edu: **{item.get('education_percent', 'N/A')}%**)"
            with st.expander(header_text):
                # Use st.code to display the LLM output with good formatting
                st.code(item['full_analysis'], language='markdown')


# --- New Filter JD Tab Function ---

def filter_jd_tab_content():
    st.header("🔍 Filter Job Descriptions by Criteria")
    st.markdown("Use the filters below to narrow down your saved Job Descriptions.")

    if not st.session_state.candidate_jd_list:
        st.info("No Job Descriptions are currently loaded. Please add JDs in the 'JD Management' tab.")
        if 'filtered_jds_display' not in st.session_state:
            st.session_state.filtered_jds_display = []
        return
    
    # --- Skill and Role Extraction ---
    global DEFAULT_ROLES, DEFAULT_JOB_TYPES, STARTER_KEYWORDS
    
    unique_roles = sorted(list(set(
        [item.get('role', 'General Analyst') for item in st.session_state.candidate_jd_list] + DEFAULT_ROLES
    )))
    unique_job_types = sorted(list(set(
        [item.get('job_type', 'Full-time') for item in st.session_state.candidate_jd_list] + DEFAULT_JOB_TYPES
    )))
    
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
    # --- End Extraction ---

    # --- Start Filter Form ---
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

    # --- Start Filtering Logic ---
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
            
            # 1. Role Filter
            role_match = (selected_role == "All Roles") or (selected_role == jd_role)
            
            # 2. Job Type Filter
            job_type_match = (selected_job_type == "All Job Types") or (selected_job_type == jd_job_type)
            
            # 3. Skills Filter
            skill_match = True
            if selected_skills_lower:
                if not any(k in jd_key_skills for k in selected_skills_lower):
                    skill_match = False
            
            if role_match and job_type_match and skill_match:
                filtered_jds.append(jd)
                
        st.session_state.filtered_jds_display = filtered_jds
        st.success(f"Filter applied! Found {len(filtered_jds)} matching Job Descriptions.")

    # --- Display Results ---
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


# --- New Parsed Data Tab (To house the removed content for testing) ---

def parsed_data_tab():
    st.header("✨ Parsed Resume Data View")
    st.markdown("This tab displays the loaded candidate data and provides download options.")
    st.markdown("---")

    is_data_loaded_and_valid = (
        st.session_state.get('parsed', {}).get('name') is not None and 
        st.session_state.get('parsed', {}).get('error') is None
    )

    if is_data_loaded_and_valid:
        
        candidate_name = st.session_state.parsed['name']
        
        # Determine the source display name
        source_key = st.session_state.get('current_parsing_source_name', 'Unknown Source')
        if source_key == "Pasted_Text":
            source_display = "Pasted CV Data"
        elif source_key == "Generated_CV":
            source_display = "Generated CV Data (Form Input)"
        else:
            source_display = source_key.replace('_', ' ').replace('-', ' ') 

        # Calculate filenames and URIs once
        base_filename = f"{candidate_name.replace(' ', '_')}_Parsed_Resume"
        parsed_json_data = json.dumps(st.session_state.parsed, indent=4)
        parsed_markdown_data = st.session_state.full_text
        
        json_filename = f"{base_filename}.json"
        md_filename = f"{base_filename}.md"
        html_filename = f"{base_filename}.html"
        
        json_data_uri = get_download_link(parsed_json_data, json_filename, 'json')
        md_data_uri = get_download_link(parsed_markdown_data, md_filename, 'markdown')
        html_data_uri = get_download_link(parsed_markdown_data, html_filename, 'html')
        
        
        tab_markdown, tab_json, tab_download = st.tabs([
            "📄 Markdown View", 
            "💾 JSON View", 
            "⬇️ PDF/HTML Download"
        ])

        # --- Markdown View Tab ---
        with tab_markdown:
            st.markdown(f"**Candidate:** **{candidate_name}**")
            st.caption(f"Source: {source_display}")
            st.markdown("---")
            st.markdown("### Resume Content in Markdown Format")
            st.markdown(st.session_state.full_text)
            
            if st.session_state.excel_data:
                 st.markdown("### Extracted Spreadsheet Data (if applicable)")
                 st.json(st.session_data.excel_data) # Corrected syntax here
                 
            st.markdown("---")
            st.markdown("##### Download Markdown Data")
            render_download_button(
                md_data_uri, 
                md_filename, 
                f"⬇️ Download Markdown (.md)", 
                'markdown'
            )


        # --- JSON View Tab ---
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

        # --- Download Tab ---
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

# --------------------------------------------------------------------------------------
# CHATBOT FUNCTIONALITY
# --------------------------------------------------------------------------------------

# 1. Resume Q&A Function (Original)
def qa_on_resume(question):
    """Chatbot for Resume (Q&A) using LLM."""
    # Use global access to client, model, and key
    
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

# 2. JD Q&A Function (New)
def qa_on_jd(question, jd_content):
    """Chatbot for Job Description (Q&A) using LLM."""
    # Use global access to client, model, and key
    
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

# 3. Resume Q&A Content Tab
def resume_qa_content():
    """Content for the Resume Q&A sub-tab."""
    st.subheader("👤 Resume Q&A Chatbot")
    st.markdown("Ask specific questions about the currently loaded resume.")

    is_data_loaded_and_valid = (
        st.session_state.get('parsed', {}).get('name') is not None and 
        st.session_state.get('parsed', {}).get('error') is None
    )
    
    if not is_data_loaded_and_valid:
        st.warning("⚠️ **Q&A Disabled:** Please parse a valid resume in the 'Resume Parsing' tab or generate one in the 'CV Management' tab first.")
        return
    
    if "resume_chatbot_history" not in st.session_state:
        st.session_state.resume_chatbot_history = []

    st.info(f"Chatting about: **{st.session_state.parsed['name']}**")
    st.markdown("---")
    
    # Display Chat History
    for message in st.session_state.resume_chatbot_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input and Logic
    if prompt := st.chat_input("Ask a question about the resume...", key="resume_qa_input"):
        # Add user message to history
        st.session_state.resume_chatbot_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.spinner("Thinking..."):
            ai_response = qa_on_resume(prompt)

        # Display AI response and add to history
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            
        st.session_state.resume_chatbot_history.append({"role": "assistant", "content": ai_response})
        st.rerun()

    # Clear Chat Button
    if st.session_state.resume_chatbot_history:
        st.markdown("---")
        if st.button("🗑️ Clear Resume Chat History", key="clear_resume_chatbot_history"):
            st.session_state.resume_chatbot_history = []
            st.rerun()

# 4. JD Q&A Content Tab (New Sub-tab)
def jd_qa_content():
    """Content for the JD Q&A sub-tab."""
    st.subheader("💼 JD Q&A Chatbot")
    st.markdown("Select a Job Description and ask questions about its requirements.")

    if not st.session_state.get('candidate_jd_list'):
        st.warning("⚠️ **Q&A Disabled:** Please load Job Descriptions in the 'JD Management' tab first.")
        return

    # Select JD
    jd_names = [jd['name'] for jd in st.session_state.candidate_jd_list]
    selected_jd_name = st.selectbox(
        "Select Job Description",
        options=jd_names,
        key="selected_jd_for_qa"
    )

    if "jd_chatbot_history" not in st.session_state:
        st.session_state.jd_chatbot_history = {} # History stored per JD name

    # Get the JD content
    selected_jd = next((jd for jd in st.session_state.candidate_jd_list if jd['name'] == selected_jd_name), None)
    jd_content = selected_jd['content'] if selected_jd else ""

    # Initialize history for the selected JD
    current_jd_history = st.session_state.jd_chatbot_history.setdefault(selected_jd_name, [])

    st.info(f"Chatting about: **{selected_jd_name}** (Role: {selected_jd.get('role', 'N/A')})")
    st.markdown("---")

    # Display Chat History
    for message in current_jd_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input and Logic
    if prompt := st.chat_input(f"Ask about the requirements of: {selected_jd_name}...", key="jd_qa_input"):
        # Add user message to history
        current_jd_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.spinner("Thinking..."):
            ai_response = qa_on_jd(prompt, jd_content)

        # Display AI response and add to history
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            
        current_jd_history.append({"role": "assistant", "content": ai_response})
        st.rerun()

    # Clear Chat Button for the current JD
    if current_jd_history:
        st.markdown("---")
        if st.button(f"🗑️ Clear Chat History for {selected_jd_name}", key="clear_jd_chatbot_history"):
            st.session_state.jd_chatbot_history[selected_jd_name] = []
            st.rerun()

# 5. Main Chatbot Tab Wrapper
def chatbot_tab_content():
    """Main Content for the Chatbot Tab with sub-tabs."""
    st.header("🤖 AI Chatbot Assistant")
    
    # Sub-tabs for the main Chatbot tab
    tab_resume, tab_jd = st.tabs(["👤 Resume Q&A", "💼 JD Q&A"])
    
    with tab_resume:
        resume_qa_content()
        
    with tab_jd:
        jd_qa_content()

# --------------------------------------------------------------------------------------
# END CHATBOT FUNCTIONALITY
# --------------------------------------------------------------------------------------


# -------------------------
# CANDIDATE DASHBOARD FUNCTION 
# -------------------------

def candidate_dashboard():
    st.title("🧑‍💻 Candidate Dashboard")
    
    col_header, col_logout = st.columns([4, 1])
    with col_logout:
        if st.button("🚪 Log Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                # Only keep essential session state keys for the overall app structure
                if key not in ['page', 'logged_in', 'user_type']:
                    del st.session_state[key]
            go_to("login")
            st.rerun() 
            
    st.markdown("---")

    # --- Session State Initialization ---
    
    # Parsing/Data State
    if "parsed" not in st.session_state: st.session_state.parsed = {} 
    if "full_text" not in st.session_state: st.session_state.full_text = ""
    if "excel_data" not in st.session_state: st.session_state.excel_data = None
    if "candidate_uploaded_resumes" not in st.session_state: st.session_state.candidate_uploaded_resumes = []
    if "pasted_cv_text" not in st.session_state: st.session_state.pasted_cv_text = ""
    if "current_parsing_source_name" not in st.session_state: st.session_state.current_parsing_source_name = None 
    if "structured_experience" not in st.session_state: st.session_state.structured_experience = []
    
    # JD Management / Match State
    if "candidate_jd_list" not in st.session_state: st.session_state.candidate_jd_list = []
    if "candidate_match_results" not in st.session_state: st.session_state.candidate_match_results = []
    if 'filtered_jds_display' not in st.session_state: st.session_state.filtered_jds_display = []
    if 'last_selected_skills' not in st.session_state: st.session_state.last_selected_skills = []
    
    # Chatbot State
    if "resume_chatbot_history" not in st.session_state: st.session_state.resume_chatbot_history = []
    if "jd_chatbot_history" not in st.session_state: st.session_state.jd_chatbot_history = {} 

    # --- Main Content with Tabs (7 Tabs) ---
    tab_parsing, tab_cv_manage, tab_data_view, tab_jd, tab_batch_match, tab_filter_jd, tab_chatbot = st.tabs(
        [
            "📄 Resume Parsing", 
            "🛠️ CV Management", # New Tab
            "✨ Parsed Data View", 
            "📚 JD Management", 
            "🎯 Batch JD Match", 
            "🔍 Filter JD", 
            "🤖 Chatbot"
        ]
    )
    
    with tab_parsing:
        resume_parsing_tab()
        
    with tab_cv_manage: # New Tab Content
        cv_management_tab()
        
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
