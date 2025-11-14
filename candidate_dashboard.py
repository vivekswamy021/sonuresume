import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
from pypdf import Pdfplumber # This is the fixed module import
import io
import time

# --- 1. CONFIGURATION AND INITIALIZATION ---
load_dotenv()
# Use os.environ for Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="AI-Powered HR Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State variables for data storage
if 'resumes' not in st.session_state:
    st.session_state.resumes = {}  # {filename: {'text': '...', 'summary': '...'}}
if 'jds' not in st.session_state:
    st.session_state.jds = {}      # {title: 'JD text'}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'interview_history' not in st.session_state:
    st.session_state.interview_history = []

# Groq Client Initialization
@st.cache_resource
def get_groq_client():
    """Initializes and returns the Groq client."""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Please set it in a .env file.")
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

client = get_groq_client()
LLM_MODEL = "llama3-8b-8192" # Fast Groq model for quick responses

# --- 2. HELPER FUNCTIONS ---

def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file in memory."""
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return None

def groq_chat_completion(messages, system_prompt, stream=False):
    """Handles Groq API call with system prompt."""
    if not client:
        return "LLM service is unavailable."
    
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    try:
        return client.chat.completions.create(
            model=LLM_MODEL,
            messages=full_messages,
            stream=stream
        )
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return None

# --- 3. TAB FUNCTIONS (DASHBOARD PAGES) ---

def resume_management_tab():
    st.title("📄 Resume & CV Management")
    st.markdown("Upload resumes and view the parsed data and AI summaries.")
    
    # --- Resume Upload (Combined with CV Management) ---
    with st.expander("⬆️ Upload Resume (PDF Only)", expanded=True):
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            file_name = uploaded_file.name
            st.session_state.current_upload_file = file_name # Store filename temporarily
            
            if st.button("Process & Analyze Resume"):
                with st.spinner(f"Extracting text from {file_name}..."):
                    pdf_text = extract_text_from_pdf(uploaded_file.read())
                
                if pdf_text and file_name not in st.session_state.resumes:
                    st.toast("Text extraction complete. Sending to AI for summary...", icon="🤖")
                    
                    # AI Analysis to summarize the CV
                    summary_prompt = f"Analyze the following CV text and provide a structured summary. Include key sections: Name, Contact, Summary (1-2 sentences), Top 5 Skills, and Total Experience (years).\n\nCV Text: {pdf_text[:10000]}..." # Truncate for API
                    
                    messages = [{"role": "user", "content": summary_prompt}]
                    
                    response = groq_chat_completion(messages, "You are a professional HR resume parser. Be concise and use markdown formatting.")
                    
                    if response:
                        summary = response.choices[0].message.content
                        st.session_state.resumes[file_name] = {
                            'text': pdf_text,
                            'summary': summary,
                            'timestamp': time.time()
                        }
                        st.success(f"Successfully processed and analyzed {file_name}!")
                        st.balloons()
                    else:
                        st.error("Failed to get AI summary.")
                elif file_name in st.session_state.resumes:
                    st.warning(f"Resume '{file_name}' is already in the system.")
                
    # --- Stored Resumes Display ---
    st.subheader("Stored Resumes")
    if st.session_state.resumes:
        for filename, data in st.session_state.resumes.items():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"**{filename}**")
                col1.caption(f"Added: {time.strftime('%Y-%m-%d %H:%M', time.localtime(data['timestamp']))}")

                if col2.button("View Details", key=f"view_{filename}"):
                    st.info(f"Summary for {filename}")
                    st.markdown(data['summary'])
                    if st.checkbox("Show Full Extracted Text (Caution: Very long)", key=f"full_text_{filename}"):
                        st.text_area("Full Text", data['text'], height=300)

    else:
        st.info("No resumes uploaded yet. Use the uploader above to get started.")

def jd_management_tab():
    st.title("💼 JD Management & Filtering")
    st.markdown("Add Job Descriptions and use AI to help categorize and filter them.")

    # --- JD Upload Form (Combined with JD Management) ---
    with st.expander("➕ Add New Job Description", expanded=True):
        jd_title = st.text_input("Job Title/Reference", key="jd_title_input")
        jd_text = st.text_area("Paste Job Description Text", height=250, key="jd_text_input")
        
        if st.button("Save & Analyze JD"):
            if jd_title and jd_text:
                st.session_state.jds[jd_title] = jd_text
                st.toast(f"JD '{jd_title}' saved!")
                
                # AI Analysis for categorization/filtering prep
                analysis_prompt = f"Analyze the following Job Description (JD) and extract key information in a brief format: Role Level (e.g., Senior, Mid), Required Years of Experience (numeric), Key Technologies (top 3), and Soft Skills (top 2).\n\nJD Text: {jd_text[:10000]}..."
                
                messages = [{"role": "user", "content": analysis_prompt}]
                
                response = groq_chat_completion(messages, "You are an HR JD analyst. Provide a structured, concise bulleted list.")
                
                if response:
                    st.session_state.jds[jd_title + '_analysis'] = response.choices[0].message.content
                    st.success(f"JD analysis complete for {jd_title}!")
                else:
                    st.error("Failed to get AI analysis for JD.")
            else:
                st.error("Please provide both a title and the JD text.")

    # --- Stored JDs Display ---
    st.subheader("Stored Job Descriptions")
    if st.session_state.jds:
        # Filtering (Tab 5 functionality integrated here)
        filter_col, sort_col = st.columns(2)
        filter_term = filter_col.text_input("Filter JDs by Keyword in Title/Content")

        filtered_jds = {k: v for k, v in st.session_state.jds.items() if filter_term.lower() in k.lower() or (isinstance(v, str) and filter_term.lower() in v.lower())}
        
        for title, text in filtered_jds.items():
            if not title.endswith('_analysis'):
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    if title + '_analysis' in st.session_state.jds:
                        st.caption("AI Analysis:")
                        st.markdown(st.session_state.jds[title + '_analysis'])
                    
                    if st.checkbox("Show Full JD", key=f"show_jd_{title}"):
                        st.text_area("", text, height=200, disabled=True)
    else:
        st.info("No Job Descriptions saved yet.")

def batch_jd_match_tab():
    st.title("🤖 Batch JD Match")
    st.markdown("Select a Job Description and a batch of Resumes to find the best match.")
    
    # Check for data availability
    if not st.session_state.jds or not st.session_state.resumes:
        st.warning("Please upload at least one JD and one Resume in the other tabs first.")
        return

    # User Input Selection
    col1, col2 = st.columns(2)
    selected_jd_title = col1.selectbox("Select Target Job Description", 
                                        [k for k in st.session_state.jds.keys() if not k.endswith('_analysis')])
    selected_resumes = col2.multiselect("Select Resumes for Matching", 
                                        list(st.session_state.resumes.keys()))

    if st.button("Run Batch Match & Scoring"):
        if not selected_jd_title or not selected_resumes:
            st.error("Please select a JD and at least one Resume.")
            return

        jd_text = st.session_state.jds[selected_jd_title]
        
        st.subheader(f"Results for JD: {selected_jd_title}")
        results = []
        
        # Match using Groq LLM
        for resume_name in selected_resumes:
            resume_text = st.session_state.resumes[resume_name]['text']
            
            match_prompt = f"""
            Compare the following Resume against the Job Description (JD).
            1. Provide a Match Percentage (as a number between 0 and 100).
            2. List 3 key skills that match.
            3. List 2 missing skills/gaps.
            4. Provide a concise, two-sentence rationale for the score.

            Job Description: {jd_text[:8000]}
            ---
            Resume Text: {resume_text[:8000]}
            """
            
            with st.spinner(f"Matching {resume_name}..."):
                messages = [{"role": "user", "content": match_prompt}]
                response = groq_chat_completion(messages, "You are a specialized JD-Resume matching AI. Your output MUST start with the Match Percentage number followed by the rest of the analysis.")
            
            if response:
                match_result = response.choices[0].message.content
                # Attempt to extract score for the metric display
                try:
                    score_str = match_result.split('\n')[0].strip().split('.')[0].strip()
                    score = int(score_str.split(' ')[0].split('%')[0].strip())
                except:
                    score = 0 # Default if extraction fails
                
                results.append({
                    'name': resume_name,
                    'score': score,
                    'analysis': match_result
                })
            else:
                results.append({'name': resume_name, 'score': 0, 'analysis': "AI analysis failed."})

        # Display results in a structured table/list
        results.sort(key=lambda x: x['score'], reverse=True)
        for i, res in enumerate(results):
            col_rank, col_name, col_score = st.columns([1, 4, 2])
            col_rank.metric("Rank", i + 1)
            col_name.markdown(f"**{res['name']}**")
            col_score.metric("Match Score", f"{res['score']}%", delta_color="normal")
            
            with st.expander("View AI Rationale"):
                st.markdown(res['analysis'])
            st.divider()


def chatbot_tab():
    st.title("💬 Resume / JD Chatbot")
    st.markdown("Ask questions about stored Resumes or JDs. Use this to quickly parse information or compare documents.")

    # Sidebar for context setting
    with st.sidebar:
        st.subheader("Chat Context")
        resume_options = list(st.session_state.resumes.keys())
        jd_options = [k for k in st.session_state.jds.keys() if not k.endswith('_analysis')]
        
        selected_resume = st.selectbox("Context Resume (Optional)", ["None"] + resume_options)
        selected_jd = st.selectbox("Context JD (Optional)", ["None"] + jd_options)
        
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    # Construct the full context
    context_text = ""
    if selected_resume != "None":
        context_text += f"\n\n--- RESUME CONTEXT ({selected_resume}) ---\n{st.session_state.resumes[selected_resume]['text'][:5000]}"
    if selected_jd != "None":
        context_text += f"\n\n--- JD CONTEXT ({selected_jd}) ---\n{st.session_state.jds[selected_jd][:5000]}"
    
    # System Prompt for Chatbot
    system_prompt = f"""
    You are an expert HR Analyst Chatbot. Your role is to answer user questions based on the provided context (Resume and/or JD) and your general HR knowledge. 
    If you do not have context, state that you are answering based on general knowledge.
    
    Current Context:
    {context_text}
    """

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the context or general HR topics..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Stream the response from Groq
            messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
            
            stream_response = groq_chat_completion(messages_for_api, system_prompt, stream=True)
            
            if stream_response:
                full_response = st.write_stream(stream_response)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})


def interview_preparation_tab():
    st.title("🎤 Interview Preparation (AI Mock Interviewer)")
    st.markdown("Start a mock interview where the AI acts as the interviewer. The AI will provide feedback at the end.")
    
    if 'is_interview_active' not in st.session_state:
        st.session_state.is_interview_active = False

    # Sidebar for interview setup
    with st.sidebar:
        st.subheader("Interview Setup")
        
        jd_options = [k for k in st.session_state.jds.keys() if not k.endswith('_analysis')]
        interview_jd = st.selectbox("Select Target JD (Optional for context)", ["General"] + jd_options)
        
        # System prompt for the initial setup
        if interview_jd == "General":
            interview_system_prompt = "You are a professional HR interviewer. Your first question must be: 'Tell me about yourself and why you are interested in this position.'"
        else:
            jd_text = st.session_state.jds[interview_jd][:5000]
            interview_system_prompt = f"You are a professional HR interviewer conducting a mock interview for the role defined in the following JD. Start the interview by asking the candidate to introduce themselves and discuss their interest in the role based on the JD. Be critical and specific to the JD. JD: {jd_text}"

        if st.button("Start New Interview", type="primary"):
            st.session_state.interview_history = []
            st.session_state.is_interview_active = True
            
            # Get the first question from the AI
            initial_message = [{"role": "user", "content": "Begin the interview."}]
            response = groq_chat_completion(initial_message, interview_system_prompt)
            
            if response:
                first_question = response.choices[0].message.content
                st.session_state.interview_history.append({"role": "assistant", "content": first_question})
                st.rerun()
            else:
                st.error("Failed to start interview.")
        
        if st.session_state.is_interview_active and st.button("End Interview & Get Feedback", type="secondary"):
            st.session_state.is_interview_active = False
            
            # Generate final feedback
            feedback_prompt = "The interview has ended. Review the entire conversation history and provide comprehensive feedback on the candidate's performance. Focus on: Strengths, Weaknesses/Areas for Improvement, and a final Hire/No-Hire recommendation with rationale."
            
            messages_for_feedback = [{"role": m["role"], "content": m["content"]} for m in st.session_state.interview_history]
            messages_for_feedback.append({"role": "user", "content": feedback_prompt})
            
            with st.spinner("Generating detailed feedback..."):
                response = groq_chat_completion(messages_for_feedback, "You are an HR Interview Coach. Provide feedback in a structured markdown format with clear headings.")
            
            if response:
                st.session_state.interview_history.append({"role": "assistant", "content": f"### Interview Feedback\n---\n{response.choices[0].message.content}"})
                st.rerun()

    # Display interview messages
    if st.session_state.is_interview_active or st.session_state.interview_history:
        for message in st.session_state.interview_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.info("Set up your interview context and click 'Start New Interview' to begin!")
        
    # Interview continues
    if st.session_state.is_interview_active:
        if user_answer := st.chat_input("Your response..."):
            st.session_state.interview_history.append({"role": "user", "content": user_answer})
            with st.chat_message("user"):
                st.markdown(user_answer)
                
            with st.chat_message("assistant"):
                # Use the conversation history to generate the next question
                messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.interview_history]
                
                stream_response = groq_chat_completion(messages_for_api, interview_system_prompt + " Ask only the next question and wait for the user's response.", stream=True)
                
                if stream_response:
                    full_response = st.write_stream(stream_response)
                    st.session_state.interview_history.append({"role": "assistant", "content": full_response})

    
# --- 4. MAIN APP STRUCTURE ---

def main():
    st.title("AI HR Recruitment Dashboard")
    st.caption("Powered by Streamlit and Groq LLMs")

    # Define the tabs
    tab_titles = [
        "📄 Resume/CV Management", 
        "💼 JD Management & Filtering", 
        "🤖 Batch JD Match", 
        "💬 Resume / JD Chatbot", 
        "🎤 Interview Preparation"
    ]
    
    # Create the tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)
    
    with tab1:
        resume_management_tab()
    
    with tab2:
        jd_management_tab()

    with tab3:
        batch_jd_match_tab()
        
    with tab4:
        chatbot_tab()
        
    with tab5:
        interview_preparation_tab()

if __name__ == "__main__":
    main()
