# app.py
import streamlit as st
from admin_dashboard import admin_dashboard
from candidate_dashboard import candidate_dashboard
from hiring_dashboard import hiring_dashboard

# --------------------------------------------------
# 🔥 SET YOUR LOGO HERE (GitHub RAW IMAGE LINK)
# --------------------------------------------------
LOGO_URL = "https://raw.githubusercontent.com/vivekswamy021/Pragyan_AI_resume/main/pragyan_ai_school_cover.jpg"



# --------------------------------------------------
# 🔥 LOGO FUNCTION (used across all pages)
# --------------------------------------------------
def show_logo(width=510):
    st.image(LOGO_URL, width=width)


# ------------------------------
# Utility Functions
# ------------------------------

def go_to(page_name):
    st.session_state.page = page_name

def initialize_session_state():

    if 'page' not in st.session_state: st.session_state.page = "login"

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'user_type' not in st.session_state: st.session_state.user_type = None

    # Admin Data
    if 'admin_jd_list' not in st.session_state: st.session_state.admin_jd_list = []
    if 'resumes_to_analyze' not in st.session_state: st.session_state.resumes_to_analyze = []
    if 'admin_match_results' not in st.session_state: st.session_state.admin_match_results = []
    if 'resume_statuses' not in st.session_state: st.session_state.resume_statuses = {}
    if 'vendors' not in st.session_state: st.session_state.vendors = []
    if 'vendor_statuses' not in st.session_state: st.session_state.vendor_statuses = {}

    # Candidate Data
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

    # CV Management Tab Data
    if 'cv_data' not in st.session_state: 
        st.session_state.cv_data = {
            'personal_info': {'name': '', 'email': '', 'phone': ''},
            'education': [],
            'experience': [],
            'projects': [],
            'certifications': [],
            'strengths_raw': ''
        }

    if 'form_cv_text' not in st.session_state:
        st.session_state.form_cv_text = ""

    # Hiring Manager
    if 'hiring_jds' not in st.session_state: st.session_state.hiring_jds = []


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

def login_page():

    show_logo()  # 🔥 Logo added here

    st.markdown(
        """
        <h1 style="font-size: 32px; font-weight: 700; margin-bottom: 10px;">
            PragyanAI Job Portal
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Login")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.caption("Use any email/password. Select role: Candidate / Admin / Hiring Manager.")
        
        with st.form("login_form", clear_on_submit=True):

            # Role
            st.markdown("Select Your Role")
            role = st.selectbox(
                "Select Role",
                ["Select Role", "Candidate", "Admin", "Hiring Manager"],
                label_visibility="collapsed"
            )

            # Email
            st.markdown("Email")
            email = st.text_input("Email", placeholder="Enter email", label_visibility="collapsed")

            # Password
            st.markdown("Password")
            password = st.text_input("Password", type="password", placeholder="Enter password", label_visibility="collapsed")

            submitted = st.form_submit_button("Login")

            if submitted:
                if role == "Select Role":
                    st.error("Please select your role.")
                elif not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    user_role = {
                        "Candidate": "candidate",
                        "Admin": "admin",
                        "Hiring Manager": "hiring"
                    }.get(role)

                    st.session_state.logged_in = True
                    st.session_state.user_type = user_role
                    go_to(f"{user_role}_dashboard")
                    st.rerun()

        if st.button("Don't have an account? Sign up here"):
            go_to("signup")
            st.rerun()


# --------------------------------------------------
# SIGNUP PAGE
# --------------------------------------------------

def signup_page():

    show_logo()  # 🔥 Logo added here

    st.markdown(
        """<h1 style="font-size: 30px; font-weight: 700;">Create an Account</h1>""",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("signup_form"):

            st.markdown("Email")
            email = st.text_input("Email", placeholder="Enter email", label_visibility="collapsed")

            st.markdown("Password")
            password = st.text_input("Password", type="password", placeholder="Enter password", label_visibility="collapsed")

            st.markdown("Confirm Password")
            confirm_password = st.text_input("Confirm", type="password", placeholder="Confirm password", label_visibility="collapsed")

            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if not email or not password or not confirm_password:
                    st.error("Fill all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    st.success("Account created! Please log in.")
                    go_to("login")
                    st.rerun()

        if st.button("Already have an account? Login here"):
            go_to("login")
            st.rerun()


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="PragyanAI App")

    initialize_session_state()

    current_page = st.session_state.page

    if st.session_state.logged_in:
        if st.session_state.user_type == "admin":
            show_logo()  # 🔥 Logo on Admin Dashboard
            admin_dashboard(go_to)

        elif st.session_state.user_type == "candidate":
            show_logo()  # 🔥 Logo on Candidate Dashboard
            candidate_dashboard(go_to)

        elif st.session_state.user_type == "hiring":
            show_logo()  # 🔥 Logo on Hiring Dashboard
            hiring_dashboard(go_to)

    else:
        if current_page == "signup":
            signup_page()
        else:
            login_page()
