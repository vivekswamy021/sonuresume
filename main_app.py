import streamlit as st
from admin_dashboard import admin_dashboard
from candidate_dashboard import candidate_dashboard
from hiring_dashboard import hiring_dashboard

# 🔥 SET YOUR LOGO HERE
LOGO_URL = "https://raw.githubusercontent.com/vivekswamy021/Pragyan_AI_resume/main/pragyan_ai_school_cover.jpg"

# --------------------------------------------------
# 🔥 LOGO FUNCTION
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
    if 'user_email' not in st.session_state: st.session_state.user_email = "" 
    if 'user_name' not in st.session_state: st.session_state.user_name = ""

    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            "profile_pic": None,
            "github_link": "",
            "linkedin_link": "",
            "password": "password123"
        }

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

    if 'cv_data' not in st.session_state: 
        st.session_state.cv_data = {
            'personal_info': {'name': '', 'email': '', 'phone': ''},
            'education': [],
            'experience': [],
            'projects': [],
            'certifications': [],
            'strengths_raw': ''
        }
    if 'form_cv_text' not in st.session_state: st.session_state.form_cv_text = ""

    # Hiring Manager
    if 'hiring_jds' not in st.session_state: st.session_state.hiring_jds = []

# --------------------------------------------------
# 👤 PROFILE SIDEBAR FUNCTION (With Logout)
# --------------------------------------------------
def render_profile_sidebar():
    with st.sidebar:
        st.header(f"👤 {st.session_state.user_name}")
        st.markdown(f"**Role:** {st.session_state.user_type.capitalize()}")
        st.markdown(f"**Email:** {st.session_state.user_email}")
        
        st.divider()

        # 1. Profile Picture
        col_pic, col_upload = st.columns([1, 2])
        if st.session_state.user_profile["profile_pic"]:
            st.image(st.session_state.user_profile["profile_pic"], width=100)
        else:
            st.markdown("## 👤")
            st.caption("No image")

        uploaded_pic = st.file_uploader("Update Photo", type=["jpg", "png", "jpeg"])
        if uploaded_pic is not None:
            st.session_state.user_profile["profile_pic"] = uploaded_pic
            st.success("Photo Updated!")
            st.rerun()

        st.divider()

        # 2. Edit Name
        st.subheader("✏️ Profile Name")
        new_name = st.text_input("Display Name", value=st.session_state.user_name, label_visibility="collapsed")
        if st.button("Update Name"):
            st.session_state.user_name = new_name
            st.success("Name updated!")
            st.rerun()

        st.divider()

        # 3. Professional Links
        st.subheader("🔗 Professional Links")
        new_github = st.text_input("GitHub URL", value=st.session_state.user_profile["github_link"])
        new_linkedin = st.text_input("LinkedIn URL", value=st.session_state.user_profile["linkedin_link"])

        if st.button("Save Links"):
            st.session_state.user_profile["github_link"] = new_github
            st.session_state.user_profile["linkedin_link"] = new_linkedin
            st.success("Links saved!")

        st.divider()

        # 4. Change Password
        with st.expander("🔐 Change Password"):
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            if st.button("Update Password"):
                if new_pass == confirm_pass and new_pass:
                    st.session_state.user_profile["password"] = new_pass
                    st.success("Password changed!")
                else:
                    st.error("Passwords mismatch or empty.")

        st.divider()

        # --------------------------------------------------
        # 🚪 LOGOUT BUTTON (Added here for all dashboards)
        # --------------------------------------------------
        if st.button("🚪 Log Out", type="primary", use_container_width=True):
            # Clear all session data and return to login
            st.session_state.clear()
            st.rerun()

# --------------------------------------------------
# LOGIN & SIGNUP PAGES
# --------------------------------------------------
def login_page():
    show_logo()
    st.markdown("<h1>PragyanAI Job Portal</h1>", unsafe_allow_html=True)
    st.subheader("Login")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form", clear_on_submit=True):
            role = st.selectbox("Select Role", ["Select Role", "Candidate", "Admin", "Hiring Manager"])
            email = st.text_input("Email", placeholder="Enter email")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if role == "Select Role" or not email or not password:
                    st.error("Please fill all fields.")
                else:
                    user_role = {"Candidate": "candidate", "Admin": "admin", "Hiring Manager": "hiring"}.get(role)
                    st.session_state.logged_in = True
                    st.session_state.user_type = user_role
                    st.session_state.user_email = email
                    st.session_state.user_name = email.split("@")[0].capitalize()
                    go_to(f"{user_role}_dashboard")
                    st.rerun()

        if st.button("Don't have an account? Sign up here"):
            go_to("signup")
            st.rerun()

def signup_page():
    show_logo()
    st.markdown("<h1>Create an Account</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("signup_form"):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if password == confirm and email:
                    st.success("Account created! Please log in.")
                    go_to("login")
                    st.rerun()
                else:
                    st.error("Check passwords/email.")
        if st.button("Already have an account? Login here"):
            go_to("login")
            st.rerun()

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="PragyanAI App")
    initialize_session_state()

    if st.session_state.logged_in:
        render_profile_sidebar()
        show_logo()
        
        if st.session_state.user_type == "admin":
            admin_dashboard(go_to)
        elif st.session_state.user_type == "candidate":
            candidate_dashboard(go_to)
        elif st.session_state.user_type == "hiring":
            hiring_dashboard(go_to)
    else:
        if st.session_state.page == "signup":
            signup_page()
        else:
            login_page()
