#app.py
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
    if 'user_email' not in st.session_state: st.session_state.user_email = "" 
    if 'user_name' not in st.session_state: st.session_state.user_name = ""

    # --------------------------------------------------
    # 👤 USER PROFILE DATA
    # --------------------------------------------------
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
# 👤 PROFILE SIDEBAR FUNCTION
# --------------------------------------------------
def render_profile_sidebar():
    with st.sidebar:
        # Dynamic Header with User Name
        st.header(f"👤 {st.session_state.user_name}")
        st.caption(f"Role: {st.session_state.user_type.capitalize()}")
        
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

        # 2. Edit Name Section
        with st.expander("✏️ Edit Profile Details"):
            # Allow user to update the name extracted from email
            new_name = st.text_input("Display Name", value=st.session_state.user_name)
            if st.button("Update Name"):
                st.session_state.user_name = new_name
                st.success("Name updated!")
                st.rerun()

        # 3. Professional Links
        st.subheader("🔗 Professional Links")
        
        current_github = st.session_state.user_profile["github_link"]
        current_linkedin = st.session_state.user_profile["linkedin_link"]

        new_github = st.text_input("GitHub URL", value=current_github, placeholder="https://github.com/...")
        new_linkedin = st.text_input("LinkedIn URL", value=current_linkedin, placeholder="https://linkedin.com/in/...")

        if st.button("Save Links"):
            st.session_state.user_profile["github_link"] = new_github
            st.session_state.user_profile["linkedin_link"] = new_linkedin
            st.success("Links saved successfully!")

        st.divider()

        # 4. Change Password
        with st.expander("🔐 Change Password"):
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm New Password", type="password")
            
            if st.button("Update Password"):
                if new_pass and confirm_pass:
                    if new_pass == confirm_pass:
                        st.session_state.user_profile["password"] = new_pass
                        st.success("Password changed!")
                    else:
                        st.error("Passwords do not match.")
                else:
                    st.warning("Please fill both fields.")

        st.divider()
        
        if st.button("Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_type = None
            st.session_state.user_email = "" 
            st.session_state.user_name = "" 
            # 🔥 Clear the uploaded photo/profile picture
            st.session_state.user_profile["profile_pic"] = None
            st.session_state.page = "login"
            st.rerun()


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

def login_page():

    show_logo()

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

            st.markdown("Select Your Role")
            role = st.selectbox(
                "Select Role",
                ["Select Role", "Candidate", "Admin", "Hiring Manager"],
                label_visibility="collapsed"
            )

            st.markdown("Email")
            email = st.text_input("Email", placeholder="Enter email", label_visibility="collapsed")

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

                    # Extract Name from Email (Simple Logic)
                    extracted_name = email.split("@")[0].capitalize()

                    st.session_state.logged_in = True
                    st.session_state.user_type = user_role
                    st.session_state.user_email = email
                    st.session_state.user_name = extracted_name
                    
                    go_to(f"{user_role}_dashboard")
                    st.rerun()

        if st.button("Don't have an account? Sign up here"):
            go_to("signup")
            st.rerun()


# --------------------------------------------------
# SIGNUP PAGE
# --------------------------------------------------

def signup_page():

    show_logo()

    st.markdown(
        """<h1 style="font-size: 30px; font-weight: 700;">Create an Account</h1>""",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("signup_form"):

            # Added Name field to signup for realism
            st.markdown("Full Name")
            full_name = st.text_input("Full Name", placeholder="Enter full name", label_visibility="collapsed")

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
                    st.success(f"Account created for {full_name or email}! Please log in.")
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
        
        # RENDER SIDEBAR with Profile options
        render_profile_sidebar()

        if st.session_state.user_type == "admin":
            show_logo()
            admin_dashboard(go_to)

        elif st.session_state.user_type == "candidate":
            show_logo()
            candidate_dashboard(go_to)

        elif st.session_state.user_type == "hiring":
            show_logo()
            hiring_dashboard(go_to)

    else:
        if current_page == "signup":
            signup_page()
        else:
            login_page()
