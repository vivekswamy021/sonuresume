# app.py
import streamlit as st
from admin_dashboard import admin_dashboard
from candidate_dashboard import candidate_dashboard
from hiring_dashboard import hiring_dashboard

image_url = "https://raw.githubusercontent.com/<username>/<repo>/main/logo.png"
st.image(image_url, caption="My Image from GitHub") 
# --- Utility Functions for Navigation and State Management ---

def go_to(page_name):
    """Changes the current page in Streamlit's session state."""
    st.session_state.page = page_name

def initialize_session_state():
    """Initializes all necessary session state variables for the entire application."""
    # Initialize page to 'login' or 'signup'
    if 'page' not in st.session_state: st.session_state.page = "login"
    
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'user_type' not in st.session_state: st.session_state.user_type = None

    # Admin/Global Data
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
    
    # --- CV Management Tab Data (NEWLY ADDED) ---
    if 'cv_data' not in st.session_state: 
        st.session_state.cv_data = {
            'personal_info': {'name': '', 'email': '', 'phone': ''},
            'education': [],
            'experience': [],
            'projects': [],
            'certifications': [],
            'strengths_raw': '' # Initializing the key to prevent KeyError
        }
    
    # FIX for existing sessions: If 'cv_data' exists but is missing the new key
    if 'cv_data' in st.session_state and 'strengths_raw' not in st.session_state.cv_data:
        st.session_state.cv_data['strengths_raw'] = ''

    if 'form_cv_text' not in st.session_state:
        st.session_state.form_cv_text = ""
    # ---------------------------------------------

    # Hiring Manager Data (Placeholder)
    if 'hiring_jds' not in st.session_state: st.session_state.hiring_jds = []
    
def login_page():
    """Handles the user login and redirects to the appropriate dashboard based on the selected role."""
    
    # --- Custom Header (Mimicking the image's branding) ---
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 30px;">
            <span style="font-size: 32px; margin-right: 10px;">🌐</span> 
            <h1 style="font-size: 32px; margin: 0; font-weight: 600;">PragyanAI Job Portal <span style="font-size: 20px; color: #4CAF50;">🔗</span></h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Login")
    
    # Use columns to create a centered, narrower login form area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.caption("Use any email/password, but select the role: **Candidate**, **Admin**, or **Hiring Manager** to test the dashboards.")
        
        with st.form("login_form", clear_on_submit=True):
            
            # 1. Role Selection
            st.markdown("Select Your Role")
            role = st.selectbox(
                "Select Role",
                ["Select Role", "Candidate", "Admin", "Hiring Manager"],
                label_visibility="collapsed",
                index=0
            )

            # 2. Email Input
            st.markdown("Email")
            email = st.text_input("Email", label_visibility="collapsed", placeholder="Enter your email...")

            # 3. Password Input
            st.markdown("Password")
            password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter your password...")
            
            # Login Button
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if role == "Select Role":
                    st.error("Please select your role.")
                elif not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    # Map the selected role to the session state type used in the original app
                    user_role = ""
                    if role == "Candidate":
                        user_role = "candidate"
                    elif role == "Admin":
                        user_role = "admin"
                    elif role == "Hiring Manager":
                        user_role = "hiring" # Matches the original app's internal 'hiring' type

                    if user_role:
                        st.session_state.logged_in = True
                        st.session_state.user_type = user_role
                        st.success(f"Logged in as {role}!")
                        go_to(f"{user_role}_dashboard")
                        st.rerun()
                    else:
                        st.error("Invalid role selected.")


        # 4. Sign-up Link (Clicking this now navigates to the sign-up page)
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        # Using a standard Streamlit button for clean navigation
        if st.button("Don't have an account? Sign up here", key="signup_link_from_login"):
            go_to("signup")
        st.markdown('</div>', unsafe_allow_html=True)


def signup_page():
    """Handles the user registration process, matching the 'Create an Account' design."""
    
    # --- Custom Header ---
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 30px;">
            <span style="font-size: 32px; margin-right: 10px;">🌐</span> 
            <h1 style="font-size: 32px; margin: 0; font-weight: 600;">PragyanAI Job Portal <span style="font-size: 20px; color: #4CAF50;">🔗</span></h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Create an Account")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("signup_form", clear_on_submit=False):
            
            # Email Input
            st.markdown("Email")
            email = st.text_input("Email", label_visibility="collapsed", key="signup_email", placeholder="Enter your email...")

            # Password Input
            st.markdown("Password")
            password = st.text_input("Password", type="password", label_visibility="collapsed", key="signup_password", placeholder="Enter your password...")
            
            # Confirm Password Input
            st.markdown("Confirm Password")
            confirm_password = st.text_input("Confirm Password", type="password", label_visibility="collapsed", key="confirm_password", placeholder="Confirm your password...")
            
            # Sign Up Button
            submitted = st.form_submit_button("Sign Up", use_container_width=True)

            if submitted:
                if not email or not password or not confirm_password:
                    st.error("Please fill out all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    # In a real app, you would register the user here.
                    # For this demo, we simulate success and redirect to login.
                    st.success("Account created successfully! Please log in.")
                    go_to("login")
                    st.rerun() 
        
        # Login Link
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        if st.button("Already have an account? Login here", key="login_link_from_signup"):
            go_to("login")
        st.markdown('</div>', unsafe_allow_html=True)


# -------------------------
# MAIN EXECUTION BLOCK 
# -------------------------

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="PragyanAI Multi-Dashboard")
    
    initialize_session_state()

    current_page = st.session_state.page
    
    if st.session_state.logged_in:
        # User is logged in, show the appropriate dashboard
        if st.session_state.user_type == "admin":
            admin_dashboard(go_to)
        elif st.session_state.user_type == "candidate":
            candidate_dashboard(go_to)
        elif st.session_state.user_type == "hiring":
            hiring_dashboard(go_to)
        else:
            # Fallback for logged-in user with unknown type
            st.error("Unknown user type. Logging out.")
            st.session_state.logged_in = False
            st.rerun()
    else:
        # User is not logged in, show login or signup page
        if current_page == "signup":
            signup_page()
        else: # Default to login page
            login_page()
