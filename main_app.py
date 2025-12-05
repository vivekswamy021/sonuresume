import streamlit as st
from admin_dashboard import admin_dashboard
from candidate_dashboard import candidate_dashboard
from hiring_dashboard import hiring_dashboard

# Import MongoDB client
from pymongo import MongoClient

# --- CONFIGURATION & CONSTANTS ---
# --------------------------------------------------------------------------
# 🔥 SET YOUR MONGODB CONNECTION STRING HERE
# Replace this with your actual connection string (e.g., from MongoDB Atlas)
# It's highly recommended to use Streamlit Secrets for this in a real app!
MONGO_URI = "mongodb://localhost:27017/" 
DATABASE_NAME = "pragyan_job_portal"
# --------------------------------------------------------------------------

# --------------------------------------------------
# 🔥 SET YOUR LOGO HERE (GitHub RAW IMAGE LINK)
# --------------------------------------------------
LOGO_URL = "https://raw.githubusercontent.com/vivekswamy021/Pragyan_AI_resume/main/pragyan_ai_school_cover.jpg"


# --------------------------------------------------
# 🔥 MONGODB CONNECTION (Cached to run only once)
# --------------------------------------------------
@st.cache_resource
def get_database_client():
    """Initializes and caches the MongoDB client connection."""
    try:
        # Use server_api option to suppress the DeprecationWarning for older servers
        client = MongoClient(MONGO_URI)
        
        # Ping the deployment to confirm a successful connection
        client.admin.command('ping')
        print("MongoDB connection successful!")
        
        db = client[DATABASE_NAME]
        
        # Return the database object and the client
        return client, db
        
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        st.stop() # Stop the app if DB connection fails
        
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
    """Initializes session state and establishes MongoDB connection."""
    
    # Connect to MongoDB and store the database instance in session state
    # Although the function is cached, we store the result for easy access.
    if 'db_client' not in st.session_state or 'db' not in st.session_state:
        client, db = get_database_client()
        st.session_state.db_client = client
        st.session_state.db = db
    
    # Initialize page/auth states
    if 'page' not in st.session_state: st.session_state.page = "login"
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'user_type' not in st.session_state: st.session_state.user_type = None
    if 'user_email' not in st.session_state: st.session_state.user_email = None # Store email for identification

    # --- Data Initialization (Will be replaced by MongoDB reads in dashboard files) ---
    # For now, we initialize them as empty lists/dicts as the dashboard files expect them.
    
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
        st.caption("Enter your email and password, and select your role to log in.")
        
        db = st.session_state.db
        users_collection = db["users"]
        
        with st.form("login_form", clear_on_submit=True):

            # Role Mapping
            role_map = {
                "Candidate": "candidate",
                "Admin": "admin",
                "Hiring Manager": "hiring"
            }
            
            # Role Selection (for authentication matching)
            st.markdown("Select Your Role")
            role_display = st.selectbox(
                "Select Role",
                ["Select Role", "Candidate", "Admin", "Hiring Manager"],
                label_visibility="collapsed"
            )
            role = role_map.get(role_display)

            # Email
            st.markdown("Email")
            email = st.text_input("Email", placeholder="Enter email", label_visibility="collapsed")

            # Password
            st.markdown("Password")
            password = st.text_input("Password", type="password", placeholder="Enter password", label_visibility="collapsed")

            submitted = st.form_submit_button("Login")

            if submitted:
                if role == "Select Role" or not role:
                    st.error("Please select your role.")
                elif not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    # 1. Look up user in MongoDB
                    user_data = users_collection.find_one({"email": email})
                    
                    if user_data:
                        # 2. Check password (simplistic check for demo - use hashing in production!)
                        if user_data["password"] == password:
                            
                            # 3. Check if the selected role matches the stored role
                            if user_data["role"] == role:
                                st.session_state.logged_in = True
                                st.session_state.user_type = role
                                st.session_state.user_email = email
                                st.success(f"Logged in as {role_display}!")
                                go_to(f"{role}_dashboard")
                                st.rerun()
                            else:
                                st.error(f"User email exists, but the selected role ({role_display}) does not match the registered role ({user_data['role'].capitalize()}).")
                        else:
                            st.error("Incorrect password.")
                    else:
                        st.error("User not found. Please sign up.")


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
        
        db = st.session_state.db
        users_collection = db["users"]
        
        with st.form("signup_form"):

            # Role Mapping
            role_map = {
                "Candidate": "candidate",
                "Admin": "admin",
                "Hiring Manager": "hiring"
            }

            st.markdown("Select Your Role")
            role_display = st.selectbox(
                "Select Role",
                ["Select Role", "Candidate", "Admin", "Hiring Manager"],
                label_visibility="collapsed"
            )
            role = role_map.get(role_display)
            
            st.markdown("Email")
            email = st.text_input("Email", placeholder="Enter email", label_visibility="collapsed")

            st.markdown("Password (Use a simple password for this demo)")
            password = st.text_input("Password", type="password", placeholder="Enter password", label_visibility="collapsed")

            st.markdown("Confirm Password")
            confirm_password = st.text_input("Confirm", type="password", placeholder="Confirm password", label_visibility="collapsed")

            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if role == "Select Role" or not role:
                    st.error("Please select your role.")
                elif not email or not password or not confirm_password:
                    st.error("Fill all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    # Check if user already exists
                    if users_collection.find_one({"email": email}):
                        st.error("An account with this email already exists.")
                    else:
                        # Insert new user into MongoDB
                        new_user = {
                            "email": email,
                            "password": password, # Warning: Store hashed passwords in production!
                            "role": role
                        }
                        users_collection.insert_one(new_user)
                        
                        st.success(f"Account created for {email} as {role_display}! Please log in.")
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

    # This calls the MongoDB connection setup inside get_database_client()
    initialize_session_state()

    current_page = st.session_state.page

    if st.session_state.logged_in:
        
        # Logout button visible on all dashboards
        with st.sidebar:
            if st.button("Logout", key="logout_btn"):
                st.session_state.logged_in = False
                st.session_state.user_type = None
                st.session_state.user_email = None
                go_to("login")
                st.rerun()
                
            st.markdown(f"**Logged in as:** {st.session_state.user_email} ({st.session_state.user_type.capitalize()})")


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
        # User is not logged in
        if current_page == "signup":
            signup_page()
        else:
            login_page()
