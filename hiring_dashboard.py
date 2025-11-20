import streamlit as st

import streamlit as st

def hiring_dashboard(go_to_func):
    """
    Main function for the Hiring Manager Dashboard.
    Requires go_to_func for logout.
    """
    
    # 1. Define the Logout Callback Function (Uses the passed-in go_to_func)
    def logout_callback():
        # 1. Clear authentication state
        st.session_state.logged_in = False
        st.session_state.user_type = None
        
        # 2. Set the target page using the passed function
        go_to_func("login")
        
        # 3. Force the application to re-run (CRITICAL step)
        st.rerun()

    # --- Dashboard Header and Logout Button ---
    col_title, nav_col = st.columns([10, 2])
    
    with col_title:
        st.title("👨‍💼 Hiring Manager Dashboard")
        st.caption("Manage JDs, review top candidates, and track interviews.")
    
    with nav_col:
        # Place the Log Out Button with the callback
        st.button("🚪 Log Out", use_container_width=True, on_click=logout_callback)
            
    st.markdown("---") # Visual separator after the header/logout

    st.header("Candidate Review Pipeline")
    
    # --- Check and Initialize necessary states if they somehow missed the global init ---
    if 'resumes_to_analyze' not in st.session_state: st.session_state.resumes_to_analyze = []
    if 'resume_statuses' not in st.session_state: st.session_state.resume_statuses = {}
    if 'admin_jd_list' not in st.session_state: st.session_state.admin_jd_list = []
    
    # Example: Display candidates with 'Approved' status from admin's data
    approved_candidates = [
        r['name'] for r in st.session_state.resumes_to_analyze 
        if st.session_state.resume_statuses.get(r['name']) == 'Approved'
    ]
    
    if approved_candidates:
        st.subheader(f"✅ Approved Candidates Ready for Interview ({len(approved_candidates)})")
        st.dataframe({"Candidate Name": approved_candidates})
    else:
        st.info("No candidates have been approved by the Admin yet.")

    st.markdown("---")
    
    st.header("Job Description Tracker")
    
    if st.session_state.admin_jd_list:
        st.subheader("Active Job Descriptions")
        # Extract name and clean the simulated prefix if present
        jd_names = [item['name'].replace("--- Simulated JD for: ", "") for item in st.session_state.admin_jd_list]
        st.dataframe({"Job Title": jd_names, "Status": ["Active"] * len(jd_names)}, use_container_width=True)
    else:
        st.warning("No Job Descriptions are currently loaded in the system.")
