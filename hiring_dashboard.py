import streamlit as st

def hiring_dashboard(go_to_func):
    """
    Main function for the Hiring Manager Dashboard.
    Requires go_to_func for logout.
    """
    st.title("👨‍💼 Hiring Manager Dashboard (Placeholder)")
    st.caption("Manage JDs, review top candidates, and track interviews.")
    
    col_header, col_logout = st.columns([4, 1])
    with col_logout:
        if st.button("🚪 Log Out", use_container_width=True):
            go_to_func("login")
            st.rerun() 
            
    st.markdown("---")

    st.header("Candidate Review Pipeline")
    
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
        jd_names = [item['name'].replace("--- Simulated JD for: ", "") for item in st.session_state.admin_jd_list]
        st.dataframe({"Job Title": jd_names, "Status": ["Active"] * len(jd_names)}, use_container_width=True)
    else:
        st.warning("No Job Descriptions are currently loaded in the system.")
