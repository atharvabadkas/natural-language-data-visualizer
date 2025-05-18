import streamlit as st
import requests
import pandas as pd
import json
import base64
from io import BytesIO

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8080"

# Set page title
st.title("Data Analysis and Visualization")

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Upload file to backend
    files = {"file": uploaded_file}
    response = requests.post(f"{BACKEND_URL}/upload-csv", files=files)
    
    if response.status_code == 200:
        st.success("File uploaded successfully!")
        
        # Get dataset info
        dataset_name = uploaded_file.name.rsplit('.', 1)[0]
        info_response = requests.get(f"{BACKEND_URL}/dataset/{dataset_name}/info")
        
        if info_response.status_code == 200:
            dataset_info = info_response.json()
            
            # Create tabs for different functionalities
            tab1, tab2, tab3 = st.tabs(["Query", "Visualization", "Dataset Info"])
            
            # Query tab
            with tab1:
                st.subheader("Ask Questions About Your Data")
                query = st.text_area("Enter your query:", height=100)
                
                if st.button("Submit Query", key="query_button"):
                    if query:
                        with st.spinner("Processing your query..."):
                            try:
                                # Show the query being processed
                                st.info(f"Processing query: {query}")
                                
                                # Make the API call
                                query_response = requests.post(
                                    f"{BACKEND_URL}/query",
                                    params={"query": query},
                                    timeout=180  # 3-minute timeout
                                )
                                
                                # Parse the response
                                if query_response.status_code == 200:
                                    result = query_response.json()
                                    
                                    # Handle different response statuses
                                    if result.get("status") == "error":
                                        st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
                                        st.write("Debug info:", result)
                                    elif result.get("status") == "timeout":
                                        st.warning("Query timed out. Please try a simpler question.")
                                    elif result.get("status") == "success":
                                        # Create response tabs
                                        response_tab1, response_tab2, response_tab3 = st.tabs(
                                            ["Answer", "Raw Response", "Debug Info"]
                                        )
                                        
                                        # Answer tab
                                        with response_tab1:
                                            st.markdown("### Answer")
                                            if result.get('response'):
                                                st.write(result['response'])
                                                st.write(f"Confidence: {result.get('confidence', 'N/A')}")
                                            else:
                                                st.warning("No response received")
                                        
                                        # Raw response tab
                                        with response_tab2:
                                            st.markdown("### Raw LLM Response")
                                            if result.get('raw_llm_response'):
                                                st.code(result['raw_llm_response'])
                                            else:
                                                st.info("No raw response available")
                                        
                                        # Debug tab
                                        with response_tab3:
                                            st.markdown("### Debug Information")
                                            debug_info = {
                                                "Query Type": result.get("query_type", "N/A"),
                                                "Sample Size": result.get("sample_size", "N/A"),
                                                "Status": result.get("status", "N/A"),
                                                "Confidence": result.get("confidence", "N/A")
                                            }
                                            st.json(debug_info)
                                    else:
                                        st.error("Unexpected response format")
                                        st.write("Response:", result)
                                
                                elif query_response.status_code == 400:
                                    st.warning("Please upload a dataset first")
                                else:
                                    st.error(f"Server error (Status code: {query_response.status_code})")
                                    try:
                                        error_details = query_response.json()
                                        st.write("Error details:", error_details)
                                    except:
                                        st.write("Raw response:", query_response.text)
                            
                            except requests.Timeout:
                                st.error("Request timed out. Please try again.")
                            except requests.ConnectionError:
                                st.error("Could not connect to the server. Is it running?")
                            except Exception as e:
                                st.error(f"An unexpected error occurred: {str(e)}")
                    else:
                        st.warning("Please enter a query")
            
            # Visualization tab
            with tab2:
                st.subheader("Create Visualizations")
                
                # Get available columns
                columns = dataset_info.get("columns", [])
                
                # Create two columns for inputs
                col1, col2 = st.columns(2)
                
                with col1:
                    plot_type = st.selectbox("Select plot type", ["bar", "line", "scatter"])
                    x_column = st.selectbox("Select X-axis column", columns)
                
                with col2:
                    y_column = st.selectbox("Select Y-axis column", columns)
                    title = st.text_input("Plot title (optional)")
                
                if st.button("Create Visualization", key="viz_button"):
                    viz_data = {
                        "dataset_name": dataset_name,
                        "plot_type": plot_type,
                        "x_column": x_column,
                        "y_column": y_column,
                        "title": title if title else None
                    }
                    
                    viz_response = requests.post(f"{BACKEND_URL}/visualize", json=viz_data)
                    
                    if viz_response.status_code == 200:
                        result = viz_response.json()
                        image_bytes = base64.b64decode(result['image'])
                        st.image(image_bytes)
                    else:
                        st.error("Error creating visualization")
            
            # Dataset Info tab
            with tab3:
                st.subheader("Dataset Information")
                st.json(dataset_info)
        
        else:
            st.error("Error getting dataset information")
    else:
        st.error("Error uploading file") 