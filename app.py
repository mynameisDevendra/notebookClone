
import google.generativeai as genai
import streamlit as st
import textwrap
import io
import pandas as pd
from PyPDF2 import PdfReader
import docx
import os

# --- Configuration and Setup ---
st.set_page_config(layout="wide")
st.title("My NotebookLM Clone")
st.markdown("---")

# Access Google API key securely from Streamlit secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    if not api_key:
        st.error("Google API key not found. Please add it to your Streamlit secrets.")
        st.stop()
    genai.configure(api_key=api_key)
except KeyError:
    st.error("`GOOGLE_API_KEY` not found in Streamlit secrets.")
    st.info("Please add it to your `.streamlit/secrets.toml` file or the Streamlit Cloud dashboard.")
    st.stop()
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()

# Initialize the Gemini model
try:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    st.error(f"Failed to initialize Gemini model: {e}")
    st.info("Please check your Google API key and ensure it has access to 'gemini-1.5-pro-latest'.")
    st.stop()

# --- Helper Functions (Backend Logic) ---

def extract_text_from_file(uploaded_file):
    """
    Extracts text content from various file types.
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    content = ""
    try:
        if file_type == "txt":
            content = uploaded_file.read().decode("utf-8")
        elif file_type == "pdf":
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            for page in reader.pages:
                content += page.extract_text() or ""
        elif file_type in ["docx"]:
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            for para in doc.paragraphs:
                content += para.text + "\n"
        elif file_type in ["xlsx", "csv"]:
            if file_type == "xlsx":
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else: # csv
                df = pd.read_csv(uploaded_file)
            content = df.to_string(index=False)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None
    return content

def generate_ai_overview(model, combined_content):
    """
    Generates a comprehensive summary and key topics for the combined documents.
    """
    prompt = f"""
    You are an AI assistant designed to analyze documents. Based on the following combined documents,
    provide a comprehensive summary and list the key topics. Format your response clearly.

    Combined Documents:
    {combined_content}

    Summary:
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating overview: {e}"

def interactive_qa(model, combined_content, user_query):
    """
    Answers questions based ONLY on the provided combined document content.
    """
    qa_prompt = f"""
    You are an AI assistant tasked with answering questions based ONLY on the following documents.
    If the answer is not in the documents, state that you don't have enough information from the documents.
    Do not use any external knowledge.

    Documents:
    {combined_content}

    Question:
    {user_query}

    Answer:
    """
    try:
        response = model.generate_content(qa_prompt)
        return response.text
    except Exception as e:
        return f"Error getting answer: {e}"

# --- Streamlit User Interface (Frontend) ---

if 'all_notebooks' not in st.session_state:
    st.session_state.all_notebooks = {'Default Notebook': {}}
if 'current_notebook' not in st.session_state:
    st.session_state.current_notebook = 'Default Notebook'
# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.sidebar.subheader("Notebooks")
notebook_names = list(st.session_state.all_notebooks.keys())
selected_notebook = st.sidebar.selectbox("Select a notebook", notebook_names, index=notebook_names.index(st.session_state.current_notebook))

st.session_state.current_notebook = selected_notebook
current_library = st.session_state.all_notebooks[st.session_state.current_notebook]

new_notebook_name = st.sidebar.text_input("New notebook name")
if st.sidebar.button("Create Notebook"):
    if new_notebook_name and new_notebook_name not in st.session_state.all_notebooks:
        st.session_state.all_notebooks[new_notebook_name] = {}
        st.session_state.current_notebook = new_notebook_name
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader(f"Files in '{st.session_state.current_notebook}'")
if current_library:
    for filename in current_library.keys():
        st.sidebar.write(f"- {filename}")
else:
    st.sidebar.write("No files in this notebook.")

st.subheader(f"Current Notebook: {st.session_state.current_notebook}")
uploaded_files = st.file_uploader("Upload documents to your library", type=["txt", "pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)
if uploaded_files:
    files_added = False
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in current_library:
            content = extract_text_from_file(uploaded_file)
            if content:
                current_library[uploaded_file.name] = content
                files_added = True
    if files_added:
        st.rerun()

if not current_library:
    st.info("Upload a file to enable the AI features and start a chat.")
else:
    combined_content = "\n\n---\n\n".join(current_library.values())

    st.subheader("2. Get AI Overview (Summary & Key Topics)")
    if st.button("Generate AI Overview"):
        with st.spinner("Analyzing all documents..."):
            overview = generate_ai_overview(model, combined_content)
            st.markdown(overview)
    st.markdown("---")

    st.subheader("3. Interactive Chat")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Searching documents for an answer..."):
            # Get the full chat history including the new prompt
            full_chat_history = [
                {"role": "user", "parts": [combined_content]},
                {"role": "model", "parts": ["Understood. I have reviewed the documents. How can I help?"]},
            ] + [{"role": msg["role"], "parts": [msg["content"]]} for msg in st.session_state.messages]

            # Use Gemini's chat history for context
            chat = model.start_chat(history=full_chat_history)

            # Get the response from the LLM
            response = chat.send_message(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response.text)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.text})
    st.markdown("---")
