import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise Knowledge Copilot",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Enterprise Knowledge Copilot")

st.caption(
    "Upload company documents, ask questions, "
    "and analyze knowledge using RAG and Agent workflows."
)

# -------------------------------------
# Backend Status
# -------------------------------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    health_response.raise_for_status()
    st.sidebar.success("Backend Connected")

except requests.RequestException:
    st.sidebar.error("Backend Disconnected")
    st.error(
        "FastAPI is not responding on port 8000. "
        "Start the backend and refresh this page."
    )
    st.stop()

# -------------------------------------
# Navigation Tabs
# -------------------------------------

upload_tab, chat_tab, agent_tab = st.tabs(
    [
        "📄 Upload Documents",
        "💬 Knowledge Chat",
        "🤖 AI Agent",
    ]
)

# =====================================
# TAB 1: DOCUMENT UPLOAD
# =====================================

with upload_tab:
    st.header("Upload a Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        key="document_uploader",
    )

    if uploaded_file is not None:
        st.write("File:", uploaded_file.name)

        if st.button("Process and Index Document"):
            try:
                with st.spinner(
                    "Extracting text and indexing document..."
                ):
                    response = requests.post(
                        f"{API_URL}/api/documents/process",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                                or "application/octet-stream",
                            )
                        },
                        timeout=300,
                    )

                    response.raise_for_status()
                    result = response.json()

                st.success("Document indexed successfully!")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Characters",
                    result["characters"],
                )
                col2.metric(
                    "Words",
                    result["words"],
                )
                col3.metric(
                    "Stored Chunks",
                    result["stored_chunks"],
                )

                with st.expander("View Extracted Text"):
                    st.text(result["extracted_text"])

                with st.expander("View Text Chunks"):
                    for index, chunk in enumerate(
                        result["chunks"],
                        start=1,
                    ):
                        st.markdown(f"**Chunk {index}**")
                        st.write(chunk)

            except requests.RequestException as error:
                st.error("Document processing failed.")
                st.caption(str(error))

            except (KeyError, ValueError) as error:
                st.error("Unexpected document-processing response.")
                st.caption(str(error))


# =====================================
# TAB 2: KNOWLEDGE CHAT
# =====================================

with chat_tab:
    st.header("Ask Your Documents")

    st.write(
        "Ask questions based on your indexed documents."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**Document:** {source['filename']}"
                        )
                        st.write(source["text"])

                        score = source.get("score")
                        if isinstance(score, (int, float)):
                            st.caption(
                                f"Similarity score: {score:.3f}"
                            )

    question = st.chat_input(
        "Ask about your uploaded documents...",
        key="knowledge_chat_input",
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner(
                    "Searching documents and generating answer..."
                ):
                    response = requests.post(
                        f"{API_URL}/api/chat",
                        json={
                            "question": question,
                            "top_k": 3,
                        },
                        timeout=180,
                    )

                    response.raise_for_status()
                    result = response.json()

                answer = result.get(
                    "answer",
                    "The backend did not return an answer.",
                )
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            st.markdown(
                                f"**Document:** {source['filename']}"
                            )
                            st.write(source["text"])

                            score = source.get("score")
                            if isinstance(score, (int, float)):
                                st.caption(
                                    f"Similarity score: {score:.3f}"
                                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

        except requests.RequestException as error:
            st.error("Could not get an answer from the backend.")
            st.caption(str(error))


# =====================================
# TAB 3: AGENT WORKFLOW
# =====================================

with agent_tab:
    st.header("🤖 Enterprise Knowledge Agent")

    st.write(
        "Choose a task and let the agent analyze your indexed documents."
    )

    task_options = {
        "Ask a Question": "question",
        "Summarize Documents": "summarize",
        "Generate Recommendations": "recommend",
        "Extract Action Items": "action_items",
    }

    selected_task = st.selectbox(
        "Select an Agent Task",
        options=list(task_options.keys()),
    )

    agent_question = st.text_area(
        "Enter your request",
        placeholder=(
            "Example: Summarize the important rules "
            "in our HR leave policy."
        ),
        height=120,
    )

    top_k = st.slider(
        "Number of document chunks to retrieve",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.button("Run AI Agent", type="primary"):
        if not agent_question.strip():
            st.warning("Please enter your request.")

        else:
            payload = {
                "task": task_options[selected_task],
                "question": agent_question.strip(),
                "top_k": top_k,
            }

            try:
                with st.spinner(
                    "Agent is processing your request..."
                ):
                    response = requests.post(
                        f"{API_URL}/api/agent",
                        json=payload,
                        timeout=180,
                    )

                    response.raise_for_status()
                    result = response.json()

                st.success("Agent task completed!")

                st.subheader("📝 Agent Answer")
                st.markdown(
                    result.get(
                        "answer",
                        "No answer was returned.",
                    )
                )

                sources = result.get("sources", [])

                st.subheader("📚 Supporting Sources")

                if sources:
                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):
                        with st.expander(
                            f"Source {index}: {source['filename']}"
                        ):
                            st.write(source["text"])

                            score = source.get("score")
                            if isinstance(score, (int, float)):
                                st.caption(
                                    f"Similarity score: {score:.3f}"
                                )
                else:
                    st.info("No supporting sources were returned.")

            except requests.HTTPError as error:
                st.error(
                    "The Agent API returned an HTTP error. "
                    "Check the FastAPI terminal."
                )

                if error.response is not None:
                    st.code(
                        error.response.text,
                        language="text",
                    )

            except requests.RequestException as error:
                st.error(
                    "Could not connect to the Agent API."
                )
                st.caption(str(error))

            except (KeyError, ValueError) as error:
                st.error("Unexpected Agent API response.")
                st.caption(str(error))