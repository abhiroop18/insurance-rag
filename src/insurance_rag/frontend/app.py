
import requests
import streamlit as st


# =============================================================================
# CONFIG
# =============================================================================

API_URL = "http://localhost:8000"


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Insurance RAG Assistant",
    page_icon="📄",
    layout="centered",
)


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =============================================================================
# TITLE
# =============================================================================

st.title("📄 Insurance Policy Assistant")

st.caption(
    "Ask questions about your insurance policy."
)


# =============================================================================
# DISPLAY CHAT HISTORY
# =============================================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("request_id")
        ):

            st.caption(
                f"Request ID: {message['request_id']}"
            )


# =============================================================================
# CHAT INPUT
# =============================================================================

question = st.chat_input(
    "Ask a question about your policy..."
)


# =============================================================================
# PROCESS QUESTION
# =============================================================================

if question:

    # =========================================================================
    # USER MESSAGE
    # =========================================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # =========================================================================
    # ASSISTANT RESPONSE
    # =========================================================================

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # -------------------------------------------------------------
                # Call FastAPI
                # -------------------------------------------------------------

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                    },
                    timeout=60,
                )

                response.raise_for_status()

                data = response.json()

                # -------------------------------------------------------------
                # Extract response
                # -------------------------------------------------------------

                answer = data.get(
                    "answer",
                    "",
                )

                blocked = data.get(
                    "blocked",
                    False,
                )

                blocked_at = data.get(
                    "blocked_at"
                )

                request_id = data.get(
                    "request_id"
                )

                # =============================================================
                # DISPLAY ANSWER
                # =============================================================

                # Always display the answer returned by FastAPI.
                #
                # Even if the request was blocked, FastAPI may return a useful
                # fallback message such as:
                #
                # "I'm sorry, but I couldn't find relevant information..."
                #
                st.markdown(answer)

                # =============================================================
                # BLOCKED INFORMATION
                # =============================================================

                if blocked:

                    if blocked_at:

                        st.caption(
                            f"Request blocked at: {blocked_at}"
                        )

                # =============================================================
                # REQUEST ID
                # =============================================================

                if request_id:

                    st.caption(
                        f"Request ID: {request_id}"
                    )

                # =============================================================
                # SAVE ASSISTANT MESSAGE
                # =============================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "request_id": request_id,
                    }
                )

            # =================================================================
            # CONNECTION ERROR
            # =================================================================

            except requests.exceptions.ConnectionError:

                error_message = (
                    "Could not connect to the FastAPI server. "
                    "Make sure FastAPI is running on port 8000."
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            # =================================================================
            # TIMEOUT
            # =================================================================

            except requests.exceptions.Timeout:

                error_message = (
                    "The request timed out. "
                    "The RAG pipeline took too long to respond."
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            # =================================================================
            # HTTP ERROR
            # =================================================================

            except requests.exceptions.HTTPError as error:

                error_message = (
                    f"FastAPI returned an error: {error}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            # =================================================================
            # OTHER ERROR
            # =================================================================

            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This assistant uses a Retrieval-Augmented "
        "Generation (RAG) pipeline to answer questions "
        "from insurance policy documents."
    )

    st.divider()

    st.subheader("System")

    st.write("Frontend: Streamlit")
    st.write("Backend: FastAPI")
    st.write("RAG: LangGraph")

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()
