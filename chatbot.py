import streamlit as st
import utils
import os

st.title("Chat with Repositories")

if True:

    user_repo = st.text_input("Github Link to your public codebase", "")
    if user_repo:
        st.write("You entered:", user_repo)

        ## Load the Github Repo
        embedder = utils.Embedder(user_repo)
        embedder.clone_repo()
        st.write("Your repo has been cloned")

        ## Chunk and Create DB
        st.write("Parsing the content ...")
        embedder.load_db()
        st.write("Done Loading. Ready to take your questions")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        baseprompt="You are a chat assistant which answers based on context provided from a github repo. Try to not stray away from the context provided"
        # Accept user input
        if prompt := st.chat_input("Type your question here."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": baseprompt+prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            response = embedder.retrieve_results(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
