import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from main import setup_service_context, setup_document_index

# Initialize index and service context if not already in session_state
if "index" not in st.session_state or "service_context" not in st.session_state:
    try:
        index, service_context = setup_document_index()
        st.session_state.index = index
        st.session_state.service_context = service_context
    except Exception as e:
        st.error(f"Failed to initialize the document index or service context: {e}", icon="🚨")
        st.stop()

# Initialize conversation history in session_state if not already
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Display the logo
st.image("logo.png", width=200)

# Streamlit UI for the chatbot
st.title("NavigaTORS: Oropharynx Cancer Chatbot")
st.write("Ask me anything about oropharynx cancer, treatment options, TORS surgery, and related topics.")

# Clear history button
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []

# Display conversation history with styled boxes for readability
for question, response in st.session_state.conversation_history:
    st.markdown(f"**User:** {question}")
    st.markdown(f"<div style='padding: 8px; border-radius: 8px; background-color: #f0f2f6;'><b>NavigaTORS:</b> {response}</div>", unsafe_allow_html=True)

# User input box for questions
user_query = st.text_input("Your question:")

# Function to query the index and retrieve a refined LLM response
def query_index(index, query_str, service_context, placeholder):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    query_engine = RetrieverQueryEngine(retriever=retriever)
    
    # Display loading spinner
    with placeholder.spinner("Generating Response..."):
        # Initial response from the vector index
        response = query_engine.query(query_str)
        initial_response = response.response if hasattr(response, 'response') else response
        
        # Define instructions for the LLM in the prompt
        instructions = (
            "You are a chatbot patient education assistant knowledgeable about oropharynx cancer and transoral robotic surgery at Stanford. "
            "Your role is to clarify and provide educational information and answer healthcare-related questions in a clear, empathetic, and thoughtful manner. "
            "You are not a doctor, and your responses should not be taken as medical advice. Always encourage users to consult their "
            "healthcare professionals for personal health issues. You are here to help summarize complex information, provide general guidance "
            "about treatment options, peri-operative care, rehabilitation, and support, and clarify users' questions about oropharynx cancer treatments."
            "Only answers question in this context."
        )

        # Combine instructions and query response
        prompt_with_instructions = f"{instructions}\n\nUser's Query:\n{query_str}"

        # Refined response from LLM using prompt_helper
        refined_content = initial_response  # Fallback in case 'content' isn't found
        if service_context.get('llm'):
            try:
                # Invoke the prompt helper with the prompt
                refined_llm_response = service_context['llm'].invoke(prompt_with_instructions)
                
                # Attempt to extract the actual response content
                refined_content = getattr(refined_llm_response, 'content', initial_response)
            except Exception as e:
                st.error(f"Error with prompt_helper invocation: {e}", icon="🚨")
    
    return refined_content

# Process the query and display the response
if user_query:
    try:
        # Placeholder for loading message
        loading_placeholder = st.empty()
        
        # Get the response from the chatbot
        response = query_index(st.session_state.index, user_query, st.session_state.service_context, loading_placeholder)
        
        # Display the response
        st.markdown(f"<div style='padding: 8px; border-radius: 8px; background-color: #e6f7ff;'><b>NavigaTORS:</b> {response}</div>", unsafe_allow_html=True)
        
        # Append question and response to conversation history
        st.session_state.conversation_history.append((user_query, response))
        
        # Clear the input box after submission
        st.experimental_rerun()  # Clears the input field for new entry
    except Exception as e:
        st.error("An error occurred while generating the response. Please try again later.", icon="🚨")
