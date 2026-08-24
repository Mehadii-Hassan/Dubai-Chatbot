import streamlit as st
from chatbot import ask_dynamic_rag_stream  

# Page UI configuration setup
st.set_page_config(page_title="Real Estate Smart Consultant", page_icon="🏢", layout="centered")

st.title("🏢 Real Estate Smart Chatbot")
st.write("Hello! Feel free to ask me anything about the property details or layouts!")

# Initialize session history logs array
if "messages" not in st.session_state:
    st.session_state.messages = []

# Iteratively render previous chat dialogue history logs onto screen UI layout containers
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "images" in msg and msg["images"]:
            for img in msg["images"]:
                st.image(img, use_container_width=True)

# Capture active interactive user input query
if user_input := st.chat_input("Ask me anything..."):
    
    # Render user query message box component instantly
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process assistant execution responses loops
    with st.chat_message("assistant"):
        # Create an empty placeholder container for "Thinking..." state
        status_placeholder = st.empty()
        
        # Display "Thinking..." inside the placeholder while accessing database and initial LLM setup
        with status_placeholder.container():
            with st.spinner("Thinking..."):
                # Call backend pipelines to fetch the text generator stream and image data
                text_stream, final_images = ask_dynamic_rag_stream(user_input, st.session_state.messages[:-1])
        
        # FIXED: Instantly wipe out the "Thinking..." status block BEFORE starting the stream output rendering
        status_placeholder.empty()
        
        # Stream the generated output response word-by-word into the web canvas area natively
        full_response = st.write_stream(text_stream)
        
        # Display image cleanly right below the text block area if matched
        if final_images:
            for img_path in final_images:
                st.image(img_path, use_container_width=True)
                    
    # Log everything back cleanly into stream history records sequence array blocks
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response, 
        "images": final_images
    })
