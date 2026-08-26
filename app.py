import streamlit as st
import logging
import time
from chatbot import ask_dynamic_rag_stream  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] User_Session: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Real Estate Smart Consultant", page_icon="🏢", layout="centered")
st.title("🏢 Real Estate Smart RAG Chatbot")
st.write("Welcome! Feel free to ask any questions about the project property specifications, locations, or architecture layout designs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous dialogue history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Display saved text output safely (stripping hidden structural tags if any)
        display_content = msg["content"].split("|||")[0] if "|||" in msg["content"] else msg["content"]
        st.write(display_content)
        if "images" in msg and msg["images"]:
            for img in msg["images"]:
                st.image(img, use_container_width=True)

if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        
        with status_placeholder.container():
            with st.spinner("Thinking..."):
                text_stream, final_images = ask_dynamic_rag_stream(user_input, st.session_state.messages[:-1])
        
        status_placeholder.empty()
        
        # Capture the raw streamed text entirely
        complete_llm_output = ""
        text_placeholder = st.empty()
        
        for chunk in text_stream:
            complete_llm_output += chunk.content
            # Live stream the text buildup silently without showing final isolated lines prematurely
            text_placeholder.markdown(complete_llm_output)
            
        # Parse the output into Core Description vs AI Generated Closing Question
        # Looking for double newline or question mark breakdown structures safely
        lines = complete_llm_output.strip().split("\n\n")
        
        core_description = lines[0]
        ai_closing_question = lines[-1] if len(lines) > 1 else ""
        
        # If model failed to split properly with double newline, scan dynamically
        if not ai_closing_question and "?" in complete_llm_output:
            sentences = complete_llm_output.split("?")
            ai_closing_question = sentences[-2].strip() + "?" if len(sentences) > 1 else ""
            core_description = complete_llm_output.replace(ai_closing_question, "").strip()

        # Step 1: Render ONLY the elite descriptive block initially
        text_placeholder.markdown(core_description)
        
        # Step 2: Smoothly inject the verified brochure image asset right below
        if final_images:
            for img_path in final_images:
                st.image(img_path, use_container_width=True)
        
        # Step 3: Stream the AI's completely unique follow-up question AFTER the image display loop
        if ai_closing_question:
            st.write("") # Spacer block
            question_placeholder = st.empty()
            typed_text = ""
            for word in ai_closing_question.split(" "):
                typed_text += word + " "
                question_placeholder.markdown(f"*{typed_text.strip()}*")
                time.sleep(0.07) # Natural typing rhythm speed delay simulation
                
        # Consolidate the entire raw payload response back safely into history lists
        final_history_text = f"{core_description}\n\n{ai_closing_question}"
                
    logger.info(f"QUESTION: '{user_input}' | DELIVERED_IMAGE: {final_images if final_images else 'None'}")
                    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_history_text, 
        "images": final_images
    })
