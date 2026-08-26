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
st.write("Hello! I am your personal real estate assistant. Feel free to ask me anything about the property details or layouts!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous dialogue history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
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
        
        # 1. Stream the core description text first
        full_response = st.write_stream(text_stream)
        
        # 2. Render the matched image exactly below the description
        if final_images:
            for img_path in final_images:
                st.image(img_path, use_container_width=True)
        
        # 3. PURE DYNAMIC POST-IMAGE ENGAGEMENT ENGINE
        # Generate custom questions based on what the user originally asked, shown AFTER the picture
        query_lower = user_input.lower()
        follow_up_question = "Would you like to explore more details about this section?" # Default fallback
        
        if any(w in query_lower for w in ["layout", "floor", "plan", "2 bhk", "3 bhk", "bedroom", "flat"]):
            follow_up_question = "✨ Would you like to explore the amenities or pricing plans next?"
        elif any(w in query_lower for w in ["map", "location", "route", "where", "situated"]):
            follow_up_question = "✨ Shall I guide you through the structural master design plan next?"
        elif any(w in query_lower for w in ["amenities", "facilities", "pool", "gym", "spa"]):
            follow_up_question = "✨ Would you like to check the available apartment floor layouts now?"
            
        st.write("") # Tiny visual spacing block
        
        # Stream the follow-up question smoothly to grab customer attention right after looking at the image
        question_placeholder = st.empty()
        typed_text = ""
        for word in follow_up_question.split(" "):
            typed_text += word + " "
            question_placeholder.markdown(f"**{typed_text}**")
            time.sleep(0.08) # Simulates natural premium human typing speed
            
        # Append the follow-up question seamlessly into the final saved text record
        full_response += f"\n\n{follow_up_question}"
                
    logger.info(f"QUESTION: '{user_input}' | DELIVERED_IMAGE: {final_images if final_images else 'None'}")
                    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response, 
        "images": final_images
    })
