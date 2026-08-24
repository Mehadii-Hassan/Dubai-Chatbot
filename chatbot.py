import os
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Prevent SSL bugs in local environments
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Initialize Mistral Large for text generation
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.4) # Slightly higher temperature for natural friendly speech

def ask_dynamic_rag_stream(user_query, chat_history):
    """Retrieves context and yields text chunks sequentially along with final unique image tracking data."""
    # 1. Retrieve top 4 context materials from Vector DB
    retrieved_docs = db.similarity_search(user_query, k=4)
    
    context_chunks = []
    dynamic_images = []
    
    for doc in retrieved_docs:
        if doc.metadata.get("type") == "text":
            context_chunks.append(doc.page_content)
        elif doc.metadata.get("type") == "image":
            context_chunks.append(doc.page_content) 
            if doc.metadata.get("image_path"):
                dynamic_images.append(doc.metadata["image_path"])
                
    context = "\n\n".join(context_chunks)
    
    # 2. Re-architected system prompt for high-utility friendly persona execution
    system_instructions = f"""You are a warm, welcoming, and highly professional real estate consultant assistant. 
    Your tone must be friendly, polite, energetic, and helpful. Always welcome the user nicely and treat them like a valued client.
    Answer the user's question accurately based ONLY on the provided document context map below. 
    If the client wants to see a layout, picture, plan, or map, warmly acknowledge that you are bringing it up for them on the screen right now.

    Context:
    {context}"""
    
    # 3. Format the conversation chain payload using standard LangChain Message history
    formatted_messages = [HumanMessage(content=system_instructions)]
    
    # Inject memory history loops
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_messages.append(AIMessage(content=msg["content"]))
            
    # Add current active user input query
    formatted_messages.append(HumanMessage(content=user_query))
    
    # 4. Generate text streaming pipeline using .stream() generator architecture
    text_stream = llm.stream(formatted_messages)
    
    # 5. Extract image asset data paths
    visual_keywords = ["show", "see", "pic", "picture", "image", "map", "layout", "photo", "ছবি", "দেখাও", "ম্যাপ", "লেআউট", "প্ল্যান"]
    user_wants_visual = any(word in user_query.lower() for word in visual_keywords)
    
    final_image_paths = []
    if user_wants_visual and dynamic_images:
        seen = set()
        for path in dynamic_images:
            if path not in seen:
                final_image_paths.append(path)
                break # Only fetch the single best layout image instance
                
    return text_stream, final_image_paths
