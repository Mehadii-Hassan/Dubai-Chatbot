import os
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.3)

def ask_dynamic_rag_stream(user_query, chat_history):
    """Answers queries dynamically using user's premium prompt and orchestrates AI-driven image re-ranking."""
    
    # 1. Retrieve the top 7 elements to guarantee a wide pool of candidate images
    retrieved_docs = db.similarity_search(user_query, k=7)
    
    context_chunks = []
    candidate_images_pool = []
    
    for doc in retrieved_docs:
        if doc.metadata.get("type") == "text":
            context_chunks.append(doc.page_content)
        elif doc.metadata.get("type") == "image":
            context_chunks.append(doc.page_content) 
            path = doc.metadata.get("image_path")
            if path:
                candidate_images_pool.append({
                    "image_path": path,
                    "description": doc.page_content
                })
                
    context = "\n\n".join(context_chunks)
    
    # 2. INTEGRATED YOUR PREMIUM UPGRADED PROMPT (Slightly tuned for clean text generation)
    system_instructions = f"""
    You are the premium sales consultant for 'The Pinnacle at Sobha Central'.

    GOAL:
    Make every conversation feel like the client is browsing a luxury real estate brochure.

    RULES:
    1. Keep every answer short, energetic, and engaging (maximum 2-3 short bullets or sentences).
    2. Answer ONLY using the brochure context below. Never invent information.
    3. Naturally reference that a stunning project visual, layout plan, map, or render has been displayed for them below.
       Examples:
       - "I've shown the master plan below."
       - "You can see the 3-bedroom layout below."
       - "The clubhouse rendering is displayed below."
    4. Write in a premium, elegant, brochure-like tone that builds excitement without sounding pushy.
    5. Whenever appropriate, end with a short follow-up question that encourages the next interaction.
       Examples:
       - "Would you like to explore the floor plans?"
       - "Shall I show you the amenities?"
       - "Interested in the payment plan?"

    Brochure Context:
    {context}
    """
    
    formatted_messages = [HumanMessage(content=system_instructions)]
    
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_messages.append(AIMessage(content=msg["content"]))
            
    formatted_messages.append(HumanMessage(content=user_query))
    text_stream = llm.stream(formatted_messages)
    
    # 3. PURE AI IMAGE RE-RANKER (Remains 100% Dynamic - No manual rules)
    final_image_paths = []
    
    if candidate_images_pool:
        re_rank_prompt = f"""You are an expert image retrieval router. Analyze the user's current query and the chat history context.
        From the JSON array of available images below, pick exactly ONE image that is the most relevant visual asset to accompany the answer.
        - If the user asks for a map/location, select a map.
        - If the user asks for a layout/floor plan, select that specific floor plan.
        - If the user asks about amenities or general project features, select a relevant facility, rendering, or lifestyle image.
        
        Available Images:
        {json.dumps(candidate_images_pool, indent=2)}
        
        User Query: {user_query}
        
        CRITICAL: Respond ONLY with a valid JSON containing a single key "selected_path". Do not include markdown blocks or extra text.
        Example response format:
        {{"selected_path": "extracted_data/images/page_1_img_1.png"}}
        If absolutely no images match the topic, respond with:
        {{"selected_path": ""}}"""
        
        try:
            re_rank_response = llm.invoke([HumanMessage(content=re_rank_prompt)], response_format={"type": "json_object"})
            raw_content = re_rank_response.content.strip()
            parsed_json = json.loads(raw_content)
            selected_path = parsed_json.get("selected_path", "")
            
            if selected_path and os.path.exists(selected_path):
                final_image_paths.append(selected_path)
        except Exception:
            if candidate_images_pool:
                final_image_paths.append(candidate_images_pool[0]["image_path"])

    return text_stream, final_image_paths
