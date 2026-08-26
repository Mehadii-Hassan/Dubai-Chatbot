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

# Initialize identical embeddings and load database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Using optimized mistral-small exactly as requested for fast streams and JSON compliance
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.1)

def ask_dynamic_rag_stream(user_query, chat_history):
    """Answers queries dynamically and uses advanced AI reasoning to select pure architectural assets without any hardcoding."""
    
    # 1. Retrieve the top textual context nodes
    text_docs = db.similarity_search(user_query, k=3, filter={"type": "text"})
    context_chunks = [doc.page_content for doc in text_docs]
    context = "\n\n".join(context_chunks)
    
    # 2. Retrieve a wider pool of image summaries from the database to expand choices
    image_docs = db.similarity_search(user_query, k=15, filter={"type": "image"})
    candidate_images_pool = []
    
    for doc in image_docs:
        path = doc.metadata.get("image_path")
        if path:
            if not any(item["image_path"] == path for item in candidate_images_pool):
                candidate_images_pool.append({
                    "image_path": path,
                    "description": doc.page_content
                })
            
    # 3. Premium Consultant Chat System Prompt Setup
    system_instructions = f"""
    You are the premium sales consultant for 'The Pinnacle at Sobha Central'.

    GOAL:
    Make every conversation feel like the client is browsing a luxury real estate brochure.

    RULES:
    1. RESPONSE LENGTH & TONE: Write a beautifully crafted description of about 2 to 4 full, descriptive sentences. Do not make it too short, robotic, or dry. It must feel warm, sophisticated, and premium.
    2. STRICT NO EMOJI POLICY: Do not use any emojis, icons, or decorative symbols anywhere in your response. Keep it strictly textual and professional.
    3. FACTUAL BOUNDARY: Answer ONLY using the factual brochure context provided below. Never invent or hallucinate information.
    4. IMAGE REFERENCE: Naturally mention that a relevant architectural visual, layout plan, map, or rendering from that brochure section has been brought up below.
    5. THE CLOSING QUESTION GENERATOR: At the very end of your response, separate it by a double newline, and create a completely unique, contextually relevant, and highly captivating follow-up question. This question must motivate the user and make them eager to discover the next part of their dream home. 
       - If discussing location, ask about master layout plans or lifestyle amenities.
       - If discussing layouts, ask about wellness facilities or tower features.
       Never repeat the exact same closing question across turns.

    Brochure Context:
    {context}
    """
    
    formatted_messages = [HumanMessage(content=system_instructions)]
    
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            content_to_append = msg["content"].split("|||") if "|||" in msg["content"] else msg["content"]
            formatted_messages.append(AIMessage(content=content_to_append))
            
    formatted_messages.append(HumanMessage(content=user_query))
    text_stream = llm.stream(formatted_messages)
    
    # 4. PURE AI DYNAMIC ROUTING & FILTERING ENGINE (100% Non-Hardcoded)
    final_image_paths = []
    
    if candidate_images_pool:
        re_rank_prompt = f"""You are a professional real estate asset analyzer and image routing system.
        Evaluate the user's active query, the chat history context, and analyze the candidate images pool provided in the JSON array below.
        
        CRITICAL REASONING & ROUTING RULES:
        1. ASSET TARGETING: Select exactly ONE image path that represents a core real estate asset matching the query intent. This must be a location map, floor layout plan, 3D architectural rendering, tower structure view, or physical community amenities (like pools or gyms).
        2. MANDATORY PURGING: You must actively REJECT any images whose summaries focus on human faces, lifestyle models, families, couples, decorative stock characters, musical instruments (like violins), abstract artwork, text templates, or corporate branding logos. These are completely irrelevant to property layout or destination search queries.
        
        Available Images Pool:
        {json.dumps(candidate_images_pool, indent=2)}
        
        User Query: {user_query}
        
        Respond ONLY with a valid JSON object containing a single key "selected_path". Do not include markdown wraps, code blocks, or extra text.
        Format: {{"selected_path": "extracted_data/images/page_1_img_1.png"}}
        """
        
        try:
            re_rank_response = llm.invoke([HumanMessage(content=re_rank_prompt)], response_format={"type": "json_object"})
            parsed_json = json.loads(re_rank_response.content.strip())
            selected_path = parsed_json.get("selected_path", "")
            
            if selected_path and os.environ.get("MISTRAL_API_KEY") and os.path.exists(selected_path):
                final_image_paths.append(selected_path)
        except Exception:
            pass

        # FIXED: Structural string index extractor applied safely during pure programmatic fallback routing
        if not final_image_paths and candidate_images_pool:
            fallback_target = candidate_images_pool[0]["image_path"]
            if os.path.exists(fallback_target):
                final_image_paths.append(fallback_target)

    return text_stream, final_image_paths
