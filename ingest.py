import os
import base64
import pymupdf

# --- CRITICAL FIX FOR SSL FILE NOT FOUND ERROR ---
# Removes the broken environment variable that causes the httpx FileNotFoundError
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
# ------------------------------------------------

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
# Standalone langchain_chroma package to fix deprecation warning
from langchain_chroma import Chroma 
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Constants and paths configuration
PDF_PATH = "data/brochure.pdf"
DB_DIR = "./chroma_db"
IMAGE_DIR = "extracted_data/images"

# Ensure output directory for extracted images exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Initialize Mistral Vision model and HuggingFace local embeddings framework
vision_model = ChatMistralAI(model="pixtral-12b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def encode_image_to_base64(path):
    """Encodes a local image file into a base64 string for vision API consumption."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

documents = []

# Open the target PDF document using PyMuPDF
doc = pymupdf.open(PDF_PATH)
print(f"Processing PDF: {len(doc)} pages found...")

for page_num, page in enumerate(doc):
    page_id = page_num + 1
    
    # --- A) Extract text from the page dynamically ---
    text_content = page.get_text()
    if text_content.strip():
        documents.append(Document(
            page_content=text_content,
            metadata={
                "source": PDF_PATH,
                "page": page_id,
                "type": "text",
                "image_path": ""
            }
        ))
    
    # --- B) Extract and describe images from the page dynamically ---
    image_list = page.get_images(full=True)
    for img_idx, img_info in enumerate(image_list):
        # FIXED: Extract the integer xref from the tuple object (img_info[0] is the actual xref ID)
        xref = img_info[0] 
        
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        # Generate an automated unique filename incorporating page and index structure
        img_name = f"page_{page_id}_img_{img_idx + 1}.{image_ext}"
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # Write binary image data to disk
        with open(img_path, "wb") as f:
            f.write(image_bytes)
            
        # Convert local image to base64 format
        base64_image = encode_image_to_base64(img_path)
        
        # Construct standard LangChain multi-modal structural message payload
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": "Describe this image in detail. Identify if it is a location map, master plan, floor plan, apartment layout, or building photo. Keep it highly descriptive so a vector search can match queries like 'show me the map' or '2 bedroom layout'."
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/{image_ext};base64,{base64_image}"}
                }
            ]
        )
        
        print(f"Generating dynamic summary for image asset: {img_name}")
        response = vision_model.invoke([message])
        img_description = response.content
        
        # Build Document object for the image with tracking path stored in metadata
        documents.append(Document(
            page_content=f"Visual Content Summary: {img_description}",
            metadata={
                "source": PDF_PATH,
                "page": page_id,
                "type": "image",
                "image_path": img_path
            }
        ))

# Persist all dynamic text and visual summaries into Chroma Vector Database
print("Indexing extracted data elements into Vector Store...")
vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embeddings, 
    persist_directory=DB_DIR
)
print(f"Success! Indexed {len(documents)} document objects into {DB_DIR}")
