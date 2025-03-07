import io
import os
import requests
import streamlit as st
import concurrent.futures
from PyPDF2 import PdfReader, PdfWriter
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import json
from openai import OpenAI
import urllib3
from fpdf import FPDF
import tempfile
import base64
from functools import lru_cache
import time

# Disable SSL warnings (for development; not recommended for production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()
# Ensure LANDING_API_KEY is set in your .env file.
api_key = os.getenv("LANDING_API_KEY")
if not api_key:
    st.error("LANDING_API_KEY not set in .env!")
    st.stop()

#############################
# API CALL FUNCTION
#############################
def call_api(pdf_bytes, api_key):
    url = "https://api.landing.ai/v1/tools/document-analysis"
    files = {"pdf": ("chunk.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    data = {
        "parse_text": True,
        "parse_tables": True,
        "parse_figures": True,
        "summary_verbosity": "none",
        "caption_format": "json",
        "response_format": "json",
        "return_chunk_crops": False,
        "return_page_crops": False,
    }
    headers = {"Authorization": f"Basic {api_key}"}
    response = requests.post(url, files=files, data=data, headers=headers, timeout=600, verify=False)
    try:
        return response.json()
    except Exception as e:
        return {"error": str(e), "response_text": response.text}

def call_api_with_retry(pdf_bytes, api_key, max_retries=3, backoff_factor=2):
    """Call API with retry logic"""
    for attempt in range(max_retries):
        try:
            url = "https://api.landing.ai/v1/tools/document-analysis"
            files = {"pdf": ("chunk.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
            data = {
                "parse_text": True,
                "parse_tables": True,
                "parse_figures": True,
                "summary_verbosity": "none",
                "caption_format": "json",
                "response_format": "json",
                "return_chunk_crops": False,
                "return_page_crops": False,
            }
            headers = {"Authorization": f"Basic {api_key}"}
            response = requests.post(url, files=files, data=data, headers=headers, timeout=600, verify=False)
            response.raise_for_status()  # Raise exception for bad status codes
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                return {"error": str(e), "response_text": getattr(response, 'text', str(e))}
            wait_time = backoff_factor ** attempt
            st.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

#############################
# Helper to ensure result is dict
#############################
def ensure_dict(result, idx):
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception as e:
            st.error(f"Error parsing JSON for chunk {idx+1}: {e}")
            return {}
    return result

#############################
# Helper to get answer and best chunks from ChatGPT
#############################
def get_answer_and_best_chunks(user_query, evidence):
    prompt = f"""
    Use the following JSON evidence extracted from the uploaded PDF files, answer the following question based on that evidence.
    Please return your response in JSON format with three keys: 
    1. "answer": Your detailed answer to the question
    2. "reasoning": Your step-by-step reasoning process explaining how you arrived at the answer
    3. "best_chunks": A list of objects that support your answer, where each object must include:
    - "file": The filename where the evidence was found
    - "page": The page number (1-indexed) where the evidence was found
    - "bboxes": A list of bounding boxes, where each box is [x, y, w, h]
    - "captions": A list of captions or text snippets corresponding to each bbox
    - "reason": A detailed explanation of why these specific chunks support your answer and how they connect to your reasoning

    Note: Most of the times, an answer spans multiple pages, multiple files, and have many bboxes and captions associated with it, verify if the overall answer and reasoning is derived from the best_chunks selected from the evidence without missing any chunk and don't skip returning all associated chunks with all relevant bboxes and captions.
        
    Question: {user_query}

    Evidence: {evidence}
    """
    try:
        client = OpenAI()
        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful expert that analyses the context deeply and reasons through it without assuming anything to provide a detailed and accurate answer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        raw = chat_response.choices[0].message.content.strip()
        # Remove markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        st.error(f"Error getting answer from ChatGPT: {e}")
        return {
            "answer": "Sorry, I could not retrieve an answer.", 
            "reasoning": "An error occurred during processing.",
            "best_chunks": []
        }

#############################
# Split a PDF into chunks of 3 pages
#############################
def split_pdf_into_chunks(pdf_file, chunk_size=3):
    try:
        reader = PdfReader(pdf_file)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None, 0
    total_pages = len(reader.pages)
    chunks = []
    for i in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        for j in range(i, min(i + chunk_size, total_pages)):
            writer.add_page(reader.pages[j])
        pdf_chunk_buffer = io.BytesIO()
        writer.write(pdf_chunk_buffer)
        pdf_chunk_buffer.seek(0)
        chunks.append(pdf_chunk_buffer.getvalue())
    return chunks, total_pages

#############################
# Convert a PDF to images (one per page) and return dimensions (in PDF points)
#############################
def pdf_to_images(pdf_file):
    images = []
    page_dims = []  # (pdf_width, pdf_height) per page
    try:
        import fitz  # PyMuPDF
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in pdf_document:
            rect = page.rect
            page_dims.append((rect.width, rect.height))
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.array(img))
        pdf_document.close()
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
    return images, page_dims

#############################
# Convert an annotated image to a PDF file and return its path
#############################
def image_to_pdf(image):
    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(image).save(temp_img.name)
    temp_img.close()
    pdf = FPDF(unit="mm", format="A4")
    pdf.add_page()
    pdf.image(temp_img.name, x=0, y=0, w=210)  # A4 width is 210 mm; adjust if needed.
    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(temp_pdf.name)
    temp_pdf.close()
    return temp_pdf.name

#############################
# Display a PDF in an iframe
#############################
def display_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

#############################
# Optimized chunk matching for speed
#############################
@lru_cache(maxsize=128)
def calculate_scale_factors(img_width, img_height, pdf_width, pdf_height):
    """Calculate scale factors with aspect ratio preservation"""
    # Calculate scales that would preserve aspect ratio
    scale_x = img_width / pdf_width - 0.7
    scale_y = img_height / pdf_height - 0.7
    
    return scale_x, scale_y


    # Use the smaller scale to ensure the entire content fits
    # scale = min(scale_x, scale_y)
    
    # return scale, scale  # Use same scale for both dimensions

def process_chunks_parallel(chunks_list, img, scale_factors, offset_x, offset_y, invert_y):
    """Process chunks in parallel using numpy vectorization"""
    img_height, img_width = img.shape[:2]
    scale_x, scale_y = scale_factors
    
    # Pre-allocate arrays for all boxes
    total_boxes = sum(len(chunk.get("bboxes", [])) for chunk in chunks_list)
    boxes = np.zeros((total_boxes, 4), dtype=np.int32)
    box_idx = 0
    
    # Vectorized bbox processing
    for chunk in chunks_list:
        bboxes = chunk.get("bboxes", [])
        for bbox in bboxes:
            if len(bbox) == 4:
                # Convert PDF coordinates to image coordinates
                x1 = int(bbox[0] * scale_x)
                x2 = int(bbox[2] * scale_x)
                
                if invert_y:
                    y1 = int(img_height - (bbox[3] * scale_y))
                    y2 = int(img_height - (bbox[1] * scale_y))
                else:
                    y1 = int(bbox[1] * scale_y)
                    y2 = int(bbox[3] * scale_y)
                
                # Apply offsets and ensure coordinates are within bounds
                x1 = max(0, min(x1 + offset_x, img_width - 1))
                x2 = max(0, min(x2 + offset_x, img_width - 1))
                y1 = max(0, min(y1 + offset_y, img_height - 1))
                y2 = max(0, min(y2 + offset_y, img_height - 1))
                
                boxes[box_idx] = [x1, y1, x2, y2]
                box_idx += 1
    
    # Batch draw rectangles
    for box in boxes[:box_idx]:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    return img

def process_pdf_in_batches(pdf_file, api_key, batch_size=2):
    """Process PDF in smaller batches"""
    chunks, total_pages = split_pdf_into_chunks(pdf_file, chunk_size=3)
    if chunks is None:
        return [], total_pages
    
    chunk_results = []
    
    # Process each chunk separately to maintain correct page ordering
    for i in range(0, len(chunks)):
        chunk = chunks[i]
        start_page = i * 3  # Each chunk contains 3 pages
        
        try:
            result = call_api_with_retry(chunk, api_key)
            if isinstance(result, dict):
                result['chunk_start_page'] = start_page
                chunk_results.append(result)
            else:
                st.error(f"Invalid result format for chunk {i}")
        except Exception as exc:
            st.error(f"Error processing chunk starting at page {start_page + 1}: {exc}")
            chunk_results.append({"error": str(exc), "chunk_start_page": start_page})
    
    return chunk_results, total_pages

#############################
# Main Application
#############################
st.title("Multi-PDF Research Paper QA Assistant")

st.markdown("""
Upload one or more PDFs. The app will precompute evidence (text chunks with bounding boxes) from each PDF using the Landing AI API.
You can then ask a question, and the system will answer using a ChatGPT model – displaying only the pages (as PDFs) where supporting evidence was found.
""")

# Allow multiple PDF uploads.
uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Add PDF previews in collapsible sections
if uploaded_pdfs:
    st.markdown("### PDF Previews")
    st.info("You can preview each PDF while the system processes them. Click to expand/collapse.")
    for pdf_file in uploaded_pdfs:
        with st.expander(f"Preview: {pdf_file.name}"):
            # Convert PDF to base64 for display
            pdf_bytes = pdf_file.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            # Reset file pointer after reading
            pdf_file.seek(0)

if uploaded_pdfs and api_key:
    # Check if these PDFs have already been processed
    current_pdfs = {pdf.name: pdf for pdf in uploaded_pdfs}
    
    # Only process if PDFs changed or haven't been processed
    if ("processed_pdfs" not in st.session_state or 
        current_pdfs.keys() != st.session_state.processed_pdfs.keys()):
        
        with st.status("Processing new PDFs..."):
            all_evidence = {}       
            all_images = {}         
            all_page_dims = {}      
            all_total_pages = {}    

            # Process each PDF file
            for pdf_file in uploaded_pdfs:
                filename = pdf_file.name
                st.write(f"Processing {filename}...")
                chunk_results, total_pages = process_pdf_in_batches(pdf_file, api_key, batch_size=2)
                if not chunk_results:
                    continue
                
                # Reset file pointer and get images/dimensions.
                pdf_file.seek(0)
                page_images, page_dims = pdf_to_images(pdf_file)
                if not page_images:
                    st.error(f"Failed to convert {filename} to images.")
                    continue
                
                # Save images and dims in dictionaries.
                all_images[filename] = page_images
                all_page_dims[filename] = page_dims
                all_total_pages[filename] = total_pages

                # Precompute evidence concurrently for each chunk.
                for idx, result in enumerate(chunk_results):
                    result = ensure_dict(result, idx)
                    pages_data = result.get("data", {}).get("pages", [])
                    start_page = result.get('chunk_start_page', 0)
                    for i, page_data in enumerate(pages_data):
                        actual_page = start_page + i + 1  # Convert to 1-indexed page numbers
                        composite_key = f"{filename}:{actual_page}"
                        st.write(f"Processing page {actual_page}")
                        all_evidence[composite_key] = page_data.get("chunks", [])
            
            # Save everything in session state
            st.session_state.all_evidence = all_evidence
            st.session_state.all_images = all_images
            st.session_state.all_page_dims = all_page_dims
            st.session_state.all_total_pages = all_total_pages
            st.session_state.processed_pdfs = current_pdfs
            
            st.success("PDF processing complete!")
    else:
        # Use existing processed data from session state
        all_evidence = st.session_state.all_evidence
        all_images = st.session_state.all_images
        all_page_dims = st.session_state.all_page_dims
        all_total_pages = st.session_state.all_total_pages

    st.markdown("### Precomputation Complete")
    
    # --- Chat Interface ---
    st.markdown("## Ask a Question")
    user_query = st.chat_input("Enter your question about the PDFs:")
    if user_query:
        # Append user's query to chat history.
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Start timing
        start_time = time.time()
        
        # Show progress messages
        with st.status("Analyzing your question...", expanded=True) as status:
            # Combine evidence from all files that have evidence.
            filtered_evidence = {}
            for comp_key, evidence in all_evidence.items():
                if evidence:
                    filtered_evidence[comp_key] = evidence
            combined_evidence = json.dumps(filtered_evidence, indent=2)
            
            # Debug: Show precomputed evidence
            # debug_col1, debug_col2 = st.columns(2)
            # with debug_col1:
            #     st.markdown("**Debug: Precomputed Evidence JSON**")
            #     st.code(combined_evidence, language="json")
            
            status.update(label="Analyzing...", state="running")
            
            # Get answer and best chunks from ChatGPT.
            result_json = get_answer_and_best_chunks(user_query, combined_evidence)
            answer = result_json.get("answer", "No answer provided.")
            reasoning = result_json.get("reasoning", "No reasoning provided.")
            best_chunks = result_json.get("best_chunks", [])
            
            # Debug: Show matched chunks
            # with debug_col2:
            #     st.markdown("**Debug: Matched Chunks JSON**")
            #     st.code(json.dumps(best_chunks, indent=2), language="json")
            
            status.update(label="Finding evidence in PDFs...", state="running")
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            # Display chat interface in tabs
            tab1, tab2 = st.tabs(["Current Q&A", "Chat History"])
            
            with tab1:
                # Display current question and answer
                st.chat_message("user").write(user_query)
                st.chat_message("assistant").write(answer)
                
                if best_chunks:
                    status.update(label="Results generated", state="running")
                    
                    # Process and display annotated PDFs first
                    matched = {}
                    for chunk in best_chunks:
                        key = f"{chunk.get('file')}:{chunk.get('page')}"
                        matched.setdefault(key, []).append(chunk)
                    
                    # Process pages in parallel
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_to_key = {}
                        for comp_key, chunks_list in matched.items():
                            try:
                                # Split the composite key correctly
                                filename = comp_key.split(':')[0]
                                page_num = int(comp_key.split(':')[1])
                                page_idx = page_num - 1
                                
                                if filename in all_images and page_idx < len(all_images[filename]):
                                    img = all_images[filename][page_idx].copy()
                                    img_height, img_width = img.shape[:2]
                                    pdf_width, pdf_height = all_page_dims[filename][page_idx]
                                    
                                    # Get cached scale factors
                                    scale_factors = calculate_scale_factors(
                                        img_width, img_height, 
                                        pdf_width, pdf_height
                                    )
                                    
                                    future = executor.submit(
                                        process_chunks_parallel,
                                        chunks_list,
                                        img,
                                        scale_factors,
                                        0,
                                        0,
                                        False
                                    )
                                    future_to_key[future] = (comp_key, filename, page_num)
                            except Exception as e:
                                st.error(f"Error processing {comp_key}: {e}")
                                continue
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_key):
                            comp_key, filename, page_num = future_to_key[future]
                            try:
                                annotated_img = future.result()
                                annotated_pdf_path = image_to_pdf(annotated_img)
                                st.markdown(f"**Matched Page {page_num} from {filename}**")
                                display_pdf(annotated_pdf_path)
                            except Exception as e:
                                st.warning(f"Failed to process {comp_key}: {e}")

                    # Display analysis after the PDFs
                    st.markdown("### Answer Analysis and Supporting Evidence")
                    
                    # Display the reasoning for chunk selection
                    st.markdown("**Why these chunks were selected:**")
                    st.write(reasoning)

                    # Display the supporting evidence
                    st.markdown("\n**Supporting Evidence:**")
                    for chunk in best_chunks:
                        st.markdown(f"📄 **{chunk.get('file')} - Page {chunk.get('page')}**")
                        
                        # Display all captions
                        captions = chunk.get('captions', [])
                        if captions:
                            st.markdown("**Text:**")
                            for caption in captions:
                                st.markdown(f"- {caption}")
                        
                        # Display the reason
                        st.markdown(f"**Why this supports the answer:** {chunk.get('reason')}")
                        st.markdown("---")

                    # Calculate and display total processing time at the end
                    total_time = round(time.time() - start_time, 2)
                    status.update(label=f"Thought process complete! (took {total_time} seconds)", state="complete")
                else:
                    # Calculate total processing time for error case
                    total_time = round(time.time() - start_time, 2)
                    status.update(label=f"No supporting evidence found (took {total_time} seconds)", state="error")
                    st.info("No supporting chunks identified.")

            with tab2:
                st.markdown("### Complete Chat History")
                # Display full chat history
                for chat in st.session_state.chat_history:
                    if chat["role"] == "user":
                        st.chat_message("user").write(chat["content"])
                    else:
                        st.chat_message("assistant").write(chat["content"])
else:
    st.info("Please upload one or more PDF files and ensure LANDING_API_KEY is set in .env.")
