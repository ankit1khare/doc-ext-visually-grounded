import os
import io
import time
import json
import base64
import tempfile
import concurrent.futures

import streamlit as st
from dotenv import load_dotenv
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF

# Set SDK parallelism to maximum recommended values
# These need to be set before importing the SDK
os.environ["BATCH_SIZE"] = "20"  # Higher batch size for processing multiple documents
os.environ["MAX_WORKERS"] = "5"  # Max workers per document processing
os.environ["MAX_RETRIES"] = "100"  # Maximum retry attempts
os.environ["RETRY_LOGGING_STYLE"] = "inline_block"  # More compact logging

# --- Import the Agentic Doc SDK ---
from agentic_doc.parse import parse_documents
from agentic_doc.common import ChunkType

# Optional for debugging or advanced features
# from agentic_doc.utils import viz_parsed_document
# from agentic_doc.config import VisualizationConfig

#############################
# Load environment variables
#############################
load_dotenv()
api_key = os.getenv("VISION_AGENT_API_KEY")  # The SDK expects this env variable
if not api_key:
    st.error("VISION_AGENT_API_KEY not set in .env!")
    st.stop()

#############################
# ChatGPT Answer Helper
#############################
def get_answer_and_best_chunks(user_query, evidence):
    """
    Sends user_query and evidence to ChatGPT (or GPT-4).
    Returns a dict with "answer", "reasoning", and "best_chunks".
    (This part of the code is the same as your original.)
    """
    import json
    from openai import OpenAI
    
    prompt = f"""
    Use the following JSON evidence extracted from the uploaded PDF files, answer the following question based on that evidence.
    Please return your response in JSON format with three keys: 
    1. "answer": Your detailed answer to the question
    2. "reasoning": Your step-by-step reasoning process
    3. "best_chunks": A list of objects with:
       - "file"
       - "page"
       - "bboxes" (each bbox is [x, y, w, h])
       - "captions" (list of text snippets)
       - "reason"
       
    Question: {user_query}

    Evidence: {evidence}
    """

    # Example with openai API usage. Adjust as needed (model, etc.).
    try:
        client = OpenAI()
        chat_response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4", "gpt-3.5-turbo", etc.
            messages=[
                {
                    "role": "system",
                    "content": ("You are a helpful expert that analyses context deeply "
                                "and reasons through it without assuming anything.")
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        raw = chat_response.choices[0].message.content.strip()
        # If the result is wrapped in ```json ... ``` fences, remove them:
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
# Convert PDF to images for annotation display
#############################
def pdf_to_images(pdf_file):
    """
    Convert each page of an in-memory PDF to a list of images (numpy arrays).
    Also returns each page's width/height (in PDF points).
    """
    import fitz  # PyMuPDF
    images = []
    page_dims = []
    try:
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
    pdf_file.seek(0)  # Reset pointer
    return images, page_dims

#############################
# Convert an annotated image to a PDF for inline display
#############################
def image_to_pdf(image):
    """
    Takes an annotated image (np.array) and writes it to a temp PDF file,
    returning the path to that PDF for display.
    """
    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(image).save(temp_img.name)
    temp_img.close()

    pdf = FPDF(unit="mm", format="A4")
    pdf.add_page()
    pdf.image(temp_img.name, x=0, y=0, w=210)  # A4 width is 210mm
    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(temp_pdf.name)
    temp_pdf.close()

    return temp_pdf.name

#############################
# Display a PDF in an iframe
#############################
def display_pdf(pdf_path):
    """
    Embeds a PDF into Streamlit via base64 iframe.
    """
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

#############################
# Calculate scaling for bounding boxes
#############################
def calculate_scale_factors(img_width, img_height, pdf_width, pdf_height):
    """
    Given the PDF's 'points' dimensions (pdf_width, pdf_height) and
    the rendered image dimensions (img_width, img_height), compute scale factors.
    """
    # It's common to scale by the ratio of output_image / PDF_points. Adjust as needed.
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height
    return scale_x, scale_y

#############################
# Draw bounding boxes in parallel
#############################
def process_chunks_parallel(chunks_list, img, scale_factors):
    """
    For each chunk in chunks_list, draws bounding boxes on img.
    Each chunk is expected to have "bboxes" (in PDF coords).
    We'll transform them to the image coordinate space using scale_factors.
    """
    scale_x, scale_y = scale_factors
    img_h, img_w = img.shape[:2]
    
    # Process the actual boxes from SDK
    total_boxes = []
    for chunk in chunks_list:
        for bbox in chunk.get("bboxes", []):
            if len(bbox) == 4:
                # SDK format is [x, y, w, h] where x,y,w,h are normalized 0-1
                x, y, w, h = bbox
                
                # Convert to pixel coordinates
                x1 = int(x * img_w)
                y1 = int(y * img_h)
                x2 = int((x + w) * img_w)
                y2 = int((y + h) * img_h)

                # Clip to image boundaries
                x1 = max(0, min(x1, img_w - 1))
                x2 = max(0, min(x2, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                y2 = max(0, min(y2, img_h - 1))
                
                # Only add if box has reasonable size
                if x2 > x1 and y2 > y1 and (x2-x1)*(y2-y1) > 100:  # Min 100 pixels area
                    total_boxes.append((x1, y1, x2, y2))

    # Draw boxes with a nice visible GREEN color instead of blue
    for (x1, y1, x2, y2) in total_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green, 3px thick

    return img

#############################
# Parse PDFs with the agentic-doc SDK
#############################
def parse_pdf_agentic(pdf_file, filename):
    import tempfile
    from agentic_doc.parse import parse_documents

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    pdf_file.seek(0)

    try:
        # The parse_documents function will now use the environment variables 
        # we set for BATCH_SIZE and MAX_WORKERS
        parse_results = parse_documents([tmp_path])
        if not parse_results:
            return {}

        parsed_doc = parse_results[0]
        page_map = {}

        for chunk in parsed_doc.chunks:
            if chunk.chunk_type == "error":
                # Skip or handle error chunk
                continue

            # A single chunk can have multiple groundings (boxes)
            for grounding in chunk.grounding:
                # grounding.page is 0-based
                page_idx = grounding.page + 1  # convert to 1-based if you prefer

                if page_idx not in page_map:
                    page_map[page_idx] = []

                # Fix: use box attribute instead of bounding_box
                box = grounding.box
                x1, y1 = box.l, box.t
                w, h = box.r - box.l, box.b - box.t

                # Add to your "page_map" structure
                page_map[page_idx].append({
                    "bboxes": [[x1, y1, w, h]],
                    "captions": [chunk.text],  # or any other logic
                })

        return page_map
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

#############################
# Streamlit App
#############################
st.title("Multi-PDF Research Paper QA Assistant (SDK Version)")

st.markdown("""
Upload one or more PDFs. This app uses the `agentic-doc` library to extract text, tables, and bounding boxes.
Then you can ask questions, and we'll highlight the relevant pages that ChatGPT references.
""")

# 1. Upload PDFs
uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_pdfs:
    st.markdown("### PDF Previews")
    for pdf_file in uploaded_pdfs:
        with st.expander(f"Preview: {pdf_file.name}"):
            pdf_bytes = pdf_file.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            # reset pointer
            pdf_file.seek(0)

# 2. Parse PDFs (only if user has uploaded them)
if uploaded_pdfs:
    # Check if we've already parsed these PDFs in session
    current_pdfs = {pdf.name: pdf for pdf in uploaded_pdfs}

    if ("processed_pdfs" not in st.session_state
        or current_pdfs.keys() != st.session_state.processed_pdfs.keys()):
        
        # Estimate processing time based on SDK behavior
        # Based on observations: ~3-5 minutes per PDF depending on size/complexity
        avg_minutes_per_pdf = 4
        est_total_minutes = len(uploaded_pdfs) * avg_minutes_per_pdf
        
        if est_total_minutes < 1:
            time_estimate = "less than a minute"
        elif est_total_minutes == 1:
            time_estimate = "about 1 minute"
        else:
            time_estimate = f"approximately {est_total_minutes}-{est_total_minutes + len(uploaded_pdfs)} minutes"
        
        # Simple status message with estimate
        status_message = st.empty()
        status_message.info(f"Processing {len(uploaded_pdfs)} PDF(s) with agentic-doc SDK. This might take {time_estimate}...")
        
        # Start time tracking
        start_time = time.time()
        
        all_evidence = {}
        all_images = {}
        all_page_dims = {}
        
        # Use the optimized processing function for multiple PDFs
        if len(uploaded_pdfs) > 1:
            # Process multiple PDFs in parallel using optimized function
            process_results = process_pdfs_with_sdk(uploaded_pdfs)
            
            # Process the results
            for i, pdf_file in enumerate(uploaded_pdfs):
                filename = pdf_file.name
                if i < len(process_results):
                    parsed_doc = process_results[i]
                    
                    # Map to the page-based structure
                    page_map = {}
                    for chunk in parsed_doc.chunks:
                        if chunk.chunk_type == "error":
                            continue
                            
                        for grounding in chunk.grounding:
                            page_idx = grounding.page + 1
                            
                            if page_idx not in page_map:
                                page_map[page_idx] = []
                                
                            box = grounding.box
                            x1, y1 = box.l, box.t
                            w, h = box.r - box.l, box.b - box.t
                            
                            page_map[page_idx].append({
                                "bboxes": [[x1, y1, w, h]],
                                "captions": [chunk.text],
                            })
                    
                    # Store in evidence structure
                    for page_num, chunk_list in page_map.items():
                        composite_key = f"{filename}:{page_num}"
                        all_evidence[composite_key] = chunk_list
                    
                # Create images for bounding box annotation (same as before)
                pdf_file.seek(0)
                images, page_dims = pdf_to_images(pdf_file)
                all_images[filename] = images
                all_page_dims[filename] = page_dims
        else:
            # Original process for single PDFs
            for pdf_file in uploaded_pdfs:
                filename = pdf_file.name
                
                # Parse & store page-based evidence
                page_map = parse_pdf_agentic(pdf_file, filename)
                for page_num, chunk_list in page_map.items():
                    composite_key = f"{filename}:{page_num}"
                    all_evidence[composite_key] = chunk_list
                
                # Create images for bounding box annotation
                pdf_file.seek(0)
                images, page_dims = pdf_to_images(pdf_file)
                all_images[filename] = images
                all_page_dims[filename] = page_dims
        
        # Processing complete - show total time
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        status_message.success(f"✅ All PDFs processed in {minutes}m {seconds}s")
        
        # Store in session
        st.session_state.all_evidence = all_evidence
        st.session_state.all_images = all_images
        st.session_state.all_page_dims = all_page_dims
        st.session_state.processed_pdfs = current_pdfs
    else:
        # Load from session
        all_evidence = st.session_state.all_evidence
        all_images = st.session_state.all_images
        all_page_dims = st.session_state.all_page_dims

    st.markdown("### Precomputation Complete")

    # 3. Ask a Question
    st.markdown("## Ask a Question")
    user_query = st.chat_input("Enter your question about the PDFs:")
    if user_query:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        start_time = time.time()

        with st.status("Analyzing your question..."):
            # 3a. Combine the evidence
            filtered_evidence = {k: v for k, v in all_evidence.items() if v}
            combined_evidence = json.dumps(filtered_evidence, indent=2)

            # 3b. ChatGPT call
            result_json = get_answer_and_best_chunks(user_query, combined_evidence)
            answer = result_json.get("answer", "No answer provided.")
            reasoning = result_json.get("reasoning", "No reasoning provided.")
            best_chunks = result_json.get("best_chunks", [])

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Create two tabs in the same window
        current_qa_tab, chat_history_tab = st.tabs(["Current Q&A", "Chat History"])

        # Content for the Current Q&A tab
        with current_qa_tab:
            st.subheader("Your Question:")
            st.chat_message("user").write(user_query)
            
            st.subheader("Answer:")
            st.chat_message("assistant").write(answer)
            
            if best_chunks:
                st.markdown("### Highlighted Evidence")

                # Group best_chunks by (file, page)
                matched = {}
                for chunk in best_chunks:
                    key = f"{chunk.get('file')}:{chunk.get('page')}"
                    matched.setdefault(key, []).append(chunk)

                # Annotate each matched page
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_map = {}
                    for comp_key, chunks_list in matched.items():
                        try:
                            filename, page_str = comp_key.split(":")
                            page_num = int(page_str)
                            page_idx = page_num - 1

                            if filename in all_images and page_idx < len(all_images[filename]):
                                img = all_images[filename][page_idx].copy()
                                pdf_width, pdf_height = all_page_dims[filename][page_idx]
                                scale_factors = calculate_scale_factors(
                                    img_width=img.shape[1],
                                    img_height=img.shape[0],
                                    pdf_width=pdf_width,
                                    pdf_height=pdf_height
                                )
                                future = executor.submit(
                                    process_chunks_parallel, chunks_list, img, scale_factors
                                )
                                future_map[future] = (comp_key, filename, page_num)
                        except Exception as e:
                            st.error(f"Error in parallel processing {comp_key}: {e}")

                    # Render results
                    for future in concurrent.futures.as_completed(future_map):
                        comp_key, filename, page_num = future_map[future]
                        try:
                            annotated_img = future.result()
                            annotated_pdf_path = image_to_pdf(annotated_img)
                            st.markdown(f"**Matched Page {page_num} from {filename}**")
                            display_pdf(annotated_pdf_path)
                        except Exception as e:
                            st.warning(f"Failed to process {comp_key}: {e}")

                # Reasoning & evidence
                with st.expander("Answer Analysis and Supporting Evidence", expanded=True):
                    st.markdown("**Reasoning / Thought Process**")
                    st.write(reasoning)

                    st.markdown("**Supporting Evidence:**")
                    for chunk in best_chunks:
                        fn = chunk.get("file", "")
                        pg = chunk.get("page", "")
                        st.markdown(f"#### {fn} - Page {pg}")
                        for cap in chunk.get("captions", []):
                            st.write(f"- {cap}")
                        st.markdown(f"**Why this supports the answer:** {chunk.get('reason', '')}")
                        st.markdown("---")

                total_time = round(time.time() - start_time, 2)
                st.info(f"Completed in {total_time} seconds.")
            else:
                st.info("No supporting chunks identified.")
                total_time = round(time.time() - start_time, 2)
                st.warning(f"No evidence found. (Took {total_time} seconds)")

        # Content for the Chat History tab
        with chat_history_tab:
            st.markdown("### Complete Chat History")
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.chat_message("user").write(chat["content"])
                else:
                    st.chat_message("assistant").write(chat["content"])
else:
    st.info("Please upload PDF files to begin, and ensure `VISION_AGENT_API_KEY` is set in your .env.")

def process_pdfs_with_sdk(uploaded_pdfs):
    """Process multiple PDFs using the Landing AI SDK with maximum parallelism"""
    temp_paths = []
    try:
        # Create a progress bar in the UI
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Convert streamlit uploaded files to local paths
        for pdf_file in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                temp_paths.append(tmp.name)
        
        progress_text.text(f"Preparing {len(uploaded_pdfs)} documents for parallel processing...")
        
        # Import any necessary configuration options from the SDK
        from agentic_doc.common import ChunkType
        
        # Setup progress tracking
        import threading
        import time
        progress_tracker = {"done": False, "estimated_pct": 0}
        
        def update_progress():
            start_time = time.time()
            max_time = len(uploaded_pdfs) * 240  # Estimate 4 minutes per document
            
            while not progress_tracker["done"]:
                elapsed = time.time() - start_time
                pct = min(0.99, elapsed / max_time)
                progress_bar.progress(pct)
                
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                progress_text.text(f"Parsing documents: {elapsed_min:02d}:{elapsed_sec:02d} elapsed " +
                                   f"(~{int(pct*100)}% complete)")
                
                time.sleep(1)
        
        # Start progress tracking thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Process all documents - SDK will handle parallelism based on env vars
            results = parse_documents(temp_paths)
            
            # Signal completion
            progress_tracker["done"] = True
            progress_bar.progress(1.0)
            progress_text.text("Document parsing complete!")
            
            return results
        except Exception as e:
            # Signal completion even on error
            progress_tracker["done"] = True
            progress_bar.empty()
            progress_text.error(f"Error parsing documents: {e}")
            raise
    finally:
        # Clean up temp files
        for path in temp_paths:
            try:
                os.unlink(path)
            except Exception as e:
                st.warning(f"Failed to remove temporary file {path}: {e}")
