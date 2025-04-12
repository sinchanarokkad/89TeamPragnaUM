import streamlit as st
import hashlib
from PIL import Image
from PIL import (Image)
import easyocr
from deep_translator import GoogleTranslator
import numpy as np
from pdf2image import convert_from_bytes
import os
import re
import requests
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import nltk
from nltk.tokenize import sent_tokenize


nltk.download('punkt_tab')

ORDS_BASE_URL = "https://gb679cfde2e0a96-d2w1n9hojweqjdr2.adb.ap-mumbai-1.oraclecloud.com/ords/batch/"

db_username='BATCH_FEB25'
db_password='Welcome$022025'

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

# Plsql procedure for register user
def register_user(userid, password, email):
    """Register a new user using ORDS API"""
    try:
        password_hash = hash_password(password)
        payload = {
            "p_userid": userid,
            "p_email": email,
            "p_password": password_hash
        }
        response = requests.post(
            'https://gb679cfde2e0a96-d2w1n9hojweqjdr2.adb.ap-mumbai-1.oraclecloudapps.com/ords/batch/russian_register_user_proc/',
            json=payload,
            headers={"accept": "application/json",'Content-Type':"application/json"},
            auth=(db_username,db_password)
        )
        
        if response.status_code == 200 or response.status_code==201:
            data=response.json()
            status_code = data.get("p_status_code")
            status_message = data.get("p_status_message")

            if status_code == 201:
                st.success("Registration successful! Please login.")
                return True
            elif status_code == 409:
                st.error(f"Registration failed: {status_message}")
                return False
            else:
                st.error(f"Registration failed: {status_message} (Code: {status_code})")
                return False
        else:
            st.error(f"API Error: Status code {response.status_code} {response.content}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False

def verify_login(userid, password, email):
    """Verify user login credentials using ORDS API"""
    try:
        password_hash = hash_password(password)
        print(password_hash)
        payload = {
            "p_userid": userid,
            "p_password": password_hash,
            "p_email": email
        }
        response = requests.post(
            'https://gb679cfde2e0a96-d2w1n9hojweqjdr2.adb.ap-mumbai-1.oraclecloudapps.com/ords/batch/russian_verify_login/',
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            auth=(db_username, db_password)
        )
        
        if response.status_code in (200, 201):
            data = response.json()
            
           
            login_success = data.get("p_login_success")
            status_code = data.get("p_status_code")
            status_message = data.get("p_status_message")
            
            
            if login_success and status_code == 200:
                st.success(status_message)
                return True
            else:
                st.error(f"Login failed: {status_message} (Code: {status_code})")
                return False
        else:
            st.error(f"API Error: Status code {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        st.error(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        return False

def store_extracted_text(userid, file_format, extracted_text, invoice_name):
    """Store extracted text using ORDS API"""
    try:
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        payload = {
            "p_userid": userid,
            "p_format_of_upload": file_format,
            "p_extracted_text": extracted_text,
            "p_upload_time": current_time,
            "p_invoice_name": invoice_name
        }
        
        
        response = requests.post(
            'https://gb679cfde2e0a96-d2w1n9hojweqjdr2.adb.ap-mumbai-1.oraclecloudapps.com/ords/batch/russian_store_extracted_text/',
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            auth=(db_username, db_password)
        )
        
        # Process the response
        if response.status_code in (200, 201):
            data = response.json()
            
            status_code = data.get("p_status_code")
            status_message = data.get("p_status_message")
            
            if status_code == 201: 
                st.success(f"Message from the PL/SQL:  {status_message}")
                return True
            else:
                st.error(f"Failed to store text: {status_message} (Code: {status_code})")
                return False
        else:
            st.error(f"API Error: Status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Error storing extracted text: {str(e)}")
        return False
    
def store_translated_text(userid, translated_text):
    """Store translated text using ORDS API"""
    try:
        payload = {
            "p_userid": userid,
            "p_translated_text": translated_text,
            "p_translated_time": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        }
        response = requests.post(
            'https://gb679cfde2e0a96-d2w1n9hojweqjdr2.adb.ap-mumbai-1.oraclecloudapps.com/ords/batch/russian_store_translated_text/',
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            auth=(db_username, db_password)
        )
        
        if response.status_code == 200 or response.status_code == 201:
            data = response.json()
            status_code = data.get("p_status_code")
            status_message = data.get("p_status_message")
            if data:
                st.success(f'Message from the PL/SQL: {status_message}')
                return True
            else:
                st.error(f'Failed to store translated text!:{status_message} Code:{status_code}')
                return False
        else:
            st.error(f"API Error: Status code {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Error storing translated text: {str(e)}")
        return False
def store_accuracy_metrics(userid, file_name, accuracy):
    """Store accuracy metrics using ORDS API"""
    try:
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        payload = {
            "p_user_id": userid,
            "p_file_name": file_name,
            "p_accuracy": float(accuracy),
            "p_created_on": current_time,
            "p_modified_on": current_time,
            "p_email":0
        }
        
        response = requests.post(
            'https://gb679cfde2e0a96-d2w1n9hojweqjdr2.adb.ap-mumbai-1.oraclecloudapps.com/ords/batch/russian_insert_accuracy/',
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            auth=(db_username, db_password)
        )
        
        
        if response.status_code in (200, 201):

            data = response.json()
            success_flag = data.get("p_success_flag")
            error_message = data.get("p_error_message")
            response_mail=requests.get('https://conneqtion-dev2-bmwxqal1yvum-bo.integration.ap-mumbai-1.ocp.oraclecloud.com/ic/api/integration/v1/flows/rest/RUSSIAN_SEND_EMAIL/1.0/send_mail',auth=("lokendar.singh@conneqtiongroup.com","LokendarSingh$2025"))
            print(response_mail)
            if success_flag == 'Y':
                st.success("Accuracy metrics stored successfully!")
                return True
            else:
                st.error(f"Failed to store metrics: {error_message}")
                return False
        else:
            st.error(f"API Error: Status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Error storing accuracy metrics: {str(e)}")
        return False

# OCR and Translation Functions
def extract_text_from_image(image, reader):
    """Extract text from image using EasyOCR and return with confidence scores"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)

        results = reader.readtext(image)
        
        # Collect text and corresponding confidence scores
        extracted_data = [{'text': result[1], 'confidence': result[2]} for result in results]
        
        full_text = '\n'.join([data['text'] for data in extracted_data])
        
        # Calculate average confidence score
        if extracted_data:
            avg_confidence = sum(data['confidence'] for data in extracted_data) / len(extracted_data)
        else:
            avg_confidence = 0.0
        
        return full_text, avg_confidence, extracted_data  # Return text, confidence, and all extracted data
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return "", 0.0, []


def convert_pdf_to_images(pdf_bytes, poppler_path):
    """Convert PDF to images using pdf2image"""
    try:
        if not os.path.exists(poppler_path):
            st.error(f"Poppler binaries not found in {poppler_path}")
            st.info("Please follow the setup instructions to add Poppler binaries.")
            return None

        images = convert_from_bytes(
            pdf_bytes,
            poppler_path=poppler_path,
            dpi=300
        )
        return images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        return None

def process_pdf_with_ocr(pdf_file, reader, poppler_path):
    """Process PDF file with OCR and compute confidence scores"""
    try:
        pdf_bytes = pdf_file.getvalue()
        images = convert_pdf_to_images(pdf_bytes, poppler_path)
        if images is None:
            return ""

        full_text = ""
        total_confidence = 0.0
        page_count = len(images)
        progress_bar = st.progress(0)
        all_extracted_data = []

        for idx, img in enumerate(images):
            text, confidence, extracted_data = extract_text_from_image(img, reader)
            full_text += text + "\n\n--- Page Break ---\n\n"
            total_confidence += confidence
            all_extracted_data.extend(extracted_data)
            progress_bar.progress((idx + 1) / page_count)

            with st.expander(f"Preview Page {idx + 1}", expanded=False):
                st.image(img, caption=f"Page {idx + 1}", use_column_width=True)
        
        if page_count > 0:
            avg_confidence = total_confidence / page_count
        else:
            avg_confidence = 0.0

        return full_text, avg_confidence, all_extracted_data  # Return full text, average confidence, and all extracted data

    except Exception as e:
        st.error(f"Error processing PDF with OCR: {str(e)}")
        return "", 0.0, []

def translate_text(text):
    """Translate text from Russian to English"""
    try:
        if not text.strip():
            return ""

        max_chunk_size = 5000
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        translator = GoogleTranslator(source='ru', target='en')
        translated_chunks = []

        progress_bar = st.progress(0)
        for idx, chunk in enumerate(chunks):
            translated_chunk = translator.translate(chunk)
            if translated_chunk:
                translated_chunks.append(translated_chunk)
            progress_bar.progress((idx + 1) / len(chunks))


        return ''.join(translated_chunks)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return ""

# Load Groq LLM for advanced receipt parsing
@st.cache_resource
def load_llm_client():
    """Initialize and cache the Groq LLM client"""
    try:
        # Get API key from environment variable or Streamlit secrets
        groq_api_key = "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"
        
        if not groq_api_key:
            st.warning("GROQ_API_KEY not found. LLM-enhanced parsing will not be available.")
            return None
            
        # Initialize the ChatGroq client
        client = ChatGroq(
            groq_api_key=groq_api_key,
            temperature=0,
            model_name="llama3-70b-8192"  # Using Llama3 70B model for advanced parsing
        )
        return client
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {str(e)}")
        return None

def parse_receipt_details_with_llm(text, llm_client=None):
    """
    Extract key details from receipt text using Groq LLM
    
    Args:
        text (str): The translated receipt text
        llm_client: The initialized Groq LLM client
        
    Returns:
        dict: Dictionary containing receipt details
    """
    # Default values in case parsing fails
    receipt_details = {
        'bill_number': None,
        'amount': None,
        'date': None,
        'establishment': None
    }
    
    # If no text or no LLM client available, fall back to regex-based parsing
    if not text.strip() or llm_client is None:
        return parse_receipt_details_regex(text)
    
    try:
        prompt = f"""
        Extract the following information from this receipt text:
        1. Bill number (or receipt number, check number)
        2. Total amount (just the number, without currency symbol)
        3. Date (in format DD-MM-YYYY if possible)
        4. Establishment name (name of the store, restaurant, or business)

        Return ONLY a JSON object with these keys: "bill_number", "amount", "date", "establishment"
        If information is not found, use null for that field.

        Receipt text:
        {text}
        """
        
        # Call the Groq LLM
        with st.spinner("Processing receipt with AI..."):
            messages = [HumanMessage(content=prompt)]
            llm_response = llm_client.invoke(messages)
            
            # Get the response content
            response_content = llm_response.content
            
            # Extract the JSON part from the response
            json_str = None
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "{" in response_content and "}" in response_content:
                # Extract everything between the first { and the last }
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                json_str = response_content[start:end]
                
            if json_str:
                parsed_data = json.loads(json_str)
                
                # Update receipt details with parsed data
                receipt_details.update({
                    "bill_number": parsed_data.get("bill_number"),
                    "amount": float(parsed_data.get("amount")) if parsed_data.get("amount") and parsed_data.get("amount") != "null" else None,
                    "date": parsed_data.get("date"),
                    "establishment": parsed_data.get("establishment")
                })
    except Exception as e:
        st.warning(f"AI parsing error: {str(e)}. Falling back to basic extraction.")
        return parse_receipt_details_regex(text)
        
    return receipt_details

def parse_receipt_details_regex(text):
    """
    Extract key details from the receipt text using regex patterns.
    Args:
        text (str): The OCR-extracted and translated text from the receipt
    Returns:
        dict: Dictionary containing receipt details
    """
    receipt_details = {
        'bill_number': None,
        'amount': None,
        'date': None,
        'establishment': None
    }

    # Split text into lines for easier processing
    lines = text.split('\n')

    # Process each line
    for line in lines:
        # Extract bill number
        if 'Sales check' in line or 'check' in line.lower() or 'receipt' in line.lower() or 'bill' in line.lower():
            # Look for patterns like "Sales check 123/456" or "Receipt #123"
            match = re.search(r'(?:Sales check|receipt|bill|check|#|No\.)\s*[#:]?\s*(\d+(?:[/-]\d+)?)', line, re.IGNORECASE)
            if match:
                receipt_details['bill_number'] = match.group(1)

        # Extract amount - look for currency amounts
        # This looks for patterns like "Total: 123.45" or "Amount Due: 123.45"
        amount_match = re.search(r'(?:total|amount|sum|итого)(?:\s*:)?\s*(?:\$|€|£|₽|RUB)?\s*(\d+[.,]\d{2})', line, re.IGNORECASE)
        if amount_match:
            # Convert to float, handling comma as decimal separator
            amount_str = amount_match.group(1).replace(',', '.')
            receipt_details['amount'] = float(amount_str)
        elif not receipt_details['amount']:
            # Fallback: look for standalone numbers that match currency format
            amount_fallback = re.search(r'\b(\d+[.,]\d{2})\b', line)
            if amount_fallback:
                amount_str = amount_fallback.group(1).replace(',', '.')
                receipt_details['amount'] = float(amount_str)

        # Extract date - look for common date formats
        # DD-MM-YYYY, DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD
        date_patterns = [
            r'(\d{2}[-/.]\d{2}[-/.]\d{4})',  # DD-MM-YYYY, DD.MM.YYYY, DD/MM/YYYY
            r'(\d{4}[-/.]\d{2}[-/.]\d{2})',  # YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})'  # 15 January 2023
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, line, re.IGNORECASE)
            if date_match:
                receipt_details['date'] = date_match.group(1)
                break  # Stop after first date match

        # Extract establishment name
        # Look for common indicators of business names
        establishment_indicators = ['LLC', 'Ltd', 'Inc', 'Hotel', 'Restaurant', 'Cafe', 'Store', 'Shop', 'Market']
        for indicator in establishment_indicators:
            if indicator in line:
                # Extract a reasonable length string around the indicator
                match = re.search(r'(.{5,50}' + indicator + r'.{0,20})', line)
                if match:
                    receipt_details['establishment'] = match.group(1).strip()
                    break
        
        # Special case for Hotel President
        if 'President' in line and ('Hotel' in line or 'Отель' in line):
            receipt_details['establishment'] = 'President-Hotel'

    return receipt_details

# NEW: Function to display accuracy meter for OCR confidence
def display_accuracy_meter(confidence, extracted_data=None):
    """
    Display a visually appealing accuracy meter based on confidence score
    
    Args:
        confidence (float): The confidence score (0-1)
        extracted_data (list): Optional list of extracted text items with confidence scores
    """
    st.markdown("### 📊 OCR Accuracy Analysis")
    
    # Overall confidence meter
    st.markdown("#### Overall Confidence Score")
    
    # Determine color based on confidence
    if confidence >= 0.8:
        color = "green"
        status = "High Accuracy"
    elif confidence >= 0.6:
        color = "orange"
        status = "Medium Accuracy"
    else:
        color = "red"
        status = "Low Accuracy"
    
    # Create a visually appealing progress bar
    st.markdown(
        f"""
        <div style="margin-bottom: 10px;">
            <div style="
                width: 100%;
                background-color: #f0f0f0;
                border-radius: 5px;
                height: 30px;
                position: relative;">
                <div style="
                    width: {confidence * 100}%;
                    background-color: {color};
                    height: 30px;
                    border-radius: 5px;
                    transition: width 0.5s;">
                </div>
                <div style="
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: #000;
                    font-weight: bold;">
                    {confidence * 100:.1f}% - {status}
                </div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Display detailed confidence information if available
    if extracted_data and len(extracted_data) > 0:
        with st.expander("View Detailed Text Confidence Analysis", expanded=False):
            # Create a table showing individual text elements and their confidence
            st.markdown("#### Text Elements Confidence Breakdown")
            
            # Display a table with the text elements and their confidence scores
            table_data = []
            for i, item in enumerate(extracted_data[:20]):  # Limit to first 20 items to avoid clutter
                table_data.append({
                    "Item": i+1,
                    "Text": item['text'][:50] + "..." if len(item['text']) > 50 else item['text'],
                    "Confidence": f"{item['confidence'] * 100:.1f}%",
                    "Status": "✅ High" if item['confidence'] >= 0.8 else "⚠️ Medium" if item['confidence'] >= 0.6 else "❌ Low"
                })
            
            # Use Streamlit's dataframe
            st.dataframe(table_data)
            
            if len(extracted_data) > 20:
                st.info(f"Showing 20 of {len(extracted_data)} text elements. The complete text is available in the text area above.")
            
            # Add histogram visualization of confidence distribution
            st.markdown("#### Confidence Distribution")
            
            # Prepare data for histogram
            confidence_values = [item['confidence'] for item in extracted_data]
            
            # Calculate distribution
            ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            hist_data = [0] * (len(ranges) - 1)
            
            for conf in confidence_values:
                for i in range(len(ranges) - 1):
                    if ranges[i] <= conf < ranges[i+1]:
                        hist_data[i] += 1
                        break
            
            # Add recommendations based on confidence
            st.markdown("#### Accuracy Recommendations")
            
            if confidence < 0.6:
                st.warning("""
                **Low overall accuracy detected.** Consider:
                - Using higher resolution images
                - Improving lighting or document contrast
                - Checking if the document is Russian text
                - For PDF files, try extracting the text directly if it's a digital PDF
                """)
            elif confidence < 0.8:
                st.info("""
                **Medium accuracy detected.** For better results:
                - Check for blurry or low-contrast areas in the document
                - Make sure the document is properly oriented
                - Review the extracted text for potential errors
                """)
            else:
                st.success("""
                **High accuracy detected!** The OCR process performed well.
                - Always review the translation for context-specific errors
                - Some specialized terminology may still need manual correction
                """)

# UI Pages
def login_page():
    """Display login page"""
    st.title("Login to Russian Translation App")

    with st.form("login_form"):
        userid = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Login")

        if submitted:
                if verify_login(userid, password, email):
                    st.session_state.logged_in = True
                    st.session_state.userid = userid
                    st.rerun()
                else:
                    st.error("Invalid credentials or user not registered!")
            

    st.write("---")
    if st.button("New user? Register here"):
        st.session_state.show_register = True
        st.rerun()

def register_page():
    """Display registration page"""
    st.title("Register for Russian Translation App")

    with st.form("registration_form"):
        userid = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Register")

        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Invalid email format!")
            else:
                if register_user(userid, password, email):
                    st.session_state.show_register = False
                    st.rerun()
           

    st.write("---")
    if st.button("Already have an account? Login here"):
        st.session_state.show_register = False
        st.rerun()

def segment_bills(text):
    """
    Segment a document containing multiple bills into separate bill texts
    
    Args:
        text (str): The full extracted text that may contain multiple bills
        
    Returns:
        list: List of dictionaries, each containing a separate bill's text
    """
    # If text is too short, assume it's a single bill
    if len(text) < 200:
        return [{"text": text, "confidence": 1.0}]
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Return single bill if too few sentences
    if len(sentences) < 5:
        return [{"text": text, "confidence": 1.0}]
    
    # Use heuristic segmentation first (look for common bill separators)
    bill_segments = heuristic_bill_segmentation(text)
    
    # If heuristic segmentation found multiple bills, return them
    if len(bill_segments) > 1:
        return [{"text": segment, "confidence": 0.9} for segment in bill_segments]
    
    # Otherwise try algorithmic segmentation
    bills = algorithmic_bill_segmentation(sentences)
    
    if len(bills) <= 1:
        # If still only one bill detected, return original text
        return [{"text": text, "confidence": 1.0}]
    
    return bills

def heuristic_bill_segmentation(text):
    """
    Use pattern matching to identify bill boundaries
    
    Args:
        text (str): Full text with potential multiple bills
        
    Returns:
        list: List of bill text segments
    """
    # Common bill separator patterns
    separators = [
        r'={5,}',  # Multiple equals signs
        r'-{5,}',  # Multiple hyphens
        r'\*{5,}',  # Multiple asterisks
        r'_{5,}',  # Multiple underscores
        r'#{5,}',  # Multiple hash marks
        r'NEW RECEIPT',  # Explicit indicator
        r'НОВЫЙ ЧЕК',    # Russian for NEW RECEIPT
        r'RECEIPT \d+',  # Receipt with number
        r'ЧЕК \d+',      # Russian for RECEIPT with number
        r'BILL \d+',     # Bill with number
        r'СЧЕТ \d+',     # Russian for BILL with number
        r'={3,} Page Break ={3,}',  # Page break marker
        r'--- Page Break ---',      # Another page break format
    ]
    
    # Create combined pattern
    combined_pattern = '|'.join(f'({pattern})' for pattern in separators)
    
    # Split the text by the combined pattern
    segments = re.split(combined_pattern, text)
    
    # Filter out the separators from the results and empty segments
    bills = [seg.strip() for seg in segments if seg and not any(re.match(pattern, seg) for pattern in separators)]
    
    return bills

def algorithmic_bill_segmentation(sentences):
    """
    Use clustering to identify natural document segments
    
    Args:
        sentences (list): List of sentences from the document
        
    Returns:
        list: List of dictionaries with bill texts and confidence scores
    """
    try:
        # Vectorize sentences
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        X = vectorizer.fit_transform(sentences)
        
        # Cluster sentences using DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(X)
        
        # Get cluster labels
        labels = clustering.labels_
        
        # Group sentences by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sentences[i])
        
        # Convert clusters to bill segments
        bill_segments = []
        for label, segment_sentences in clusters.items():
            if label != -1:  # Ignore noise cluster
                bill_text = " ".join(segment_sentences)
                # Only include segments with substantial content
                if len(bill_text.strip()) > 50:
                    bill_segments.append({
                        "text": bill_text,
                        "confidence": 0.7  # Algorithmic confidence is lower than heuristic
                    })
        
        # If no valid segments found, return original text
        if not bill_segments:
            combined_text = " ".join(sentences)
            return [{"text": combined_text, "confidence": 1.0}]
            
        return bill_segments
    
    except Exception as e:
        st.warning(f"Error in algorithmic segmentation: {str(e)}")
        # Fallback: return full text as one segment
        combined_text = " ".join(sentences)
        return [{"text": combined_text, "confidence": 1.0}]

# Update the receipt parsing to handle multiple bills
def process_multiple_bills(extracted_text, avg_confidence, llm_client=None):
    """
    Process text that may contain multiple bills
    
    Args:
        extracted_text (str): The full extracted text
        avg_confidence (float): Average OCR confidence
        llm_client: The LLM client for advanced parsing
        
    Returns:
        list: List of bill information dictionaries
    """
    # Segment the text into potential multiple bills
    bill_segments = segment_bills(extracted_text)
    
    bills_data = []
    
    for i, segment in enumerate(bill_segments):
        bill_text = segment["text"]
        segment_confidence = segment["confidence"] * avg_confidence  # Adjust confidence
        
        # Translate the bill segment
        translated_text = translate_text(bill_text)
        
        # Parse the bill details
        receipt_details = parse_receipt_details_with_llm(translated_text, llm_client) if llm_client else parse_receipt_details_regex(translated_text)
        
        # Add additional metadata
        bill_data = {
            "bill_id": i + 1,
            "original_text": bill_text,
            "translated_text": translated_text,
            "receipt_details": receipt_details,
            "confidence": segment_confidence
        }
        
        bills_data.append(bill_data)
    
    return bills_data

# Main Translation App
def translation_app():
    """Main translation application interface with multi-bill support"""
    try:
        user_msg = requests.get(f"https://conneqtion-dev2-bmwxqal1yvum-bo.integration.ap-mumbai-1.ocp.oraclecloud.com/ic/api/integration/v1/flows/rest/RT_RUSSIAN_TRANSLATOR/1.0/RussianUsers?username={st.session_state.userid}",auth=("lokendar.singh@conneqtiongroup.com","LokendarSingh$2025"))
        welcome_msg = user_msg.json()['message']
    except:
        welcome_msg = "Welcome"

    # Add a custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .sub-header {
        color: #2C5282;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    .ai-badge {
        background-color: #4F46E5;
        color: white;
        font-size: 0.7rem;
        padding: 3px 8px;
        border-radius: 10px;
        margin-left: 8px;
    }
    .bill-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        border-left: 4px solid #3182CE;
    }
    .bill-header {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 8px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header with personalized welcome message
    st.markdown(f'<h1 class="main-header">📚 Russian to English OCR Translator</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size: 1.2rem; color: #4A5568;">{welcome_msg}</p>', unsafe_allow_html=True)

    # Initialize paths and models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    poppler_path = r"C:\Users\sinch\Downloads\NewProject\poppler"

    if not os.path.exists(poppler_path):
        st.warning("""
        Poppler binaries not found!
        Please ensure you have installed the required dependencies.
        """)

    # Initialize EasyOCR
    @st.cache_resource
    def load_ocr():
        return easyocr.Reader(['ru'], gpu=False)

    with st.spinner("Loading OCR model... (this may take a minute on first run)"):
        reader = load_ocr()
        
    # Initialize Groq LLM
    llm_client = load_llm_client()
    
    # Show AI status
    if llm_client:
        st.sidebar.markdown("🧠 **AI Status:** <span style='color:green'>Connected</span>", unsafe_allow_html=True)
        st.sidebar.markdown("Using Groq LLM for enhanced receipt parsing")
    else:
        st.sidebar.markdown("🧠 **AI Status:** <span style='color:orange'>Limited</span>", unsafe_allow_html=True)
        st.sidebar.markdown("Using regex-based receipt parsing")

    # Add a card-like container for file upload
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload Document</p>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a Russian PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    
    # Add multi-bill processing option
    enable_multi_bill = st.checkbox("Enable multiple bill detection", value=True, 
                                   help="Automatically detect and segment multiple bills in a single document")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        file_name=uploaded_file.name
        with st.spinner("Processing your file..."):
            extracted_text = ""
            file_format = uploaded_file.type  # Determine the format of the uploaded file
            
            avg_confidence = 0.0
            extracted_data = []

            # Process based on file type
            if uploaded_file.type == "application/pdf":
                st.write("📄 Processing PDF with OCR...")
                extracted_text, avg_confidence, extracted_data = process_pdf_with_ocr(uploaded_file, reader, poppler_path)
            else:
                st.write("🖼️ Processing Image...")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                extracted_text, avg_confidence, extracted_data = extract_text_from_image(image, reader)
            
            # Store Extracted Text using ORDS API
            if extracted_text:
                store_extracted_text(st.session_state.userid, file_format, extracted_text,file_name)
            
            # Process for multiple bills if enabled
            if enable_multi_bill:
                # Process multiple bills
                bills_data = process_multiple_bills(extracted_text, avg_confidence, llm_client)
                
                # Display number of bills detected
                if len(bills_data) > 1:
                    st.success(f"Successfully detected {len(bills_data)} separate bills in the document!")
                else:
                    st.info("Detected a single bill in the document.")
                
                # Display each bill in a separate section
                for bill in bills_data:
                    with st.expander(f"Bill #{bill['bill_id']} {bill['receipt_details']['establishment'] or 'Unknown Establishment'}", expanded=bill['bill_id'] == 1):
                        # Display in three columns for each bill
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.markdown('<p class="sub-header">📜 Russian Text</p>', unsafe_allow_html=True)
                            st.text_area(f"Russian Text (Bill #{bill['bill_id']})", bill['original_text'], height=200)
                        
                        with col2:
                            st.markdown('<p class="sub-header">🌍 English Translation</p>', unsafe_allow_html=True)
                            st.text_area(f"English Translation (Bill #{bill['bill_id']})", bill['translated_text'], height=200)
                        
                        with col3:
                            # Add the AI badge to the header if LLM is available
                            ai_badge = '<span class="ai-badge">AI Enhanced</span>' if llm_client else ''
                            st.markdown(f'<p class="sub-header">📋 Receipt Details {ai_badge}</p>', unsafe_allow_html=True)
                            
                            # Display formatted details
                            receipt_details = bill['receipt_details']
                            receipt_html = "<div style='background-color: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1E88E5;'>"
                            
                            if receipt_details['establishment']:
                                receipt_html += f"<p><strong>Establishment:</strong> {receipt_details['establishment']}</p>"
                            
                            if receipt_details['bill_number']:
                                receipt_html += f"<p><strong>Bill/Receipt Number:</strong> {receipt_details['bill_number']}</p>"
                            
                            if receipt_details['date']:
                                receipt_html += f"<p><strong>Date:</strong> {receipt_details['date']}</p>"
                            
                            if receipt_details['amount']:
                                receipt_html += f"<p><strong>Amount:</strong> {receipt_details['amount']}</p>"
                            
                            receipt_html += "</div>"
                            st.markdown(receipt_html, unsafe_allow_html=True)
                            
                            # Add download button for this specific bill
                            if st.button(f"Download Bill #{bill['bill_id']} as JSON", key=f"dl_bill_{bill['bill_id']}"):
                                json_data = json.dumps(bill, indent=4)
                                st.download_button(
                                    label=f"Save Bill #{bill['bill_id']} JSON",
                                    data=json_data,
                                    file_name=f"bill_{bill['bill_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                
                # Add option to download all bills as a single JSON
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="sub-header">📥 Export All Bills</p>', unsafe_allow_html=True)
                
                if st.button("Download All Bills as JSON"):
                    all_bills_json = json.dumps({
                        "total_bills": len(bills_data),
                        "extraction_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "bills": bills_data
                    }, indent=4)
                    
                    st.download_button(
                        label="Save Complete JSON",
                        data=all_bills_json,
                        file_name=f"all_bills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                # Original single-bill display code
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.markdown('<p class="sub-header">📜 Extracted Russian Text</p>', unsafe_allow_html=True)
                    st.text_area("Russian Text", extracted_text, height=300)

                with col2:
                    st.markdown('<p class="sub-header">🌍 English Translation</p>', unsafe_allow_html=True)
                    translated_text = ""
                    if extracted_text.strip():
                        with st.spinner("Translating..."):
                            translated_text = translate_text(extracted_text)
                            st.text_area("English Translation", translated_text, height=300)
                            if translated_text:
                                store_translated_text(st.session_state.userid, translated_text)
                    else:
                        st.warning("No text was extracted from the file.")

                with col3:
                    if translated_text.strip():
                        # Add the AI badge to the header if LLM is available
                        ai_badge = '<span class="ai-badge">AI Enhanced</span>' if llm_client else ''
                        st.markdown(f'<p class="sub-header">📋 Receipt Details {ai_badge}</p>', unsafe_allow_html=True)
                        
                        # Parse receipt details with AI if available
                        receipt_details = parse_receipt_details_with_llm(translated_text, llm_client) if llm_client else parse_receipt_details_regex(translated_text)
                        
                        # Display formatted details with better styling
                        receipt_html = "<div style='background-color: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1E88E5;'>"
                        
                        if receipt_details['establishment']:
                            receipt_html += f"<p><strong>Establishment:</strong> {receipt_details['establishment']}</p>"
                        
                        if receipt_details['bill_number']:
                            receipt_html += f"<p><strong>Bill/Receipt Number:</strong> {receipt_details['bill_number']}</p>"
                        
                        if receipt_details['date']:
                            receipt_html += f"<p><strong>Date:</strong> {receipt_details['date']}</p>"
                        
                        if receipt_details['amount']:
                            receipt_html += f"<p><strong>Amount:</strong> {receipt_details['amount']}</p>"
                        
                        receipt_html += "</div>"
                        st.markdown(receipt_html, unsafe_allow_html=True)
                
                # Display the accuracy meter after text extraction
                if avg_confidence > 0:
                    store_accuracy_metrics(st.session_state.userid, file_name, avg_confidence*100)
                    display_accuracy_meter(avg_confidence, extracted_data)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add Export options
                if translated_text.strip():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<p class="sub-header">📥 Export Options</p>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Download Translation as TXT"):
                            # Prepare download button for translation
                            txt_download = translated_text
                            st.download_button(
                                label="Save TXT",
                                data=txt_download,
                                file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                            
                    with col2:
                        # Create a JSON of receipt details and translation
                        if st.button("Download Receipt Details as JSON"):
                            receipt_data = {
                                "original_text": extracted_text,
                                "translated_text": translated_text,
                                "receipt_details": receipt_details,
                                "extracted_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            json_download = json.dumps(receipt_data, indent=4)
                            st.download_button(
                                label="Save JSON",
                                data=json_download,
                                file_name=f"receipt_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                            
                    st.markdown('</div>', unsafe_allow_html=True)
def main():
    """Main app initialization"""
    # Set page config
    st.set_page_config(
        page_title="Russian to English Translator",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'userid' not in st.session_state:
        st.session_state.userid = None
    
    # Display sidebar
    st.sidebar.title("📚 App Navigation")
    
    # Show user status in sidebar
    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in as: {st.session_state.userid}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.userid = None
            st.rerun()
    else:
        st.sidebar.warning("Not logged in")
    
    # Add help and info section
    with st.sidebar.expander("ℹ️ App Information"):
        st.markdown("""
        This app translates Russian documents (PDFs and images) to English using:
        - OCR technology to extract text
        - Machine translation to convert to English
        - AI-powered receipt parsing for documents containing financial information
        
        **Supported file types:**
        - PDF documents (.pdf)
        - Images (.png, .jpg, .jpeg)
        """)
    
    # App settings
    with st.sidebar.expander("⚙️ Settings"):
        st.markdown("**OCR & Translation Settings**")
        st.write("OCR model: EasyOCR (Russian)")
        st.write("Translation engine: Google Translator API")
        st.write("Receipt parser: Groq Llama3-70B")
    
    # Show appropriate page based on login state
    if st.session_state.logged_in:
        translation_app()
    else:
        if st.session_state.show_register:
            register_page()
        else:
            login_page()

if __name__ == "__main__":
    main()