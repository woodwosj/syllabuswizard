import os
import json
import tempfile
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI
import requests
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
JINA_API_KEY = 'jina_744f2c5c58084713bfc7f2f5372246afY2w6X8kWUwHlzadQPFV34oreSDCs'
NGROK_DOMAIN = 'obviously-quick-osprey.ngrok-free.app'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'syllabi' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('syllabi')
    
    if not files:
        logging.error("No files uploaded")
        return jsonify({"error": "No files uploaded"}), 400

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            full_url = f"https://{NGROK_DOMAIN}/uploads/{filename}"
            uploaded_files.append(full_url)
            logging.info(f"File uploaded: {full_url}")

    if not uploaded_files:
        logging.error("No valid files uploaded")
        return jsonify({"error": "No valid files uploaded"}), 400

    return jsonify({"message": "Files uploaded successfully", "files": uploaded_files})

@app.route('/process', methods=['POST'])
def process_files():
    files = request.json.get('files', [])
    if not files:
        logging.error("No files to process")
        return jsonify({"error": "No files to process"}), 400

    processed_data = []
    jina_errors = []
    for file_url in files:
        try:
            # Use Jina to process the file URL directly
            jina_url = f"https://r.jina.ai/{file_url}"
            headers = {
                'Authorization': f'Bearer {JINA_API_KEY}',
                'ngrok-skip-browser-warning': 'true',
                'User-Agent': 'SyllabusWizard/1.0'
            }
            logging.info(f"Sending request to Jina API for file: {file_url}")
            response = requests.get(jina_url, headers=headers, timeout=30)
            response.raise_for_status()
            processed_content = response.text
            logging.info(f"Received response from Jina API for file: {file_url}")
            
            # Save processed content
            processed_filename = f"processed_{os.path.basename(file_url)}.txt"
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            with open(processed_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            processed_data.append(processed_content)
        except requests.RequestException as e:
            error_msg = f"Error processing file {file_url}: {str(e)}"
            logging.error(error_msg)
            jina_errors.append(error_msg)

    if jina_errors:
        return jsonify({
            "error": "Jina API errors occurred",
            "details": jina_errors
        }), 500

    if not processed_data:
        logging.error("No files were successfully processed")
        return jsonify({"error": "No files were successfully processed"}), 500

    try:
        # Compile all processed data
        compiled_data = "\n\n".join(processed_data)
        logging.info("Compiled data from all processed files")

        # Process with LLM
        final_schedule = generate_final_schedule(compiled_data)
        logging.info("Generated final schedule from LLM")
        
        # Convert the schedule to a more user-friendly format
        formatted_schedule = format_schedule(final_schedule)
        logging.info("Formatted the schedule for display")
    except Exception as e:
        error_msg = f"Error compiling or processing data: {str(e)}"
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 500

    # Cleanup
    cleanup_files([os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(f)) for f in files])

    return jsonify(formatted_schedule)

def generate_final_schedule(compiled_data):
    prompt = f"""
    Analyze the following syllabus information and generate a comprehensive academic schedule:

    {compiled_data}

    Create a JSON object with the following structure:
    {{
        "schedule": [
            {{
                "date": "YYYY-MM-DD",
                "name": "string",
                "class": "string",
                "type": "string",
                "weight": "string or null"
            }}
        ],
        "grading_rubrics": {{
            "ClassName1": {{
                "Component1": "Weight1",
                "Component2": "Weight2"
            }}
        }},
        "conflicts": [
            {{
                "description": "string",
                "classes_involved": ["string"]
            }}
        ],
        "notes": ["string"]
    }}

    Ensure the schedule is sorted chronologically and includes every single graded component and important date without summarization.
    If no valid data is found, include appropriate notes explaining the situation.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a highly detail-oriented assistant that creates exhaustive academic schedules from extracted syllabus information. Your goal is to ensure no graded components or important dates are overlooked."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            max_tokens=4000
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Error generating final schedule: {str(e)}")
        return {"error": f"Error generating final schedule: {str(e)}"}

def format_schedule(schedule_data):
    formatted = ""

    # Format schedule
    formatted += "Your Academic Schedule:\n\n"
    if schedule_data.get("schedule"):
        for item in schedule_data["schedule"]:
            formatted += f"{item['date']}: {item['name']} ({item['class']} - {item['type']}) {f'[{item['weight']}]' if item['weight'] else ''}\n"
    else:
        formatted += "No schedule items found.\n"

    # Format grading rubrics
    formatted += "\nGrading Rubrics:\n\n"
    if schedule_data.get("grading_rubrics"):
        for class_name, rubric in schedule_data["grading_rubrics"].items():
            formatted += f"{class_name}:\n"
            for component, weight in rubric.items():
                formatted += f"  - {component}: {weight}\n"
    else:
        formatted += "No grading rubrics found.\n"

    # Format conflicts
    if schedule_data.get("conflicts"):
        formatted += "\nConflicts:\n\n"
        for conflict in schedule_data["conflicts"]:
            formatted += f"- {conflict['description']} (Classes involved: {', '.join(conflict['classes_involved'])})\n"

    # Format notes
    if schedule_data.get("notes"):
        formatted += "\nNotes:\n\n"
        for note in schedule_data["notes"]:
            formatted += f"- {note}\n"

    return formatted

def cleanup_files(file_list):
    for file in file_list:
        try:
            os.remove(file)
        except Exception as e:
            logging.error(f"Error deleting file {file}: {str(e)}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['ngrok-skip-browser-warning'] = 'true'
    response.headers['User-Agent'] = 'SyllabusWizard/1.0'
    return response

if __name__ == '__main__':
    app.run(debug=True)