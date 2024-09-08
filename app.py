import os
import json
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import tempfile
from datetime import datetime

app = Flask(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_info_from_syllabus(content, filename):
    prompt = f"""
    Follow this process step by step to extract information from the syllabus:
    1. Carefully read through the entire syllabus.
    2. Identify ALL graded components, including but not limited to:
       - Assignments
       - Quizzes
       - Tests/Exams
       - Projects
       - Labs (including Packet Tracer labs)
       - Discussions
       - Participation grades
       - Any other component that contributes to the final grade
    3. For each graded component, extract:
       - Exact due date (in YYYY-MM-DD format)
       - Full, detailed name of the component
       - Type of component (e.g., Assignment, Quiz, Exam, Project, Lab, Discussion)
       - Weight in the grading rubric (if available)
    4. Identify the course start and end dates.
    5. Extract the complete grading rubric or weights table for the class.
    6. Note any important dates or deadlines, even if they're not graded.

    It is CRITICAL that you do not omit any graded components, no matter how small. Each component must be listed separately.

    Output the information in JSON format with the following structure:
    {{
        "class_name": "string",
        "course_start_date": "YYYY-MM-DD",
        "course_end_date": "YYYY-MM-DD",
        "graded_components": [
            {{
                "date": "YYYY-MM-DD",
                "name": "string",
                "type": "string",
                "weight": "string or null"
            }}
        ],
        "grading_rubric": {{
            "component": "weight"
        }},
        "important_dates": [
            {{
                "date": "YYYY-MM-DD",
                "description": "string"
            }}
        ]
    }}

    Syllabus content:
    {content}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a meticulous assistant that extracts every single piece of relevant academic information from syllabus documents. Your goal is to ensure absolutely no graded components or important dates are missed."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            max_tokens=4000
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Error processing syllabus {filename}: {str(e)}"}

def verify_extracted_info(original_content, extracted_info, filename):
    prompt = f"""
    Follow this process step by step to verify the extracted information:
    1. Carefully read through the entire original syllabus content.
    2. Compare the extracted information with the original syllabus content.
    3. Check for any missing graded components, including:
       - Assignments
       - Quizzes
       - Tests/Exams
       - Projects
       - Labs (including Packet Tracer labs)
       - Discussions
       - Participation grades
       - Any other component that contributes to the final grade
    4. Verify that all due dates are correct and in YYYY-MM-DD format.
    5. Ensure that the grading rubric is complete and accurate.
    6. Check that all important dates are included.
    7. If you find any missing information, add it to the JSON structure.
    8. If you remove any information, document your reasoning in a "notes" field.

    Original syllabus content:
    {original_content}

    Extracted information (JSON):
    {json.dumps(extracted_info, indent=2)}

    Output the final, verified, and potentially updated information in the same JSON format as the input, with an additional "notes" field if needed.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a detail-oriented assistant tasked with verifying and completing extracted syllabus information. Your goal is to ensure all graded components and important dates are captured accurately."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            max_tokens=4000
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Error verifying information for {filename}: {str(e)}"}

def generate_final_schedule(extracted_info_list):
    prompt = f"""
    Follow this process step by step to generate the final comprehensive academic schedule:
    1. Read through all the JSON files containing extracted and verified syllabus information.
    2. Combine all graded components and important dates into a single, chronological list.
    3. For each item, include:
       - Exact Date (YYYY-MM-DD format)
       - Full, detailed name of the component
       - Class Name
       - Type of component (e.g., Assignment, Quiz, Exam, Project, Lab, Discussion)
       - Weight in the grading rubric (if available)
    4. Ensure that EVERY SINGLE graded component is included, even if multiple are due on the same date.
    5. Do not summarize or group components. Each should be its own line item.
    6. Include a separate section for each class's complete grading rubric.
    7. Note any conflicts or potential issues in a separate section.
    8. Include any notes or concerns about potentially missing information in a separate section.

    Extracted information from all syllabi:
    {json.dumps(extracted_info_list, indent=2)}

    Output the final comprehensive academic schedule as a JSON object with the following structure:
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

    Ensure the schedule is sorted chronologically and includes every single graded component without summarization.
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
        return {"error": f"Error generating final schedule: {str(e)}"}

def format_schedule(schedule_json):
    formatted_output = "Comprehensive Academic Schedule:\n"
    for item in schedule_json['schedule']:
        formatted_output += f"{item['date']}: {item['name']} ({item['class']} - {item['type']})"
        if item['weight']:
            formatted_output += f" [{item['weight']}]"
        formatted_output += "\n"
    
    formatted_output += "\nGrading Rubrics:\n"
    for class_name, rubric in schedule_json['grading_rubrics'].items():
        formatted_output += f"{class_name}\n"
        for component, weight in rubric.items():
            formatted_output += f"- {component}: {weight}\n"
        formatted_output += "\n"
    
    if schedule_json['conflicts']:
        formatted_output += "Conflicts:\n"
        for conflict in schedule_json['conflicts']:
            formatted_output += f"- {conflict['description']} (Classes involved: {', '.join(conflict['classes_involved'])})\n"
    
    if schedule_json['notes']:
        formatted_output += "\nNotes and Concerns:\n"
        for note in schedule_json['notes']:
            formatted_output += f"- {note}\n"
    
    return formatted_output

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('syllabi')
        
        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        extracted_info_list = []
        for file in files:
            if file.filename == '':
                continue
            if file and file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    file.save(temp_file.name)
                    temp_file.flush()

                    with open(temp_file.name, 'rb') as f:
                        content = f.read()
                    
                    syllabus_content = content.decode('utf-8', errors='ignore')
                    
                    extracted = extract_info_from_syllabus(syllabus_content, file.filename)
                    verified = verify_extracted_info(syllabus_content, extracted, file.filename)
                    
                    extracted_info_list.append(verified)
                
                os.unlink(temp_file.name)

        final_schedule_json = generate_final_schedule(extracted_info_list)
        
        # Format the JSON into a readable string
        formatted_schedule = format_schedule(final_schedule_json)

        return jsonify({"schedule": formatted_schedule, "raw_data": final_schedule_json})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)