from flask import Flask, render_template, request, jsonify
from student_routine import generate_mind_map, generate_learning_plan  # Explicitly import generate_learning_plan
from teacher_syllabus import generate_syllabus
from teacher_marks_report import generate_marks_report
from student_handwriting import load_user_image, generate_dyslexia_report
import os
import numpy as np
import tensorflow as tf
import ollama
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model for dyslexia prediction
MODEL_PATH = "dyslexia_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/student_routine', methods=['GET', 'POST'])
def student_routine():
    if request.method == 'POST':
        file = request.files['file']
        days_remaining = request.form['days_remaining']
        subject_focus = request.form['subject_focus']
        
        # Check which form was submitted
        form_type = request.form.get('form_type', 'mind_map')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            try:
                if form_type == 'learning_plan':
                    # Additional parameters for learning plan
                    learning_style = request.form.get('learning_style', '')
                    challenges = request.form.get('challenges', '')
                    goals = request.form.get('goals', '')
                    
                    result = generate_learning_plan(filepath, days_remaining, subject_focus, 
                                                  learning_style, challenges, goals)
                    return render_template('student_routine.html', learning_plan=result, file_path=filepath)
                else:
                    # Regular mind map generation
                    result = generate_mind_map(filepath, days_remaining, subject_focus)
                    return render_template('student_routine.html', result=result, file_path=filepath)
            except Exception as e:
                return render_template('student_routine.html', error=str(e))
    return render_template('student_routine.html')

@app.route('/teacher_syllabus', methods=['GET', 'POST'])
def teacher_syllabus():
    if request.method == 'POST':
        age = request.form['age']
        difficulty = request.form['difficulty']
        study_hours = request.form['study_hours']
        additional_notes = request.form.get('additional_notes', '')
        try:
            syllabus_pdf = generate_syllabus(age, difficulty, study_hours, additional_notes)
            with open(syllabus_pdf, 'rb') as f:  # Open in binary mode to avoid decoding issues
                syllabus_content = f.read().decode('latin1')  # Decode using 'latin1' to handle non-UTF-8 characters
            return render_template('teacher_syllabus.html', syllabus_content=syllabus_content, syllabus_pdf=syllabus_pdf)
        except Exception as e:
            return render_template('teacher_syllabus.html', error=str(e))
    return render_template('teacher_syllabus.html')

@app.route('/teacher_marks_report', methods=['GET', 'POST'])
def teacher_marks_report():
    if request.method == 'POST':
        files = request.files.getlist('files')
        age = request.form['age']
        previous_test_marks = request.form['previous_test_marks']
        target_marks = request.form['target_marks']
        additional_notes = request.form.get('additional_notes', '')
        learning_style = request.form.get('learning_style', '')
        preferred_subjects = request.form.get('preferred_subjects', '')
        challenges = request.form.get('challenges', '')
        goals = request.form.get('goals', '')

        filepaths = [os.path.join(app.config['UPLOAD_FOLDER'], f.filename) for f in files if f]
        for f, path in zip(files, filepaths):
            f.save(path)

        try:
            report, pdf_path = generate_marks_report(filepaths, age, previous_test_marks, target_marks, additional_notes, learning_style, preferred_subjects, challenges, goals)
            if "Error" in report:
                return render_template('teacher_marks_report.html', error=report)
            return render_template('teacher_marks_report.html', report=report, pdf_path=pdf_path)
        except Exception as e:
            return render_template('teacher_marks_report.html', error=str(e))
    return render_template('teacher_marks_report.html')

@app.route('/student_dyslexia', methods=['GET', 'POST'])
def student_dyslexia():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('student_dyslexia.html', error="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('student_dyslexia.html', error="No selected file.")
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        img_array = load_user_image(file_path)
        if img_array is None:
            return render_template('student_dyslexia.html', error="Invalid image.")
        
        prediction = model.predict(img_array)
        dyslexia_confidence = int(np.max(prediction) * 100)  # Get confidence percentage
        
        # Get additional parameters from the form
        age = request.form.get('age', 'Unknown')
        learning_style = request.form.get('learning_style', 'Unknown')
        challenges = request.form.get('challenges', 'Unknown')
        goals = request.form.get('goals', 'Unknown')
        
        # Generate a report using gemma:2b
        report = generate_dyslexia_report(dyslexia_confidence, age, learning_style, challenges, goals)
        
        return render_template("student_dyslexia.html", 
                               confidence=dyslexia_confidence, 
                               report=report)
    return render_template('student_dyslexia.html')

# Add a new route for generating quiz questions with Gemma
@app.route('/generate_quiz', methods=['GET'])
def generate_quiz():
    try:
        # Generate quiz questions using gemma:2b
        response = ollama.chat(model='gemma:2b', messages=[
            {"role": "system", "content": "Generate educational quiz questions about learning disabilities, particularly dyslexia."},
            {"role": "user", "content": "Generate 6 quiz questions about dyslexia and learning disabilities with 4 options each and indicate the correct answer. Format each question with options labeled A, B, C, D and mark the correct answer."}
        ])
        
        # Parse the response to extract questions
        raw_content = response['message']['content']
        
        # Simple parsing logic (adjust based on actual format)
        questions = []
        current_question = {}
        
        for line in raw_content.split('\n'):
            if line.strip() == '':
                continue
                
            if line.strip().startswith(('Question', 'Q', '1.', '2.', '3.', '4.', '5.', '6.')):
                if current_question and 'question' in current_question and 'options' in current_question:
                    questions.append(current_question)
                current_question = {'question': line.split(':', 1)[-1].strip() if ':' in line else line.strip(), 'options': []}
            
            elif line.strip().startswith(('A.', 'B.', 'C.', 'D.', 'A)', 'B)', 'C)', 'D)')):
                if 'options' in current_question:
                    current_question['options'].append(line.split('.', 1)[-1].strip() if '.' in line else line.split(')', 1)[-1].strip())
            
            elif line.strip().startswith(('Answer:', 'Correct Answer:')):
                answer_letter = line.split(':')[-1].strip()
                current_question['correctAnswer'] = 'ABCD'.index(answer_letter[0]) if answer_letter else 0
        
        if current_question and 'question' in current_question and 'options' in current_question:
            questions.append(current_question)
            
        # Ensure we have at least some questions
        if not questions:
            # Fallback to default questions
            questions = [
                {
                    "question": "What is dyslexia primarily associated with?",
                    "options": [
                        "Difficulty with mathematics",
                        "Difficulty with reading and writing",
                        "Difficulty with social interactions",
                        "Difficulty with physical coordination"
                    ],
                    "correctAnswer": 1
                },
                # ...add more fallback questions...
            ]
        
        return jsonify({"success": True, "questions": questions})
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
