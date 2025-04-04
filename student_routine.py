import fitz
import ollama
import wandb
import time
import os

wandb.login(key="e4c42d08e0619b99531d9876746df3d9ce5026b4")

def extract_text(file_path):
    try:
        doc = fitz.open(file_path)
        return " ".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"Error extracting text: {e}")
        return "Error extracting text from file."

def generate_mind_map(file_path, days_remaining, subject_focus):
    start_time = time.time()  # Start timing
    wandb.init(project="dyslexia-app", name="generate-mind-map", reinit=True)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return "Error: File does not exist."
    
    content = extract_text(file_path)
    if content.startswith("Error"):
        return content

    try:
        response = ollama.chat(model='gemma:2b', messages=[
            {"role": "system", "content": "Create a structured study mind map."},
            {"role": "user", "content": f"Text: {content}, Days: {days_remaining}, Focus: {subject_focus}"}
        ])
        
        result = response['message']['content']
        execution_time = time.time() - start_time  # Calculate execution time

        # Log metrics and results
        wandb.log({
            "input_file": file_path,
            "days_remaining": days_remaining,
            "subject_focus": subject_focus,
            "execution_time": execution_time,
            "mind_map_content": result
        })
        wandb.finish()
        return result
    except Exception as e:
        print(f"Error generating mind map: {e}")
        wandb.finish()
        return f"Error generating mind map: {str(e)}"

def generate_learning_plan(file_path, days_remaining, subject_focus, learning_style, challenges, goals):
    """Generate a personalized learning plan for dyslexic students."""
    start_time = time.time()  # Start timing
    wandb.init(project="dyslexia-app", name="generate-learning-plan", reinit=True)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return "Error: File does not exist."
    
    content = extract_text(file_path)
    if content.startswith("Error"):
        return content

    try:
        response = ollama.chat(model='gemma:2b', messages=[
            {"role": "system", "content": "Create a detailed learning plan for a dyslexic student that includes daily activities, reading strategies, writing exercises, and self-assessment tools. Use clear structure, simple language, and colorful visuals."},
            {"role": "user", "content": f"Curriculum Content: {content}, Days Available: {days_remaining}, Subject Focus: {subject_focus}, Learning Style: {learning_style}, Challenges: {challenges}, Goals: {goals}. Create a day-by-day dyslexia-friendly learning plan."}
        ])
        
        result = response['message']['content']
        execution_time = time.time() - start_time  # Calculate execution time

        # Log metrics and results
        wandb.log({
            "input_file": file_path,
            "days_remaining": days_remaining,
            "subject_focus": subject_focus,
            "learning_style": learning_style,
            "challenges": challenges,
            "goals": goals,
            "execution_time": execution_time,
            "learning_plan_content": result
        })
        wandb.finish()
        return result
    except Exception as e:
        print(f"Error generating learning plan: {e}")
        wandb.finish()
        return f"Error generating learning plan: {str(e)}"
