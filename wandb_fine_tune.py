import wandb

wandb.login(key="e4c42d08e0619b99531d9876746df3d9ce5026b4")

# Fine-tuning student routine
run = wandb.init(project="dyslexia-app", name="fine-tune-student-routine", reinit=True)
artifact = wandb.Artifact("student_routine", type="dataset")
artifact.add_file("fine_tune/student_routine.json")
wandb.log_artifact(artifact)
wandb.finish()

# Fine-tuning teacher syllabus
run = wandb.init(project="dyslexia-app", name="fine-tune-teacher-syllabus", reinit=True)
artifact = wandb.Artifact("teacher_syllabus", type="dataset")
artifact.add_file("fine_tune/teacher_syllabus.json")
wandb.log_artifact(artifact)
wandb.finish()

# Fine-tuning teacher marks report
run = wandb.init(project="dyslexia-app", name="fine-tune-teacher-marks", reinit=True)
artifact = wandb.Artifact("teacher_marks_report", type="dataset")
artifact.add_file("fine_tune/teacher_marks_report.json")
wandb.log_artifact(artifact)
wandb.finish()
