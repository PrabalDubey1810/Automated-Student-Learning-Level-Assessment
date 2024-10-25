import pandas as pd
import random

# Define the subjects and topics
subjects = ['physics', 'chemistry', 'math']
physics_topics = ['Newton\'s laws', 'kinematics', 'conservation laws']
chemistry_topics = ['stoichiometry', 'chemical bonding', 'kinetics']
math_topics = ['algebra', 'calculus', 'geometry']

# Generate random questions and topics for 100 exams
random.seed(42)  # for reproducibility
exams = []
for _ in range(100):
    subject = random.choice(subjects)
    if subject == 'physics':
        topic = random.choice(physics_topics)
    elif subject == 'chemistry':
        topic = random.choice(chemistry_topics)
    else:
        topic = random.choice(math_topics)
    question = f"What is {topic}?"
    exams.append({'question': question, 'subject': subject, 'topic': topic})

# Create a DataFrame from the generated data
df = pd.DataFrame(exams)

# Save the DataFrame to a CSV file
df.to_csv('random_dataset.csv', index=False)

print("Random dataset generated and saved to 'random_dataset.csv'")
