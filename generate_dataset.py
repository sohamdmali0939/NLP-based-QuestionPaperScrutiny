import random
import csv
import os
from sklearn.model_selection import train_test_split

bloom_actions = {
    1: [  # Remember (Recall)
        "define", "list", "recall", "identify", "state", "recognize", "name", "explain basic concepts"
    ],
    2: [  # Understand (Explain)
        "explain", "describe", "summarize", "clarify", "illustrate", "compare", "distinguish", 
        "demonstrate understanding of", "outline", "identify"
    ],
    3: [  # Apply (Solve, Use)
        "apply", "solve", "demonstrate", "compute", "use", "implement", "design", "develop", 
        "construct", "simulate", "calculate"
    ],
    4: [  # Analyze (Examine, Differentiate)
        "analyze", "compare", "categorize", "distinguish", "evaluate", "decompose", "examine", 
        "organize", "break down", "identify patterns", "study"
    ],
    5: [  # Evaluate (Assess, Justify)
        "evaluate", "assess", "justify", "criticize", "validate", "debate", "defend", "recommend", 
        "estimate", "rate", "argue", "debate", "defend the design", "justify the choice"
    ],
    6: [  # Create (Design, Formulate)
        "design", "create", "formulate", "construct", "build", "develop", "design a solution", 
        "create an algorithm", "develop a framework", "generate a model", "build a system"
    ]
}

# Topics for generating questions 
topics = [
    "C programming for problem solving", "Problem solving with data structures", "Discrete mathematical structures",
    "Computer organization and architecture", "DBMS", "Data Structures and Algorithm", 
    "Operating System principles and programming", "Principles of Compiler Design", "Exploratory Data Analysis",
    "Object-Oriented Programming (OOP)", "Software Engineering", "Computer Networks", "Machine Learning", 
    "Internet of Things (IoT)", "Blockchain", "Operating Systems", "Network Security", "Database Management Systems",
    "File Systems", "Python programming", "SQL queries", "TCP/IP protocol", "the OSI model", "Sorting algorithms",
    "Binary search", "Memory management", "Encryption techniques", "Artificial Intelligence", "Cloud Computing",
    "Big Data", "Data Mining", "Deep Learning", "Natural Language Processing (NLP)", "Computer Graphics", 
    "Digital Logic Design", "Embedded Systems", "Software Testing", "Agile methodologies", "Software Architecture",
    "Distributed Systems", "Parallel Computing", "Parallel Algorithms", "Automata Theory", "Theory of Computation",
    "Compiler Design", "Formal Languages", "Linear Algebra", "Multivariable Calculus", "Mathematical Logic",
    "Advanced Algorithms", "Optimization Techniques", "Web Development", "Mobile App Development", "Cyber Security",
    "Cryptography", "Cloud Security", "DevOps", "Blockchain Technology", "Augmented Reality", "Virtual Reality",
    "Data Structures in Python", "Graph Theory", "Advanced Database Systems", "Embedded C programming", 
    "Real-time Systems", "Digital Signal Processing", "Computer Vision", "Pattern Recognition", "Bioinformatics",
    "Human-Computer Interaction", "Computational Biology", "Cloud Infrastructure", "Database Design", "Linux programming",
    "Object-Oriented Design", "Networking Protocols", "Parallel Programming", "Distributed Databases"
]

def generate_question(verb, topic, level):
    if level == 1:  # Level 1: Remember
        return f"Define {random.choice(topics)}."
    elif level == 2:  # Level 2: Understand
        return f"Explain {random.choice(topics)} and describe its applications."
    elif level == 3:  # Level 3: Apply
        return f"Apply {random.choice(topics)} to solve a real-world problem."
    elif level == 4:  # Level 4: Analyze
        return f"Analyze the implementation of {random.choice(topics)} in a given scenario."
    elif level == 5:  # Level 5: Evaluate
        return f"Evaluate the effectiveness of {random.choice(topics)} in solving industry-specific problems."
    elif level == 6:  # Level 6: Create
        return f"Design a {random.choice(topics)} system to handle X problem in an engineering context."

data = []

for label, verbs in bloom_actions.items():
    for _ in range(80):  # Generate 80 questions for each Bloom level
        verb = random.choice(verbs)
        topic = random.choice(topics)
        question = generate_question(verb, topic, label)
        data.append((question, label))

# Shuffle data and split into 80% train and 20% test using train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create the "data" directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save to train.csv
with open("data/train.csv", "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["question", "bloom_label"])
    writer.writerows(train_data)

# Save to test.csv
with open("data/test.csv", "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["question", "bloom_label"])
    writer.writerows(test_data)

print("✅ Dataset split and saved: data/train.csv, data/test.csv")
