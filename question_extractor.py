import re
from docx import Document
import fitz  # PyMuPDF

def extract_questions_docx(path):
    doc = Document(path)
    lines = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return _extract_subquestions(lines)

def extract_questions_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return _extract_subquestions(lines)

def _extract_subquestions(lines):
    # Matches 1a, 2b, 3c etc., at the beginning of a line or after "Q.No" structure
    pattern = re.compile(r'^(?:Q\.?\s*\d+\s*)?(\d+[a-cA-C])[\).]?\s*(.*)', re.IGNORECASE)

    # Expanded keywords indicating header or metadata lines to skip
    header_keywords = [
    "SCHOOL", "Course Code", "Course Title", "Duration", "Max. Marks",
    "Note", "Q.No.", "Questions", "Marks", "CO", "BL", "PO", "PI code", "Page",
    "USN", "Semester", "Date of Exam", "Bloom:", "Earlier known as", "PI",
    "Internal Assessment", "Mark s", "Programming",
    "Model Question Paper", "Minor Examination", "ISA-1",  
    "Q.No", "Q: Q.No", "75 Mins", "40", "Exam", "Assessment", "22ECAC302", " Mark s", "15" 
]


    questions = []
    current_q = ""
    current_id = ""

    for line in lines:
        # Skip headerlines
        if any(keyword.lower() in line.lower() for keyword in header_keywords):
            continue

        match = pattern.match(line)
        if match:
            if current_q:
                questions.append(current_q.strip())
            current_id = match.group(1)
            current_q = f"{current_id}. {match.group(2)}"
        else:
            current_q += " " + line

    if current_q:
        questions.append(current_q.strip())

    return questions
