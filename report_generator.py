from fpdf import FPDF
import os

def clean_text(text):
    """Replace unsupported smart characters with ASCII equivalents."""
    replacements = {
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "—": "-", "–": "-"
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

def generate_report(questions_info, output_path="report.pdf"):
    pdf = FPDF()

    # Add a Unicode font (DejaVuSans)
    font_path = os.path.join("fonts", r"C:\Users\Soham\Documents\NLP&GEN_AI(Sem 6)\Course Project\scrutinynew\fonts\dejavu-fonts-ttf-2.37")
    pdf.add_font('DejaVu', '', font_path, uni=True)
    pdf.set_font('DejaVu', '', 12)

    pdf.add_page()

    # Title
    pdf.set_font('DejaVu', 'B', 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, txt="Question Paper Scrutiny Report", ln=True, align='C')
    pdf.ln(10)

    # Body
    for i, q in enumerate(questions_info, 1):
        question = clean_text(q['question'])
        bloom = q.get('bloom', 'N/A')
        marks = q.get('marks', 'N/A')
        co = q.get('co', 'N/A')

        # Question
        pdf.set_font('DejaVu', '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 10, f"{i}. {question}")

        # Metadata
        pdf.set_font('DejaVu', 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, f"   Bloom Level: {bloom} | Marks: {marks} | CO: {co}", ln=True)

        pdf.ln(5)

    # Output
    pdf.output(output_path)
