import streamlit as st
import os
import base64
import matplotlib.pyplot as plt
from app.question_extractor import extract_questions_docx, extract_questions_pdf
from app.bloom_classifier import classify_bloom
from app.marks_allocator import get_marks
from app.co_mapper import map_to_co
from app.report_generator import generate_report

# ---- Page Config ----
st.set_page_config(page_title="Question Paper Scrutiny", layout="centered")

# ---- Add Background Image from System ----
image_path = r"C:\Users\Soham\Pictures\Cute_dog.jpg"
if os.path.exists(image_path):
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
            <style>
                html, body, [class*="css"] {{
                    background-image: url("data:image/jpeg;base64,{encoded_image}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
            </style>
        """, unsafe_allow_html=True)

# ---- Title ----
st.markdown("""
    <h1 style='text-align: center; color: #003366;'>📋 Question Paper Scrutiny Tool</h1>
""", unsafe_allow_html=True)

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload Question Paper (DOCX or PDF)", type=["pdf", "docx"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded successfully.")

    # ---- Extract Questions ----
    try:
        if file_path.endswith(".docx"):
            questions = extract_questions_docx(file_path)
        elif file_path.endswith(".pdf"):
            questions = extract_questions_pdf(file_path)
        else:
            st.error("❌ Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to extract questions: {str(e)}")
        st.stop()

    if not questions:
        st.warning("⚠️ No valid questions found in the document.")
        st.stop()

    st.markdown("<h2 style='color:#003366;'>📑 Scrutiny Results</h2>", unsafe_allow_html=True)

    results = []
    bloom_counts = {}

    with st.spinner("⏳ Classifying questions and allocating marks..."):
        for q in questions:
            bloom, confidence, bloom_explanation = classify_bloom(q, return_explanation=True)
            marks = get_marks(bloom)
            co, co_explanation = map_to_co(q, return_explanation=True)

            # Tally for Bloom Distribution
            bloom_counts[bloom] = bloom_counts.get(bloom, 0) + 1

            results.append({
                "question": q,
                "bloom": bloom,
                "confidence": confidence,
                "marks": marks,
                "co": co,
                "bloom_explanation": bloom_explanation,
                "co_explanation": co_explanation
            })

    # ---- Tabs ----
    tab1, tab2 = st.tabs(["📊 Summary", "📄 Detailed Questions"])

    with tab1:
        st.markdown("<h3 style='color:#003366;'>📊 Bloom Level Distribution</h3>", unsafe_allow_html=True)

        # Multi-color bar plot
        fig, ax = plt.subplots()
        bloom_labels = list(bloom_counts.keys())
        counts = list(bloom_counts.values())
        colors = plt.cm.Set3.colors[:len(bloom_labels)]  # Distinct colors from colormap
        ax.bar(bloom_labels, counts, color=colors)
        ax.set_xlabel("Bloom Level")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Questions by Bloom Level")
        st.pyplot(fig)

    with tab2:
        for i, res in enumerate(results, start=1):
            st.markdown(f"### Question {i}")
            st.markdown(f"> {res['question']}")

            # Progress bar for confidence
            st.progress(res["confidence"])

            # Compact column display
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**🧠 Bloom Level**: `{res['bloom']}`")
            col2.markdown(f"**📝 Marks**: `{res['marks']}`")
            col3.markdown(f"**📘 Mapped CO**: `{res['co']}`")

            with st.expander("🔍 Explain Bloom Classification"):
                st.write(res["bloom_explanation"])

            with st.expander("📚 Explain CO Mapping"):
                st.write(res["co_explanation"])

            st.markdown("---")

    # ---- Generate PDF Report ----
    if st.button("📥 Generate PDF Report"):
        try:
            pdf_bytes = generate_report(results)
            st.download_button(
                "⬇️ Download Scrutiny Report",
                pdf_bytes,
                file_name="scrutiny_report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"❌ Report generation failed: {str(e)}")
