from flask import Flask, request, jsonify
from transformers import BertForQuestionAnswering, BertTokenizerFast, pipeline
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import os

# Load the fine-tuned model
model_path = "./final_model"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

# Create a pipeline for question answering
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Initialize Flask app
app = Flask(__name__)

# Path to Tesseract (if Tesseract is not installed in the default location)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Extract text from images (OCR)
def extract_text_from_images(pdf_path):
    text = ""
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)

    # Run OCR on each image (page)
    for image in images:
        text += pytesseract.image_to_string(image)

    return text

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save and extract text
    file_path = os.path.join("./uploads", file.filename)
    file.save(file_path)

    # Extract text from PDF
    context = extract_text_from_pdf(file_path)

    # Extract text from images (OCR)
    ocr_text = extract_text_from_images(file_path)

    # Combine regular text and OCR text
    full_context = context + "\n" + ocr_text

    return jsonify({"context": full_context}), 200

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    if "context" not in data or "question" not in data:
        return jsonify({"error": "Missing context or question"}), 400

    context = data["context"]
    question = data["question"]

    # Get the answer
    result = qa_pipeline(question=question, context=context)
    return jsonify({
        "answer": result["answer"],
        "score": result["score"]
    }), 200

if __name__ == "__main__":
    if not os.path.exists("./uploads"):
        os.makedirs("./uploads")
    app.run(host="0.0.0.0", port=5000)
