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

# Determine device (GPU if available, otherwise CPU)
device = 0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1  # Use GPU if available, otherwise CPU

# Create a pipeline for question answering with device argument
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

# Initialize Flask app
app = Flask(__name__)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if not text:
            raise ValueError("No text extracted from PDF.")
    except Exception as e:
        text = None
        print(f"Error extracting text from PDF: {e}")
    return text

# Extract text from images (OCR)
def extract_text_from_images(pdf_path):
    text = ""
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)

        # Run OCR on each image (page)
        for image in images:
            text += pytesseract.image_to_string(image)
        if not text:
            raise ValueError("No text extracted from OCR.")
    except Exception as e:
        text = None
        print(f"Error extracting text from images: {e}")
    return text

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save the file to the uploads directory
    file_path = os.path.join("./uploads", file.filename)
    file.save(file_path)

    # Extract text from PDF and OCR
    context = extract_text_from_pdf(file_path)
    ocr_text = extract_text_from_images(file_path)

    # Fallback to OCR text if PDF extraction is empty
    full_context = context if context else "" + "\n" + (ocr_text if ocr_text else "")

    if not full_context.strip():
        return jsonify({"error": "Unable to extract any text from the PDF."}), 400

    return jsonify({"context": full_context}), 200

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    if "context" not in data or "question" not in data:
        return jsonify({"error": "Missing context or question"}), 400

    context = data["context"]
    question = data["question"]

    # Ensure that the context is not empty before processing
    if not context.strip():
        return jsonify({"error": "Context is empty."}), 400

    # Get the answer from the model
    result = qa_pipeline(question=question, context=context)

    return jsonify({
        "answer": result["answer"],
        "score": result["score"]
    }), 200

if __name__ == "__main__":
    if not os.path.exists("./uploads"):
        os.makedirs("./uploads")
    app.run(host="0.0.0.0", port=5000)
