from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
import PyPDF2
import pytesseract
from transformers import pipeline
from pdf2image import convert_from_path
import os

# Load the fine-tuned t5
model_path = './fine_tuned_t5_large'
tokenizer = T5Tokenizer.from_pretrained(model_path)
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load the GPT-2 large model and tokenizer for fallback
gpt2_large_model_path = "./gpt2-large_model"
fallback_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_large_model_path)
fallback_model = GPT2LMHeadModel.from_pretrained(gpt2_large_model_path)

# Ensure the pad token is set for both models
tokenizer.pad_token = tokenizer.eos_token
fallback_tokenizer.pad_token = fallback_tokenizer.eos_token

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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

# Prepare input for GPT-2
def prepare_input(question, context):
    """
    Prepare the input string in the same format as used during training.
    """
    input_text = f"question: {question} context: {context} answer:"
    return input_text

# Split the context into chunks of tokens
def split_into_chunks_with_overlap(text, max_tokens, overlap):
    """
    Splits a text into chunks with a specified token overlap.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(tokens[i:i + max_tokens])
        i += max_tokens - overlap
        if i >= len(tokens):
            break
    return chunks

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
    full_context = (context if context else "") + "\n" + (ocr_text if ocr_text else "")

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

    if not context.strip():
        return jsonify({"error": "Context is empty."}), 400

    # Tokenize the context and split into chunks of 1000 tokens
    max_tokens = 1000
    overlap = 200  # Optional overlap between chunks
    chunks = split_into_chunks_with_overlap(context, max_tokens, overlap)

    all_answers = []  # List to store answers from all chunks
    confidence_scores = []  # List to store confidence scores

    for chunk in chunks:
        # Decode chunk back to text
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)

        # Prepare input
        input_text = prepare_input(question, chunk_text)
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).input_ids
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

        # Generate output for each chunk
        try:
            outputs = fine_tuned_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,  # Enable sampling
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True  # Enable output scores
            )

            # Decode and extract the answer
            output_ids = outputs.sequences[0]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            answer = output_text.split("answer:")[-1].strip()

            # Calculate the confidence score
            scores = outputs.scores  # List of logits for each generation step
            probabilities = []
            for step_logits, token_id in zip(scores, output_ids[1:]):  # Skip the first token (input prompt)
                step_probs = step_logits.softmax(dim=-1)  # Convert logits to probabilities
                token_prob = step_probs[0, token_id].item()  # Probability of the selected token
                probabilities.append(token_prob)

            # Compute the average confidence score
            average_score = sum(probabilities) / len(probabilities) if probabilities else 0.0

            # Append the answer and confidence score to their respective lists
            all_answers.append(answer)
            confidence_scores.append(average_score)

        except Exception as e:
            print(f"Error during generation: {e}")

    # Combine all answers
    combined_answer = " ".join(all_answers)  # Concatenate all answers
    print(f"Combined Answer: {combined_answer}")
    # Fallback to GPT-2 Large model if no valid answers or low confidence
    if not all_answers or max(confidence_scores, default=0) < 0.99:
        print("Low confidence detected, switching to GPT-2 Large model...")
        inputs = fallback_tokenizer.encode(question, return_tensors="pt")
        try:
            outputs = fallback_model.generate(inputs, max_length=150, num_return_sequences=1)
            fallback_answer = fallback_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Summarize fallback answer
            summary = summarizer(fallback_answer, max_length=200, min_length=25, do_sample=False)
            print(f"Summarized Answer: {summary[0]['summary_text']}")

            return jsonify({
                "answer": summary[0]['summary_text'],
                "question": question,
                "score": "fallback"
            }), 200
        except Exception as e:
            print(f"Error during fallback generation: {e}")
            return jsonify({"error": "Both models failed to generate a valid answer."}), 500

    # Summarize the combined answers
    summary = summarizer(combined_answer, max_length=100, min_length=25, do_sample=False)

    return jsonify({
        "answer": summary[0]['summary_text'],
        "question": question,
        "score": max(confidence_scores, default=0)
    }), 200

if __name__ == "__main__":
    if not os.path.exists("./uploads"):
        os.makedirs("./uploads")
    app.run(host="0.0.0.0", port=5000)
