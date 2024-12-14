const express = require("express");
const multer = require("multer");
const axios = require("axios");
const bodyParser = require("body-parser");
const path = require("path");
const fs = require("fs");
const FormData = require("form-data");

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(bodyParser.json());
app.use(express.static("public"));

// Upload PDF or Image and extract context
app.post("/upload", upload.single("file"), async (req, res) => {
  const file = req.file;

  try {
    const formData = new FormData();
    formData.append("file", fs.createReadStream(file.path)); // Read the file with fs.createReadStream

    // Pass formData headers explicitly
    const response = await axios.post("http://localhost:5000/upload", formData, {
      headers: formData.getHeaders(), // Correctly attach headers from form-data
    });

    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to upload and process file" });
  } finally {
    // Optional: Delete the uploaded file after processing
    fs.unlink(file.path, (err) => {
      if (err) console.error("Failed to delete file:", err);
    });
  }
});

// Ask a question
app.post("/ask", async (req, res) => {
  const { context, question } = req.body;

  try {
    const response = await axios.post("http://localhost:5000/ask", {
      context,
      question,
    });

    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to get answer" });
  }
});

// Serve HTML
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});
