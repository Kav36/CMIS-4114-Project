const express = require("express");
const multer = require("multer");
const axios = require("axios");
const bodyParser = require("body-parser");
const path = require("path");
const fs = require("fs");
const FormData = require("form-data");
const mysql = require("mysql2/promise");

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(bodyParser.json());
app.use(express.static("public"));

// MySQL Database Connection
const db = mysql.createPool({
  host: "localhost",     // Your MySQL host (e.g., localhost)
  user: "root",          // Your MySQL username
  password: "",          // Your MySQL password
  database: "ai",        // Your MySQL database name
});

// Create the QA table if it doesn't exist
async function initializeDatabase() {
  const createTableQuery = `
    CREATE TABLE IF NOT EXISTS qa (
      id INT AUTO_INCREMENT PRIMARY KEY,
      question TEXT NOT NULL,
      answer TEXT NOT NULL
    )
  `;
  await db.query(createTableQuery);
}
initializeDatabase(); // Ensure table exists when server starts

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
    // Check if the question already exists in the database
    const selectQuery = `SELECT answer FROM qa WHERE question = ?`;
    const [rows] = await db.query(selectQuery, [question.toLowerCase()]); // Convert question to lowercase for lookup

    if (rows.length > 0) {
      // If the question exists, return the answer from the database
      console.log("Answer retrieved from the database:", rows[0].answer);
      return res.json({ answer: rows[0].answer, source: "database" });
    }

    // If the question does not exist, call the external API
    const response = await axios.post("http://localhost:5000/ask", {
      context,
      question,
    });

    const { answer } = response.data;

    // Save the new question and answer to the database (convert both to lowercase)
    const insertQuery = `INSERT INTO qa (question, answer) VALUES (?, ?)`;
    await db.query(insertQuery, [question.toLowerCase(), answer.toLowerCase()]);

    console.log("Answer retrieved from the API and saved to the database:", response.data);

    // Return the answer from the API
    res.json({ ...response.data, source: "API" });
  } catch (error) {
    console.error("Error processing the /ask route:", error);
    res.status(500).json({ error: "Failed to process your request" });
  }
});

// Serve HTML
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(2000, () => {
  console.log("Server running on http://localhost:2000");
});
