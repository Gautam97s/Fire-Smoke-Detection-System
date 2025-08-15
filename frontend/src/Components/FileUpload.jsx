import React, { useState } from "react";
import axios from "axios";

export default function FileUpload({ onResult }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const res = await axios.post("http://localhost:8000/detect", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      onResult(res.data); // pass response up to App.jsx
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Upload failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button
        onClick={handleUpload}
        disabled={loading}
        style={{
          marginLeft: "10px",
          padding: "6px 12px",
          cursor: loading ? "not-allowed" : "pointer",
        }}
      >
        {loading ? "Processing..." : "Upload"}
      </button>
    </div>
  );
}
