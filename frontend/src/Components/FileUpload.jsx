import React, { useState, useRef } from "react";
import axios from "axios";
import { Button } from "./ui/button";

export default function FileUpload({ onResult }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleBrowse = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click(); // programmatically open file picker
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select an image first!");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const res = await axios.post("http://localhost:8000/detect", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("Server response:", res.data);
      onResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Upload failed ❌");
    } finally {
      setLoading(false);
    }
  };

  

  return (
    <div className="w-full max-w-xl mx-auto bg-black/60 rounded-lg shadow p-4">
      <div className="flex items-center justify-between">
        {/* Hidden file input */}
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          className="hidden"
          onChange={(e) => setFile(e.target.files[0])}
        />

        {/* Browse button */}
        <Button
          type="button"
          onClick={handleBrowse}
          className="cursor-pointer text-white px-4 py-2 rounded-lg shadow transition"
        >
          Browse
        </Button>

        {/* File name display */}
        <span className="text-sm text-gray-300 ml-4 flex-1 truncate">
          {file ? file.name : "No file selected"}
        </span>

        {/* Upload button */}
        <Button
          onClick={handleUpload}
          disabled={loading || !file}
          className="ml-4 text-white rounded-lg shadow px-4 py-2 transition"
        >
          {loading ? "Processing..." : "Upload & Detect"}
        </Button>
      </div>
    </div>
  );
}
