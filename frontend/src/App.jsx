import React, { useState } from "react";
import FileUpload from "./components/FileUpload";
import DetectionResult from "./components/DetectionResult";

export default function App() {
  const [result, setResult] = useState(null);

  return (
    <div>
      <h1 style={{ textAlign: "center" }}>🔥 Fire & Smoke Detection</h1>
      <FileUpload onResult={setResult} />
      <DetectionResult result={result} />
    </div>
  );
}
