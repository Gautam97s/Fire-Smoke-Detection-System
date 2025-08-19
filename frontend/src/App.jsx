import React, { useState } from "react";
import FileUpload from "./components/FileUpload";
import { BackgroundLines } from "./Components/ui/background-lines";
import ResultDisplay from "./components/DetectionResult";

export default function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="relative min-h-screen text-white flex items-center justify-center">
      {/* Background */}
      <BackgroundLines className="absolute inset-0" />

      {/* Foreground content */}
      <div className="relative z-10 text-center">
        <h1 className="inline-block text-6xl font-extrabold mb-8 bg-gradient-to-b from-gray-500 via-gray-400 to-white bg-clip-text text-transparent">
          Fire & Smoke Detection
        </h1>        
        <FileUpload onResult={setResult} />
        <ResultDisplay result={result} />
      </div>
    </div>
  );
}
