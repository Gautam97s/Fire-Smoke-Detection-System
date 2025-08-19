import React from "react";
import { Card, CardContent } from "./ui/card";

export default function ResultDisplay({ result }) {
  if (!result) return null;

  return (
    <Card className="w-full max-w-4xl mx-auto mt-8 bg-transparent border border-transparent shadow-xl rounded-2xl backdrop-blur">
      <CardContent className="p-6 text-white">
        <h2 className="text-2xl font-bold text-center mb-6 tracking-wide">
          Detection Results
        </h2>

        <div className="flex flex-col lg:flex-row gap-6">
          {result.result_image && (
            <div className="flex-1 flex justify-center">
              <img
                src={result.result_image}
                alt="Detection Result"
                className="rounded-xl border border-gray-600 shadow-lg max-h-[420px] object-contain bg-black"
                onError={(e) => {
                  console.error("Image failed to load:", e.target.src);
                  e.target.style.display = 'none';
                }}
              />
            </div>
          )}

          <div className="flex-1 bg-gradient-to-br from-gray-900/70 to-gray-800/50 rounded-xl p-4 border border-gray-700 shadow-inner">
            <h3 className="text-lg font-semibold mb-3">Objects Detected</h3>
            {result.detections && result.detections.length > 0 ? (
              <ul className="space-y-3">
                {result.detections.map((det, idx) => (
                  <li
                    key={idx}
                    className="flex justify-between items-center bg-gray-800/70 px-4 py-2 rounded-lg shadow-md hover:bg-gray-700/70 transition"
                  >
                    <span className="font-medium text-red-400">{det.class}</span>
                    <span className="text-gray-300">
                      {Math.round(det.confidence * 100)}%
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-400 italic">No objects detected.</p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}