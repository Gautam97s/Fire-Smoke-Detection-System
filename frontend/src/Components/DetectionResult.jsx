import React from "react";

export default function DetectionResult({ result }) {
  if (!result) return null;

  return (
    <div style={{ textAlign: "center", marginTop: "30px" }}>
      <h2>Detections</h2>

      {result.detections.length === 0 ? (
        <p>No fire or smoke detected ✅</p>
      ) : (
        <ul style={{ listStyle: "none", padding: 0 }}>
          {result.detections.map((d, i) => (
            <li key={i}>
              <strong>{d.class}</strong> - {(d.confidence * 100).toFixed(2)}%
            </li>
          ))}
        </ul>
      )}

      {result.result_image && (
        <div style={{ marginTop: "20px" }}>
          <h3>Processed Image</h3>
          <img
            src={`http://localhost:8000${result.result_image}`}
            alt="Detection result"
            style={{
              maxWidth: "500px",
              border: "2px solid #444",
              borderRadius: "6px",
            }}
          />
        </div>
      )}
    </div>
  );
}
