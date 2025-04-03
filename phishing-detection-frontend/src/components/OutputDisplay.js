import React from "react";
import { useLocation, useNavigate } from "react-router-dom";

const OutputDisplay = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const result = location.state?.result || "No result available"; // ✅ Get result from navigation state

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h2>Detection Result</h2>
      <p><strong>Prediction:</strong> {result}</p>

      <button onClick={() => navigate("/")}>Check Another URL</button>
    </div>
  );
};

export default OutputDisplay;
