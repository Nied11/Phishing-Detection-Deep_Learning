import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const InputForm = ({ setResult }) => {
  const [url, setUrl] = useState("");
  const navigate = useNavigate(); // Hook for navigation

  const handleSubmit = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", { url });
      setResult(response.data.prediction);

      // Redirect to OutputDisplay
      navigate("/output");
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  return (
    <div>
      <h2>Phishing Detection</h2>
      <input 
        type="text" 
        placeholder="Enter URL" 
        value={url} 
        onChange={(e) => setUrl(e.target.value)}
      />
      <button onClick={handleSubmit}>Check</button>
    </div>
  );
};

export default InputForm;
