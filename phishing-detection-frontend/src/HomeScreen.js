import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { checkPhishing } from "./api"; // Import API function

const HomeScreen = () => {
    const navigate = useNavigate();
    const [url, setUrl] = useState("");
    const [prediction, setPrediction] = useState("");

    const handleCheck = async () => {
        if (!url.trim()) {
            alert("Please enter a valid URL.");
            return;
        }
        const result = await checkPhishing(url);
        if (result.error) {
            setPrediction("Error: Unable to connect to the backend");
        } else {
            setPrediction(`Prediction: ${result.prediction}`);
        }
    };

    const handleStoreOutput = () => {
        if (prediction) {
            localStorage.setItem("phishingDetectionResult", prediction);
            alert("Output stored successfully!");
        } else {
            alert("No prediction available to store.");
        }
    };

    return (
        <div style={{ textAlign: "center", padding: "20px", backgroundColor: "#eef5d7" }}>
            <h1>DEPARTMENT OF INFORMATION TECHNOLOGY</h1>
            <h2>NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA, SURATHKAL-575025</h2>

            <h2>Information Assurance and Security (IT352) Course Project</h2>
            <h3>Title: “DEPHIDES: Deep Learning Based Phishing Detection System”</h3>

            <h3>
                Carried out by: <br />
                <b>Nidhi Kumari (221IT047)</b> <br />
                <b>Sneha Singh (221IT063)</b> <br />
                During Academic Session: January – April 2025
            </h3>

            <div style={{ marginTop: "20px" }}>
                <input
                    type="text"
                    placeholder="Enter URL for detection"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    style={{
                        padding: "10px",
                        width: "300px",
                        marginBottom: "10px",
                        borderRadius: "5px",
                        border: "1px solid #ccc",
                    }}
                />
                <button onClick={handleCheck} style={{ marginLeft: "10px", padding: "10px" }}>
                    Check URL
                </button>
            </div>

            {prediction && <h3 style={{ color: "blue", marginTop: "10px" }}>{prediction}</h3>}

            <div style={{ marginTop: "20px" }}>
                <button className="button" onClick={() => navigate("/input")} style={{ marginRight: "10px" }}>
                    Enter Input
                </button>
                <button 
                    className="button" 
                    onClick={() => navigate("/output", { state: { result: prediction } })}
                    style={{ marginRight: "10px" }}>
                    Display Output
                </button>
                <button className="button" onClick={handleStoreOutput}>
                    Store Output
                </button>
            </div>
        </div>
    );
};

export default HomeScreen;
