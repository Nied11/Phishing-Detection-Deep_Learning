function checkPhishing() {
    let url = document.getElementById("urlInput").value;

    fetch("http://127.0.0.1:8000/predict", {  // Ensure port 8000
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Prediction: " + data.prediction;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Failed to connect to backend.";
    });
}
