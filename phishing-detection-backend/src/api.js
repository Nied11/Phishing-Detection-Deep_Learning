const API_BASE_URL = "http://127.0.0.1:5000";  // Ensure Flask is running

export async function checkPhishing(url) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ url: url }),
            mode: "cors",  // Add this
        });

        if (!response.ok) {
            throw new Error("Failed to fetch prediction");
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error:", error);
        return { error: "Unable to connect to the API" };
    }
}


