import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:5000";  // Ensure this matches the backend URL

export const checkPhishing = async (url) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/predict`, { url });
        return response.data;
    } catch (error) {
        console.error("Error fetching data:", error);
        return { error: "Failed to connect to backend" };
    }
};
