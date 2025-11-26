// src/api/client.js
import axios from "axios";

const apiClient = axios.create({
  baseURL: "http://127.0.0.1:8000", // FastAPI backend
  timeout: 15000,
});

export default apiClient;
