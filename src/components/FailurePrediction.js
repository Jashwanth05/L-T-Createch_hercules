import React, { useState, useEffect } from "react";
import axios from "axios";

const FailurePrediction = () => {
  const [prediction, setPrediction] = useState("Loading...");
  const [sensorData, setSensorData] = useState({});

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5000/predict/machine");
        console.log("API Response:", response.data); // Debugging
        setPrediction(response.data?.prediction?.trim() || "Unknown");
        setSensorData(response.data?.input_data || {});
      } catch (error) {
        console.error("❌ Error fetching prediction:", error);
        setPrediction("Error fetching data");
      }
    };

    fetchPrediction();
    const interval = setInterval(fetchPrediction, 5000); // Fetch every 5 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-8 bg-gray-100 shadow-lg rounded-xl text-center max-w-6xl mx-auto w-full -mt-8">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">IoT Failure Prediction</h2>
  
      {/* ✅ Dynamic Failure Status */}
      <p className={`text-xl font-semibold ${prediction !== "No Failure" ? "text-red-500" : "text-green-500"}`}>
        {prediction}
      </p>
  
      {/* ✅ Sensor Data Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-6 mt-6">
        {Object.entries(sensorData || {}).map(([key, value]) => (
          <div key={key} className="p-6 bg-white shadow-md rounded-xl text-center w-full">
            <h3 className="text-base font-semibold text-gray-800">{key}</h3>
            <p className="text-xl font-bold text-gray-600">
              {typeof value === "number" ? value.toFixed(2) : value}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
  
};

export default FailurePrediction;
