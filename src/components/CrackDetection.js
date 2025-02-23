import React, { useState, useEffect } from "react";

const CrackDetection = () => {
    const [crackImages, setCrackImages] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchCrackImages();
    }, []);

    const fetchCrackImages = async () => {
        setLoading(true);
        try {
            const response = await fetch("http://127.0.0.1:5000/predict/crack");
            const data = await response.json();
            setCrackImages(data);
        } catch (error) {
            console.error("Error fetching cracked images:", error);
        }
        setLoading(false);
    };

    return (
        <div style={{ textAlign: "center", padding: "20px" }}>
            <h2>Crack Detection Results</h2>
            {loading && <p>Loading detected cracks...</p>}
            
            {crackImages.length > 0 ? (
                <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center" }}>
                    {crackImages.map((img, index) => (
                        <div key={index} style={{ margin: "10px", textAlign: "center" }}>
                            <img src={img.image_url} alt={`Crack ${index}`} width="300px" />
                            <p>{img.filename}</p>
                        </div>
                    ))}
                </div>
            ) : (
                !loading && <p>No cracks detected ✅</p>
            )}
        </div>
    );
};

export default CrackDetection;