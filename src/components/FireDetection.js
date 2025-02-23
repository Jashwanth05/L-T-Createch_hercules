import React, { useState, useEffect } from "react";

const FireDetection = () => {
  const [fireImages, setFireImages] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:5000/predict/fire")
      .then((res) => res.json())
      .then((data) => {
        setFireImages(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching fire data:", error);
        setLoading(false);
      });
  }, []);

  return (
    <div className="p-5 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
      {loading ? (
        <>
          <p className="text-center text-lg font-semibold text-red-600 col-span-full">
            🔥 Loading fire images...
          </p>
          {Array(8).fill(0).map((_, index) => (
            <div key={index} className="w-full h-40 bg-gray-300 animate-pulse rounded-lg" />
          ))}
        </>
      ) : fireImages.length === 0 ? (
        <p className="text-gray-500 text-center col-span-full">No fire detected.</p>
      ) : (
        fireImages.map((image, index) => (
          <div key={index} className="rounded-lg overflow-hidden shadow-lg">
            <img src={`http://localhost:5000${image.image_url}`} alt="Fire Detected" className="w-full h-40 object-cover" />
          </div>
        ))
      )}
    </div>
  );
};

export default FireDetection;
