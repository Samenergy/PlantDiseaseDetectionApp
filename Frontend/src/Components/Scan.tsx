"use client";
import { useState } from "react";
import { FaTimes, FaUpload, FaSpinner } from "react-icons/fa";

interface ScanModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const ScanModal: React.FC<ScanModalProps> = ({ isOpen, onClose }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<string | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setScanResult(null); // Reset scan result on new upload
    }
  };

  const handleScan = async () => {
    if (!selectedFile) return;
  
    setIsScanning(true);
    setScanResult(null); // Clear previous results
  
    const formData = new FormData();
    formData.append("file", selectedFile);
  
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error("Failed to scan image");
      }
  
      const result = await response.json();
  
      // Ensure response structure is handled correctly
      if (result && result.prediction) {
        setScanResult(`Predicted Disease: ${result.prediction.replace(/_/g, " ")}`);
      } else {
        setScanResult("Unexpected response format from server.");
      }
    } catch (error) {
      console.error("Error scanning image:", error);
      setScanResult("Error scanning image. Please try again.");
    } finally {
      setIsScanning(false);
    }
  };
  

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 backdrop-blur-sm flex items-center justify-center z-50 animate-fadeIn">
      <div className="bg-white rounded-2xl p-8 max-w-2xl w-full mx-4 relative shadow-2xl transform transition-all duration-300 ease-in-out animate-slideUp">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition-colors duration-200"
        >
          <FaTimes size={24} />
        </button>

        <h2 className="text-3xl font-bold mb-6 text-center bg-gradient-to-r from-[#145b2f] to-[#2e7d32] bg-clip-text text-transparent">
          Scan Plant Leaf
        </h2>

        <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center mb-6 hover:border-[#145b2f] hover:bg-gray-50">
          {previewUrl ? (
            <div className="relative group">
              <img src={previewUrl} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-lg" />
              <button
                onClick={() => {
                  setSelectedFile(null);
                  setPreviewUrl(null);
                  setScanResult(null);
                }}
                className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full shadow-lg hover:bg-red-600"
              >
                <FaTimes size={16} />
              </button>
            </div>
          ) : (
            <div className="space-y-6">
              <FaUpload size={48} className="mx-auto text-gray-400 animate-bounce" />
              <p className="text-gray-600 text-lg">Drag and drop your leaf image here, or click to select</p>
              <input type="file" accept="image/*" onChange={handleFileSelect} className="hidden" id="file-upload" />
              <label htmlFor="file-upload" className="inline-block bg-[#145b2f] text-white px-8 py-3 rounded-full cursor-pointer hover:bg-[#0f4423]">
                Select Image
              </label>
            </div>
          )}
        </div>

        {isScanning && (
          <div className="text-center space-y-4 animate-fadeIn">
            <FaSpinner className="animate-spin mx-auto text-[#145b2f]" size={32} />
            <p className="text-gray-600 font-medium">Analyzing leaf image...</p>
          </div>
        )}

        {scanResult && (
          <div className="mt-6 text-center p-4 bg-gray-100 rounded-lg border border-gray-300">
            <h3 className="text-xl font-semibold text-gray-700">Prediction:</h3>
            <p className="text-lg text-[#145b2f] font-medium">{scanResult}</p>
          </div>
        )}

        <div className="flex justify-center space-x-6 mt-6">
          <button
            onClick={onClose}
            className="px-8 text-[#145b2f] py-3 border-2 border-gray-300 rounded-full hover:bg-gray-100"
          >
            Cancel
          </button>
          <button
            onClick={handleScan}
            disabled={!selectedFile || isScanning}
            className={`px-8 py-3 rounded-full text-white font-medium shadow-lg ${
              !selectedFile || isScanning ? "bg-gray-400 cursor-not-allowed" : "bg-[#145b2f] hover:bg-[#0f4423]"
            }`}
          >
            Start Scan
          </button>
        </div>
      </div>
    </div>
  );
};

export default ScanModal;
