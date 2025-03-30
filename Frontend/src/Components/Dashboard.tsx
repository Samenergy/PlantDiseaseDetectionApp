import React, { useState, ChangeEvent, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { FaLeaf, FaSync, FaHistory, FaSignOutAlt, FaArrowLeft } from "react-icons/fa";
import Swal from "sweetalert2";
import { DISEASE_TREATMENTS } from "./diseaseTreatments";

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<string>("detect");
  const [leafImage, setLeafImage] = useState<File | null>(null);
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<
    { id: number; text: string; treatment: string; date: string }[]
  >([]);
  const [retrainHistory, setRetrainHistory] = useState<
    { id: number; text: string; date: string }[]
  >([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);
  const [visualizationImages, setVisualizationImages] = useState<{
    classification_report: string | null;
    confusion_matrix: string | null;
    loss_plot: string | null;
    accuracy_plot: string | null;
  }>({
    classification_report: null,
    confusion_matrix: null,
    loss_plot: null,
    accuracy_plot: null,
  });

  // Refs for file inputs
  const leafInputRef = useRef<HTMLInputElement>(null);
  const zipInputRef = useRef<HTMLInputElement>(null);

  // Base URL for API (configurable via environment variable for cloud deployment)
  const API_BASE_URL ="http://127.0.0.1:8000";

  // Get token from localStorage
  const getToken = () => localStorage.getItem("token");

  // Toggle sidebar collapse
  const toggleSidebar = () => setIsSidebarCollapsed((prev) => !prev);

  // Trigger file input click with debugging
  const triggerFileInput = (ref: React.RefObject<HTMLInputElement>) => {
    console.log("Triggering file input:", ref.current);
    if (ref.current) {
      ref.current.click();
    } else {
      console.error("File input ref is null or undefined");
    }
  };

  // Handle image upload with /predict API
  const handleImageUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      console.log("No file selected");
      setIsProcessing(false);
      return;
    }

    console.log("File selected:", file.name);
    setLeafImage(file);
    setIsProcessing(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log("Sending request to /predict");
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${getToken()}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Prediction failed: ${errorText}`);
      }

      const data = await response.json();
      console.log("Prediction response:", data);

      const confidencePercentage = Number((data.confidence * 100).toFixed(2));
      const predictedDisease = data.prediction || "healthy";
      const treatment = DISEASE_TREATMENTS[predictedDisease] || "No specific treatment available.";

      const prediction = {
        id: Date.now(),
        text: `Predicted disease for ${file.name}: ${predictedDisease} (${confidencePercentage}%)`,
        treatment: treatment,
        date: new Date().toLocaleString(),
      };
      setPredictionHistory((prev) => [...prev, prediction]);

      Swal.fire({
        icon: "success",
        title: "Prediction Successful",
        html: `Disease predicted: ${predictedDisease} with ${confidencePercentage}% confidence<br><strong>Treatment:</strong> ${treatment}`,
        timer: 3000,
        showConfirmButton: false,
      });
    } catch (error: any) {
      console.error("Error during prediction:", error.message);
      Swal.fire({
        icon: "error",
        title: "Prediction Failed",
        text: "Unable to process the image. Please try again.",
      });
    } finally {
      setIsProcessing(false);
      console.log("Processing finished");
    }
  };

  // Handle zip file upload with /retrain API
  const handleZipUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      console.log("No file selected");
      setIsProcessing(false);
      return;
    }

    console.log("File selected:", file.name);
    setZipFile(file);
    setIsProcessing(true);

    const formData = new FormData();
    formData.append("files", file);

    try {
      console.log("Sending request to /retrain");
      const response = await fetch(`${API_BASE_URL}/retrain`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${getToken()}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Retraining failed");
      }

      const data = await response.json();
      console.log("Retraining response:", data);

      const retrainLog = {
        id: data.retraining_id || Date.now(),
        text: `Model retrained with ${file.name} (${data.num_classes} classes)`,
        date: new Date().toLocaleString(),
      };
      setRetrainHistory((prev) => [...prev, retrainLog]);

      // Update visualization images with all available plots
      setVisualizationImages({
        classification_report: data.visualization_files.classification_report,
        confusion_matrix: data.visualization_files.confusion_matrix,
        loss_plot: data.visualization_files.loss_plot,
        accuracy_plot: data.visualization_files.accuracy_plot,
      });

      Swal.fire({
        icon: "success",
        title: "Retraining Successful",
        text: data.message || "Model has been retrained successfully!",
        timer: 2000,
        showConfirmButton: false,
      });
    } catch (error: any) {
      console.error("Error during retraining:", error.message);
      Swal.fire({
        icon: "error",
        title: "Retraining Failed",
        text: error.message || "Unable to retrain the model. Please try again.",
      });
    } finally {
      setIsProcessing(false);
      console.log("Processing finished");
    }
  };

  // Fetch prediction history from /prediction_history API
  const fetchPredictionHistory = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/prediction_history`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${getToken()}`,
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction history");
      }

      const data = await response.json();
      setPredictionHistory(
        data.map((item: any) => ({
          id: item.id,
          text: item.text,
          treatment:
            item.treatment ||
            DISEASE_TREATMENTS[item.text.split(": ")[1]?.split(" (")[0]] ||
            "No treatment recorded.",
          date: new Date(item.date).toLocaleString(),
        }))
      );
    } catch (error) {
      Swal.fire({
        icon: "error",
        title: "History Fetch Failed",
        text: "Unable to load prediction history.",
      });
    }
  };

  // Fetch retraining history from /retraining_history API
  const fetchRetrainHistory = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/retraining_history`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${getToken()}`,
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Failed to fetch retraining history");
      }

      const data = await response.json();
      setRetrainHistory(
        data.map((item: any) => ({
          id: item.id,
          text: item.text,
          date: new Date(item.date).toLocaleString(),
        }))
      );
    } catch (error) {
      Swal.fire({
        icon: "error",
        title: "History Fetch Failed",
        text: "Unable to load retraining history.",
      });
    }
  };

  // Load history when switching to "history" tab
  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    if (tab === "history") {
      fetchPredictionHistory();
      fetchRetrainHistory();
    }
  };

  // Logout with animation
  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
    Swal.fire({
      icon: "success",
      title: "Logged Out",
      text: "You have been successfully logged out.",
      timer: 1500,
      showConfirmButton: false,
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white flex">
      {/* Sidebar */}
      <aside
        className={`bg-gray-800 shadow-2xl p-6 flex flex-col justify-between transition-all duration-300 ${
          isSidebarCollapsed ? "w-24" : "w-64"
        }`}
      >
        <div>
          <div className="flex items-center justify-between mb-10">
            {!isSidebarCollapsed && <img src="/logo.png" alt="Logo" />}
            <button onClick={toggleSidebar} className="text-gray-300 hover:text-[#7fd95c]">
              {isSidebarCollapsed ? (
                <img src="/g.png" alt="Expand" className="w-16 h-auto" />
              ) : (
                <FaArrowLeft size={24} />
              )}
            </button>
          </div>
          <nav className="space-y-4">
            {[
              { id: "detect", label: "Detect Disease", icon: <FaLeaf /> },
              { id: "retrain", label: "Retrain Model", icon: <FaSync /> },
              { id: "history", label: "History", icon: <FaHistory /> },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => handleTabChange(item.id)}
                className={`w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-200 ${
                  activeTab === item.id
                    ? "bg-[#25c656] shadow-lg scale-105"
                    : "bg-gray-700 hover:bg-gray-600 hover:scale-102"
                } ${isSidebarCollapsed ? "justify-center" : ""}`}
                title={isSidebarCollapsed ? item.label : undefined}
              >
                <span className="text-xl">{item.icon}</span>
                {!isSidebarCollapsed && <span className="text-lg font-medium">{item.label}</span>}
              </button>
            ))}
          </nav>
        </div>
        <button
          onClick={handleLogout}
          className={`flex items-center space-x-3 p-3 bg-red-600 rounded-xl hover:bg-red-700 transition-all duration-200 ${
            isSidebarCollapsed ? "justify-center" : ""
          }`}
          title={isSidebarCollapsed ? "Logout" : undefined}
        >
          <FaSignOutAlt className="text-xl" />
          {!isSidebarCollapsed && <span className="text-lg font-medium">Logout</span>}
        </button>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="max-w-5xl">
          <header className="mb-8">
            <h2 className="text-4xl font-bold tracking-wide animate-fade-in-down">
              {activeTab === "detect"
                ? "Plant Disease Detection"
                : activeTab === "retrain"
                ? "Model Retraining"
                : "Activity History"}
            </h2>
            <p className="text-gray-400 mt-2">Advanced tools for your Farming needs</p>
          </header>

          {activeTab === "detect" && (
            <section className="bg-gray-800 p-6 rounded-2xl shadow-xl animate-slide-up">
              <h3 className="text-2xl font-semibold mb-4">Upload Leaf Image</h3>
              <div className="relative">
                <button
                  onClick={() => triggerFileInput(leafInputRef)}
                  className={`w-full p-3 bg-gray-700 rounded-lg text-white font-medium flex items-center justify-center space-x-2 transition-all duration-200 ${
                    isProcessing
                      ? "opacity-50 cursor-not-allowed"
                      : "hover:bg-gray-600 hover:border-green-500 border border-gray-600"
                  }`}
                  disabled={isProcessing}
                >
                  <FaLeaf className="text-xl" />
                  <span>Choose Leaf Image</span>
                </button>
                <input
                  type="file"
                  accept="image/*"
                  ref={leafInputRef}
                  onChange={handleImageUpload}
                  className="hidden"
                />
                {isProcessing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 rounded-lg">
                    <div className="w-12 h-12 border-4 border-green-500 border-t-transparent rounded-full animate-spin"></div>
                  </div>
                )}
              </div>
              {leafImage && !isProcessing && (
                <div className="mt-4 p-4 bg-gray-700 rounded-lg animate-fade-in">
                  <p className="text-green-400">Uploaded: {leafImage.name}</p>
                  <p className="mt-2 text-gray-300">
                    Prediction:{" "}
                    {predictionHistory[predictionHistory.length - 1]?.text.split(": ")[1] || "Healthy"}
                  </p>
                  <p className="mt-2 text-gray-300">
                    Treatment: {predictionHistory[predictionHistory.length - 1]?.treatment || "No treatment available."}
                  </p>
                  <img
                    src={URL.createObjectURL(leafImage)}
                    alt="Leaf Preview"
                    className="mt-4 max-w-xs rounded-lg shadow-md"
                  />
                </div>
              )}
            </section>
          )}

          {activeTab === "retrain" && (
            <section className="bg-gray-800 p-6 rounded-2xl shadow-xl animate-slide-up">
              <h3 className="text-2xl font-semibold mb-4">Retrain Model</h3>
              <div className="relative">
                <button
                  onClick={() => triggerFileInput(zipInputRef)}
                  className={`w-full p-3 bg-gray-700 rounded-lg text-white font-medium flex items-center justify-center space-x-2 transition-all duration-200 ${
                    isProcessing
                      ? "opacity-50 cursor-not-allowed"
                      : "hover:bg-gray-600 hover:border-green-500 border border-gray-600"
                  }`}
                  disabled={isProcessing}
                >
                  <FaSync className="text-xl" />
                  <span>Choose Zip File</span>
                </button>
                <input
                  type="file"
                  accept=".zip"
                  ref={zipInputRef}
                  onChange={handleZipUpload}
                  className="hidden"
                />
                {isProcessing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 rounded-lg">
                    <div className="w-12 h-12 border-4 border-green-500 border-t-transparent rounded-full animate-spin"></div>
                  </div>
                )}
              </div>
              {zipFile && !isProcessing && (
                <div className="mt-4 p-4 bg-gray-700 rounded-lg animate-fade-in">
                  <p className="text-green-400">Uploaded: {zipFile.name}</p>
                  <p className="mt-2 text-gray-300">Status: Retraining Complete</p>
                  {visualizationImages.classification_report && (
                    <div className="mt-4">
                      <h4 className="text-lg font-medium text-green-400">Classification Report</h4>
                      <img
                        src={visualizationImages.classification_report}
                        alt="Classification Report"
                        className="mt-2 max-w-full rounded-lg shadow-md"
                      />
                    </div>
                  )}
                  {visualizationImages.confusion_matrix && (
                    <div className="mt-4">
                      <h4 className="text-lg font-medium text-green-400">Confusion Matrix</h4>
                      <img
                        src={visualizationImages.confusion_matrix}
                        alt="Confusion Matrix"
                        className="mt-2 max-w-full rounded-lg shadow-md"
                      />
                    </div>
                  )}
                  {visualizationImages.loss_plot && (
                    <div className="mt-4">
                      <h4 className="text-lg font-medium text-green-400">Loss Plot</h4>
                      <img
                        src={visualizationImages.loss_plot}
                        alt="Loss Plot"
                        className="mt-2 max-w-full rounded-lg shadow-md"
                      />
                    </div>
                  )}
                  {visualizationImages.accuracy_plot && (
                    <div className="mt-4">
                      <h4 className="text-lg font-medium text-green-400">Accuracy Plot</h4>
                      <img
                        src={visualizationImages.accuracy_plot}
                        alt="Accuracy Plot"
                        className="mt-2 max-w-full rounded-lg shadow-md"
                      />
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          {activeTab === "history" && (
            <section className="bg-gray-800 p-6 rounded-2xl shadow-xl animate-slide-up">
              <h3 className="text-2xl font-semibold mb-6">Activity History</h3>
              <div className="mb-8">
                <h4 className="text-xl font-medium mb-3 text-green-400">Prediction History</h4>
                {predictionHistory.length > 0 ? (
                  <ul className="space-y-3">
                    {predictionHistory.map((entry) => (
                      <li
                        key={entry.id}
                        className="p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-all duration-200"
                      >
                        <p>{entry.text}</p>
                        <p className="text-sm text-gray-300">Treatment: {entry.treatment}</p>
                        <p className="text-sm text-gray-400">{entry.date}</p>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-400">No predictions recorded yet.</p>
                )}
              </div>
              <div>
                <h4 className="text-xl font-medium mb-3 text-green-400">Retrain History</h4>
                {retrainHistory.length > 0 ? (
                  <ul className="space-y-3">
                    {retrainHistory.map((entry) => (
                      <li
                        key={entry.id}
                        className="p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-all duration-200"
                      >
                        <p>{entry.text}</p>
                        <p className="text-sm text-gray-400">{entry.date}</p>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-400">No retraining events recorded yet.</p>
                )}
              </div>
            </section>
          )}
        </div>
      </main>

      <style>{`
        @keyframes fade-in-down {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slide-up {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .animate-fade-in-down { animation: fade-in-down 0.5s ease-out; }
        .animate-slide-up { animation: slide-up 0.5s ease-out; }
        .animate-fade-in { animation: fade-in 0.5s ease-out; }
      `}</style>
    </div>
  );
};

export default Dashboard;