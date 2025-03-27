import React, { useState, ChangeEvent } from "react";
import { useNavigate } from "react-router-dom";
import { FaLeaf, FaSync, FaHistory, FaSignOutAlt, FaArrowLeft } from "react-icons/fa";

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<string>("detect");
  const [leafImage, setLeafImage] = useState<File | null>(null);
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<{ id: number; text: string; date: string }[]>([]);
  const [retrainHistory, setRetrainHistory] = useState<{ id: number; text: string; date: string }[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);

  // Toggle sidebar collapse
  const toggleSidebar = () => setIsSidebarCollapsed((prev) => !prev);

  // Handle image upload with simulated processing
  const handleImageUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setLeafImage(file);
      setIsProcessing(true);
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const prediction = {
        id: Date.now(),
        text: `Predicted disease for ${file.name}: Healthy`,
        date: new Date().toLocaleString(),
      };
      setPredictionHistory((prev) => [...prev, prediction]);
      setIsProcessing(false);
    }
  };

  // Handle zip file upload with simulated processing
  const handleZipUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setZipFile(file);
      setIsProcessing(true);
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const retrainLog = {
        id: Date.now(),
        text: `Model retrained with ${file.name}`,
        date: new Date().toLocaleString(),
      };
      setRetrainHistory((prev) => [...prev, retrainLog]);
      setIsProcessing(false);
    }
  };

  // Logout with animation
  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white flex">
      {/* Sidebar */}
      <aside
        className={`bg-gray-800 shadow-2xl p-6 flex flex-col justify-between transition-all duration-300 ${
          isSidebarCollapsed ? "w-24" : "w-64 "
        }`}
      >
        <div>
          {/* Toggle Button and Title */}
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

          {/* Navigation */}
          <nav className="space-y-4">
            {[
              { id: "detect", label: "Detect Disease", icon: <FaLeaf /> },
              { id: "retrain", label: "Retrain Model", icon: <FaSync /> },
              { id: "history", label: "History", icon: <FaHistory /> },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-200 ${
                  activeTab === item.id
                    ? "bg-[#25c656] shadow-lg scale-105"
                    : "bg-gray-700 hover:bg-gray-600 hover:scale-102"
                } ${isSidebarCollapsed ? "justify-center" : ""}`}
                title={isSidebarCollapsed ? item.label : undefined} // Tooltip for collapsed state
              >
                <span className="text-xl">{item.icon}</span>
                {!isSidebarCollapsed && <span className="text-lg font-medium">{item.label}</span>}
              </button>
            ))}
          </nav>
        </div>

        {/* Logout Button */}
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
          {/* Header */}
          <header className="mb-8">
            <h2 className="text-4xl font-bold tracking-wide animate-fade-in-down">
              {activeTab === "detect"
                ? "Plant Disease Detection"
                : activeTab === "retrain"
                ? "Model Retraining"
                : "Activity History"}
            </h2>
            <p className="text-gray-400 mt-2">Advanced tools for your gardening needs</p>
          </header>

          {/* Content Sections */}
          {activeTab === "detect" && (
            <section className="bg-gray-800 p-6 rounded-2xl shadow-xl animate-slide-up">
              <h3 className="text-2xl font-semibold mb-4">Upload Leaf Image</h3>
              <div className="relative">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="w-full p-3 bg-gray-700 rounded-lg border border-gray-600 hover:border-green-500 transition-all duration-200"
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
                  <p className="mt-2 text-gray-300">Prediction: Healthy (Mock Result)</p>
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
                <input
                  type="file"
                  accept=".zip"
                  onChange={handleZipUpload}
                  className="w-full p-3 bg-gray-700 rounded-lg border border-gray-600 hover:border-green-500 transition-all duration-200"
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
                  <p className="mt-2 text-gray-300">Status: Retraining Complete (Mock Result)</p>
                </div>
              )}
            </section>
          )}

          {activeTab === "history" && (
            <section className="bg-gray-800 p-6 rounded-2xl shadow-xl animate-slide-up">
              <h3 className="text-2xl font-semibold mb-6">Activity History</h3>

              {/* Prediction History */}
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
                        <p className="text-sm text-gray-400">{entry.date}</p>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-400">No predictions recorded yet.</p>
                )}
              </div>

              {/* Retrain History */}
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

      
    </div>
  );
};

export default Dashboard;