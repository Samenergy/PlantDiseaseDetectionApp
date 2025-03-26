import React from 'react';

const StatisticsSection = () => {
  return (
    <section className="relative py-20 px-4 bg-[url('/back.webp')] bg-cover bg-center">
      {/* Overlay for reduced opacity */}
      <div className="absolute inset-0 bg-black opacity-50"></div>

      <div className="relative max-w-7xl mx-auto text-white">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
          <div className="p-6">
            <h3 className="text-4xl font-bold text-green-400 mb-2">98%</h3>
            <p className="text-gray-300">Detection Accuracy</p>
          </div>
          <div className="p-6">
            <h3 className="text-4xl font-bold text-green-400 mb-2">10+</h3>
            <p className="text-gray-300">Plant Diseases in Database</p>
          </div>
          <div className="p-6">
            <h3 className="text-4xl font-bold text-green-400 mb-2">
              50,000+
            </h3>
            <p className="text-gray-300">Happy Users</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default StatisticsSection; 