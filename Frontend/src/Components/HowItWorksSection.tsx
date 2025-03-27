import React from 'react';
import {
  FaLeaf,
  FaCloudUploadAlt,
  FaRegLightbulb,
} from 'react-icons/fa';

const HowItWorksSection = () => {
  return (
    <section className="py-20 px-4 bg-green-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-800 dark:text-white mb-4">
            How It Works
          </h2>
          <p className="max-w-2xl mx-auto text-gray-600 dark:text-gray-300">
            Detect plant diseases in just three simple steps
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Step 1 */}
          <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg relative">
            <div className="absolute -top-5 -left-5 w-12 h-12 rounded-full bg-green-500 flex items-center justify-center text-white font-bold">
              1
            </div>
            <div className="text-green-500 mb-4">
              <FaCloudUploadAlt className="w-10 h-10" />
            </div>
            <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-3">
              Upload Photo
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Take a clear photo of the affected plant leaf and upload it to
              our system.
            </p>
          </div>

          {/* Step 2 */}
          <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg relative">
            <div className="absolute -top-5 -left-5 w-12 h-12 rounded-full bg-green-500 flex items-center justify-center text-white font-bold">
              2
            </div>
            <div className="text-green-500 mb-4">
              <FaLeaf className="w-10 h-10" />
            </div>
            <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-3">
              AI Analysis
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Our advanced AI analyzes the image to identify the plant and
              detect any diseases.
            </p>
          </div>

          {/* Step 3 */}
          <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg relative">
            <div className="absolute -top-5 -left-5 w-12 h-12 rounded-full bg-green-500 flex items-center justify-center text-white font-bold">
              3
            </div>
            <div className="text-green-500 mb-4">
              <FaRegLightbulb className="w-10 h-10" />
            </div>
            <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-3">
              Get Solutions
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Receive detailed information about the disease and personalized
              treatment recommendations.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorksSection; 