
import { FaMicroscope, FaMobileAlt, FaDatabase } from "react-icons/fa";

const FeaturesSection = () => {
  return (
    <section className="py-12 md:py-20 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col lg:flex-row gap-8 lg:gap-12">
          {/* Left side - Images */}
          <div>
            <img className="w-auto h-[800px]" src="/pic.webp" alt="" />
          </div>

          {/* Right side - Features */}
          <div className="w-full lg:w-3/5">
            <div className="text-left mb-12 md:mb-16">
              <h2 className="flex items-center gochi-hand-regular md:mb-20">
                <div className="mx-2 h-0.5 w-6 bg-[#25c656]"></div>
                <span className="text-2xl text-gray-400 md:text-2xl">
                  Why Choose LeafSense
                </span>
              </h2>
              <h3 className="text-2xl md:text-3xl lg:text-4xl font-bold text-gray-800 dark:text-white -mt-6 md:-mt-16">
                Advanced Technology for Healthier Plants and Better Yields
              </h3>
              <p className="max-w-2xl text-gray-600 dark:text-gray-300 mt-5">
                Our advanced AI technology provides accurate plant disease
                detection to help you maintain healthy crops and gardens.
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8 -mt-5">
              {/* Feature 1 */}
              <div className="bg-white dark:bg-gray-800 p-6 md:p-8 rounded-xl shadow-lg hover:shadow-xl transition-all">
                <div className="text-green-500 mb-4">
                  <FaMicroscope className="w-8 h-8 md:w-10 md:h-10" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-3">
                  Accurate Diagnosis
                </h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Our AI model is trained on thousands of images to provide
                  highly accurate plant disease identification with 95%+
                  accuracy.
                </p>
              </div>

              {/* Feature 2 */}
              <div className="bg-white dark:bg-gray-800 p-6 md:p-8 rounded-xl shadow-lg hover:shadow-xl transition-all">
                <div className="text-green-500 mb-4">
                  <FaMobileAlt className="w-8 h-8 md:w-10 md:h-10" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-3">
                  Easy to Use
                </h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Simply upload a photo of your plant leaf and get instant
                  disease detection results with treatment recommendations.
                </p>
              </div>

              {/* Feature 3 */}
              <div className="bg-white dark:bg-gray-800 p-6 md:p-8 rounded-xl shadow-lg hover:shadow-xl transition-all">
                <div className="text-green-500 mb-4">
                  <FaDatabase className="w-8 h-8 md:w-10 md:h-10" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-3">
                  Comprehensive Database
                </h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Access our extensive library of plant diseases, symptoms, and
                  treatment methods to better understand your plants' health.
                </p>
              </div>

              {/* Feature 4 */}
              <div className="bg-white dark:bg-gray-800 p-6 md:p-8 rounded-xl shadow-lg hover:shadow-xl transition-all">
                <div className="text-green-500 mb-4">
                  <svg
                    className="w-8 h-8 md:w-10 md:h-10"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M10 3.5a1.5 1.5 0 013 0V4a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-.5a1.5 1.5 0 000 3h.5a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-.5a1.5 1.5 0 00-3 0v.5a1 1 0 01-1 1H6a1 1 0 01-1-1v-3a1 1 0 00-1-1h-.5a1.5 1.5 0 010-3H4a1 1 0 001-1V6a1 1 0 011-1h3a1 1 0 001-1v-.5z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-3">
                  Personalized Insights
                </h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Get personalized recommendations based on your garden's
                  history, local climate, and seasonal patterns for optimal
                  plant health.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
