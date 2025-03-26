"use client";
import { FaChevronRight } from "react-icons/fa6";
import { useState, useEffect, useCallback } from "react";
import ScanModal from "./Scan";
import CountUp from "react-countup";

const images = ["/banner.webp", "/banner-bg.webp", "/banner-bg-2.webp"];

// Animation settings
const TRANSITION_DURATION = 1000;
const IMAGE_DISPLAY_DURATION = 5000;

const Header: React.FC = () => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [nextImageIndex, setNextImageIndex] = useState(1);
  const [isFading, setIsFading] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isScanModalOpen, setIsScanModalOpen] = useState(false);
  const [showDetect, setShowDetect] = useState(false);
  const [showDiagnose, setShowDiagnose] = useState(false);
  const [showDefend, setShowDefend] = useState(false);
  const [showDescription, setShowDescription] = useState(false);

  // Sequential animations on load
  useEffect(() => {
    const detectTimer = setTimeout(() => setShowDetect(true), 500);
    const diagnoseTimer = setTimeout(() => setShowDiagnose(true), 1000);
    const defendTimer = setTimeout(() => setShowDefend(true), 1500);
    const descriptionTimer = setTimeout(() => setShowDescription(true), 2000);

    return () => {
      clearTimeout(detectTimer);
      clearTimeout(diagnoseTimer);
      clearTimeout(defendTimer);
      clearTimeout(descriptionTimer);
    };
  }, []);

  const advanceImage = useCallback(() => {
    setIsFading(true);
    setTimeout(() => {
      setCurrentImageIndex(nextImageIndex);
      setNextImageIndex((nextImageIndex + 1) % images.length);
      setIsFading(false);
    }, TRANSITION_DURATION);
  }, [nextImageIndex]);

  useEffect(() => {
    if (isPaused) return;
    const interval = setInterval(() => {
      advanceImage();
    }, IMAGE_DISPLAY_DURATION);
    return () => clearInterval(interval);
  }, [nextImageIndex, isPaused, advanceImage]);

  useEffect(() => {
    images.forEach((imgSrc) => {
      const img = new Image();
      img.src = imgSrc;
    });
  }, []);

  return (
    <header
      className="relative text-center px-8 py-24 bg-gray-100 dark:bg-gradient-to-bl dark:text-white transition-colors"
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      {/* Background images */}
      {images.map((imgSrc, index) => (
        <div
          key={imgSrc}
          className="absolute inset-0 transition-opacity duration-1000 dark:mix-blend-overlay"
          style={{
            backgroundImage: `url(${imgSrc})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
            opacity:
              index === currentImageIndex
                ? isFading
                  ? 0
                  : 1
                : index === nextImageIndex && isFading
                ? 1
                : 0,
            zIndex: 0,
          }}
        />
      ))}

      {/* Overlay for light/dark mode */}
      <div className="absolute inset-0 bg-gray-100 opacity-70 dark:bg-black dark:opacity-0"></div>

      {/* Content */}
      <div className="relative z-10">
        <div className="flex items-center justify-between gap-72">
          <h1 className="text-7xl text-left font-bold mt-20 text-gray-900 dark:text-white overflow-hidden">
            <div
              className={`transform transition-transform duration-700 ${
                showDetect ? "translate-x-0" : "-translate-x-full"
              }`}
            >
              Detect,
            </div>
            <div
              className={`transform transition-transform duration-700 ${
                showDiagnose ? "translate-x-0" : "-translate-x-full"
              }`}
            >
              Diagnose and
            </div>
            <div
              className={`transform transition-transform duration-700 ${
                showDefend ? "translate-x-0" : "-translate-x-full"
              }`}
            >
              <i className="font-light text-[#25c656] dark:text-[#25c656]">Defend</i>
            </div>
          </h1>
          <div
            className={`transform transition-transform duration-1000 ${
              showDescription ? "translate-x-0" : "translate-x-full"
            }`}
          >
            <p className="text-lg text-gray-700 dark:text-gray-300 text-left max-w-lg mt-8">
              LeafSense uses advanced AI technology to instantly identify plant
              diseases from simple leaf photos. Our cutting-edge system helps
              farmers, gardeners, and agricultural professionals detect problems
              early, reduce crop losses, and implement targeted treatments for
              healthier plants.
            </p>
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between">
            <button
              onClick={() => setIsScanModalOpen(true)}
              className="rounded-md bg-[#25c656] -mt-10 text-left px-6 py-4 flex items-center space-x-5 hover:bg-gray-900 dark:hover:bg-white hover:text-white dark:hover:text-black transition-colors"
            >
              Start scanning
            </button>
            <div className="space-x-20 flex items-center mt-36 justify-between">
              <div>
                <p className="text-8xl text-gray-900 dark:text-white">
                  <CountUp start={0} end={95} duration={5} />
                  <span>%</span>
                </p>
                <p className="uppercase text-gray-700 dark:text-gray-300">
                  Detection accuracy rate
                </p>
              </div>
              <div>
                <p className="text-8xl text-gray-900 dark:text-white">
                  <CountUp start={0} end={10} duration={5} />
                  <span>+</span>
                </p>
                <p className="uppercase text-gray-700 dark:text-gray-300">
                  Plant diseases identified
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <ScanModal
        isOpen={isScanModalOpen}
        onClose={() => setIsScanModalOpen(false)}
      />
    </header>
  );
};

export default Header;