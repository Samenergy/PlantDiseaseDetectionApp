import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom"; // Import Link from react-router-dom

const Navbar: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  return (
    <nav
      className={`fixed w-full z-10 transition-all duration-300 ${
        isScrolled ? "backdrop-blur-md" : "bg-transparent"
      }`}
    >
      <div className="max-w-7xl relative flex items-center justify-between p-4 gap-5">
        {/* Logo & Navigation */}
        <div className="flex items-center gap-5">
          {/* Logo */}
          <div className="text-black dark:text-white text-xl rounded-full border border-gray-500">
            <Link to="/">
              <img
                src="/logo.png"
                alt="LeafSense Logo"
                className="max-w-60 h-auto -my-6"
              />
            </Link>
          </div>

          {/* Desktop Navigation Links */}
          <div className="hidden md:flex space-x-8 text-black dark:text-white rounded-full border border-gray-500 px-4 py-2 text-md">
            <Link to="/">Home</Link>
            <Link to="/detect">Detect Disease</Link>
            <Link to="/diseases">Disease Database</Link>
            <Link to="/contact">Contact</Link>
          </div>
        </div>

        {/* Mobile Menu Button */}
        <div className="md:hidden flex items-center">
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="text-black dark:text-white"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              className="h-6 w-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
        </div>

        {/* Login & SignUp positioned to the far right */}
        <div className="absolute -right-[600px] top-1/2 -translate-y-1/2 hidden md:flex space-x-8 text-black dark:text-white rounded-full border border-gray-500 px-4 py-2 text-md">
          <Link to="/login">Login</Link>
          <Link to="/signup">SignUp</Link>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="md:hidden flex flex-col items-center bg-gray-900 text-black dark:text-white py-4 space-y-4">
          <Link to="/" className="text-md">Home</Link>
          <Link to="/detect" className="text-md">Detect Disease</Link>
          <Link to="/diseases" className="text-md">Disease Database</Link>
          <Link to="/contact" className="text-md">Contact</Link>
          <div className="space-x-8 text-black dark:text-white">
            <Link to="/login" className="text-md">Login</Link>
            <Link to="/signup" className="text-md">SignUp</Link>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
