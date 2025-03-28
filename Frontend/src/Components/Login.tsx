import React, { useState, FormEvent } from "react";
import { useNavigate } from "react-router-dom";

// Importing icons for Google and Apple
import { FcGoogle } from "react-icons/fc";
import { FaApple } from "react-icons/fa";

const Login: React.FC = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      const response = await fetch("https://plantdiseasedetectionapp.onrender.com/token", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          username: email,
          password,
        }),
      });

      if (!response.ok) {
        throw new Error("Invalid credentials");
      }

      const data = await response.json();
      localStorage.setItem("token", data.access_token);
      navigate("/dashboard");
    } catch (err) {
      setError("Failed to login. Please check your credentials.");
    } finally {
      setIsLoading(false);
    }
  };

  // Function to fill demo credentials
  const fillDemoCredentials = () => {
    setEmail("marvin@teacher.com");
    setPassword("123");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900">
      <div className="flex rounded-2xl shadow-lg max-w-5xl w-full">
        {/* Left Side: Login Form */}
        <div className="w-1/2 p-12 bg-gray-800">
          <h2 className="text-4xl font-bold text-white mb-3">Welcome back</h2>
          <p className="text-gray-400 text-lg mb-8">Login to continue your Detecting journey</p>

          {error && (
            <div className="bg-red-100 text-red-700 themep-4 rounded-lg mb-6">
              {error}
            </div>
          )}

          {/* Social Login Buttons */}
          <div className="flex space-x-6 mb-8">
            <button className="flex-1 flex items-center justify-center py-3 border border-gray-600 rounded-lg text-gray-300 hover:bg-gray-700">
              <FcGoogle className="mr-3" size={24} />
              <span className="text-lg">Log in with Google</span>
            </button>
            <button className="flex-1 flex items-center justify-center py-3 border border-gray-600 rounded-lg text-gray-300 hover:bg-gray-700">
              <FaApple className="mr-3" size={24} />
              <span className="text-lg">Log in with Apple</span>
            </button>
          </div>

          {/* Divider */}
          <div className="relative mb-8">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-600"></div>
            </div>
            <div className="relative flex justify-center text-base">
              <span className="px-3 bg-gray-800 text-gray-400">OR</span>
            </div>
          </div>

          {/* Email and Password Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-base font-medium text-gray-300">
                Email Address
              </label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="mt-2 block w-full px-4 py-3 border border-gray-600 rounded-lg shadow-sm focus:outline-none focus:ring-green-500 focus:border-green-500 bg-gray-700 text-white text-lg placeholder-gray-400"
                placeholder="hello@yourcompany.com"
                required
              />
            </div>

            <div>
              <div className="flex justify-between">
                <label htmlFor="password" className="block text-base font-medium text-gray-300">
                  Password
                </label>
                <a href="/forgot-password" className="text-base text-gray-400 hover:text-gray-200">
                  Forgot password?
                </a>
              </div>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-2 block w-full px-4 py-3 border border-gray-600 rounded-lg shadow-sm focus:outline-none focus:ring-green-500 focus:border-green-500 bg-gray-700 text-white text-lg placeholder-gray-400"
                required
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 px-4 bg-green-600 text-white rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 text-lg font-medium"
            >
              {isLoading ? "Logging in..." : "LOGIN"}
            </button>

            {/* Demo Credentials Button with Animation */}
            <button
              type="button"
              onClick={fillDemoCredentials}
              className="w-full py-3 px-4 bg-gray-600 text-white rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 text-lg font-medium animate-bounce"
            >
              Fill Demo Credentials
            </button>
          </form>

          <p className="mt-6 text-center text-base text-gray-400">
            Don't have an account?{" "}
            <a href="/signup" className="text-green-500 hover:text-green-400">
              Sign up
            </a>
          </p>
        </div>

        {/* Right Side: Image */}
        <div className="w-1/2">
          <img
            src="m.jpg"
            alt="Gardening"
            className="h-full w-full object-cover rounded-r-2xl"
          />
        </div>
      </div>
    </div>
  );
};

export default Login;