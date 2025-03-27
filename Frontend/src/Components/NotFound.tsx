// Components/NotFound.tsx
import React from 'react';
import { Link } from 'react-router-dom';

const NotFound = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
      <div className="text-center p-8 bg-black dark:bg-gray-800 rounded-lg shadow-lg">
        <h1 className="text-6xl font-bold text-white">404</h1>
        <p className="text-2xl text-gray-300">Page Not Found</p>
        <p className="mt-4">
          <Link to="/" className="text-blue-600 dark:text-blue-400 hover:underline">
            Go back to Home
          </Link>
        </p>
      </div>
    </div>
  );
};

export default NotFound;
