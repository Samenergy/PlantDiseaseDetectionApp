import React from 'react';
import { Route, Routes } from 'react-router-dom'; // Only import Route and Routes, no need for Router
import Home from './Components/Home';
import Login from './Components/Login';
import NotFound from './Components/NotFound'; // Import the NotFound component
import Dashboard from './Components/Dashboard';
import './App.css';
const App = () => {
  return (
    <Routes> {/* Define Routes directly */}
      <Route path="/" element={<Home />} /> {/* Home page route */}
      <Route path="/login" element={<Login />} /> {/* Login page route */}
      <Route path='/dashboard' element={<Dashboard/>} /> {/* Dashboard page route */}
      <Route path="*" element={<NotFound />} /> {/* Catch-all route for 404 page */}
    </Routes>
  );
};

export default App;
