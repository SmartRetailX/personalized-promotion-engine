import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AdminDashboard from "./components/AdminDashboard.jsx";
import CustomerDashboard from "./components/CustomerDashboard.jsx";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AdminDashboard />} />
        <Route path="/admin" element={<AdminDashboard />} />
        <Route path="/customer" element={<CustomerDashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
