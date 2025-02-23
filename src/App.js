import React, { useState } from "react";
import {  Train, Home, Info, HelpCircle } from "lucide-react";
import FailurePrediction from "./components/FailurePrediction";
import CrackDetection from "./components/CrackDetection";
import FireDetection from "./components/FireDetection";
import { Flame, GaugeCircle,AlertTriangle  } from "lucide-react";
// Page Components
const Pages = {
  Home: () => (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-gray-900">Welcome to MetroAI Vision</h1>
    </div>
  ),

  About: () => (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-gray-900">About Us</h1>
    </div>
  ),

  RoutesStations: () => (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-gray-900">Routes & Stations</h1>
    </div>
  ),

  Fares: () => (
    <div className="min-h-screen flex items-center justify-center bg-gray-200 ">
      <FailurePrediction />
    </div>
  ),

  Timings: () => (
    <div className="min-h-screen flex items-center justify-center bg-gray-200 ">
      <CrackDetection/>
    </div>
  ),

  Contact: () => (
    <div className="min-h-screen flex items-center justify-center bg-gray-200 ">
      <FireDetection/>
    </div>
  ),

  FAQs: () => (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-gray-900">FAQs</h1>
    </div>
  )
};

const NavLink = ({ to, children, icon: Icon, currentPage, setCurrentPage }) => {
  const isActive = currentPage === to;
  
  return (
    <li>
      <button
        onClick={() => setCurrentPage(to)}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
          isActive
            ? "bg-blue-600 text-white"
            : "text-gray-700 hover:bg-blue-50"
        }`}
      >
        {Icon && <Icon size={18} />}
        {children}
      </button>
    </li>
  );
};

const Navbar = ({ currentPage, setCurrentPage }) => (
  <nav className="bg-white shadow-lg">
    <div className="max-w-7xl mx-auto px-4">
      <div className="flex justify-between items-center h-16">
        <div className="flex-shrink-0 flex items-center">
          <span className="text-2xl font-bold text-blue-600">MetroAI Vision</span>
        </div>
        <ul className="flex space-x-4">
          <NavLink to="home" icon={Home} currentPage={currentPage} setCurrentPage={setCurrentPage}>Home</NavLink>
          <NavLink to="about" icon={Info} currentPage={currentPage} setCurrentPage={setCurrentPage}>Supply</NavLink>
          <NavLink to="routes" icon={Train} currentPage={currentPage} setCurrentPage={setCurrentPage}>Automation</NavLink>
          <NavLink to="fares" icon={ GaugeCircle} currentPage={currentPage} setCurrentPage={setCurrentPage}>Machine_Performance</NavLink>
          <NavLink to="timings" icon={AlertTriangle } currentPage={currentPage} setCurrentPage={setCurrentPage}>Crack Dectction</NavLink>
          <NavLink to="contact" icon={Flame} currentPage={currentPage} setCurrentPage={setCurrentPage}>Fire Detection</NavLink>
          <NavLink to="faqs" icon={HelpCircle} currentPage={currentPage} setCurrentPage={setCurrentPage}>FAQs</NavLink>
        </ul>
      </div>
    </div>
  </nav>
);

const App = () => {
  const [currentPage, setCurrentPage] = useState("home");

  const getPageComponent = () => {
    switch (currentPage) {
      case "home":
        return <Pages.Home />;
      case "about":
        return <Pages.About />;
      case "routes":
        return <Pages.RoutesStations />;
      case "fares":
        return <Pages.Fares />;
      case "timings":
        return <Pages.Timings />;
      case "contact":
        return <Pages.Contact />;
      case "faqs":
        return <Pages.FAQs />;
      default:
        return <Pages.Home />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      {getPageComponent()}
    </div>
  );
};

export default App;