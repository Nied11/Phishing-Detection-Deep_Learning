import React, { Suspense, lazy } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

// Lazy loading components
const HomeScreen = lazy(() => import("./HomeScreen"));
const InputForm = lazy(() => import("./components/InputForm"));
const OutputDisplay = lazy(() => import("./components/OutputDisplay"));

function App() {
  return (
    <Router>
      <Suspense fallback={<div style={{ textAlign: "center", marginTop: "20px" }}>Loading...</div>}>
        <Routes>
          <Route path="/" element={<HomeScreen />} />
          <Route path="/input" element={<InputForm />} />
          <Route path="/output" element={<OutputDisplay />} />
        </Routes>
      </Suspense>
    </Router>
  );
}

export default App;
