import { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useFilterStore } from './stores/filterStore';
import { LandingPage } from './components/LandingPage';
import { Dashboard } from './components/Dashboard';
import './styles/app.css';

function AppContent() {
  const { isLoading, loadError, presentationMode, loadDataset } =
    useFilterStore();

  useEffect(() => {
    loadDataset();
  }, [loadDataset]);

  const containerClass = [
    'app-container',
    presentationMode ? 'presentation-mode' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={containerClass}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route
          path="/dashboard"
          element={
            isLoading ? (
              <div className="loading-container">
                <div className="spinner" />
                <p>Loading dataset...</p>
              </div>
            ) : loadError ? (
              <div className="loading-container">
                <p className="load-error">Failed to load dataset: {loadError}</p>
              </div>
            ) : (
              <Dashboard />
            )
          }
        />
      </Routes>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
