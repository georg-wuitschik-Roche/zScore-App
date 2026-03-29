import { lazy, Suspense, useEffect } from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import { useFilterStore } from './stores/filterStore';
import { LandingPage } from './components/LandingPage';
import './styles/app.css';

const dashboardImport = () =>
  import('./components/Dashboard').then((m) => ({ default: m.Dashboard }));

const Dashboard = lazy(dashboardImport);

function AppContent() {
  const { isFullDataLoaded, loadError, presentationMode, theme, loadDataset } =
    useFilterStore();

  const dropdownIndex = useFilterStore((s) => s.dropdownIndex);

  useEffect(() => {
    loadDataset();
  }, [loadDataset]);

  // Apply theme on mount
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Preload Dashboard + Plotly chunk once landing page is interactive
  useEffect(() => {
    if (dropdownIndex) {
      dashboardImport();
    }
  }, [dropdownIndex]);

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
            !isFullDataLoaded && !loadError ? (
              <div className="loading-container">
                <div className="spinner" />
                <p>Loading dataset...</p>
              </div>
            ) : loadError ? (
              <div className="loading-container">
                <p className="load-error">Failed to load dataset: {loadError}</p>
              </div>
            ) : (
              <Suspense fallback={
                <div className="loading-container">
                  <div className="spinner" />
                  <p>Loading dashboard...</p>
                </div>
              }>
                <Dashboard />
              </Suspense>
            )
          }
        />
      </Routes>
    </div>
  );
}

export default function App() {
  return (
    <HashRouter>
      <AppContent />
    </HashRouter>
  );
}
