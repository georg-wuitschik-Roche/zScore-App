/**
 * Plotly wrapper that uses the pre-built browser bundle (plotly.js-dist-min)
 * to avoid Vite/Rollup issues with Node.js builtins (buffer, stream, etc.).
 */

// @ts-expect-error — plotly.js-dist-min has no type declarations
import Plotly from 'plotly.js-dist-min';
import _createPlotlyComponent from 'react-plotly.js/factory';

// Handle CJS→ESM interop: factory may be { default: fn } or fn directly
const createPlotlyComponent =
  typeof _createPlotlyComponent === 'function'
    ? _createPlotlyComponent
    : (_createPlotlyComponent as { default: typeof _createPlotlyComponent }).default;

const Plot = createPlotlyComponent(Plotly);
export { Plotly };
export default Plot;
