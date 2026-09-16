/**
 * Plotly event subscription.
 *
 * Kept out of components/Plot.tsx so that file only exports components (fast
 * refresh), but this is the single place that knows how to reach the emitter.
 */

/**
 * Subscribe to a Plotly event on a graph div.
 *
 * Plotly augments the div with an event emitter that @types/plotly.js does not
 * model on HTMLElement, so the cast lives here rather than at each call site.
 *
 * Use this instead of react-plotly.js's `onHover`/`onUnhover`/`onRelayout` props:
 * those record the handler in the library's own bookkeeping, but the listener does
 * not survive on the graph div's emitter here, so the callback never fires. Bind
 * from the plot's `onInitialized` callback instead.
 */
export function bindPlotlyEvent<T>(
  graphDiv: HTMLElement,
  event: string,
  handler: (data: T) => void,
): void {
  (graphDiv as unknown as { on: (e: string, h: (d: T) => void) => void }).on(event, handler);
}
