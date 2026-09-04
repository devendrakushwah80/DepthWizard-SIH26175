import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Sparkles } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackModeSwitch?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('DepthWizard ErrorBoundary caught an unhandled exception:', error, errorInfo);
  }

  private handleReload = () => {
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen w-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6 select-none font-sans">
          <div className="max-w-lg w-full bg-slate-900 border border-rose-900/60 rounded-xl p-6 shadow-2xl space-y-5">
            {/* Header */}
            <div className="flex items-center space-x-3 border-b border-slate-800 pb-4">
              <div className="w-10 h-10 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <h2 className="text-lg font-bold text-white tracking-wide">DepthWizard Workspace Alert</h2>
                  <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800">
                    SIH26175
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  An unexpected error occurred during rendering.
                </p>
              </div>
            </div>

            {/* Error Message */}
            <div className="bg-slate-950 border border-slate-800 rounded-lg p-3.5 text-xs font-mono text-rose-300 break-words max-h-36 overflow-y-auto">
              {this.state.error?.message || 'Unknown runtime error occurred.'}
            </div>

            {/* Guidance */}
            <p className="text-xs text-slate-400 leading-relaxed">
              If the 3D analytical workspace failed due to a missing scene or backend communication error, you can retry loading or switch to the cloud estimator.
            </p>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-center gap-3 pt-2">
              <button
                onClick={this.handleReload}
                className="w-full sm:w-auto flex-1 flex items-center justify-center space-x-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-4 py-2.5 rounded-lg text-xs font-medium transition-colors cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Reload Workspace</span>
              </button>

              {this.props.fallbackModeSwitch && (
                <button
                  onClick={() => {
                    this.setState({ hasError: false, error: null });
                    this.props.fallbackModeSwitch?.();
                  }}
                  className="w-full sm:w-auto flex-1 flex items-center justify-center space-x-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold px-4 py-2.5 rounded-lg text-xs transition-colors cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>ZeroGPU Estimator</span>
                </button>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
