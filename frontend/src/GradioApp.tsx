import React, { useState } from 'react';
import { Loader2, Upload, Box, Satellite, ShieldCheck } from 'lucide-react';
import { predictAgl, type AglPredictionResult } from './api/gradio';

function getOutputUrl(value: unknown): string | null {
    if (typeof value === 'string') return value;

    if (value && typeof value === 'object' && 'url' in value) {
        const url = (value as { url?: unknown }).url;
        return typeof url === 'string' ? url : null;
    }

    return null;
}

interface GradioAppProps {
    onSwitchToWorkspace?: () => void;
}

export const GradioApp: React.FC<GradioAppProps> = ({ onSwitchToWorkspace }) => {
    const [file, setFile] = useState<File | null>(null);
    const [gsd, setGsd] = useState('');
    const [result, setResult] = useState<AglPredictionResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleRun = async () => {
        if (!file) {
            setError('Please select a PNG or JPG image.');
            return;
        }

        const gsdValue = gsd.trim() ? Number(gsd) : undefined;

        if (
            gsdValue !== undefined &&
            (!Number.isFinite(gsdValue) || gsdValue <= 0)
        ) {
            setError('GSD must be a positive number.');
            return;
        }

        try {
            setLoading(true);
            setError(null);
            setResult(null);

            const output = await predictAgl(file, gsdValue);
            setResult(output);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setLoading(false);
        }
    };

    const heatmapUrl = getOutputUrl(result?.heatmap);
    const rawFileUrl = getOutputUrl(result?.rawAglFile);

    return (
        <div className="min-h-screen bg-slate-950 text-white p-6">
            <div className="max-w-6xl mx-auto">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
                    <div>
                        <div className="flex items-center space-x-3">
                            <div className="w-9 h-9 rounded bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                                <Satellite className="w-5 h-5" />
                            </div>
                            <div>
                                <div className="flex items-center space-x-2">
                                    <h1 className="text-2xl font-bold tracking-wide">DepthWizard</h1>
                                    <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
                                        SIH26175 • ISRO
                                    </span>
                                </div>
                                <p className="text-slate-400 text-xs mt-0.5">
                                    M3-FINAL · Metric AGL Height Estimation · Hugging Face ZeroGPU Inference
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center space-x-3">
                        <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg text-xs font-mono text-emerald-400">
                            <ShieldCheck className="w-4 h-4 text-emerald-400" />
                            <span>M3-FINAL (Verified)</span>
                        </div>

                        {onSwitchToWorkspace && (
                            <button
                                onClick={onSwitchToWorkspace}
                                className="flex items-center space-x-2 bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 hover:border-cyan-500/50 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all shadow-sm cursor-pointer"
                            >
                                <Box className="w-4 h-4" />
                                <span>3D Terrain Workspace</span>
                            </button>
                        )}
                    </div>
                </div>

                <div className="grid md:grid-cols-2 gap-6 mt-8">
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                        <h2 className="font-semibold text-lg mb-4">
                            Input Satellite Image
                        </h2>

                        <label className="border-2 border-dashed border-slate-700 rounded-lg p-8 flex flex-col items-center cursor-pointer hover:border-cyan-500">
                            <Upload className="w-8 h-8 text-cyan-400 mb-3" />

                            <span>
                                {file ? file.name : 'Choose PNG / JPG image'}
                            </span>

                            <input
                                type="file"
                                accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                                className="hidden"
                                onChange={(e) => {
                                    setFile(e.target.files?.[0] ?? null);
                                    setResult(null);
                                    setError(null);
                                }}
                            />
                        </label>

                        <div className="mt-5">
                            <label className="text-sm text-slate-300">
                                GSD (metres / pixel) — optional
                            </label>

                            <input
                                type="number"
                                value={gsd}
                                onChange={(e) => setGsd(e.target.value)}
                                placeholder="Example: 0.5"
                                step="0.01"
                                min="0.01"
                                className="w-full mt-2 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2"
                            />

                            <p className="text-xs text-slate-500 mt-2">
                                Leave blank if GSD is unknown.
                            </p>
                        </div>

                        <button
                            onClick={handleRun}
                            disabled={!file || loading}
                            className="w-full mt-5 bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-700 text-slate-950 font-bold py-2.5 rounded-lg"
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    Running M3-FINAL...
                                </span>
                            ) : (
                                'Estimate Height'
                            )}
                        </button>

                        {error && (
                            <div className="mt-4 bg-rose-950 border border-rose-800 text-rose-200 p-3 rounded-lg">
                                {error}
                            </div>
                        )}
                    </div>

                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                        <h2 className="font-semibold text-lg mb-4">
                            Metric AGL Prediction
                        </h2>

                        {!result && (
                            <div className="h-72 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-center text-slate-500">
                                Run inference to generate height map
                            </div>
                        )}

                        {result && (
                            <>
                                {heatmapUrl && (
                                    <img
                                        src={heatmapUrl}
                                        alt="Predicted AGL"
                                        className="w-full rounded-lg"
                                    />
                                )}

                                <h3 className="mt-5 mb-2 font-semibold">
                                    Inference Metadata
                                </h3>

                                <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs overflow-auto max-h-72">
                                    {JSON.stringify(result.metadata, null, 2)}
                                </pre>

                                {rawFileUrl && (
                                    <a
                                        href={rawFileUrl}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="inline-block mt-4 text-cyan-400 hover:text-cyan-300"
                                    >
                                        Download raw AGL (.npy)
                                    </a>
                                )}
                            </>
                        )}
                    </div>
                </div>

                <p className="text-xs text-slate-500 mt-6">
                    Depth Anything V2 provides a scale-agnostic structural prior.
                    M3-FINAL predicts metric above-ground-level height.
                </p>
            </div>
        </div>
    );
};

export default GradioApp;