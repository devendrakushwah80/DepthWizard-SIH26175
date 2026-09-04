import { Client, handle_file } from '@gradio/client';

const SPACE_ID =
    import.meta.env.VITE_GRADIO_SPACE_ID || 'Devendra80/depthwizard-api';

export type AglMetadata = Record<string, unknown>;

export interface AglPredictionResult {
    heatmap: unknown;
    metadata: AglMetadata;
    rawAglFile: unknown;
}

let appPromise: ReturnType<typeof Client.connect> | null = null;

async function getGradioApp() {
    if (!appPromise) {
        appPromise = Client.connect(SPACE_ID);
    }

    return appPromise;
}

export async function predictAgl(
    file: File,
    gsdM?: number
): Promise<AglPredictionResult> {
    if (!file) {
        throw new Error('RGB image is required.');
    }

    if (
        gsdM !== undefined &&
        (!Number.isFinite(gsdM) || gsdM <= 0)
    ) {
        throw new Error('GSD must be a positive number.');
    }

    const app = await getGradioApp();

    const result = await app.predict('/predict_agl', [
        handle_file(file),
        gsdM ?? 0
    ]);

    const data = result.data as unknown[];

    if (!Array.isArray(data) || data.length < 3) {
        throw new Error('Unexpected response from DepthWizard inference API.');
    }

    return {
        heatmap: data[0],
        metadata: data[1] as AglMetadata,
        rawAglFile: data[2]
    };
}