import { Client, handle_file } from '../frontend/node_modules/@gradio/client/dist/index.js';
import fs from 'fs';
import path from 'path';

async function testRemoteSpace() {
  console.log('Connecting to remote Hugging Face Space: Devendra80/depthwizard-api...');
  const t0 = Date.now();
  const app = await Client.connect('Devendra80/depthwizard-api');
  console.log(`Connected successfully in ${(Date.now() - t0) / 1000}s.`);

  // 1. Health check
  console.log('\n--- 1. Testing /health Endpoint ---');
  try {
    const healthResult = await app.predict('/health', []);
    console.log('Health Result:', JSON.stringify(healthResult.data, null, 2));
  } catch (err) {
    console.error('Health check failed:', err.message);
  }

  // 2. Predict AGL
  console.log('\n--- 2. Testing /predict_agl Endpoint ---');
  const testImagePath = path.resolve('data/test_sample_512.png');
  console.log('Using test image:', testImagePath);

  try {
    const fileBuffer = fs.readFileSync(testImagePath);
    const blob = new Blob([fileBuffer], { type: 'image/png' });
    const tInference = Date.now();
    console.log(`Sending predict_agl request (image size: ${fileBuffer.length} bytes, GSD: 0.5m/px)...`);
    
    const result = await app.predict('/predict_agl', [
      blob,
      0.5
    ]);
    
    const latency = (Date.now() - tInference) / 1000;
    console.log(`\nInference completed in ${latency.toFixed(2)}s!`);
    console.log('Result Data items count:', result.data.length);
    console.log('Metadata:', JSON.stringify(result.data[1], null, 2));
    console.log('Heatmap URL:', result.data[0]?.url || result.data[0]);
    console.log('Raw File URL:', result.data[2]?.url || result.data[2]);
  } catch (err) {
    console.error('predict_agl failed:', err);
  }
}

testRemoteSpace().catch(console.error);
