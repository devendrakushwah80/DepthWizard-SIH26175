/**
 * DepthWizard (SIH26175) — Centralized HTTP Client
 * Player 5: Frontend Integration & Product UI Lead
 */

import axios, { AxiosError } from 'axios';
import type { APIErrorResponse } from '../types';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Accept': 'application/json'
  }
});

// Format API Error Helper
export function extractErrorMessage(err: unknown): { code: string; message: string } {
  if (axios.isAxiosError(err)) {
    const axErr = err as AxiosError<APIErrorResponse>;
    if (axErr.response?.data?.error) {
      return {
        code: axErr.response.data.error.code || 'API_ERROR',
        message: axErr.response.data.error.message || axErr.message
      };
    }
    return {
      code: `HTTP_${axErr.response?.status || 'NETWORK_ERROR'}`,
      message: axErr.message || 'Network communication error'
    };
  }
  if (err instanceof Error) {
    return { code: 'CLIENT_ERROR', message: err.message };
  }
  return { code: 'UNKNOWN_ERROR', message: String(err) };
}
