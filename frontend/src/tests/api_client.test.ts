import { describe, it, expect } from 'vitest';
import { extractErrorMessage } from '../api/client';

describe('API Client Error Handler', () => {
  it('formats standard Error instance', () => {
    const err = new Error('Connection refused');
    const res = extractErrorMessage(err);
    expect(res.code).toBe('CLIENT_ERROR');
    expect(res.message).toBe('Connection refused');
  });

  it('formats unknown string errors', () => {
    const res = extractErrorMessage('Unexpected token');
    expect(res.code).toBe('UNKNOWN_ERROR');
    expect(res.message).toBe('Unexpected token');
  });
});
