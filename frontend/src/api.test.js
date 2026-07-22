import test from 'node:test';
import assert from 'node:assert/strict';

import { requestJson } from './api.js';

test('requestJson returns JSON from a successful response', async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/model-card');
    assert.equal(options?.method, undefined);
    return {
      ok: true,
      json: async () => ({ currency: 'INR' }),
    };
  };

  try {
    assert.deepEqual(await requestJson('/api/model-card'), { currency: 'INR' });
  } finally {
    globalThis.fetch = previousFetch;
  }
});

test('requestJson surfaces backend detail for failed responses', async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: false,
    status: 503,
    json: async () => ({ detail: 'model unavailable' }),
  });

  try {
    await assert.rejects(
      () => requestJson('/api/predict'),
      error => error.message === 'model unavailable',
    );
  } finally {
    globalThis.fetch = previousFetch;
  }
});
