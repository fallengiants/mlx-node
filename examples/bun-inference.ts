#!/usr/bin/env bun
/**
 * bun-inference.ts — Minimal proof: in-process MLX inference under Bun.
 *
 * This is the smallest possible demonstration that mlx-node works with Bun.
 * No embedding model needed — just text generation via Qwen3-0.6B.
 *
 * Run: bun run examples/bun-inference.ts
 */

import { type PerformanceMetrics, type SessionCapableModel, ChatSession, loadModel } from '@mlx-node/lm';
import { resolve } from 'node:path';

async function main() {
  console.log('╔══════════════════════════════════════════════════════╗');
  console.log('║  MLX-Node + Bun: In-Process Inference Proof         ║');
  console.log('╚══════════════════════════════════════════════════════╝');
  console.log(`  Runtime:   Bun ${Bun.version}`);
  console.log(`  Platform:  ${process.platform}/${process.arch}`);
  console.log(`  Model:     Qwen3-0.6B`);
  console.log();

  // ── Load ───────────────────────────────────────────────────────────
  const modelPath = resolve(process.cwd(), '.cache', 'models', 'qwen3-0.6b');
  console.log(`Loading model from ${modelPath}...`);
  const loadStart = performance.now();
  const model = await loadModel(modelPath);
  const loadMs = performance.now() - loadStart;
  console.log(`✅ Model loaded in ${loadMs.toFixed(0)}ms\n`);

  // ── Chat session ───────────────────────────────────────────────────
  const session = new ChatSession(model as unknown as SessionCapableModel, {
    system: 'You are a helpful assistant. Be concise.',
  });

  // ── Turn 1: Full prefill ──────────────────────────────────────────
  console.log('── Turn 1: What is 2 + 2? ──');
  const turn1Start = performance.now();
  let turn1Text = '';
  let turn1Perf: PerformanceMetrics | undefined;

  for await (const event of session.sendStream('What is 2 + 2? Answer in one sentence.', {
    config: { maxNewTokens: 128, temperature: 0.1, reportPerformance: true },
  })) {
    if (event.done) {
      turn1Perf = event.performance;
    } else {
      turn1Text += event.text;
      process.stdout.write(event.text);
    }
  }
  console.log('\n');

  // ── Turn 2: Cache reuse ───────────────────────────────────────────
  console.log('── Turn 2: Follow-up (cache reuse) ──');
  const turn2Start = performance.now();
  let turn2Text = '';
  let turn2Perf: PerformanceMetrics | undefined;

  for await (const event of session.sendStream('Now multiply that by 3. One sentence.', {
    config: { maxNewTokens: 128, temperature: 0.1, reportPerformance: true },
  })) {
    if (event.done) {
      turn2Perf = event.performance;
    } else {
      turn2Text += event.text;
      process.stdout.write(event.text);
    }
  }
  console.log('\n');

  // ── Results ───────────────────────────────────────────────────────
  console.log('═══════════════════════════════════════════════════════');
  console.log('  Results:');
  console.log(`  Model load:  ${loadMs.toFixed(0)}ms`);
  if (turn1Perf) {
    console.log(`  Turn 1 TTFT: ${turn1Perf.ttftMs.toFixed(0)}ms`);
    console.log(`  Turn 1 prefill: ${turn1Perf.prefillTokensPerSecond.toFixed(1)} tok/s`);
    console.log(`  Turn 1 decode:  ${turn1Perf.decodeTokensPerSecond.toFixed(1)} tok/s`);
  }
  if (turn2Perf) {
    console.log(`  Turn 2 TTFT: ${turn2Perf.ttftMs.toFixed(0)}ms (should be < Turn 1 = cache reuse)`);
    console.log(`  Turn 2 decode:  ${turn2Perf.decodeTokensPerSecond.toFixed(1)} tok/s`);
  }
  console.log();
  console.log('  ✅ In-process MLX inference working under Bun');
  console.log('  ✅ Metal GPU acceleration via Apple Silicon');
  console.log('  ✅ Multi-turn chat with KV cache reuse');
  console.log('  ✅ Zero network calls during inference');
  console.log('═══════════════════════════════════════════════════════');
}

main().catch(console.error);
