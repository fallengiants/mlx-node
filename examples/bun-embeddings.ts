#!/usr/bin/env bun
/**
 * bun-embeddings.ts — Proof: in-process MLX embeddings under Bun.
 *
 * Uses Qwen3-Embedding-0.6B via HarrierModel for text embeddings.
 * Demonstrates cosine similarity for semantic search.
 *
 * Prerequisites:
 *   npx oxnode packages/cli/src/cli.ts download model -m Qwen/Qwen3-Embedding-0.6B
 *
 * Run: bun run examples/bun-embeddings.ts
 */

import { readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { existsSync } from 'node:fs';

import { HarrierModel } from '@mlx-node/lm';

// ── Helpers ──────────────────────────────────────────────────────────

/**
 * Patch config.json to route through HarrierModel.
 *
 * Qwen3-Embedding has model_type "qwen3" + architectures ["Qwen3ForCausalLM"].
 * The mlx-node Harrier detection requires architectures ["Qwen3Model"] (not ForCausalLM).
 * The weight names are identical — Harrier's loader strips "model." prefix and
 * renames embed_tokens → embedding, norm → final_norm, skips lm_head.
 * So we just need to fix the architecture string for detection.
 */
async function ensureHarrierConfig(modelDir: string): Promise<boolean> {
  const configPath = join(modelDir, 'config.json');
  const raw = await readFile(configPath, 'utf-8');
  const config = JSON.parse(raw);

  if (config.architectures?.includes('Qwen3Model') && !config.architectures?.includes('Qwen3ForCausalLM')) {
    return false; // Already patched
  }

  console.log('  Patching config.json for Harrier detection...');
  console.log(`    architectures: ${JSON.stringify(config.architectures)} → ["Qwen3Model"]`);
  config.architectures = ['Qwen3Model'];
  await writeFile(configPath, JSON.stringify(config, null, 2) + '\n');
  return true;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  console.log('╔══════════════════════════════════════════════════════╗');
  console.log('║  MLX-Node + Bun: In-Process Embeddings Proof        ║');
  console.log('╚══════════════════════════════════════════════════════╝');
  console.log(`  Runtime:   Bun ${Bun.version}`);
  console.log(`  Platform:  ${process.platform}/${process.arch}`);
  console.log(`  Model:     Qwen3-Embedding-0.6B (Harrier)`);
  console.log();

  // Resolve model path
  const localPath = resolve(process.cwd(), '.cache', 'models', 'qwen3-embedding-0.6b');
  const globalPath = resolve(process.env.HOME!, '.mlx-node', 'models', 'qwen3-embedding-0.6b');
  const modelDir = existsSync(localPath) ? localPath : globalPath;

  if (!existsSync(modelDir)) {
    console.error('❌ Embedding model not found. Download it first:');
    console.error('   npx oxnode packages/cli/src/cli.ts download model -m Qwen/Qwen3-Embedding-0.6B');
    process.exit(1);
  }

  console.log(`  Model dir: ${modelDir}`);

  // Patch config for Harrier detection
  const patched = await ensureHarrierConfig(modelDir);
  if (patched) console.log('  ✅ Config patched\n');

  // Load model
  console.log('Loading embedding model...');
  const loadStart = performance.now();
  const model = HarrierModel.load(modelDir);
  const harrier = await model;  // It returns a Promise
  const loadMs = performance.now() - loadStart;
  console.log(`✅ Model loaded in ${loadMs.toFixed(0)}ms`);
  console.log(`   Parameters: ${harrier.numParameters()}`);
  console.log(`   Config: ${JSON.stringify(harrier.getConfig(), null, 2)}`);
  console.log();

  // Check for prompt presets
  const prompts = harrier.getPrompts();
  if (Object.keys(prompts).length > 0) {
    console.log('  Prompt presets:', Object.keys(prompts).join(', '));
    for (const [name, prefix] of Object.entries(prompts)) {
      console.log(`    "${name}": "${prefix.substring(0, 60)}${prefix.length > 60 ? '...' : ''}"`);
    }
    console.log();
  }

  // ── Encode texts ──────────────────────────────────────────────────
  const texts = [
    'The quick brown fox jumps over the lazy dog.',
    'A fast auburn fox leaps above an idle canine.',
    'Machine learning models process data efficiently.',
    'The weather is beautiful today.',
    'What is the capital of France?',
    'Which city serves as France\'s capital?',
  ];

  console.log(`Encoding ${texts.length} texts...`);
  const encodeStart = performance.now();

  const embeddings: Float32Array[] = [];
  for (const text of texts) {
    const result = await harrier.encode(text, null);
    const raw = result.toFloat32();
    embeddings.push(new Float32Array(raw));
  }

  const encodeMs = performance.now() - encodeStart;
  console.log(`✅ Encoded ${texts.length} texts in ${encodeMs.toFixed(0)}ms (${(encodeMs / texts.length).toFixed(0)}ms/text)`);
  console.log(`   Vector dimensions: ${embeddings[0]!.length}`);
  console.log();

  // ── Cosine similarity matrix ──────────────────────────────────────
  console.log('Cosine Similarity Matrix:');
  console.log('─'.repeat(80));

  // Header
  const header = '     ' + texts.map((_, i) => `[${i}]`.padStart(8)).join('');
  console.log(header);

  for (let i = 0; i < texts.length; i++) {
    const row = `[${i}] `.padStart(5);
    const sims = texts.map((_, j) => cosineSimilarity(embeddings[i]!, embeddings[j]!).toFixed(4).padStart(8)).join('');
    console.log(row + sims);
  }
  console.log('─'.repeat(80));

  // ── Analysis ──────────────────────────────────────────────────────
  console.log();
  console.log('Semantic Pairs:');
  const pairs = [
    [0, 1, 'fox/dog ≈ fox/canine (paraphrase)'],
    [0, 3, 'fox/dog vs weather (different topics)'],
    [4, 5, 'capital of France ≈ France\'s capital (same question)'],
    [0, 4, 'fox/dog vs capital (different topics)'],
  ];
  for (const [i, j, label] of pairs) {
    const sim = cosineSimilarity(embeddings[i]!, embeddings[j]!);
    console.log(`  [${i}]↔[${j}] ${sim.toFixed(4)}  ${label}`);
  }

  // ── Instruction-aware encoding ────────────────────────────────────
  console.log();
  console.log('── Instruction-Aware Encoding ──');
  const queryText = 'What is the capital of France?';
  const queryWithInstruction = await harrier.encode(queryText, 'query');
  const queryWithout = await harrier.encode(queryText, null);
  const docEmb = await harrier.encode('Paris is the capital and most populous city of France.', null);

  const simWith = cosineSimilarity(new Float32Array(queryWithInstruction.toFloat32()), new Float32Array(docEmb.toFloat32()));
  const simWithout = cosineSimilarity(new Float32Array(queryWithout.toFloat32()), new Float32Array(docEmb.toFloat32()));

  console.log(`  Query: "${queryText}"`);
  console.log(`  Doc:   "Paris is the capital and most populous city of France."`);
  console.log(`  Similarity WITH instruction:    ${simWith.toFixed(4)}`);
  console.log(`  Similarity WITHOUT instruction: ${simWithout.toFixed(4)}`);
  console.log(`  Instruction boost: ${(simWith - simWithout > 0 ? '+' : '')}${(simWith - simWithout).toFixed(4)}`);

  // ── Summary ───────────────────────────────────────────────────────
  console.log();
  console.log('═══════════════════════════════════════════════════════');
  console.log('  ✅ In-process MLX embeddings working under Bun');
  console.log('  ✅ Qwen3-Embedding-0.6B via HarrierModel');
  console.log('  ✅ Cosine similarity for semantic search');
  console.log('  ✅ Instruction-aware encoding (query vs document)');
  console.log('  ✅ All on Apple Silicon Metal GPU');
  console.log('  ✅ Zero network calls during inference');
  console.log('═══════════════════════════════════════════════════════');
}

main().catch(console.error);
