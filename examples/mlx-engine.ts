#!/usr/bin/env bun
/**
 * MLXEngine — In-process MLX inference for Bun on Apple Silicon.
 *
 * Demonstrates the target API: text generation and embeddings running
 * entirely inside a Bun process, accelerated by Metal GPU on Apple Silicon.
 * No API calls, no network dependencies during inference.
 *
 * Usage:
 *   bun run examples/mlx-engine.ts
 */

import { resolve } from 'node:path';

// ── Native + TypeScript imports ──────────────────────────────────────
// These import from the workspace packages. In a published npm package,
// these would be `import { ... } from "mlx-node"`.
import {
  type ChatResult,
  type PerformanceMetrics,
  type SessionCapableModel,
  ChatSession,
  HarrierModel,
  loadModel,
} from '@mlx-node/lm';

// ── Types ────────────────────────────────────────────────────────────

export interface GenerateOptions {
  system?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface EmbeddingResult {
  vector: Float32Array;
  dimensions: number;
}

// ── MLXEngine ────────────────────────────────────────────────────────

/**
 * In-process MLX inference engine for Bun on Apple Silicon.
 *
 * Loads models from the local `.cache/models/` directory (populated by
 * `mlx download model`) and runs inference entirely on-device using
 * Metal GPU acceleration via the MLX framework.
 *
 * @example
 * ```typescript
 * const engine = new MLXEngine("qwen3-0.6b");
 * const output = await engine.generate("Hello, how are you?");
 * console.log(output);
 *
 * const embedding = await engine.embed("some text");
 * console.log(embedding.dimensions); // 1024
 * ```
 */
export class MLXEngine {
  private model: Awaited<ReturnType<typeof loadModel>> | null = null;
  private session: ChatSession<SessionCapableModel> | null = null;
  private readonly modelPath: string;
  private readonly isEmbedding: boolean;

  /**
   * @param modelName Model directory name or absolute path.
   *   Resolution order:
   *   1. If absolute path, use directly
   *   2. Check .cache/models/ relative to cwd
   *   3. Check ~/.mlx-node/models/
   */
  constructor(
    modelName: string,
    modelsDir?: string,
  ) {
    if (modelsDir) {
      this.modelPath = resolve(modelsDir, modelName);
    } else if (modelName.startsWith('/') || modelName.startsWith('~')) {
      this.modelPath = modelName.replace(/^~/, process.env.HOME ?? '~');
    } else {
      // Check both locations
      const localPath = resolve(process.cwd(), '.cache', 'models', modelName);
      const globalPath = resolve(process.env.HOME ?? '/root', '.mlx-node', 'models', modelName);
      // Prefer local, fall back to global
      const { existsSync } = require('node:fs') as typeof import('node:fs');
      this.modelPath = existsSync(localPath) ? localPath : globalPath;
    }
    this.isEmbedding = false;
  }

  /**
   * Load the model into memory. Call once before inference.
   * Subsequent calls are no-ops if already loaded.
   */
  async load(): Promise<void> {
    if (this.model) return;
    const t0 = performance.now();
    this.model = await loadModel(this.modelPath);
    const elapsed = performance.now() - t0;
    const isEmbedding = this.model instanceof HarrierModel;

    if (!isEmbedding) {
      this.session = new ChatSession(this.model as unknown as SessionCapableModel, {
        system: 'You are a helpful assistant. Be concise.',
      });
    }

    console.log(
      `[MLXEngine] Loaded ${isEmbedding ? 'embedding' : 'generation'} model ` +
        `in ${elapsed.toFixed(0)}ms from ${this.modelPath}`,
    );
  }

  /**
   * Generate text from a prompt. Streaming internally, returns full text.
   */
  async generate(prompt: string, options?: GenerateOptions): Promise<string> {
    await this.load();
    if (!this.session) {
      throw new Error('This model does not support text generation (it is an embedding model). Use embed() instead.');
    }

    // Override system prompt if provided
    if (options?.system) {
      this.session = new ChatSession(this.model as unknown as SessionCapableModel, {
        system: options.system,
      });
    }

    let fullText = '';
    for await (const event of this.session.sendStream(prompt, {
      config: {
        maxNewTokens: options?.maxTokens ?? 512,
        temperature: options?.temperature ?? 0.7,
        reportPerformance: true,
      },
    })) {
      if (event.done) {
        return event.rawText;
      }
      fullText += event.text;
    }
    return fullText;
  }

  /**
   * Generate text with streaming. Yields text chunks as they arrive.
   */
  async *generateStream(
    prompt: string,
    options?: GenerateOptions,
  ): AsyncGenerator<{ text: string; done: boolean; performance?: PerformanceMetrics }> {
    await this.load();
    if (!this.session) {
      throw new Error('This model does not support text generation (it is an embedding model). Use embed() instead.');
    }

    for await (const event of this.session.sendStream(prompt, {
      config: {
        maxNewTokens: options?.maxTokens ?? 512,
        temperature: options?.temperature ?? 0.7,
        reportPerformance: true,
      },
    })) {
      yield event;
    }
  }

  /**
   * Embed text into a vector. Requires an embedding model (Harrier/Qwen3-Embedding).
   */
  async embed(text: string): Promise<EmbeddingResult> {
    await this.load();
    if (!(this.model instanceof HarrierModel)) {
      throw new Error('This model does not support embeddings (it is a generation model). Use generate() instead.');
    }

    const result = await this.model.encode(text, null);
    const raw = result.toFloat32();
    return {
      vector: new Float32Array(raw),
      dimensions: raw.length,
    };
  }

  /**
   * Embed multiple texts. Returns one vector per text.
   */
  async embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
    await this.load();
    if (!(this.model instanceof HarrierModel)) {
      throw new Error('This model does not support embeddings (it is a generation model). Use generate() instead.');
    }

    const batchResult = await this.model.encodeBatch(texts, null);
    const raw = batchResult.toFloat32();
    const hiddenSize = raw.length / texts.length;

    return texts.map((_, i) => ({
      vector: new Float32Array(raw.slice(i * hiddenSize, (i + 1) * hiddenSize)),
      dimensions: hiddenSize,
    }));
  }

  /**
   * Compute cosine similarity between two embedding vectors.
   */
  static cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
    }
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i]! * b[i]!;
      normA += a[i]! * a[i]!;
      normB += b[i]! * b[i]!;
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

// ── Demo / Proof-of-Concept ──────────────────────────────────────────

async function main() {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  MLXEngine — In-Process Inference for Bun');
  console.log('  Platform: Bun ' + Bun.version + ' on ' + process.platform + '/' + process.arch);
  console.log('═══════════════════════════════════════════════════════════\n');

  // ── Phase 1: Text Generation ─────────────────────────────────────
  console.log('── Phase 1: Text Generation (Qwen3-0.6B) ──\n');
  const genEngine = new MLXEngine('qwen3-0.6b');

  const t0 = performance.now();
  const output = await genEngine.generate('What is the meaning of life? Answer in two sentences.');
  const genTime = performance.now() - t0;

  console.log(`Response: ${output}`);
  console.log(`\nGeneration time: ${genTime.toFixed(0)}ms\n`);

  // ── Phase 2: Streaming Generation ────────────────────────────────
  console.log('── Phase 2: Streaming Generation ──\n');
  process.stdout.write('Stream: ');
  let tokenCount = 0;
  let perf: PerformanceMetrics | undefined;
  for await (const chunk of genEngine.generateStream('Count from 1 to 10.')) {
    if (chunk.done) {
      perf = chunk.performance;
    } else {
      process.stdout.write(chunk.text);
      tokenCount++;
    }
  }
  process.stdout.write('\n');
  if (perf) {
    console.log(
      `  ${tokenCount} chunks | TTFT ${perf.ttftMs.toFixed(0)}ms | ` +
        `Decode ${perf.decodeTokensPerSecond.toFixed(1)} tok/s`,
    );
  }

  // ── Phase 3: Embeddings ──────────────────────────────────────────
  console.log('\n── Phase 3: Embeddings (if model available) ──\n');

  let embedEngine: MLXEngine;
  try {
    embedEngine = new MLXEngine('qwen3-embedding-0.6b');
    await embedEngine.load();
  } catch (e) {
    console.log('Embedding model not found or failed to load.');
    console.log('Download it with:');
    console.log('  npx oxnode packages/cli/src/cli.ts download model -m Qwen/Qwen3-Embedding-0.6B');
    console.log(`Error: ${e instanceof Error ? e.message : String(e)}`);
    console.log('Skipping embedding demo.\n');
    return;
  }

  const texts = [
    'The quick brown fox jumps over the lazy dog.',
    'A fast auburn fox leaps above an idle canine.',
    'The weather is beautiful today.',
  ];

  const embedStart = performance.now();
  const embeddings = await embedEngine.embedBatch(texts);
  const embedTime = performance.now() - embedStart;

  console.log(`Embedded ${texts.length} texts in ${embedTime.toFixed(0)}ms`);
  console.log(`Vector dimensions: ${embeddings[0]!.dimensions}\n`);

  // Cosine similarity matrix
  console.log('Cosine similarity matrix:');
  console.log('─'.repeat(60));
  for (let i = 0; i < texts.length; i++) {
    const sims: string[] = [];
    for (let j = 0; j < texts.length; j++) {
      const sim = MLXEngine.cosineSimilarity(embeddings[i]!.vector, embeddings[j]!.vector);
      sims.push(sim.toFixed(4));
    }
    console.log(`  [${i}] ${sims.join('  ')}`);
  }
  console.log('─'.repeat(60));
  console.log(`\n  [0] vs [1] (similar meaning): ${MLXEngine.cosineSimilarity(embeddings[0]!.vector, embeddings[1]!.vector).toFixed(4)}`);
  console.log(`  [0] vs [2] (different meaning): ${MLXEngine.cosineSimilarity(embeddings[0]!.vector, embeddings[2]!.vector).toFixed(4)}`);

  // ── Summary ──────────────────────────────────────────────────────
  console.log('\n═══════════════════════════════════════════════════════════');
  console.log('  ✅ In-process MLX inference working under Bun');
  console.log('  ✅ Text generation via Qwen3-0.6B');
  console.log('  ✅ Embeddings via Qwen3-Embedding-0.6B');
  console.log('  ✅ Cosine similarity for semantic search');
  console.log('  ✅ All on Apple Silicon Metal GPU');
  console.log('  ✅ Zero network calls during inference');
  console.log('═══════════════════════════════════════════════════════════');
}

main().catch(console.error);
