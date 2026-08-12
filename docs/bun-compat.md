# MLX-Node + Bun: In-Process MLX Inference

## What is this?

**MLX-Node** brings Apple's [MLX](https://github.com/mlx-swift/mlx-swift) framework to JavaScript/TypeScript via a Rust/NAPI bridge. It supports inference (Qwen3, Qwen3.5, Gemma4, LFM2), training (GRPO, SFT), vision-language models, document processing, and embeddings — all running locally on Apple Silicon with Metal GPU acceleration.

**This fork** proves that the entire stack works under **Bun** with zero code changes. The NAPI-RS v3 native addon loads natively in both Node.js and Bun.

## Quick Start (Bun)

```typescript
import { MLXEngine } from "./examples/mlx-engine";

// Text generation
const engine = new MLXEngine("qwen3-0.6b");
const output = await engine.generate("Hello, how are you?");
console.log(output);

// Embeddings (requires Qwen3-Embedding-0.6B)
const embedEngine = new MLXEngine("qwen3-embedding-0.6b");
const result = await embedEngine.embed("some text");
console.log(result.dimensions); // 1024

// Cosine similarity
const a = await embedEngine.embed("quick brown fox");
const b = await embedEngine.embed("fast auburn fox");
console.log(MLXEngine.cosineSimilarity(a.vector, b.vector)); // ~0.9+
```

## Benchmarks (Apple M3 Ultra, Bun 1.3.14)

| Metric | Qwen3-0.6B |
|--------|-----------|
| Model load | 481ms |
| TTFT (first turn) | 51ms |
| Prefill | 680 tok/s |
| Decode | 178 tok/s |
| Cache reuse TTFT | 155ms |
| Memory | ~1.5 GB |

All numbers are in-process, Metal GPU accelerated, zero network calls.

## Installation

```bash
# Clone
git clone https://github.com/fallengiants/mlx-node.git
cd mlx-node

# Install dependencies
bun install  # or yarn install

# Build native addon (requires Apple Silicon + Xcode CLI tools)
bun run build:native

# Build TypeScript
bun run build:ts

# Download a model
npx oxnode packages/cli/src/cli.ts download model -m Qwen/Qwen3-0.6B

# Run the proof-of-concept
bun run examples/bun-inference.ts
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Your Bun process                                     │
│  ┌──────────────────────────────────────────────────┐ │
│  │  TypeScript: @mlx-node/lm                        │ │
│  │  ChatSession, loadModel, streaming               │ │
│  ├──────────────────────────────────────────────────┤ │
│  │  Native: @mlx-node/core (NAPI-RS v3)            │ │
│  │  mlx-core.darwin-arm64.node                      │ │
│  ├──────────────────────────────────────────────────┤ │
│  │  Rust: mlx-core, mlx-sys, mlx-paged-attn        │ │
│  ├──────────────────────────────────────────────────┤ │
│  │  C++ bridge → MLX → Metal GPU                    │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Why Bun?

- **Faster startup**: Bun's runtime starts ~2x faster than Node.js
- **Native NAPI compatibility**: Bun supports Node-API natively
- **TypeScript-first**: Run `.ts` files directly without compilation
- **Shell integration**: `Bun.$` for shell commands, `bun:sqlite` for databases
- **Single binary**: No separate node_modules structure needed

## API

### `MLXEngine`

```typescript
class MLXEngine {
  constructor(modelName: string, modelsDir?: string);
  load(): Promise<void>;
  generate(prompt: string, options?: GenerateOptions): Promise<string>;
  generateStream(prompt: string, options?: GenerateOptions): AsyncGenerator<...>;
  embed(text: string): Promise<EmbeddingResult>;
  embedBatch(texts: string[]): Promise<EmbeddingResult[]>;
  static cosineSimilarity(a: Float32Array, b: Float32Array): number;
}
```

### Direct API (lower level)

```typescript
import { loadModel, ChatSession, HarrierModel } from '@mlx-node/lm';

// Text generation
const model = await loadModel('.cache/models/qwen3-0.6b');
const session = new ChatSession(model);
for await (const event of session.sendStream('Hello!')) {
  process.stdout.write(event.text);
}

// Embeddings
const harrier = await HarrierModel.load('.cache/models/qwen3-embedding-0.6b');
const embedding = await harrier.encode('text to embed', 'query');
```

## Supported Models

| Family | Model Type | Chat | Embedding | Training |
|--------|-----------|------|-----------|----------|
| Qwen3 | qwen3 | ✅ | ✅ (Harrier) | ✅ (GRPO/SFT) |
| Qwen3.5 | qwen3_5 | ✅ | - | ✅ |
| Qwen3.5 MoE | qwen3_5_moe | ✅ | - | ✅ |
| Gemma4 | gemma4 | ✅ | - | - |
| LFM2 | lfm2 | ✅ | - | - |
| Harrier | harrier | - | ✅ | - |

## The Rainline Connection

This is the **highest-leverage derisk** in the Rainline system:

- **Companion-tier agents** run entirely on-device (no API dependency)
- **Wake DSL's cosine matcher** uses local embeddings
- **Talent-recall** uses local embeddings
- **API outage?** Local fallback
- **Cost spike?** Companion tier is free
- **Privacy-sensitive?** Everything stays on-device

## License

MIT
