System Architecture
├── server.ts                  # Main MCP server entry point
├── core/
│   ├── registry.ts            # Method registry and factory
│   ├── analyzer.ts            # Task & dataset analyzer
│   ├── trainer.ts             # Training orchestration
│   ├── evaluator.ts           # Model evaluation suite
│   └── exporter.ts            # Model export utilities
├── methods/                   # Implementation of all 25 methods
│   ├── low_rank/              # LoRA, DoRA, ReFT, etc.
│   ├── architectural/         # Adapter methods
│   ├── quantization/          # QLoRA, LoftQ, etc.
│   ├── moe/                   # MoRA, MoELoRA, etc.
│   └── specialized/           # Domain-specific methods
├── utils/
│   ├── metrics.ts             # Performance metrics
│   ├── optimization.ts        # Training optimizations
│   ├── data_processing.ts     # Dataset handling
│   └── gpu_manager.ts         # GPU resource allocation
└── configs/
    ├── method_configs/        # Default configs for methods
    ├── deployment.ts          # Deployment configurations
    └── benchmarks.ts          # Benchmark definitions
