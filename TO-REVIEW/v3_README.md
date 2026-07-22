Our ecosystem consists of two complementary MCP servers that work together to solve different aspects of the fine-tuning challenge:
1. FineTuneX Script Generator
A specialized MCP server that generates optimal fine-tuning scripts and configurations.
2. FineTuneX Compatibility Inspector
A complementary MCP server that performs deep dependency analysis and compatibility verification.

FineTuneX Script Generator MCP
Show Image
Show Image
An intelligent MCP server that connects you to the right fine-tuning approach with perfectly configured scripts.
🎯 Purpose
Generate optimal fine-tuning scripts tailored to your specific:

Model architecture
Training task
Dataset characteristics
Hardware constraints

🧠 How It Works
User Request → Method Selection → Hyperparameter Optimization → Script Generation
The Script Generator MCP delivers:

Python scripts ready for execution
Optimized hyperparameters for your specific scenario
Command-line arguments with sensible defaults
Integration templates for various execution environments

✨ Key Features

Method Selection Engine: Choose from 25 state-of-the-art fine-tuning approaches
Script Customization: Tailor scripts to your exact requirements
Hyperparameter Optimization: Get recommended learning rates, batch sizes, etc.
Export Options: Generate scripts for local execution or cloud deployment

# FineTuneX MCP Ecosystem

Our ecosystem consists of two complementary MCP servers that work together to solve different aspects of the fine-tuning challenge:

## 1. FineTuneX Script Generator

A specialized MCP server that generates optimal fine-tuning scripts and configurations.

## 2. FineTuneX Compatibility Inspector

A complementary MCP server that performs deep dependency analysis and compatibility verification.

---

# FineTuneX Script Generator MCP

[![Model Context Protocol](https://img.shields.io/badge/MCP-Compatible-blue)](https://github.com/anthropics/anthropic-cookbook/tree/main/mcp)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An intelligent MCP server that connects you to the right fine-tuning approach with perfectly configured scripts.

## 🎯 Purpose

Generate optimal fine-tuning scripts tailored to your specific:
- Model architecture
- Training task
- Dataset characteristics
- Hardware constraints

## 🧠 How It Works

```
User Request → Method Selection → Hyperparameter Optimization → Script Generation
```

The Script Generator MCP delivers:
- Python scripts ready for execution
- Optimized hyperparameters for your specific scenario
- Command-line arguments with sensible defaults
- Integration templates for various execution environments

## ✨ Key Features

- **Method Selection Engine**: Choose from 25 state-of-the-art fine-tuning approaches
- **Script Customization**: Tailor scripts to your exact requirements
- **Hyperparameter Optimization**: Get recommended learning rates, batch sizes, etc.
- **Export Options**: Generate scripts for local execution or cloud deployment

## 🛠️ Usage

```typescript
// Connect to the MCP server
const scriptGenerator = new McpClient('http://localhost:8000');

// Get a fine-tuning script
const script = await scriptGenerator.invoke('generate_script', {
  method: 'qlora',
  model: 'llama-2-7b',
  task: 'instruction-following',
  dataset_format: 'alpaca',
  hardware: {
    gpu_memory: '24GB',
    architecture: 'nvidia'
  }
});

// Save the script
fs.writeFileSync('train_script.py', script.code);
console.log('Recommended hyperparameters:', script.hyperparameters);
```

## 💰 Pricing

- **Starter**: $9/month - 50 script generations, 5 methods
- **Pro**: $29/month - Unlimited scripts, all 25 methods, priority support
- **Enterprise**: Custom pricing - Custom method integration, private deployment

---

# FineTuneX Compatibility Inspector MCP

[![Model Context Protocol](https://img.shields.io/badge/MCP-Compatible-blue)](https://github.com/anthropics/anthropic-cookbook/tree/main/mcp)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An MCP server that ensures all dependencies, versions, and configurations work seamlessly together in your fine-tuning environment.

## 🎯 Purpose

Eliminate compatibility headaches by validating:
- Python package versions
- CUDA/ROCm requirements
- Hardware compatibility
- Framework-specific dependencies
- Conflicting requirements

## 🧠 How It Works

```
Script + Environment Specs → Deep Dependency Analysis → Compatibility Report → Conflict Resolution
```

The Compatibility Inspector MCP analyzes:
- Your fine-tuning script
- Your execution environment
- Package compatibility matrices
- Hardware/software requirements

## ✨ Key Features

- **Dependency Resolution**: Identifies and resolves package conflicts
- **Environment Validation**: Ensures your execution environment is properly configured
- **Compatibility Matrices**: Maintains databases of known-good configurations
- **Upgrade Pathways**: Suggests minimal changes to resolve conflicts
- **Docker Generation**: Creates containerized environments with guaranteed compatibility

## 🛠️ Usage

```typescript
// Connect to the MCP server
const compatibilityInspector = new McpClient('http://localhost:8001');

// Check compatibility
const report = await compatibilityInspector.invoke('check_compatibility', {
  script_path: './train_qlora.py',
  environment: {
    python_version: '3.10',
    cuda_version: '11.8',
    hardware: 'nvidia-a100'
  },
  packages: {
    'transformers': '4.34.0',
    'torch': '2.0.1'
  }
});

// Review issues
if (report.has_conflicts) {
  console.log('Conflicts detected:', report.conflicts);
  console.log('Suggested resolution:', report.resolution);
  
  // Generate fixed requirements
  const fixedReqs = await compatibilityInspector.invoke('generate_requirements', {
    resolve_conflicts: true,
    base_requirements: './requirements.txt',
    compatibility_report: report.id
  });
  
  fs.writeFileSync('requirements.txt', fixedReqs.content);
}
```

## 💰 Pricing

- **Starter**: $7/month - 20 compatibility checks, basic resolution
- **Pro**: $19/month - Unlimited checks, advanced resolution, Docker generation
- **Enterprise**: Custom pricing - Private deployments, custom compatibility matrices

---

## 🔄 Complementary Services

While these MCPs can be used independently, they provide maximum value when used together:

1. **Script Generator** creates the optimal fine-tuning approach
2. **Compatibility Inspector** ensures it runs smoothly in your environment

Together, they form a complete solution for the two most challenging aspects of LLM fine-tuning:
- Finding the right approach with optimal parameters
- Ensuring everything works together without conflicts

## 🚀 Deployment Options

Both MCPs can be deployed via:

- **Docker**: Containerized deployments for local or cloud environments
- **Fly.io**: Edge deployments for low-latency access
- **Railway**: One-click deployments with simple scaling
- **Self-hosted**: Run on your own infrastructure for maximum control

## 🌐 Integration Examples

### Complete Fine-Tuning Workflow

```typescript
// Generate script
const script = await scriptGenerator.invoke('generate_script', { 
  method: 'lora',
  /* ... other parameters ... */
});

// Check compatibility
const report = await compatibilityInspector.invoke('check_compatibility', {
  script_content: script.code,
  environment: { /* environment details */ }
});

// Resolve conflicts
if (report.has_conflicts) {
  const fixedEnvironment = await compatibilityInspector.invoke('fix_environment', {
    compatibility_report: report.id
  });
  
  console.log('Updated requirements:', fixedEnvironment.requirements);
  console.log('Environment setup commands:', fixedEnvironment.setup_commands);
}

// Ready to execute!
```

## 📚 Documentation

- [Full Documentation](https://docs.finetunex.com)
- [Script Generator API Reference](https://docs.finetunex.com/script-generator)
- [Compatibility Inspector API Reference](https://docs.finetunex.com/compatibility-inspector)

## 🤝 Support

Need help? Contact us:
- Email: support@finetunex.com
- Discord: [FineTuneX Community](https://discord.gg/finetunex)
- GitHub Issues: [Report bugs](https://github.com/finetunex/mcp-servers/issues)

## 📄 License

Both MCP servers are available under the MIT License.
