# Main Server Implementation
// server.ts
import { McpServer, HttpServerTransport } from '@modelcontextprotocol/sdk';
import { MethodRegistry } from './core/registry';
import { TaskAnalyzer } from './core/analyzer';
import { ModelTrainer } from './core/trainer';
import { ModelEvaluator } from './core/evaluator';
import { ModelExporter } from './core/exporter';

// Initialize core components
const registry = new MethodRegistry();
const analyzer = new TaskAnalyzer(registry);
const trainer = new ModelTrainer(registry);
const evaluator = new ModelEvaluator();
const exporter = new ModelExporter();

// Create MCP server with all tools
const server = new McpServer({
  name: 'finetunex',
  version: '1.0.0',
  capabilities: {
    tools: {
      // Analysis tools
      analyze_task: {
        description: 'Analyze dataset and task to recommend fine-tuning methods',
        parameters: {
          dataset_path: { type: 'string', description: 'Path to training dataset' },
          task_type: { type: 'string', enum: ['classification', 'generation', 'qa', 'summarization', 'other'] },
          model_name: { type: 'string', description: 'Base model to fine-tune' },
          compute_budget: { type: 'string', enum: ['low', 'medium', 'high'] },
          memory_constraints: { type: 'string', enum: ['severe', 'moderate', 'none'] }
        },
        handler: async (params) => analyzer.analyzeTask(params)
      },
      
      // Method selection tools
      get_method_info: {
        description: 'Get detailed information about a specific fine-tuning method',
        parameters: {
          method_name: { type: 'string', description: 'Name of the fine-tuning method' }
        },
        handler: async (params) => registry.getMethodInfo(params.method_name)
      },
      
      list_available_methods: {
        description: 'List all available fine-tuning methods with filtering options',
        parameters: {
          filter_by: { type: 'string', enum: ['category', 'performance', 'memory_usage', 'all'] },
          filter_value: { type: 'string', description: 'Value to filter by' }
        },
        handler: async (params) => registry.listMethods(params.filter_by, params.filter_value)
      },
      
      // Training tools
      train_model: {
        description: 'Fine-tune a model using specified method',
        parameters: {
          method: { type: 'string', description: 'Fine-tuning method to use' },
          base_model: { type: 'string', description: 'Base model to fine-tune' },
          dataset_path: { type: 'string', description: 'Path to training dataset' },
          eval_path: { type: 'string', description: 'Path to evaluation dataset' },
          output_dir: { type: 'string', description: 'Directory to save outputs' },
          config: { type: 'object', description: 'Method-specific configuration' }
        },
        handler: async (params) => trainer.trainModel(params)
      },
      
      // Method-specific training endpoints (examples)
      train_lora: {
        description: 'Fine-tune model using LoRA',
        parameters: {
          // LoRA-specific parameters
          base_model: { type: 'string' },
          dataset_path: { type: 'string' },
          rank: { type: 'integer', default: 8 },
          alpha: { type: 'integer', default: 16 },
          target_modules: { type: 'array', items: { type: 'string' } },
          // Common training parameters
          batch_size: { type: 'integer', default: 16 },
          learning_rate: { type: 'number', default: 3e-4 },
          num_epochs: { type: 'integer', default: 3 }
        },
        handler: async (params) => trainer.trainWithMethod('lora', params)
      },
      
      train_dora: {
        description: 'Fine-tune model using DoRA',
        parameters: {
          // DoRA-specific parameters alongside common ones
          base_model: { type: 'string' },
          dataset_path: { type: 'string' },
          rank: { type: 'integer', default: 8 },
          alpha: { type: 'integer', default: 16 },
          target_modules: { type: 'array', items: { type: 'string' } },
          // DoRA specific
          regularization_strength: { type: 'number', default: 0.1 },
          // Common parameters
          batch_size: { type: 'integer', default: 16 },
          learning_rate: { type: 'number', default: 3e-4 },
          num_epochs: { type: 'integer', default: 3 }
        },
        handler: async (params) => trainer.trainWithMethod('dora', params)
      },
      
      // Similar endpoints for all 25 methods
      
      // Evaluation tools
      evaluate_model: {
        description: 'Evaluate fine-tuned model on benchmarks',
        parameters: {
          model_path: { type: 'string' },
          benchmark: { type: 'string', enum: ['mmlu', 'hellaswag', 'gsm8k', 'truthfulqa', 'custom'] },
          custom_eval_path: { type: 'string' },
          metrics: { type: 'array', items: { type: 'string' } }
        },
        handler: async (params) => evaluator.evaluateModel(params)
      },
      
      // Exporting and deployment tools
      export_model: {
        description: 'Export fine-tuned model for deployment',
        parameters: {
          model_path: { type: 'string' },
          format: { type: 'string', enum: ['gguf', 'safetensors', 'pytorch', 'onnx'] },
          quantization: { type: 'string', enum: ['none', '8bit', '4bit'] },
          target_device: { type: 'string', enum: ['cpu', 'gpu', 'edge'] }
        },
        handler: async (params) => exporter.exportModel(params)
      },
      
      deploy_model: {
        description: 'Deploy fine-tuned model to inference endpoint',
        parameters: {
          model_path: { type: 'string' },
          platform: { type: 'string', enum: ['huggingface', 'replicate', 'custom'] },
          requirements: { type: 'object' }
        },
        handler: async (params) => exporter.deployModel(params)
      }
    }
  }
});

// Start the server
const port = process.env.PORT || 8000;
const transport = new HttpServerTransport({ port });
server.start(transport).then(() => {
  console.log(`FineTuneX MCP Server running on port ${port}`);
});
