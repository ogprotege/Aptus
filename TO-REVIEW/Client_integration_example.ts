// Example of using the MCP client with FineTuneX
import { McpClient } from '@modelcontextprotocol/sdk';

async function fineTuneModel() {
  // Connect to FineTuneX MCP server
  const client = new McpClient('http://localhost:8000');
  
  // Analyze task and get method recommendations
  const analysisResult = await client.invoke('analyze_task', {
    dataset_path: '/path/to/my/dataset',
    task_type: 'summarization',
    model_name: 'llama2-7b',
    compute_budget: 'medium',
    memory_constraints: 'moderate'
  });
  
  console.log('Recommended methods:', analysisResult.recommended_methods);
  
  // Use the top recommended method
  const topMethod = analysisResult.recommended_methods[0].name;
  
  // Start training
  const trainingResult = await client.invoke('train_model', {
    method: topMethod,
    base_model: 'llama2-7b',
    dataset_path: '/path/to/my/dataset',
    eval_path: '/path/to/eval/data',
    output_dir: './outputs',
    config: {
      learning_rate: 3e-4,
      batch_size: 16,
      num_epochs: 3
    }
  });
  
  console.log('Training complete:', trainingResult);
  
  // Evaluate trained model
  const evalResult = await client.invoke('evaluate_model', {
    model_path: trainingResult.model_path,
    benchmark: 'mmlu',
    metrics: ['accuracy', 'f1']
  });
  
  console.log('Evaluation results:', evalResult);
  
  // Export for deployment
  const exportResult = await client.invoke('export_model', {
    model_path: trainingResult.model_path,
    format: 'gguf',
    quantization: '4bit',
    target_device: 'cpu'
  });
  
  console.log('Model exported:', exportResult);
}

fineTuneModel().catch(console.error);
