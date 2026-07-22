# Training Orchestration
// core/trainer.ts
import { MethodRegistry } from './registry';
import { GPUManager } from '../utils/gpu_manager';
import * as fs from 'fs';
import * as path from 'path';

export class ModelTrainer {
  private registry: MethodRegistry;
  private gpuManager: GPUManager;
  
  constructor(registry: MethodRegistry) {
    this.registry = registry;
    this.gpuManager = new GPUManager();
  }
  
  async trainModel(params: any) {
    const { method, ...trainingParams } = params;
    return this.trainWithMethod(method, trainingParams);
  }
  
  async trainWithMethod(methodName: string, params: any) {
    // Get method implementation
    const method = this.registry.getMethod(methodName);
    
    // Check resource requirements
    const requirements = await method.getResourceRequirements(params);
    const resourceAvailable = await this.gpuManager.checkAvailability(requirements);
    
    if (!resourceAvailable.available) {
      return {
        status: 'error',
        error: 'Insufficient resources',
        required: requirements,
        available: resourceAvailable.resources
      };
    }
    
    // Reserve GPU resources
    const gpuAllocation = await this.gpuManager.allocateResources(requirements);
    
    try {
      // Set environment variables for GPU allocation
      process.env.CUDA_VISIBLE_DEVICES = gpuAllocation.gpus.join(',');
      
      // Create training run directory
      const runId = `${methodName}_${Date.now()}`;
      const runDir = path.join(params.output_dir, runId);
      fs.mkdirSync(runDir, { recursive: true });
      
      // Save training parameters
      fs.writeFileSync(
        path.join(runDir, 'params.json'),
        JSON.stringify({ method: methodName, ...params }, null, 2)
      );
      
      // Execute training
      const trainingResult = await method.train({
        ...params,
        output_dir: runDir
      });
      
      return {
        status: trainingResult.status,
        run_id: runId,
        output_dir: runDir,
        model_path: trainingResult.model_path,
        metrics: trainingResult.metrics,
        logs_path: path.join(runDir, 'logs.txt')
      };
    } catch (error) {
      return {
        status: 'error',
        error: error.message
      };
    } finally {
      // Release GPU resources
      await this.gpuManager.releaseResources(gpuAllocation.id);
    }
  }
}
