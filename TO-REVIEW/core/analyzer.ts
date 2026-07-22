# Task Analysis Module
// core/analyzer.ts
import { MethodRegistry } from './registry';
import { readDataset } from '../utils/data_processing';

export class TaskAnalyzer {
  private registry: MethodRegistry;
  
  constructor(registry: MethodRegistry) {
    this.registry = registry;
  }
  
  async analyzeTask(params: any) {
    const { 
      dataset_path, 
      task_type, 
      model_name, 
      compute_budget, 
      memory_constraints 
    } = params;
    
    // Analyze dataset statistics
    const datasetStats = await readDataset(dataset_path);
    
    // Determine optimal methods based on task and constraints
    const recommendedMethods = this.recommendMethods(
      task_type,
      model_name,
      compute_budget,
      memory_constraints,
      datasetStats
    );
    
    return {
      dataset_stats: datasetStats,
      recommended_methods: recommendedMethods,
      rationale: this.generateRationale(recommendedMethods, params)
    };
  }
  
  private recommendMethods(
    taskType: string,
    modelName: string,
    computeBudget: string,
    memoryConstraints: string,
    datasetStats: any
  ) {
    // Get all available methods
    const allMethods = Array.from(this.registry.listMethods());
    
    // Filter and rank methods based on constraints
    let filteredMethods = allMethods;
    
    // Filter by memory constraints
    if (memoryConstraints === 'severe') {
      // For severe memory constraints, prioritize QLoRA, IA³, LoftQ
      filteredMethods = filteredMethods.filter(method => 
        ['qlora', 'ia3', 'loftq', 'bitfit'].includes(method.name)
      );
    } else if (memoryConstraints === 'moderate') {
      // For moderate constraints, exclude high-memory methods
      filteredMethods = filteredMethods.filter(method => 
        !['full_ft', 'moelora', 'mora'].includes(method.name)
      );
    }
    
    // Filter by compute budget
    if (computeBudget === 'low') {
      // For low compute, prioritize faster methods
      filteredMethods = filteredMethods.filter(method =>
        !['full_ft', 'reft', 'moelora'].includes(method.name)
      );
    }
    
    // Rank methods by suitability for task type
    const rankedMethods = filteredMethods.map(method => {
      let score = 0;
      
      // Score based on task suitability
      if (method.suitedFor.includes(taskType)) {
        score += 5;
      }
      
      // Score based on performance profile
      if (method.performanceProfile.includes('high_accuracy')) {
        score += 3;
      }
      
      // Additional scoring factors can be added
      
      return {
        ...method,
        suitability_score: score
      };
    });
    
    // Sort by suitability score
    rankedMethods.sort((a, b) => b.suitability_score - a.suitability_score);
    
    // Return top 3 recommended methods
    return rankedMethods.slice(0, 3);
  }
  
  private generateRationale(recommendedMethods: any[], params: any) {
    // Generate explanation for recommendations
    return `Based on your ${params.task_type} task with ${params.compute_budget} compute and ${params.memory_constraints} memory constraints, we've ranked the methods by suitability. ${recommendedMethods[0].name} is the top recommendation because it provides the best balance of performance and resource efficiency for your constraints.`;
  }
}
