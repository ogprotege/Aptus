# Method Registry and Implementation
// core/registry.ts
import { LoRAMethod } from '../methods/low_rank/lora';
import { DoRAMethod } from '../methods/low_rank/dora';
import { ReFTMethod } from '../methods/low_rank/reft';
import { AdaLoRAMethod } from '../methods/low_rank/adalora';
// Import all 25 method implementations

export class MethodRegistry {
  private methods: Map<string, FineTuningMethod>;
  
  constructor() {
    this.methods = new Map();
    this.registerAllMethods();
  }
  
  private registerAllMethods() {
    // Register all 25 methods
    this.registerMethod(new LoRAMethod());
    this.registerMethod(new DoRAMethod());
    this.registerMethod(new ReFTMethod());
    this.registerMethod(new AdaLoRAMethod());
    // Continue for all 25 methods
  }
  
  private registerMethod(method: FineTuningMethod) {
    this.methods.set(method.name, method);
  }
  
  getMethod(name: string): FineTuningMethod {
    const method = this.methods.get(name);
    if (!method) {
      throw new Error(`Fine-tuning method "${name}" not found`);
    }
    return method;
  }
  
  getMethodInfo(name: string) {
    const method = this.getMethod(name);
    return {
      name: method.name,
      description: method.description,
      category: method.category,
      performance_profile: method.performanceProfile,
      memory_usage: method.memoryUsage,
      suited_for: method.suitedFor,
      paper_reference: method.paperReference,
      default_config: method.defaultConfig
    };
  }
  
  listMethods(filterBy?: string, filterValue?: string) {
    let filteredMethods = Array.from(this.methods.values());
    
    if (filterBy && filterValue) {
      filteredMethods = filteredMethods.filter(method => {
        switch (filterBy) {
          case 'category':
            return method.category === filterValue;
          case 'performance':
            return method.performanceProfile.includes(filterValue);
          case 'memory_usage':
            return method.memoryUsage <= parseInt(filterValue);
          default:
            return true;
        }
      });
    }
    
    return filteredMethods.map(method => ({
      name: method.name,
      category: method.category,
      description: method.description
    }));
  }
}
