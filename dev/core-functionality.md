# Core Functionality Specification

## Overview
This document specifies the core functionality of the WillItLLM application, focusing on data structures, calculations, and the underlying logic that powers the tool.

## Data Structures

### Model
```javascript
{
  "name": "Model Name",
  "ollama_tag": "llama2",
  "context_length": 4096,
  "params_b": 7,
  "block_count": 32,
  "head_count": 32,
  "head_count_kv": 8,
  "embedding_length": 4096,
  "key_length": 256,
  "value_length": 256,
  "variants": [
    {
      "quantization": "q4_0",
      "size": 3.7
    }
  ]
}
```

### GPU
```javascript
{
  "name": "NVIDIA RTX 4090",
  "vram": 24,
  "tflops_fp16": 82,
  "tflops_fp32": 40,
  "tflops_int8": 164,
  "tflops_int4": 328,
  "architecture": "Ada Lovelace"
}
```

### Library
```javascript
{
  "name": "PyTorch",
  "version": "2.0.1",
  "url": "https://pytorch.org/"
}
```

### Quantization
```javascript
{
  "name": "q4_0",
  "description": "4-bit quantization with 0 groups",
  "precision": "4-bit"
}
```

### KV Cache
```javascript
{
  "name": "Standard KV Cache",
  "description": "Standard key-value cache for attention mechanism",
  "type": "standard"
}
```

## Core Calculations

### TFLOPS Calculation
The TFLOPS calculation is based on the attention mechanism and model parameters:
- For FP16: `tflops_fp16 * (context_length / 1024) * (head_count / 8)`
- For FP32: `tflops_fp32 * (context_length / 1024) * (head_count / 8)`
- For INT8: `tflops_int8 * (context_length / 1024) * (head_count / 8)`
- For INT4: `tflops_int4 * (context_length / 1024) * (head_count / 8)`

### VRAM Usage Calculation
VRAM usage is calculated based on model size and context length:
- `vram_usage = model_size * (context_length / 4096) * (batch_size / 1)`

### Context Window Handling
The context window is determined by the model's `context_length` parameter and is used to calculate memory requirements and performance characteristics.

## Data Loading Pipeline

### Data Sources
- `data/models.js` - Contains model specifications
- `data/gpus.js` - Contains GPU specifications
- `data/libraries.js` - Contains library specifications
- `data/quantizations.js` - Contains quantization specifications
- `data/kv-cache.js` - Contains KV cache specifications

### Data Loading Process
1. Load JSON data from source files
2. Parse and validate data structures
3. Store in memory for quick access
4. Provide to UI components for rendering

## Calculation Functions

### `app.calc.js`
- `calculateTflops(model, gpu)` - Calculate TFLOPS for a model-GPU combination
- `calculateVramUsage(model, gpu, contextLength)` - Calculate VRAM usage
- `calculateContextWindow(model)` - Calculate context window size
- `calculateMemoryRequirements(model, gpu)` - Calculate total memory requirements

## Implementation Details

### `app.js` - Main Application Logic
- Initializes the application
- Loads data from source files
- Sets up event handlers
- Manages state

### `app.ui.js` - UI Components
- Creates UI elements
- Handles user interactions
- Manages event listeners
- Updates display based on state

### `app.render.js` - Data Rendering
- Renders model data
- Renders GPU data
- Renders comparison results
- Updates UI based on calculations

### `app.calc.js` - Calculations
- Implements all calculation logic
- Provides utility functions
- Handles edge cases

## Error Handling

### Validation
- Validate model data structure
- Validate GPU data structure
- Validate context length values
- Validate VRAM values

### Edge Cases
- Handle missing data
- Handle invalid values
- Handle unsupported GPUs
- Handle unsupported models

## Performance Optimization

### Caching
- Cache calculation results
- Cache rendered UI elements
- Cache data structures

### Efficiency
- Optimize calculation loops
- Minimize DOM updates
- Use efficient data structures

## Data Update Process

### Model Data Updates
1. Update `data/models.js` with new model data
2. Verify data structure matches specification
3. Test with various GPU configurations
4. Commit changes to repository

### GPU Data Updates
1. Update `data/gpus.js` with new GPU data
2. Verify data structure matches specification
3. Test with various model configurations
4. Commit changes to repository

## Design Principles

### Minimal Dependencies
- No external libraries required beyond basic web technologies
- Pure JavaScript implementation
- No build tools or bundlers needed

### No Build Process
- Application runs directly in browser
- No compilation or transpilation required
- Simple HTML, CSS, and JavaScript files

### No Server-Side Components
- Completely client-side application
- All data stored in JSON files
- No backend or API required

### Self-Hostable
- Can be hosted on any web server
- Can be run locally by opening index.html
- No deployment complexity

## Conclusion

This specification provides a complete overview of the core functionality of the WillItLLM application, including data structures, calculations, implementation details, and error handling strategies.
