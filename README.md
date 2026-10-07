# Benchmark and Statistical Evaluation of Particle Swarm Optimization (PSO) Variants

A Python-based framework designed to implement, benchmark, and statistically evaluate four variants of the **Particle Swarm Optimization (PSO)** algorithm across standard benchmark functions.

The project automates the execution pipeline (N=50 runs per configuration), collects convergence metrics, performs non-parametric statistical hypothesis testing, and generates analytical visualizations.

---

## Project Overview

The objective of this project is to analyze the trade-offs between exploration and exploitation across different PSO variants. 

### Key Features
* **4 PSO Variants Implemented**:
  * Standard Particle Swarm Optimization
  * Competitive Swarm Optimizer
  * Gregarious Particle Swarm Optimizer
  * Social Learning Particle Swarm Optimizer
* **Automated Experimentation Pipeline**: Runs each algorithm variant 50 times per test function with fixed seed management for reproducible results.
* **Benchmark Test Suite**: Evaluates algorithms on continuous optimization landscapes.
* **Statistical Analysis**: Performs automated non-parametric hypothesis tests to verify statistically significant performance differences.
* **Visualizations**: Automatic generation of convergence curves and search space trajectory plots.

---

## Tech Stack & Tools

* **Language**: Python 3.x
* **Numerical & Statistical Computing**: `NumPy`, `SciPy`, `scikit-posthocs`
* **Visualization**: `Matplotlib`

---

## Analytical Methodology

1. **Experimental Setup**:
   * Each variant is run 50 times per objective function to account for stochastic variation.
   * All algorithm runs use a fixed number of particles and maximum iterations.
   
2. **Statistical Validation**:
   * Performance difference significance tested using Kruskal-Wallis and Levene tests across variants.
   * Pairwise post-hoc analysis via the Nemenyi test to confirm which variant statistically dominates.

3. **Metrics Tracked**:
   * **Mean, Median, and Best Fitness** 
   * **Standard Deviation** 
   * **Success Rate** 
   * **Average Distance**
   * **Number of Iterations** 

---

## Repository Structure

```text
├── benchmark/            # Support tools for testing and visualization
│   ├── config.py         # Utility functions for visualization and statistical tests
│   ├── test_functions.py # Benchmark continuous functions
│   └── test_run.py       # Script for running individual algorithms and collecting metrics    
├── modifications/        # Implementations of PSO algorithm variants
│   ├── cso.py            # Competitive Swarm Optimizer Algorithm
│   ├── g_pso.py          # Gregarious Particle Swarm Optimizer Algorithm
│   ├── pso.py            # Standard Particle Swarm Optimization Algorithm
│   └── sl_pso.py         # Social Learning Particle Swarm Optimizer Algorithm
├── main.py               # Main test script to set parameters and display results
├── requirements.txt      # Project dependencies
└── README.md