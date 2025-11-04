"""
QUICK START DEMO
Run this to test the DQN Manufacturing System with minimal parameters
"""

from train import ManufacturingOptimizer

print("🚀 DQN Manufacturing System - Quick Demo")
print("=" * 60)
print("This will run a minimal test to verify everything works.\n")

# Create optimizer with minimal parameters for quick testing
optimizer = ManufacturingOptimizer(
    n_jobs=30,           # 30 jobs (quick test)
    n_machines=8,        # 8 machines
    n_episodes=10,       # 10 episodes (fast training)
    max_steps=100,       # 100 steps max
    test_split=0.25,     # 25% test set
    output_dir="demo_results"
)

# Run complete pipeline
agent, kpis = optimizer.run()

print("\n" + "=" * 60)
print("✅ DEMO COMPLETE!")
print("=" * 60)
print("\nTo run full-scale optimization:")
print(">>> from train import ManufacturingOptimizer")
print(">>> optimizer = ManufacturingOptimizer(")
print("...     n_jobs=200, n_machines=15, n_episodes=150,")
print("...     max_steps=250, test_split=0.2")
print("... )")
print(">>> agent, kpis = optimizer.run()")
print("\nCheck 'demo_results/' for visualizations!")
