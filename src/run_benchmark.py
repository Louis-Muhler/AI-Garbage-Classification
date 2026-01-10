import sys
import os
import subprocess
import glob
import json
import matplotlib.pyplot as plt

def get_latest_history_file(model_name):
    """Finds the latest history.json for a given model."""
    # We assume the script is started from the root directory,
    # so 'models' is relative to the root.
    model_dir = os.path.join("models", model_name)
    
    # Search for all timestamp folders
    run_dirs = glob.glob(os.path.join(model_dir, "*"))
    if not run_dirs:
        return None
    
    # Sort by creation date (newest first)
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    
    # Check if history.json exists
    history_path = os.path.join(latest_run_dir, "history.json")
    if os.path.exists(history_path):
        return history_path
    return None

def plot_benchmark_results(models):
    """Plots the validation accuracy of all models in a single diagram."""
    print("Creating comparison plot...")
    plt.figure(figsize=(10, 6))
    
    found_data = False
    for model in models:
        history_file = get_latest_history_file(model)
        if history_file:
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                val_acc = history.get('val_acc', [])
                
                epochs = range(1, len(val_acc) + 1)
                
                # Plot only validation accuracy for comparison
                plt.plot(epochs, val_acc, label=f"{model} (Val Acc)", linewidth=2)
                found_data = True
            except Exception as e:
                print(f"Could not read history for {model}: {e}")
        else:
            print(f"No history found for {model}.")
            
    if found_data:
        plt.title('Benchmark: Validation Accuracy Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        output_path = os.path.join("models", "benchmark_comparison.png")
        plt.savefig(output_path)
        print(f"Comparison plot saved at: {output_path}")
    else:
        print("No data found for plotting.")

def run_benchmark():
    # List of models to be trained
    models = ["logistic", "simple_cnn", "custom_resnet", "resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large"]
    
    # Number of epochs
    epochs = 80
    
    # Patience for Early Stopping
    patience = 0 #10

    print("Starting Benchmark Suite...")
    print(f"Models: {models}")
    print(f"Epochs: {epochs}")
    print("-" * 40)

    # Determine path to main.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script_path = os.path.join(script_dir, "main.py")

    for model in models:
        print(f"\n========================================")
        print(f"Starting training for: {model}")
        print(f"========================================")
        
        # Construct the command
        cmd = [
            sys.executable, main_script_path,
            "--model", model,
            "--epochs", str(epochs),
            "--patience", str(patience),
            "--pin_memory"
        ]
        
        try:
            # Execute training and wait for completion
            # check=True raises an error if the script fails
            subprocess.run(cmd, check=True)
            print(f"Training for {model} completed.")
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR training {model}: {e}")
        except KeyboardInterrupt:
            print("\nBenchmark cancelled by user.")
            return

    print("\n" + "="*40)
    print("All benchmark runs completed!")
    
    # Plot results at the end
    plot_benchmark_results(models)
    
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
