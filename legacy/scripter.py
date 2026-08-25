import sys
import os
from tensorflow.python.profiler import profiler_client

def analyze_profile_cli(log_dir):
    """
    Analyzes a TensorBoard profile from the command line and prints
    a summary of the top GPU kernel statistics using a more robust method.
    """
    print(f"Analyzing profile in: {log_dir}")
    
    try:
        op_stats = profiler_client.get_op_stats(log_dir)
    except Exception as e:
        print(f"Error: Could not process profiler data.")
        print(f"Please ensure '{log_dir}' contains valid profiler output ('...xplane.pb').")
        return

    if not op_stats.kernel_stats_db:
        print("No GPU kernel stats found in the profile.")
        print("Did you run the code on a GPU?")
        return
        
    # Create a list of kernel data directly from the protobuf object
    kernel_list = []
    for kernel_name, report in op_stats.kernel_stats_db.reports.items():
        kernel_list.append({
            'name': kernel_name,
            'duration_ps': report.total_duration_ps,
            'occurrences': report.occurrences,
        })
    
    # Sort the kernels by duration in descending order
    sorted_kernels = sorted(kernel_list, key=lambda k: k['duration_ps'], reverse=True)
    
    # Print the header for our summary table
    print("\n--- Top 10 GPU Kernels by Execution Time ---")
    print(f"{'Kernel Name':<80} {'Duration (ms)':>15} {'Occurrences':>15}")
    print("-" * 112)
    
    # Print the top 10 longest-running kernels
    total_time_ms = op_stats.host_op_time_ps.total_ps / 1e9 if op_stats.host_op_time_ps else 0
    
    for kernel in sorted_kernels[:10]:
        name = kernel['name']
        duration_ms = kernel['duration_ps'] / 1e9  # Convert picoseconds to milliseconds
        occurrences = kernel['occurrences']
        
        # Truncate long names for better formatting
        if len(name) > 78:
            name = name[:75] + "..."
            
        print(f"{name:<80} {duration_ms:>15.4f} {occurrences:>15}")
        
    print("-" * 112)
    if total_time_ms > 0:
      print(f"\nTotal captured time: {total_time_ms:.4f} ms")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 scripter.py <path_to_log_directory>")
        sys.exit(1)
        
    log_directory = sys.argv[1]
    if not os.path.isdir(log_directory):
        print(f"Error: Directory not found at '{log_directory}'")
        sys.exit(1)
        
    analyze_profile_cli(log_directory)
