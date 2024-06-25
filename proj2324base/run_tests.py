import subprocess
import time
import glob

def run_test(file_path, timeout=2):
    start_time = time.time()
    try:
        subprocess.run(["python", "./pipe.py", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        end_time = time.time()
        return end_time - start_time, True
    except subprocess.TimeoutExpired:
        end_time = time.time()
        return end_time - start_time, False

def main():
    # Get all test files from both directories
    test_files_1_9 = glob.glob("../test pipe 1-9/test-*.txt")
    test_files_10x10_50x50 = glob.glob("../test pipe 10x10-50x50/test-*.txt")
    
    # Combine the lists of test files
    all_test_files = test_files_1_9 + test_files_10x10_50x50

    total_time = 0

    for test_file in all_test_files:
        print(f"Running {test_file}...")
        elapsed_time, completed = run_test(test_file)
        total_time += elapsed_time
        if completed:
            print(f"Time taken for {test_file}: {elapsed_time:.2f} seconds\n")
        else:
            print(f"{test_file} timed out after {elapsed_time:.2f} seconds\n")

    print(f"Total time for all tests: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
