import time

import torch


# https://pytorch.org/docs/stable/notes/mps.html
if not torch.backends.mps.is_available():
    print("MPS not available because")
    if not torch.backends.mps.is_built():
        print("the current PyTorch install was not built with MPS enabled.")
    else:
        print("the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    
    exit()

else:
    print("MPS is available.")


# Define a simple function to benchmark
def matrix_multiplication(_size: int, device: torch.device, epochs: int = 1) -> float:
    total_time = 0

    for _ in range(epochs):
        # Create random matrices
        a = torch.randn(_size, _size, dtype=torch.float32, device=device)
        b = torch.randn(_size, _size, dtype=torch.float32, device=device)

        # Start the timer
        start_time = time.time()
        
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        
        # Stop the timer
        end_time = time.time()
        del a, b, c
        torch.cuda.empty_cache()

        total_time += end_time - start_time

        return total_time


def main():
    for _size in (
        100, 300, 1000, 3000, 10000, 15000, 20000, 25000, 30000
    ):
        # Measure time on CPU
        print("Running on CPU...", end="")
        cpu_time = matrix_multiplication(_size, 'cpu')
        print(f"\rCPU computation time ({_size}): {cpu_time:.4f} seconds")

        # Measure time on MPS
        print("Running on MPS...", end="")
        mps_time = matrix_multiplication(_size, 'mps')
        print(f"\rMPS computation time ({_size}): {mps_time:.4f} seconds")

        # Print acceleration ratio
        print(f"MPS acceleration ratio: {cpu_time / mps_time:.2f}")
        print()


if __name__ == "__main__":
    main()