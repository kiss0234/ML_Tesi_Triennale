import psutil

def print_memory_usage():
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"Used memory: {memory.used / (1024 ** 3):.2f} GB")
    print(f"Free memory: {memory.free / (1024 ** 3):.2f} GB")
    print(f"Memory usage percentage: {memory.percent}%")

