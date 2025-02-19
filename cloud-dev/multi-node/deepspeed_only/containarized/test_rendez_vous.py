# test_rendezvous.py
import os
import torch
import torch.distributed as dist

def main():
    # Initialize the distributed process group with gloo backend.
    # The rendezvous (synchronization) happens here.
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Process {rank} out of {world_size} processes is running.")

    # Perform a barrier to synchronize all processes.
    dist.barrier()
    
    if rank == 0:
        print("Rendezvous complete: All processes are synchronized!")
    
    # Optional: Cleanup the process group.
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

