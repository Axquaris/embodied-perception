import pathlib as pth
import inspect

import torch


PROJECT_ROOT = pth.Path(__file__).parents[1]
print("PROJECT_ROOT at", PROJECT_ROOT)

TEST_SVO2_PATH = "data/zed_recordings/HD720_SN33087127_15-44-16.svo2"


def pgmem():
    # Get the current frame and go back one level to get the caller's frame
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    
    # Get information about the caller
    caller_info = inspect.getframeinfo(caller_frame)
    print(f"{caller_info.function} : {caller_info.lineno}")
    print("\ttorch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("\ttorch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("\ttorch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
