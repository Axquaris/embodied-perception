from collections import defaultdict
import time
from pprint import pprint

import torch
import numpy as np


class Timer:
    recorder = defaultdict(list)

    def __init__(self, des="", verbose=False, record=True, debug=True) -> None:
        self.des = des
        self.verbose = verbose
        self.record = record
        self.debug = debug

    def __enter__(self):
        if not self.debug:
            return self
        self.start = time.time()
        self.start_cuda = torch.cuda.Event(enable_timing=True)
        self.end_cuda = torch.cuda.Event(enable_timing=True)
        self.start_cuda.record()

        return self

    def __exit__(self, *args):
        if not self.debug:
            return 
        self.end_cuda.record()
        self.end = time.time()
        torch.cuda.synchronize()
        self.interval = self.start_cuda.elapsed_time(self.end_cuda)/1000.

        if self.verbose:
            print(f"[cudasync]{self.des} consuming {self.interval:.8f}")
        if self.record:
            Timer.recorder[self.des].append(self.interval)

    @staticmethod
    def show_recorder():
        pprint({k: f"average time {np.mean(v)}s" for k, v in Timer.recorder.items()})
        pprint({k: f"average FPS {1. / np.mean(v)}" for k, v in Timer.recorder.items()})