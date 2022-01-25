# import multiprocessing as mp
import torch.multiprocessing as mp
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_sharing_strategy('file_descriptor')

import queue
import time
from typing import List, Dict

import fastrand
import numpy as np


class MPReplayMemory(object):

    def __init__(self, seed, capacity, batch_size, reuse_batch):

        ctx = mp.get_context('spawn')
        mp_manager = ctx.Manager()
        self.push_queue = mp_manager.Queue()
        self.sample_queue = mp_manager.Queue()
        self.save_queue = mp_manager.Queue()
        self.batch_size = batch_size

        np.random.seed(seed)
        self.memory_manager = ctx.Process(target=self._memory_manager,
                                          args=(capacity, batch_size, reuse_batch, self.push_queue, self.sample_queue,
                                                self.save_queue))
        self.memory_manager.daemon = True
        self.memory_manager.start()

    def load(self, replay_memory_dict):
        self.push_queue.put(replay_memory_dict)

    def save(self):
        self.push_queue.put("SAVE")
        try:
            save_dict = self.save_queue.get(timeout=10)
            return save_dict
        except queue.Empty:
            print("save failed")
            return "no_save"

    @staticmethod
    def _memory_manager(capacity: int, batch_size: int, reuse_batch: int, push_queue: mp.Queue, sample_queue: mp.Queue,
                        save_queue: mp.Queue):
        memory = []
        position = 0

        while True:
            if not push_queue.empty():
                queue_output = push_queue.get()
                if queue_output == "QUIT":
                    return

                elif queue_output == "SAVE":
                    save_queue.put({"memory": memory, "position": position})

                elif isinstance(queue_output, Dict):
                    memory = queue_output["memory"]
                    position = queue_output["position"]

                elif isinstance(queue_output, List):
                    for transition in queue_output:
                        if len(memory) < capacity:
                            memory.append(transition)
                        else:
                            memory[position] = transition
                            position = (position + 1) % capacity
                else:
                    if len(memory) < capacity:
                        memory.append(queue_output)
                    else:
                        memory[position] = queue_output
                        position = (position + 1) % capacity

            if sample_queue.qsize() < 20 and len(memory) > batch_size:

                transistion_list = []
                for _ in range(batch_size):
                    idx = fastrand.pcg32bounded(len(memory))
                    transistion_list.append(memory[idx])
                for _ in range(reuse_batch):
                    sample_queue.put(transistion_list)

    def close(self):
        print("CLOSE REPLAY MEMORY")
        self.push_queue.put("QUIT")
        while not self.push_queue.empty():
            time.sleep(1)

    def get_push_queue(self):
        return self.push_queue

    def get_sample_queue(self):
        return self.sample_queue


class ReplayMemory(object):

    def __init__(self, capacity: int, batch_size: int):
        self.storage = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.ptr = 0

    def add(self, transistions):
        if isinstance(transistions, List):
            for transition in transistions:
                self._add(transition)
        else:
            self._add(transistions)

    def put(self, transistions):
        self.add(transistions)

    def _add(self, transistion):
        if len(self.storage) == self.capacity:
            self.storage[int(self.ptr)] = transistion
            self.ptr = (self.ptr + 1) % self.capacity
        else:
            self.storage.append(transistion)

    def get(self):
        return self.sample()

    def sample(self):
        ind = np.random.randint(0, len(self.storage), size=self.batch_size)

        transition_list = []
        for i in ind:
            transition_list.append(self.storage[i])

        return transition_list
