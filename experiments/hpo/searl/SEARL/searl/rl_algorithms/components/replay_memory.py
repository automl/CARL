import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transistion):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transistion
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transistion)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)

        transition_list = []
        for i in ind:
            transition_list.append(self.storage[i])

        return transition_list
