import numpy as np


class BatchLoader:
    def __init__(self,
                 batch_size: int,
                 shuffle: bool = True,
                 data: list = None,
                 sequential: bool = False):
        self.batch_size = batch_size
        self.sequential = sequential
        self.shuffle = shuffle
        self.data = data
        self.num_batches = len(self.data) // self.batch_size
        self.current_index = 0
        self.indices = np.arange(len(self.data))

    def _randomize_batches(self):
        batches = np.array_split(self.data, self.num_batches)
        np.random.shuffle(batches)
        self.data = np.concatenate(batches)

    def create_batches(self, discard_last_batch: bool = False):
        if self.shuffle and not self.sequential:
            self._randomize_batches()

        if not discard_last_batch and len(self.data) % self.batch_size != 0:
            self.num_batches += 1

        if self.sequential:
            self.batches = np.array([self.data[i:i+self.batch_size]
                                    for i in range(0, len(self.data),
                                    self.batch_size)])
        else:
            self.batches = np.array_split(self.data, self.num_batches)

        return self.batches

    def __len__(self):
        if len(self.data) % self.batch_size != 0:
            self.num_batches += 1
        return self.num_batches

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.batches):
            raise StopIteration

        batch_data = self.batches[self.current_index]
        self.current_index += 1
        return batch_data

# batch = BatchLoader(batch_size=2, data=[1,2,3,4,5,6,7,8,9,10])

# batch.create_batches(discard_last_batch=True)

# print(len(batch))
# # print(batch.batches[2])
# for i in batch:
#     print(i)
