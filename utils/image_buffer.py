import numpy as np

class ImageBuffer():
    def __init__(self,
                 max_capacity=50):

        self.max_capacity = max_capacity
        self.im_path_buffer = []
        self.im_buffer = []
        self._size = 0

    def size(self):
        return self._size

    def add(self, img, path):
        # if buffer has reached max capacity, remove oldest image
        if len(self.im_buffer) > self.max_capacity:
            self.im_buffer.pop(0)
            self.im_path_buffer.pop(0)

        self.im_buffer.append(img)
        self.im_path_buffer.append(path)
        self._size += 1

    def get_img(self, img, path):
        if self.size() == 0:
            self.add(img, path)
            return img
        else:
            sample = np.random.choice(self.buffer)
            self.add(img, path)
            return sample

    def sample(self):
        idx = np.random.randint(low=0, high=self.size())
        im = self.im_buffer[idx]
        im_path = self.im_path_buffer[idx]
        return (im, im_path)

    def clear(self):
        self.im_buffer = []
        self.im_path_buffer = []