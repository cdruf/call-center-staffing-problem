import unittest
from functools import reduce

import matplotlib.pyplot as plt


# %%

class StepFunction:

    def __init__(self, xs: list, ys: list):
        assert len(xs) == len(ys) + 1
        self.xs = xs
        self.ys = ys

    def append_step(self, x, y, x_end=None):
        """
        Add another bar to the end of the function.
        """
        self.xs.pop()
        self.xs.append(x)
        self.ys.append(y)
        if x_end is None:
            self.xs.append(x)
        else:
            self.xs.append(x_end)

    def get_area(self):
        """
        Gets the area under the step-function.
        """
        ret = 0
        for idx, x in enumerate(self.xs[1:]):
            ret += self.ys[idx] * (x - self.xs[idx])
        return ret

    def get_average(self):
        return self.get_area() / (self.xs[-1] - self.xs[0])

    def plot(self, title=None):
        xx = self.xs[0:1] + reduce(lambda i, j: i + [j, j], self.xs[1:-1], []) + self.xs[-1:]
        yy = reduce(lambda i, j: i + [j, j], self.ys, [])
        fig, ax = plt.subplots()
        ax.plot(xx, yy)
        plt.axhline(self.get_average())
        if title is not None:
            ax.set_title(title)
        plt.show()


# unittest will test all the methods whose name starts with 'test'
class TestStepFunction(unittest.TestCase):

    def test(self):
        fkt = StepFunction(xs=[1, 2, 5, 6], ys=[1, 2, 1])
        area = fkt.get_area()
        self.assertEqual(area, 8)

        fkt = StepFunction(xs=[0, 1, 2, 5, 6], ys=[0, 1, 2, 1])
        area = fkt.get_area()
        fkt.plot()
        self.assertEqual(area, 8)

        fkt = StepFunction(xs=[0, 1, 4, 7], ys=[1, 2, 1])
        area = fkt.get_area()
        self.assertEqual(area, 10)
        fkt.append_step(6, 6, 7)
        fkt.plot()


if __name__ == '__main__':
    unittest.main()
