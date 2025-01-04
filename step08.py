import numpy as np

class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            print(data)
            print(isinstance(data, np.ndarray))
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported. Only support ndarray'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self, func):
        self.creator = func
    def backward(self):
        print(self.creator)
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        f = self.creator
        print(f)
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def forward(self, x):
        raise NotImplementedError
    def backward(self, input):
        raise NotImplementedError
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        self.input = input
        self.output = output
        output.set_creator(self)
        return output
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    def backward(self, gy):
        print(f"square input={self}")
        x = self.input.data
        return 2 * x * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        print(f"exp input={self}")
        x = self.input.data
        return np.exp(x) * gy

def square(x):
    f = Square()
    return f(x)

def exp(x):
    return Exp()(x)


x = Variable(np.array(0.5))
y = square(exp(square(x)))

y.backward()
print(x.grad)
