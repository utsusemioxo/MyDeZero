import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
    
class Function:
    def forward(self, x):
        raise NotImplementedError
    def backward(self, gy):
        raise NotImplementedError
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input # remember input for every call
        return output
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)      # dydy
b.grad = C.backward(y.grad) # dydb
a.grad = B.backward(b.grad) # dyda
x.grad = A.backward(a.grad) # dydx