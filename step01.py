class Variable:
    def __init__(self, data) -> None:
        self.data = data

class Function:
    def __call__(self, input) -> Variable:
        x = input.data
        y = x ** 2
        output = Variable(y)
        return output