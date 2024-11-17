import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, n_ins):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_ins)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum([xi * wi for xi, wi in zip(x, self.w)], self.b)
        out = act.tanh()
        return act

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):

    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):

    def __init__(self, n_in, n_outs):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
