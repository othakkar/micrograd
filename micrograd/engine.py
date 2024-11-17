import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # internal variables
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        self._backward = _backward
        
        return out

    def __mul__(self, other): # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        self._backward = _backward
        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int & float powers"
        out = Value(math.pow(self.data, other), (self, ), 'pow')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        
        self._backward = _backward
        
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        
        self._backward = _backward
        
        return out

    def tanh(self):
        e = math.exp(2 * self.data)
        t = (e - 1) / (e + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        self._backward = _backward
        
        return out

    def relu(self):
        out = Value(max(self.data, 0.0), (self, ), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        
        self._backward = _backward
        
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __neg__(self): # -self
        return -1.0 * self

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rtruediv__(self, other): # other / self
        return other * (self ** -1)

    def __truediv__(self, other): # self / other
        return self * (other ** -1)
