from micrograd.nn import MLP
from micrograd.engine import Value

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
n = MLP(3, [4, 4, 1])

for i in range(500):
    ypreds = [n(x) for x in xs]
    loss = sum([(ypred - ytgt)**2 for ypred, ytgt in zip(ypreds, ys)])
    n.zero_grad()
    loss.backward()
    for p in n.parameters():
        p.data += (-0.01) * p.grad

print("Preds: ", ypreds)
print("Target: ", ys)
