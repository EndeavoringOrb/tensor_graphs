# graph
`a * (b^2 + c^2)`

# operations
```
a = addcpu(0, 1) 5
0 = copy2cpu(2) 8
2 = gpusquare(3) 1
3 = copy2gpu(b) 2
1 = copy2cpu(5) 8
5 = gpusquare(6) 1
6 = copy2gpu(c) 2
```

# successors
```
a -> {}
0 -> {a}
2 -> {0}
3 -> {2}
b -> {3}
1 -> {a}
5 -> {1}
6 -> {5}
c -> {6}
```

# upward rank
```
rank(a) = 5 + max({}) = 5
rank(0) = 8 + max({rank(a)}) = 8 + 5 = 13
...
```

# determine dispatch order
```
ranks = {a: 5, 0: 13, ...}
remaining = {a, 0, 2, 3, b, 1, 5, 6, c}
ordered = []
while remaining:
  ready = []
  for node in remaining:
    node_ready = True
    for child in node.children:
      if child in remaining:
        node_ready = False
        break
    if node_ready:
      ready.append(node)
  if len(ready) == 0:
    error
  best_node = ready[0]
  best_cost = ranks[ready[0]]
  for node in ready[1:]:
    node_cost = ranks[node]
    if node_cost > best_cost:
      best_cost = node_cost
      best_node = node
  ordered.append(best_node)
  remaining.pop(best_node)
```