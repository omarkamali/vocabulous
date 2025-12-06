# Benchmark Results

Date: 2025-11-05
Environment: uv run (isolated, dependencies from pyproject.toml)
Command:

```bash
uv run env PYTHONPATH=src python benchmarks/benchmark_vocabulous.py | tee benchmarks/benchmark_output.txt
```

## Highlights

- **Throughput (20k rows, auto clean+score)**: ~454k rows/s
- **Apply vs Vectorized**: Comparable across sizes; both around 450–500k rows/s on 5k–50k rows
- **Duplication ratio effect**: Minimal variance; token-cache friendly duplicates don’t significantly change throughput
- **Sentence length impact**: Longer sentences reduce throughput (e.g., len=50 ~190k rows/s)
- **Dictionary size impact**: Small effect from 50 to 5000 vocab per language; stays near ~450–480k rows/s
- **Large-n (vectorized, batched 100k)**: ~403k rows/s
- **Large-n compare (200k, dict=1000, len=20)**: ~224k–225k rows/s across apply/vectorized/numba/sparse
- **Clean-only speed**: Very fast; ~12M rows/s reported on synthetic text cleaning

## Full Output

```
Auto Score (clean+score) on 20k rows: 0.044s  -> 453686.6 rows/s
Clean only on 20k rows: 0.002s  -> 12304216.9 rows/s
Cycle clean on 20k rows (pre-scored): 0.000s

== Benchmark: apply vs vectorized ==
n=  5000 | apply: 0.012s (429296.6 rows/s) | vectorized: 0.012s (416969.2 rows/s)
n= 20000 | apply: 0.043s (469603.9 rows/s) | vectorized: 0.043s (459786.4 rows/s)
n= 50000 | apply: 0.102s (490536.5 rows/s) | vectorized: 0.100s (501237.4 rows/s)

== Benchmark: duplication ratio (token cache effect) ==
dup=0.0 | apply: 0.043s (466747.2 rows/s)
dup=0.5 | apply: 0.041s (484631.6 rows/s)
dup=0.9 | apply: 0.043s (460130.2 rows/s)

== Benchmark: sentence length ==
len= 5 | apply: 0.044s (456052.2 rows/s) | vectorized: 0.044s (458804.6 rows/s) | numba: 0.044s (453538.7 rows/s) | sparse: 0.043s (467304.7 rows/s)
len=20 | apply: 0.055s (360927.8 rows/s) | vectorized: 0.056s (359071.8 rows/s) | numba: 0.056s (354975.0 rows/s) | sparse: 0.057s (351535.0 rows/s)
len=50 | apply: 0.106s (188067.0 rows/s) | vectorized: 0.104s (193229.1 rows/s) | numba: 0.102s (196878.9 rows/s) | sparse: 0.102s (196570.1 rows/s)

== Benchmark: dictionary size ==
dict=   50 | apply: 0.044s (451660.1 rows/s) | vectorized: 0.042s (472461.9 rows/s) | numba: 0.044s (454363.9 rows/s) | sparse: 0.043s (468693.7 rows/s)
dict=  500 | apply: 0.044s (456773.1 rows/s) | vectorized: 0.043s (468660.7 rows/s) | numba: 0.042s (472982.8 rows/s) | sparse: 0.042s (478899.5 rows/s)
dict= 5000 | apply: 0.042s (478216.3 rows/s) | vectorized: 0.043s (466494.5 rows/s) | numba: 0.042s (471936.0 rows/s) | sparse: 0.043s (466270.7 rows/s)

== Benchmark: large-n (vectorized batched) ==
n=100000 vectorized-batched: 0.248s (402731.8 rows/s)

== Benchmark: high match density (dataset built from dict) ==
high-match dict=   50 | apply: 0.100s (199700.5 rows/s) | vectorized: 0.099s (201175.3 rows/s) | numba: 0.098s (204148.7 rows/s) | sparse: 0.099s (201167.2 rows/s)
high-match dict=  500 | apply: 0.099s (202409.1 rows/s) | vectorized: 0.100s (200466.0 rows/s) | numba: 0.098s (204049.9 rows/s) | sparse: 0.097s (205293.4 rows/s)
high-match dict= 5000 | apply: 0.109s (183111.9 rows/s) | vectorized: 0.107s (187090.1 rows/s) | numba: 0.109s (184243.5 rows/s) | sparse: 0.107s (187256.8 rows/s)

== Benchmark: large-n compare (apply vs vectorized batched) ==
n=200000 dict=1000 len=20 | apply: 0.892s (224336.2 rows/s) | vectorized-batched: 0.903s (221568.9 rows/s) | numba: 0.890s (224807.0 rows/s) | sparse: 0.894s (223819.6 rows/s)
```
