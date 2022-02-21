# Examples

## Steps to run a ge-spmm example

**Requirement**: GPU Compute-Capability >= SM 70, nvcc >= 11.0

You could build the whole project once by the following code

```
cd ..
make exp
```

Or you could follow these steps to run spmm only.

**Step 1**: build ge-spmm library

```bash
cd ../src/ge-spmm
make
cd ../../example
```

**Step 2**: build the example

```bash
cd ge-spmm
make
# will generate spmm.out
```

**Step 3**: run example

```bash
./spmm.out ../data/p2p-Gnutella31.mtx


./spmm.out ../data/p2p-Gnutella31.mtx 32 # set arbitrary #columns in rhs dense matrix

```

Example output (on V100, cuda v11.1)

```bash
Finish reading matrix 62586 rows, 62586 columns, 147892 nnz.
Ignore original values and use randomly generated values.
[Cusparse] Report: spmm A(62586 x 62586) * B(62586 x 32) sparsity 0.000038 (nnz=147892)
 Time 0.076032 (ms), Throughput 124.487694 (gflops).
[GE-SpMM][Alg: 0] Report: spmm A(62586 x 62586) * B(62586 x 32) sparsity 0.000038 (nnz=147892)
 Time 0.045675 (ms), Throughput 207.228882 (gflops).
[GE-SpMM][Alg: 1] Report: spmm A(62586 x 62586) * B(62586 x 32) sparsity 0.000038 (nnz=147892)
 Time 0.231848 (ms), Throughput 40.824486 (gflops).
[GE-SpMM][Alg: 2] Report: spmm A(62586 x 62586) * B(62586 x 32) sparsity 0.000038 (nnz=147892)
 Time 0.076493 (ms), Throughput 123.738281 (gflops).
[GE-SpMM][Alg: 3] Report: spmm A(62586 x 62586) * B(62586 x 32) sparsity 0.000038 (nnz=147892)
 Time 0.857259 (ms), Throughput 11.041107 (gflops).
[GE-SpMM][Alg: 8] Report: spmm A(62586 x 62586) * B(62586 x 32) sparsity 0.000038 (nnz=147892)
 Time 0.226008 (ms), Throughput 41.879375 (gflops).
[GE-SpMM][Alg: 9] Report: spmm A(62586 x 62586) * B(62586 x 32) sparsity 0.000038 (nnz=147892)
 Time 0.044950 (ms), Throughput 210.568878 (gflops).
```

## Steps to run a gspmm-fp example

Make sure you have installed the requirements as follows:
* torch >= 1.8.0
* ninja
* scipy

**Step**
Enter the example's folder:
``` bash
cd gspmm-fp 
```
Then you could run our example by 
``` python
python gspmm-exp.py [k]
```
Here k is the feature length of the input dense matrix.

Example output (on RTX3090, cuda v11.2)

```bash 
Loading extension module spmm...
running u_sub_e_sum our time is: 0.0004
```

Note here we use JIT, so it is normal to wait longer when compiling the project the first time.