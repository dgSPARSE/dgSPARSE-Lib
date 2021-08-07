使用方法

`make` 会生成libgespmm.a库，可修改makefile令其生成动态库

API

参看gespmm.h，只能用spmm_cuda_alg<0,1,2,3>这四个函数，其他不保证测试通过。