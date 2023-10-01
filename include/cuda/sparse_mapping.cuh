#define COLLISION_BOUND 20

extern "C"
#define _INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])

    inline __device__ int
    buffer_encoder(const int k_id, const int k_map_id) {
  return (k_id * 1186111 + k_map_id);
}

inline __device__ uint64_t coord_hash(const int b, const int ix, const int iy,
                                      const int iz) {
  // +1 to avoid val==0
  return ((uint64_t)b * 23783141 + (uint64_t)ix * 73856093 +
          (uint64_t)iy * 19349669 + (uint64_t)iz * 83492791 + 1);
}

inline __device__ uint64_t shift_hash(const int size, const uint64_t value) {
  return ((value + 1) % ((uint64_t)size - 2));
}

inline __device__ int kernel_decoder(int code) { return (code / 1186111); }

inline __device__ int kernel_map_decoder(int code) { return (code % 1186111); }

__global__ void insertHash(const int nnz, const int size,
                           const int *__restrict__ coord, int *idx) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // exclude illegal id number
  if (id >= nnz) {
    return;
  }
  uint64_t temp_val = coord_hash(coord[4 * id], coord[4 * id + 1],
                                 coord[4 * id + 2], coord[4 * id + 3]);
  // temp_val is unique
  uint64_t table_id = temp_val % (uint64_t)size;
  // cuckoo hashing
  int old_idx = atomicExch(&idx[table_id], id);
  // handle collision
  while (old_idx > -1) {
    table_id = (table_id + 97) % size;
    old_idx = atomicExch(&idx[table_id], old_idx);
  }
}

__global__ void insertVal(const int nnz, const int size,
                          const int *__restrict__ coord, const int *idx,
                          uint64_t *val) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) {
    return;
  }
  int input_id = idx[id];
  if (input_id < nnz && input_id > -1) {
    val[id] = coord_hash(coord[4 * input_id], coord[4 * input_id + 1],
                         coord[4 * input_id + 2], coord[4 * input_id + 3]);
  }
}

/*
The query function to build mapping for D1.
*/
template <int _NNZS_PER_BLOCK, int _KOFS_THREADS>
__global__ void _queryhash_subm(
    // input nnz
    const int innz,
    // output nnz
    const int onnz,
    // input coords hash table size
    const int size,
    // output coords, (onnz, 3)
    const int *__restrict__ coord,
    // coded kernel size, f = 311x + 17y + z
    const int ksx, const int ksy, const int ksz,
    // kernel volume
    const int kv,
    // hash table (value), (size, )
    const uint64_t *val,
    // hash table (index), (size, )
    const int *idx,
    // input-major mapping, (innz * (kv - 1))
    int *map,
    // the counter of nnz for each each kernel offsets, (kv - 1)
    int *knnz,
    // whether to compute center kernel offset separately
    bool separate_mid) {
  // a thread for a coord
  int output_id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.y;
  // exclude illegal id number
  if (output_id >= onnz) {
    return;
  }
  // shared mem to store the output coord
  // TODO: with batch idx we can use int4 to read
  __shared__ int shm[_NNZS_PER_BLOCK * 4];
  shm[threadIdx.y * 4] = coord[output_id * 4];
  shm[threadIdx.y * 4 + 1] = coord[output_id * 4 + 1];
  shm[threadIdx.y * 4 + 2] = coord[output_id * 4 + 2];
  shm[threadIdx.y * 4 + 3] = coord[output_id * 4 + 3];
  // decode kernel size
  int mid_ks = (ksx - 1) / 2 * ksy * ksz + (ksy - 1) / 2 * ksz + (ksz - 1) / 2;
#pragma unroll
  for (int k = 0;; k += _KOFS_THREADS) {
    int kp = k + threadIdx.x;
    // 0 <= kp < kv
    if (kp >= kv) {
      break;
    }
    // ignore w[0, 0, 0]
    if (separate_mid && kp == mid_ks) {
      continue;
    }
    int kx = kp / (ksz * ksy) - (ksx - 1) / 2;
    int ky = (kp / ksz) % ksy - (ksy - 1) / 2;
    int kz = kp % ksz - (ksz - 1) / 2;
    // hash query
    uint64_t target_val = coord_hash(
        shm[threadIdx.y * 4], shm[threadIdx.y * 4 + 1] + kx,
        shm[threadIdx.y * 4 + 2] + ky, shm[threadIdx.y * 4 + 3] + kz);
    uint64_t target_id = target_val % (uint64_t)size;
    // find target or empty
    while (val[target_id] != target_val && idx[target_id] > -1) {
      target_id = (target_id + 97) % size;
    }
    // set map = input id or -1
    int input_id = idx[target_id];
    if (input_id < 0 || input_id >= innz) {
      continue;
    }
    // writing into the map
    int buffer_pos = atomicAdd(&knnz[kp], 1);
    int buffer_code = buffer_encoder(kp, buffer_pos);
    map[kp * innz + input_id] = output_id;
  }
}

/*
The query function to build mapping for D2.
*/
template <int _NNZS_PER_BLOCK, int _KOFS_THREADS>
__global__ void _queryhash_sp(
    // input nnz
    const int innz,
    // output nnz
    const int onnz,
    // input coords hash table size
    const int size,
    // output coords, (onnz, 3)
    const int *__restrict__ coord,
    // coded kernel size, f = 311x + 17y + z
    const int ksx, const int ksy, const int ksz,
    // kernel volume
    const int kv,
    // stride
    const int stride_x, const int stride_y, const int stride_z,
    // padding
    const int *__restrict__ padding,
    // hash table (value), (size, )
    const uint64_t *val,
    // hash table (index), (size, )
    const int *idx,
    // input-major mapping, (innz * (kv - 1))
    int *map,
    // the counter of nnz for each each kernel offsets, (kv - 1)
    int *knnz,
    // whether to compute center kernel offset separately
    bool separate_mid) {
  // a thread for a coord
  int output_id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.y;
  // exclude illegal id number
  if (output_id >= onnz) {
    return;
  }
  // shared mem to store the output coord
  // TODO: with batch idx we can use int4 to read
  __shared__ int shm[_NNZS_PER_BLOCK * 4];
  shm[threadIdx.y * 4] = coord[output_id * 4];
  shm[threadIdx.y * 4 + 1] = coord[output_id * 4 + 1] * stride_x - padding[0];
  shm[threadIdx.y * 4 + 2] = coord[output_id * 4 + 2] * stride_y - padding[1];
  shm[threadIdx.y * 4 + 3] = coord[output_id * 4 + 3] * stride_z - padding[2];
  // decode kernel size
  int mid_ks = (ksx - 1) / 2 * ksy * ksz + (ksy - 1) / 2 * ksz + (ksz - 1) / 2;
#pragma unroll
  for (int k = 0;; k += _KOFS_THREADS) {
    int kp = k + threadIdx.x;
    // 0 <= kp < kv
    if (kp >= kv) {
      break;
    }
    // ignore w[0, 0, 0]
    if (separate_mid && kp == mid_ks) {
      continue;
    }
    // corresponds to TorchSparse & SpConv
    int kx = kp / (ksz * ksy) - (ksx - 1) / 2;
    int ky = (kp / ksz) % ksy - (ksy - 1) / 2;
    int kz = kp % ksz - (ksz - 1) / 2;
    kx += (ksx % 2 == 0 || ksx == 1) ? 0 : 1;
    ky += (ksy % 2 == 0 || ksy == 1) ? 0 : 1;
    kz += (ksz % 2 == 0 || ksz == 1) ? 0 : 1;
    // hash query
    uint64_t target_val = coord_hash(
        shm[threadIdx.y * 4], shm[threadIdx.y * 4 + 1] + kx,
        shm[threadIdx.y * 4 + 2] + ky, shm[threadIdx.y * 4 + 3] + kz);
    uint64_t target_id = target_val % (uint64_t)size;
    // find target or empty
    while (val[target_id] != target_val && idx[target_id] > -1) {
      target_id = (target_id + 97) % size;
    }
    // set map = input id or -1
    int input_id = idx[target_id];
    if (input_id < 0 || input_id >= innz) {
      continue;
    }
    // writing into the map
    int buffer_pos = atomicAdd(&knnz[kp], 1);
    int buffer_code = buffer_encoder(kp, buffer_pos);
    map[kp * innz + input_id] = output_id;
  }
}

__global__ void mapping_counter(const int nnz, const int kv, const int *map,
                                int *nnz_neighbor) {
  // a thread for a nnz
  int nid = blockIdx.x * blockDim.x + threadIdx.x;
  if (nid >= nnz) {
    return;
  }
  int counter = 0;
  for (int k = 0; k < kv; k++) {
    bool effective = map[nid * kv + k] >= 0;
    counter += effective ? 1 : 0;
  }
  nnz_neighbor[nid] = counter;
}

__global__ void mapping_decoder(const int sum_nnz,
                                const int *__restrict__ kernel_pos, int *map) {
  // a thread for a mapping
  int mid = blockIdx.x * blockDim.x + threadIdx.x;
  if (mid >= sum_nnz) {
    return;
  }
  int kinf = map[mid];
  int kofs = kernel_decoder(kinf);
  int buf_ofs = kernel_map_decoder(kinf);
  int buf_start = __ldg(&kernel_pos[kofs]);
  map[mid] = buf_start + buf_ofs;
}

/*
The amount of kernel offsets is a small number [1, 125],
so no more optimization is needed. The hand-written
kernel can effectively remove the overhead to call
Thrust::exclusive_scan function.
*/
__global__ void exclusive_scan_for_kernel(const int kv, const int *input,
                                          int *output) {
  // a thread for a scan
  const int id = threadIdx.x + 1;
  if (id >= kv) {
    return;
  }
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < id; i++) {
    acc += input[i];
  }
  output[id] = acc;
}

__global__ void exclusive_scan_for_kernel_quantified(const int kv,
                                                     const int *input,
                                                     const int q, int *output,
                                                     int *qoutput) {
  // a thread for a scan
  const int id = threadIdx.x + 1;
  if (id >= kv) {
    return;
  }
  int acc = 0;
  int qacc = 0;
#pragma unroll
  for (int i = 0; i < id; i++) {
    acc += input[i];
    qacc += (input[i] + q - 1) / q * q;
  }
  output[id] = acc;
  qoutput[id] = qacc;
}

/*
Make sure the coordinates are decoded by batch-first order.
*/
template <int _BOUND>
__global__ void coordsDownsample(
    // amount of non-zeros in input
    const int innz,
    // stride of each dimension
    const int stride_x, const int stride_y, const int stride_z,
    // input coordinates, (innz, 4)
    const int *__restrict__ icoords,
    // coded downsampled output coords, (innz, 1)
    int64_t *ocoords_code) {
  const int nid = blockIdx.x * blockDim.x + threadIdx.x;
  if (nid >= innz) {
    return;
  }
  int64_t code;
  // manually accumulation
  // code = b
  code = icoords[nid * 4];
  // code = b * s + x
  code = code * _BOUND + (icoords[nid * 4 + 1] / stride_x * stride_x);
  // code = (b * s + x) * s + y
  code = code * _BOUND + (icoords[nid * 4 + 2] / stride_y * stride_y);
  // code = ((b * s + x) * s + y) * s + z
  code = code * _BOUND + (icoords[nid * 4 + 3] / stride_z * stride_z);
  ocoords_code[nid] = code;
}

/*
Make sure the coordinates are decoded by batch-first order.
*/
template <int _BOUND, int _NNZS_PER_BLOCK>
__global__ void coordsDownsampleExpand(
    // amount of non-zeros in input
    const int innz,
    // kernel volume
    const int kv,
    // kernel size
    const int ksx, const int ksy, const int ksz,
    // tensor stride of each dimension
    const int t_stride_x, const int t_stride_y, const int t_stride_z,
    // stride of each dimension
    const int stride_x, const int stride_y, const int stride_z,
    // padding of each dimension
    const int *__restrict__ padding,
    // coords boundary
    const int *__restrict__ min, const int *__restrict__ max,
    // input coordinates, (innz, 4)
    const int *__restrict__ icoords,
    // coded downsampled output coords hashtable
    int64_t *ocoords_code) {
  const int nid = blockIdx.x * blockDim.y + threadIdx.y;
  if (nid >= innz) {
    return;
  }
  // input coords
  __shared__ int64_t shm[_NNZS_PER_BLOCK * 4];
  shm[threadIdx.y * 4] = icoords[nid * 4];
  shm[threadIdx.y * 4 + 1] = icoords[nid * 4 + 1];
  shm[threadIdx.y * 4 + 2] = icoords[nid * 4 + 2];
  shm[threadIdx.y * 4 + 3] = icoords[nid * 4 + 3];
#pragma unroll
  for (int k = 0;; k += blockDim.x) {
    int kp = k + threadIdx.x;
    if (kp >= kv) {
      break;
    }
    // int kx = kp / (ksz * ksy) - (ksx - 1) / 2;
    // int ky = (kp / ksz) % ksy - (ksy - 1) / 2;
    // int kz = kp % ksz - (ksz - 1) / 2;
    // corresponds to SpConv & TorchSparse
    int kx = kp / (ksz * ksy) - (ksx - 1) / 2;
    int ky = (kp / ksz) % ksy - (ksy - 1) / 2;
    int kz = kp % ksz - (ksz - 1) / 2;
    kx += (ksx % 2 == 0 || ksx == 1) ? 0 : 1;
    ky += (ksy % 2 == 0 || ksy == 1) ? 0 : 1;
    kz += (ksz % 2 == 0 || ksz == 1) ? 0 : 1;
    // expand
    // kx *= t_stride_x;
    // ky *= t_stride_y;
    // kz *= t_stride_z;
    // candidate
    int cand_x = shm[threadIdx.y * 4 + 1] - kx + padding[0];
    int cand_y = shm[threadIdx.y * 4 + 2] - ky + padding[0];
    int cand_z = shm[threadIdx.y * 4 + 3] - kz + padding[0];
    int cand_x_div = cand_x / stride_x;
    int cand_y_div = cand_y / stride_y;
    int cand_z_div = cand_z / stride_z;
    // condition
    if ((cand_x_div < min[0]) || (cand_x_div > max[0]) ||
        (cand_x % stride_x != 0)) {
      continue;
    }
    if ((cand_y_div < min[1]) || (cand_y_div > max[1]) ||
        (cand_y % stride_y != 0)) {
      continue;
    }
    if ((cand_z_div < min[2]) || (cand_z_div > max[2]) ||
        (cand_z % stride_z != 0)) {
      continue;
    }
    // write into hashtable
    ocoords_code[nid * kv + kp] =
        ((shm[threadIdx.y * 4] * _BOUND + cand_x_div) * _BOUND + cand_y_div) *
            _BOUND +
        cand_z_div;
  }
}

/*
The weights used for linear coding limit the coordinates to
the range of [0, _BOUND).
*/
template <int _BOUND>
__global__ void coordsGenerator(
    // amount of non-zeros in output
    const int onnz,
    // coded downsampled output coords, (onnz, 1)
    const int64_t *__restrict__ ocoords_code,
    // decoded output coordinates, (onnz, 4)
    int *ocoords) {
  const int nid = blockIdx.x * blockDim.x + threadIdx.x;
  if (nid >= onnz) {
    return;
  }
  // TODO: coalesced memory access
  int64_t code = ocoords_code[nid];
  // code = ((b * s + x) * s + y) * s + z
  ocoords[nid * 4 + 3] = code % _BOUND;
  code /= _BOUND;
  // code = (b * s + x) * s + y
  ocoords[nid * 4 + 2] = code % _BOUND;
  code /= _BOUND;
  // code = b * s + x
  ocoords[nid * 4 + 1] = code % _BOUND;
  code /= _BOUND;
  // code = b
  ocoords[nid * 4] = code;
}
