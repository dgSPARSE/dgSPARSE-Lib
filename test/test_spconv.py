import torch
import os
import argparse
import time
import dgsparse


def kpos_quantized(knnz: torch.Tensor, kpos: torch.Tensor, k_vol: int, q: int):
    kpos_quantized = torch.zeros_like(kpos)
    for k in range(k_vol):
        kpos_quantized[k + 1] = (
            kpos_quantized[k] + torch.div(knnz[k] + q - 1, q, rounding_mode="floor") * q
        )

    snnz_quantized = kpos_quantized[-1].cpu().int().item()

    return kpos_quantized, snnz_quantized


def cpu_compute(
    gpu_feats: torch.Tensor,
    gpu_weights: torch.Tensor,
    out_size: int,
    knnz: torch.Tensor,
    dev_imap: torch.Tensor,
    dev_omap: torch.Tensor,
    precompute: bool,
):
    feats = gpu_feats.cpu()
    weights = gpu_weights.cpu()
    imap = dev_imap.cpu()
    omap = dev_omap.cpu()
    k_vol = weights.size(0)
    c_in = weights.size(1)
    c_out = weights.size(2)
    mid_k = k_vol // 2 if (k_vol % 2 == 1) else 0
    output = torch.zeros((out_size, c_out), dtype=feats.dtype)
    nbaddr = 0
    for k in range(k_vol):
        nbsize = knnz[k]
        if nbsize == 0:
            continue
        for i in range(nbsize):
            in_index = imap[nbaddr]
            out_index = omap[nbaddr]
            for co in range(c_out):
                for ci in range(c_in):
                    output[out_index, co] += feats[in_index, ci] * weights[k, ci, co]
            nbaddr += 1
    if precompute:
        for i in range(out_size):
            for co in range(c_out):
                for ci in range(c_in):
                    output[i, co] += feats[i, ci] * weights[mid_k, ci, co]
    return output


def remove_mid(
    nnz: int,
    knnz: torch.Tensor,
    kpos: torch.Tensor,
    k_vol: int,
    q: int,
    imap: torch.Tensor,
    omap: torch.Tensor,
):
    kpos_quantized = torch.zeros_like(kpos)
    kpos_new = torch.zeros_like(kpos)
    mid_k = (k_vol // 2) if k_vol % 2 == 1 else 0
    change_pos = kpos[mid_k]
    imap_new = torch.cat(
        [
            imap[0:change_pos],
            torch.linspace(0, nnz - 1, nnz, device=imap.device),
            imap[change_pos:],
        ]
    )
    omap_new = torch.cat(
        [
            omap[0:change_pos],
            torch.linspace(0, nnz - 1, nnz, device=omap.device),
            omap[change_pos:],
        ]
    )
    imap_new = imap_new.int().contiguous()
    omap_new = omap_new.int().contiguous()
    for k in range(k_vol):
        if k == mid_k and knnz[k] == 0:
            knnz[k] = nnz
        kpos_quantized[k + 1] = (
            kpos_quantized[k] + torch.div(knnz[k] + q - 1, q, rounding_mode="floor") * q
        )
        kpos_new[k + 1] = kpos_new[k] + knnz[k]

    snnz_quantized = kpos_quantized[-1].cpu().int().item()

    return kpos_quantized, snnz_quantized, kpos_new, imap_new, omap_new


def test_spconv():
    root = "../example/data/sample-data"
    precision = "fp32"
    dir = os.path.join(root, precision, "minkunet-semantickitti")
    file_list = os.listdir(dir)
    file_list.sort()
    if precision == "fp32":
        Dtype = torch.float
    elif precision == "fp16":
        Dtype = torch.half
    else:
        raise NotImplementedError
    for i, file in enumerate(file_list):
        # loading data info from file ...
        conv_data = torch.load(os.path.join(dir, file))
        sum_nnz = conv_data["sum_nnz"]
        out_nnz = conv_data["out_nnz"]
        knnz = conv_data["knnz"]
        kpos = conv_data["kpos"]
        in_map = conv_data["imap"]
        out_map = conv_data["omap"]
        in_nnz = conv_data["in_nnz"]
        in_channel = conv_data["c_in"]
        out_channel = conv_data["c_out"]
        k_vol = conv_data["k_vol"]

        separate_mid = in_nnz == out_nnz

        if separate_mid:
            qkpos, sum_nnz, kpos, in_map, out_map = remove_mid(
                in_nnz, knnz, kpos, k_vol, 128, in_map, out_map
            )
            separate_mid = False
        else:
            qkpos, sum_nnz = kpos_quantized(knnz, kpos, k_vol, 128)

            # generate random input features and kernel weights
        in_feats = torch.rand((in_nnz, in_channel), dtype=Dtype, device="cuda")
        kernel = torch.rand(
            (k_vol, in_channel, out_channel), dtype=Dtype, device="cuda"
        )

        dev_output = torch.ops.dgsparse_spconv.spconv(
            in_feats,
            kernel,
            kpos,
            qkpos,
            in_map,
            out_map,
            out_nnz,
            sum_nnz,
            separate_mid,
            True,
        )

        output = dev_output.clone().cpu()
        out_sum = out_nnz * out_channel

        # output_cpu = cpu_compute(
        #     in_feats, kernel, out_nnz, knnz, in_map, out_map, separate_mid
        # )

        # assert output.shape == output_cpu.shape

        # assert torch.allclose(output, output_cpu, rtol=1e-2, atol=1e-4) == True
