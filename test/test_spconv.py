import torch
import os
import argparse
import time

def kpos_quantized(knnz: torch.Tensor, kpos: torch.Tensor, k_vol: int, q: int):
    kpos_quantized = torch.zeros_like(kpos)
    for k in range(k_vol):
        kpos_quantized[k + 1] = kpos_quantized[k] \
            + torch.div(knnz[k] + q - 1, q, rounding_mode='floor') * q

    snnz_quantized = kpos_quantized[-1].cpu().int().item()

    return kpos_quantized, snnz_quantized

def remove_mid(nnz: int, knnz: torch.Tensor, kpos: torch.Tensor, 
               k_vol: int, q: int, imap: torch.Tensor, omap: torch.Tensor):
    kpos_quantized = torch.zeros_like(kpos)
    kpos_new = torch.zeros_like(kpos)
    mid_k = (k_vol // 2) if k_vol % 2 == 1 else 0
    change_pos = kpos[mid_k]
    imap_new = torch.cat([imap[0:change_pos], 
        torch.linspace(0, nnz-1, nnz, device=imap.device), imap[change_pos:]])
    omap_new = torch.cat([omap[0:change_pos], 
        torch.linspace(0, nnz-1, nnz, device=omap.device), omap[change_pos:]])
    imap_new = imap_new.int().contiguous()
    omap_new = omap_new.int().contiguous()
    for k in range(k_vol):
        if k == mid_k and knnz[k] == 0:
            knnz[k] = nnz
        kpos_quantized[k + 1] = kpos_quantized[k] \
            + torch.div(knnz[k] + q - 1, q, rounding_mode='floor') * q
        kpos_new[k + 1] = kpos_new[k] + knnz[k]
    
    snnz_quantized = kpos_quantized[-1].cpu().int().item()

    return kpos_quantized, snnz_quantized, kpos_new, imap_new, omap_new

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='minkunet-semantickitti')
    parser.add_argument('--precision', type=str, default='fp32')
    parser.add_argument('--fusion', default=False, action='store_true')
    args = parser.parse_args()
    root = './sample-data'
    dir = os.path.join(root, args.precision.lower(), args.benchmark.lower())
    file_list = os.listdir(dir)
    file_list.sort()
    if args.precision == 'fp32':
        Dtype = torch.float
    elif args.precision == 'fp16':
        Dtype = torch.half
    else:
        raise NotImplementedError
    for i, file in enumerate(file_list):
        # loading data info from file ...
        conv_data = torch.load(os.path.join(dir, file))
        sum_nnz = conv_data['sum_nnz']
        out_nnz = conv_data['out_nnz']
        knnz = conv_data['knnz']
        dev_kpos = conv_data['kpos']
        dev_imap = conv_data['imap']
        dev_omap = conv_data['omap']
        in_nnz = conv_data['in_nnz']
        in_channel = conv_data['c_in']
        out_channel = conv_data['c_out']
        k_vol = conv_data['k_vol']
        # generate random input features and kernel weights
        dev_feats = torch.rand((in_nnz, in_channel), dtype=Dtype, device='cuda')
        dev_weights = torch.rand((k_vol, in_channel, out_channel), dtype=Dtype, device='cuda')
    
        separate_mid = in_nnz == out_nnz

        # create output tensor
        dev_output = torch.zeros((out_nnz, out_channel), 
                dtype=dev_feats.dtype, device=dev_feats.device)

        if separate_mid:
            dev_qkpos, qsum_nnz, dev_kpos, dev_imap, dev_omap = remove_mid(
                in_nnz, knnz, dev_kpos, dev_weights.shape[0], 128, dev_imap, dev_omap
            )
            separate_mid = False
        else:
            dev_qkpos, qsum_nnz = kpos_quantized(knnz, dev_kpos, dev_weights.shape[0], 128)
        
        if not args.fusion:
            # print('seq!')
            with torch.no_grad(): 
                for _ in range(10):
                    torch.ops.dgsparse.spconv_fwd_seq(
                        dev_feats, dev_weights, sum_nnz, dev_output, knnz.cpu(), dev_kpos, 
                        dev_imap, dev_omap, separate_mid, True)
                torch.cuda.synchronize()
                st = time.time()
                for i in range(100):
                    torch.ops.dgsparse.spconv_fwd_seq(
                        dev_feats, dev_weights, sum_nnz, dev_output, knnz.cpu(), dev_kpos, 
                        dev_imap, dev_omap, separate_mid, True)
                torch.cuda.synchronize()
                ed = time.time()

        else:
            # print('fusion!')
            with torch.no_grad(): 
                for _ in range(10):
                    torch.ops.dgsparse.spconv_fwd_fused(
                        dev_feats, dev_weights, qsum_nnz, dev_output, dev_kpos, dev_qkpos, 
                        dev_imap, dev_omap, separate_mid, True)
                torch.cuda.synchronize()
                st = time.time()
                for i in range(100):
                    torch.ops.dgsparse.spconv_fwd_fused(
                        dev_feats, dev_weights, qsum_nnz, dev_output, dev_kpos, dev_qkpos, 
                        dev_imap, dev_omap, separate_mid, True)
                torch.cuda.synchronize()
                ed = time.time()
        
        print(
            "[input size=%d, mapping size=%d, in channel=%d, out channel=%d, fusion=%s] duration: %.4f ms" 
            %(in_nnz, sum_nnz, in_channel, out_channel, args.fusion, (ed - st) * 1000 / 100))

        