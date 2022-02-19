
import torch
import os
from torch.utils.cpp_extension import load
from torch.nn import Parameter, init
import torch.nn.functional as F

path=os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

spmm = load(name='spmm', sources=[
            os.path.join(path, 'src/gspmm-fp/gspmm.cc'), os.path.join(path, 'src/gspmm-fp/gspmm.cu')],  verbose=True)

def u_add_e_sum(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.SUM, spmm.COMPUTEOP.ADD)

def u_sub_e_sum(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.SUM, spmm.COMPUTEOP.SUB)

def u_mul_e_sum(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.SUM, spmm.COMPUTEOP.MUL)
    
def u_div_e_sum(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.SUM, spmm.COMPUTEOP.DIV)
    
def u_add_e_max(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MAX, spmm.COMPUTEOP.ADD)
    
def u_sub_e_max(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MAX, spmm.COMPUTEOP.SUB)
    
def u_mul_e_max(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MAX, spmm.COMPUTEOP.MUL)
    
def u_div_e_max(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MAX, spmm.COMPUTEOP.DIV)
    
def u_add_e_min(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MIN, spmm.COMPUTEOP.ADD)
    
def u_sub_e_min(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MIN, spmm.COMPUTEOP.SUB)
    
def u_mul_e_min(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MIN, spmm.COMPUTEOP.MUL)
    
def u_div_e_min(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MIN, spmm.COMPUTEOP.DIV)
    
def u_add_e_mean(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MEAN, spmm.COMPUTEOP.ADD)
    
def u_sub_e_mean(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MEAN, spmm.COMPUTEOP.SUB)
    
def u_mul_e_mean(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MEAN, spmm.COMPUTEOP.MUL)
    
def u_div_e_mean(rowptr, colind, edge_feature, node_feat):
    return spmm.GSpMM_u_e(rowptr, colind, edge_feature, node_feat, spmm.REDUCEOP.MEAN, spmm.COMPUTEOP.DIV)

def copy_u_sum(rowptr, colind, node_feat):
    return spmm.GSpMM_u(rowptr, colind, node_feat, spmm.REDUCEOP.SUM)

def copy_u_max(rowptr, colind, node_feat):
    return spmm.GSpMM_u(rowptr, colind, node_feat, spmm.REDUCEOP.MAX)

def copy_u_min(rowptr, colind, node_feat):
    return spmm.GSpMM_u(rowptr, colind, node_feat, spmm.REDUCEOP.MIN)

def copy_u_mean(rowptr, colind, node_feat):
    return spmm.GSpMM_u(rowptr, colind, node_feat, spmm.REDUCEOP.MEAN)
