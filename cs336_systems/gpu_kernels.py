import torch
import numpy as np
from einops import einsum, rearrange
import triton
import triton.language as tl

class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        q_tile_size = 16
        k_tile_size = 32
        Tq = int(np.ceil(Q.shape[1]/q_tile_size))
        Tk = int(np.ceil(K.shape[1]/k_tile_size))

        Q_blocks = torch.split(Q, q_tile_size, dim=1)
        K_blocks = torch.split(K, k_tile_size, dim=1)
        V_blocks = torch.split(V, k_tile_size, dim=1)

        o = torch.empty(Q.shape[0], 0, Q.shape[2], device=Q.device, dtype=torch.float32)
        l = torch.empty(Q.shape[0], 0, 1, device=Q.device, dtype=torch.float32)
        
        for i in range(Tq):
            q_i = Q_blocks[i]
            m_i = torch.full((q_i.shape[0], q_i.shape[1], 1), -float('inf'), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((q_i.shape[0], q_i.shape[1], 1), device=Q.device, dtype=torch.float32)
            o_i = torch.zeros((q_i.shape[0], q_i.shape[1], q_i.shape[2]), device=Q.device, dtype=torch.float32)
            
            for j in range(Tk):
                k_j = K_blocks[j]
                v_j = V_blocks[j]
                attn_scores = einsum(q_i, k_j, "batch_size q_tile_size dim, batch_size k_tile_size dim -> batch_size q_tile_size k_tile_size") * (1/np.sqrt(q_i.shape[2]))
                m_i_new = torch.maximum(torch.max(attn_scores, dim=2, keepdim=True).values, m_i)
                p_i = torch.exp(attn_scores - m_i_new)
                l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.sum(p_i, dim=2, keepdim=True)
                o_i_new = torch.exp(m_i - m_i_new) * o_i + einsum(p_i, v_j, "batch_size q_tile_size k_tile_size, batch_size k_tile_size dim -> batch_size q_tile_size dim")
                m_i = m_i_new
                l_i = l_i_new
                o_i = o_i_new
            o_i = o_i / l_i
            l_i = m_i + torch.log(l_i)
            o = torch.cat((o, o_i), dim=1)
            l = torch.cat((l, l_i), dim=1)
        ctx.save_for_backward(Q, K, V, o, rearrange(l, 'batch_size seq_len 1 -> batch_size seq_len'))
        ctx.is_causal = is_causal
        return o.to(Q.dtype)

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    q = tl.load(Q_block_ptr)
    offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    for j in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_j = tl.load(K_block_ptr)
        v_j = tl.load(V_block_ptr)
        attn_scores = tl.dot(q, tl.trans(k_j)) * scale
        if is_causal:
            offs_k = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = offs_q[:, None] >= offs_k[None, :]
            attn_scores = tl.where(mask, attn_scores, -1e6)
        m_i_new = tl.maximum(tl.max(attn_scores, axis=1), m_i)
        p_i = tl.exp(attn_scores - m_i_new[:, None])
        l_i_new = tl.exp(m_i - m_i_new) * l_i + tl.sum(p_i, axis=1)
        o_i_new = tl.exp(m_i - m_i_new)[:, None] * o_i + tl.dot(p_i.to(v_j.dtype), v_j)
        m_i = m_i_new
        l_i = l_i_new
        o_i = o_i_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    tl.store(O_block_ptr, (o_i / l_i[:, None]).to(O_ptr.type.element_ty))
    tl.store(L_block_ptr, m_i + tl.log(l_i))

        
    
class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        q_tile_size = 16
        k_tile_size = 32
        Tq = int(np.ceil(Q.shape[1]/q_tile_size))
        Tk = int(np.ceil(K.shape[1]/k_tile_size))

        o = torch.empty(Q.shape[0], Q.shape[1], Q.shape[2], device=Q.device, dtype=Q.dtype)
        l = torch.empty(Q.shape[0], Q.shape[1], device=Q.device, dtype=torch.float32)

        grid = (Tq, Q.shape[0])
        flash_fwd_kernel[grid](Q, K, V, o, l,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        l.stride(0), l.stride(1),
        Q.shape[1], K.shape[1],
        1/np.sqrt(Q.shape[2]),
        D=Q.shape[2],
        Q_TILE_SIZE=q_tile_size,
        K_TILE_SIZE=k_tile_size,
        is_causal=is_causal)

        ctx.save_for_backward(Q, K, V, o, l)
        return o 

        
            

            

                
                
                

        