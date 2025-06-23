#include "operators.h"
#include <cmath>

float q_buf[4096], k_buf[4096];
// TODO: optimize this with multithreading
void RotaryPosEmb::apply(Matrix3D<float> &query, Matrix3D<float> &key,
                           int start_idx, int len) {
  PROFILE_START(profile_name);
  int num_q_head = query.m_dim_x;
  int num_k_head = key.m_dim_x;
  int head_dim = cos.m_dim_z;
  int max_sqlen = cos.m_dim_y;

  assert(query.m_dim_z == cos.m_dim_z);
  assert(key.m_dim_z == cos.m_dim_z);
  assert(max_sqlen > len + start_idx);

  // cos, sin = self.rotary_emb(key_states, seq_len=kv_seq_len)
  // query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
  // cos, sin, position_ids) cos = cos[position_ids].unsqueeze(1)  # [bs, 1,
  // seq_len, dim] sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
  // q_embed = (q * cos) + (rotate_half(q) * sin)
  // k_embed = (k * cos) + (rotate_half(k) * sin)
  // x1 = x[..., : x.shape[-1] // 2]
  // x2 = x[..., x.shape[-1] // 2 :]
  // rotate_half: torch.cat((-x2, x1), dim=-1)

  int half = head_dim / 2;
  // q
  for (int b = 0; b < num_q_head; b++) 
  {
    for (int i = 0; i < len; i++) 
    {
      // first half
      for (int j = 0; j < half; j++) 
      {
        q_buf[j] = -1 * query(b, i, j + half);
      }
      // second half
      for (int j = half; j < head_dim; j++) 
      {
        q_buf[j] = query(b, i, j - half);
      }

      for (int j = 0; j < head_dim; j++) 
      {
        query(b, i, j) = (
          (query(b, i, j) * cos(0, i + start_idx, j)) +
          (q_buf[j] * sin(0, i + start_idx, j))
        );
      }
    }
  }

  // k
  for (int b = 0; b < num_k_head; b++) 
  {
    for (int i = 0; i < len; i++) 
    {
      // first half
      for (int j = 0; j < half; j++) 
      {
        k_buf[j] = -1 * key(b, i, j + half);
      }
      // second half
      for (int j = half; j < head_dim; j++) 
      {
        k_buf[j] = key(b, i, j - half);
      }

      for (int j = 0; j < head_dim; j++) 
      {
        key(b, i, j) = ((key(b, i, j) * cos(0, i + start_idx, j)) +
                        (k_buf[j] * sin(0, i + start_idx, j)));
      }
    }
  }
  PROFILE_END(profile_name);
}
