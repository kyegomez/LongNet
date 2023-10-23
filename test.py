import torch
import torch.nn.functional as F


class DilatedAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, segment_length, dilation_rates):
        super(DilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.segment_length = segment_length
        self.dilation_rates = dilation_rates
        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        batch_size, seq_len, _ = Q.size()
        O_outputs = []
        for r in self.dilation_rates:
            O_list = []
            for i in range(0, seq_len, self.segment_length):
                Q_segment = Q[:, i : min(i + self.segment_length, seq_len), :]
                K_segment = K[:, i : min(i + self.segment_length, seq_len), :]
                V_segment = V[:, i : min(i + self.segment_length, seq_len), :]

                Q_sparsed = Q_segment[:, ::r, :]
                K_sparsed = K_segment[:, ::r, :]
                V_sparsed = V_segment[:, ::r, :]

                attn_weights = F.softmax(
                    Q_sparsed.matmul(K_sparsed.transpose(-2, -1)), dim=-1
                )
                Oi = attn_weights.matmul(V_sparsed)

                Oi_padded = torch.zeros(
                    batch_size, self.segment_length, self.d_model
                ).to(Q.device)

                Oi_padded[:, ::r, :] = Oi

                O_list.append(Oi_padded)
            Oi_concatenated = torch.cat(O_list, dim=1)
            O_outputs.append(Oi_concatenated)

        # Weighted sum of the attentions
        O = torch.stack(O_outputs).sum(0)

        return O

    def visual_forward(self, Q, K, V):
        batch_size, seq_len, _ = Q.size()
        O_outputs = []
        visualizations = []

        for r in self.dilation_rates:
            O_list = []
            attn_viz_list = []

            for i in range(0, seq_len, self.segment_length):
                Q_segment = Q[:, i : min(i + self.segment_length, seq_len), :]
                K_segment = K[:, i : min(i + self.segment_length, seq_len), :]
                V_segment = V[:, i : min(i + self.segment_length, seq_len), :]

                Q_sparsed = Q_segment[:, ::r, :]
                K_sparsed = K_segment[:, ::r, :]
                V_sparsed = V_segment[:, ::r, :]

                attn_weights = F.softmax(
                    Q_sparsed.matmul(K_sparsed.transpose(-2, -1)), dim=-1
                )
                attn_viz_list.append(attn_weights)

                Oi = attn_weights.matmul(V_sparsed)

                Oi_padded = torch.zeros(
                    batch_size, self.segment_length, self.d_model
                ).to(Q.device)

                Oi_padded[:, ::r, :] = Oi

                O_list.append(Oi_padded)

            Oi_concatenated = torch.cat(O_list, dim=1)
            O_outputs.append(Oi_concatenated)

            visualizations.append(torch.cat(attn_viz_list, dim=1))

        # Weighted sum of the attentions
        O = torch.stack(O_outputs).sum(0)

        return O, visualizations

        return O


# # Example usage:
# d_model = 512
# num_heads = 8
# segment_length = 4
# dilation_rates = [1, 2, 4]
# attention = DilatedAttention(d_model, num_heads, segment_length, dilation_rates)

# Q = torch.rand(32, 128, d_model)
# K = torch.rand(32, 128, d_model)
# V = torch.rand(32, 128, d_model)
# output = attention(Q, K, V)
# print(output)
