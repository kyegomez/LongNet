import time
import unittest
import torch

from LongNet import DilatedAttention, MultiModalDilationAttention

class TestDilatedAttention(unittest.TestCase):

    def test_output_shape(self):
        # Setup
        input_tensor = torch.randn(2, 128, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64)

        # Action
        output = dilated_attention(input_tensor)

        # Assert
        self.assertEqual(output.shape, (2, 128, 512))

    def test_xpos(self):
        # Setup
        input_tensor = torch.randn(2, 128, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64, use_xpos=True)

        # Action
        output = dilated_attention(input_tensor)

        # Assert
        self.assertEqual(output.shape, (2, 128, 512))

    def test_relative_position_bias(self):
        # Setup
        input_tensor = torch.randn(2, 128, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64, use_rel_pos_bias=True)

        # Action
        output = dilated_attention(input_tensor)

        # Assert
        self.assertEqual(output.shape, (2, 128, 512))

    
    def test_attention_consistency(self):
        # Setup
        input_tensor = torch.randn(2, 128, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64)

        # Action
        output = dilated_attention(input_tensor)

        # Assert
        self.assertTrue((output.std(dim=-1) > 0).all())

    def test_speed(self):
        # Setup
        input_tensor = torch.randn(2, 1024, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64)

        # Action
        start_time = time.time()
        output = dilated_attention(input_tensor)
        end_time = time.time()

        # Assert
        self.assertLess(end_time - start_time, 1)

    def test_gradient_flow(self):
        # Setup
        input_tensor = torch.randn(2, 128, 512, requires_grad=True)
        dilated_attention = DilatedAttention(512, 8, 2, 64)

        # Action
        output = dilated_attention(input_tensor)
        output.sum().backward()
        grad_norm = input_tensor.grad.norm().item()

        # Assert
        self.assertLess(grad_norm, 1e6)
        self.assertGreater(grad_norm, 1e-6)

    def test_scaling(self):
        input_tensor = torch.randn(2, 1024, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64)
        start_time = time.time()
        _ = dilated_attention(input_tensor)
        time_for_1024 = time.time() - start_time
        
        input_tensor = torch.randn(2, 2048, 512)
        start_time = time.time()
        _ = dilated_attention(input_tensor)
        time_for_2048 = time.time() - start_time
        
        self.assertLessEqual(time_for_2048/time_for_1024, 2)
    
    def test_reproducibility(self):
        torch.manual_seed(0)
        input_tensor = torch.randn(2, 128, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64)
        output1 = dilated_attention(input_tensor)
        
        torch.manual_seed(0)
        input_tensor = torch.randn(2, 128, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64)
        output2 = dilated_attention(input_tensor)
        
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_attention_distribution(self):
        input_tensor = torch.randn(2, 128, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64)
        _, attn_weights = dilated_attention(input_tensor)
        
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.tensor(1.)))

        def setUp(self):
            self.d_model = 128
            self.num_heads = 4
            self.dilation_rate = 2
            self.segment_size = 32
            self.dropout = 0.1
            self.casual = False
            self.use_xpos = False
            self.use_rel_pos_bias = False

            self.batch_size = 10
            self.seq_len = 100

            self.x = torch.rand(self.batch_size, self.seq_len, self.d_model)

            self.sparse_dilated_attention = DilatedAttention(self.d_model, self.num_heads, self.dilation_rate, self.segment_size, self.dropout, self.casual, self.use_xpos, self.use_rel_pos_bias)

    def test_forward_pass(self):
        output = self.sparse_dilated_attention(self.x)
        self.assertEqual(output.size(), (self.batch_size, self.seq_len, self.d_model))

    def test_attention_outputs(self):
        output = self.sparse_dilated_attention(self.x)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_dropout(self):
        self.sparse_dilated_attention.dropout.p = 1.0
        output = self.sparse_dilated_attention(self.x)
        self.assertTrue(torch.all(output == 0))
    




# class TestMultiModalDilationAttention(unittest.TestCase):

#     def test_output_shape(self):
#         # Setup
#         input_tensor = [torch.randn(2, 128, 512), torch.randn(2, 128, 512)]
#         multi_modal_attention = MultiModalDilationAttention(512, 8, 2, 64, num_modalities=2)

#         # Action
#         output = multi_modal_attention(input_tensor)

#         # Assert
#         self.assertEqual(output.shape, (2, 128, 512))

#     def test_single_modality(self):
#         # Setup
#         input_tensor = [torch.randn(2, 128, 512)]
#         multi_modal_attention = MultiModalDilationAttention(512, 8, 2, 64, num_modalities=1)

#         # Action
#         output = multi_modal_attention(input_tensor)

#         # Assert
#         self.assertEqual(output.shape, (2, 128, 512))


if __name__ == '__main__':
    unittest.main()



