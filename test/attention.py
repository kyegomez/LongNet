import time
import unittest
import torch

from LongNet.attention import DilatedAttention, MultiModalDilationAttention


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




class TestMultiModalDilationAttention(unittest.TestCase):

    def test_output_shape(self):
        # Setup
        input_tensor = [torch.randn(2, 128, 512), torch.randn(2, 128, 512)]
        multi_modal_attention = MultiModalDilationAttention(512, 8, 2, 64, num_modalities=2)

        # Action
        output = multi_modal_attention(input_tensor)

        # Assert
        self.assertEqual(output.shape, (2, 128, 512))

    def test_single_modality(self):
        # Setup
        input_tensor = [torch.randn(2, 128, 512)]
        multi_modal_attention = MultiModalDilationAttention(512, 8, 2, 64, num_modalities=1)

        # Action
        output = multi_modal_attention(input_tensor)

        # Assert
        self.assertEqual(output.shape, (2, 128, 512))


if __name__ == '__main__':
    unittest.main()



