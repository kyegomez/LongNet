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
