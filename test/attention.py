import unittest 
import torch 


from LongNet.attention import DilatedAttention, MultiModalDilationAttention



class TestDilatedAttention(unittest.TestCase):

    def test_output_shape(self):
        #setup
        input_tensor = torch.randn(2, 128, 518)
        dilated_attention = DilatedAttention(512, 8, 2, 64)

        #action 
        output = dilated_attention(input_tensor)

        #assert
        self.assertEqual(output.shape, (2, 128, 512))

    def test_xpos(self):
        #setup
        input_tensor = torch.randn(2, 128, 512)

        dilated_attention = DilatedAttention(512, 8, 2, 64, use_xpos=True)

        #action 
        output = dilated_attention(input_tensor)

        #assert 
        self.assertEqual(output.shape, (2, 128, 512))

    def test_relative_position_bias(self):
        #setup
        input_tensor = torch.randn(2, 128, 512)
        dilated_attention = DilatedAttention(512, 8, 2, 64, use_rel_pos_bias=True)

        #action
        output = dilated_attention(input_tensor)

        #assetr
        self.assertEqual(output.shape, (2, 128, 512))


class TestMultiModalDilatedAttention(unittest.TestCase):
    def test_output_shape(self):
        #setup
        input_tensor = [torch.randn(2, 128, 512), torch.randn(2, 128, 512)]
        multi_modal_attention = MultiModalDilationAttention(512, 8, 2, 64, num_modalities=2)

        #action 
        output = multi_modal_attention(input_tensor)

        #assert
        self.assertEqual(output.shape, (2, 128, 512))

    def test_single_modal(self):
        #setup
        input_tensor = [torch.randn(2, 128, 512)]
        multi_modal_attention = MultiModalDilationAttention(512, 8, 2, 64, num_modalities=1)


        #action
        output = multi_modal_attention(input_tensor)

        #assert
        self.assertEqual(output.shape, (2, 128, 512))


if __name__ == "__main__":
    unittest.main()