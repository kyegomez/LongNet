import unittest
import torch
from LongNet import DilatedLongNet


class TestDilatedLongNet(unittest.TestCase):
    def setUp(self):
        self.model = DilatedLongNet()

    def test_model_shape(self):
        # Test input and output dimensions
        x = torch.randint(0, 16000, (4, 1024))
        out = self.model(x)
        self.assertEqual(out.shape, (4, 1024, 16000))

    def test_generate(self):
        # Test sample generation
        out = self.model.generate(x, temperature=1.0, filter_thres=0.9)
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 1024)
        self.assertEqual(out.shape[2], 4)

    def test_dilation(self):
        # Test dilated attention
        self.assertEqual(self.model.dilation_rate, 1)
        self.assertEqual(self.model.segment_size, 0)
        self.assertFalse(self.model.casual)

    def test_gradients(self):
        # Test backward pass
        x = torch.randint(0, 16000, (4, 1024))
        out = self.model(x)
        out.backward()
        for name, param in self.model.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertFalse(torch.isnan(param.grad).any())
            param.grad.zero_()

    def test_training(self):
        # End-to-end training test
        optim = torch.optim.Adam(self.model.parameters())
        for _ in range(100):
            x = torch.randint(0, 16000, (4, 1024))
            loss = self.model(x).loss
            optim.zero_grad()
            loss.backward()
            optim.step()
        self.assertLess(loss.item(), 10)


if __name__ == "__main__":
    unittest.main()
