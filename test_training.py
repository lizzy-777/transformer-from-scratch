"""Regression checks: python -m unittest -v test_training.py"""
import unittest
import torch
import torch.nn.functional as F
from mask import create_causal_mask
from multi_head_attention import MultiHeadAttention
from transformer import Transformer
from train import BOS, EOS, PAD, collate, greedy_decode, make_data
import random


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = Transformer(16, 16, d_model=16, num_heads=2, d_ff=32,
                                 num_layers=2, max_len=16, padding_idx=PAD).eval()

    def test_shift_and_disjoint_data(self):
        src, decoder_in, labels = collate([(4, 5), (6, 7, 8)], 'cpu')
        self.assertEqual(decoder_in.tolist(), [[BOS, 4, 5, EOS], [BOS, 6, 7, 8]])
        self.assertEqual(labels.tolist(), [[4, 5, EOS, PAD], [6, 7, 8, EOS]])
        self.assertEqual(src.tolist(), [[4, 5, EOS, PAD], [6, 7, 8, EOS]])
        train = make_data(50, 6, 16, random.Random(1))
        val = make_data(50, 6, 16, random.Random(2), train)
        self.assertFalse(set(train) & set(val))

    def test_attention_and_cross_attention(self):
        attention = MultiHeadAttention(16, 2)
        query, memory = torch.randn(2, 4, 16), torch.randn(2, 6, 16)
        mask = create_causal_mask(4)
        output, weights = attention(query, mask=mask)
        self.assertEqual(output.shape, (2, 4, 16))
        self.assertTrue((weights.masked_select(~mask) == 0).all())
        output, weights = attention(query, memory, memory)
        self.assertEqual(weights.shape, (2, 2, 4, 6))

    def test_full_model_causality(self):
        src = torch.tensor([[4, 5, EOS]])
        first = torch.tensor([[BOS, 6, 7, 8]])
        changed = torch.tensor([[BOS, 6, 9, 10]])
        with torch.no_grad():
            a, b = self.model(src, first), self.model(src, changed)
        torch.testing.assert_close(a[:, :2], b[:, :2])

    def test_padding_invariance_and_gradients(self):
        src, decoder_in, labels = collate([(4, 5), (6, 7, 8)], 'cpu')
        with torch.no_grad():
            alone = self.model(src[:1, :3], decoder_in[:1, :3])
            batched = self.model(src, decoder_in)
        torch.testing.assert_close(alone, batched[:1, :3], atol=1e-6, rtol=1e-5)
        loss = F.cross_entropy(self.model(src, decoder_in).flatten(0, 1),
                               labels.flatten(), ignore_index=PAD)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_decode_stops_at_eos(self):
        with torch.no_grad():
            self.model.fc_out.weight.zero_()
            self.model.fc_out.bias.zero_()
            self.model.fc_out.bias[EOS] = 10
        prediction = greedy_decode(self.model, torch.tensor([[4, EOS], [5, EOS]]), 8)
        self.assertEqual(prediction.tolist(), [[EOS], [EOS]])


if __name__ == '__main__':
    unittest.main()
