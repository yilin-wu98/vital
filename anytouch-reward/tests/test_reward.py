import unittest

import torch
import torch.nn.functional as F

from anytouch_reward.reward import pool_tokens, score_patch_embeddings


class RewardTests(unittest.TestCase):
    def test_special_tokens_are_excluded(self):
        tokens = torch.zeros(1, 8, 2)
        tokens[:, :6] = torch.tensor([0., 100.])
        tokens[:, 6:] = torch.tensor([2., 0.])
        torch.testing.assert_close(pool_tokens(tokens), torch.tensor([[1., 0.]]))

    def test_cosine_matches_liquid_driver_formula(self):
        torch.manual_seed(9)
        patches = torch.randn(3, 7, 8)
        positive, negative = torch.randn(1, 8), torch.randn(1, 8)
        pooled = F.normalize(patches.mean(dim=1), dim=-1)
        expected = pooled @ F.normalize(positive, dim=-1).T - pooled @ F.normalize(negative, dim=-1).T
        torch.testing.assert_close(score_patch_embeddings(patches, positive, negative), expected)

    def test_phase_reversal_flips_contrastive_reward(self):
        patches = torch.tensor([[[1., 0.], [1., 0.]]])
        light, heavy = torch.tensor([1., 0.]), torch.tensor([0., 1.])
        a = score_patch_embeddings(patches, light, heavy)
        b = score_patch_embeddings(patches, heavy, light)
        torch.testing.assert_close(a, -b)
        self.assertEqual(a.item(), 1.)

    def test_gradient_reaches_predicted_patches(self):
        patches = torch.randn(2, 4, 8, requires_grad=True)
        score_patch_embeddings(patches, torch.randn(1, 8)).sum().backward()
        self.assertTrue(torch.isfinite(patches.grad).all())
        self.assertGreater(patches.grad.abs().sum().item(), 0)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            pool_tokens(torch.zeros(1, 6, 8))
        with self.assertRaises(ValueError):
            score_patch_embeddings(torch.zeros(1, 4, 8), torch.ones(1, 7))


if __name__ == "__main__":
    unittest.main()
