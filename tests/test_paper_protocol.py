import unittest

import torch

from dataset.dataset import DG360LiteDataset, paper_fixation_frame_ids
from models.drivergaze_net import DriverGazeNet
from utils.losses import SaliencyLoss


class PaperProtocolTests(unittest.TestCase):
    def test_sampling_matches_local_source_end_exclusive_range(self):
        dataset = DG360LiteDataset.__new__(DG360LiteDataset)
        dataset.paper_protocol = True
        dataset.data_len = [17, 30]
        dataset.T = 4
        dataset.stride = 1
        dataset.skip_first = 3
        dataset.clip_step = 4
        self.assertEqual(dataset._make_samples(), [
            (0, 3), (0, 7), (0, 11),
            (1, 3), (1, 7), (1, 11), (1, 15), (1, 19), (1, 23),
        ])

    def test_source_fixation_uses_input_and_following_clip(self):
        ids = list(range(1, 41))
        self.assertEqual(
            paper_fixation_frame_ids(ids, start=10, T=4, stride=2),
            [11, 13, 15, 17, 19, 21, 23, 25],
        )
        self.assertEqual(
            paper_fixation_frame_ids(ids, start=34, T=4, stride=2),
            [35, 37, 39],
        )

    def test_old_pe_state_loads_into_model_without_pe(self):
        torch.manual_seed(3)
        model = DriverGazeNet(
            backbone_name="mobilenet_v3_small", num_segments=2,
            temporal_mode="attn", use_tsm=True, pretrained=False,
            decoder_width=16, apply_sigmoid=False,
        )
        state = model.state_dict()
        state["temporal_pe.temporal_pe"] = torch.randn(1, 2, 576, 1, 1)
        model.load_checkpoint_weights(state)
        self.assertFalse(any("temporal_pe" in key for key in model.state_dict()))
        output = model(torch.randn(1, 2, 3, 32, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_source_saliency_loss_matches_kld_plus_one_minus_cc(self):
        prediction = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], requires_grad=True)
        target = torch.tensor([[[[0.4, 0.3], [0.2, 0.1]]]])
        eps = 1e-8
        p = prediction.flatten(1)
        q = target.flatten(1)
        p = (p / (p.sum(1, keepdim=True) + eps)).clamp_min(eps)
        q = (q / (q.sum(1, keepdim=True) + eps)).clamp_min(eps)
        kld = (q * (q.log() - p.log())).sum(1)
        p_centered = p - p.mean(1, keepdim=True)
        q_centered = q - q.mean(1, keepdim=True)
        cc = ((p_centered * q_centered).sum(1) /
              (p_centered.square().sum(1) * q_centered.square().sum(1)).sqrt())
        expected = (kld + 1 - cc).mean()
        actual = SaliencyLoss(lambda_kld=1, lambda_cc=1, profile="drivergaze360")(
            prediction, target
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
