import unittest
from types import SimpleNamespace

import torch

from models.drivergaze_net import DriverGazeNet
from train import Trainer


class FrozenBackboneTest(unittest.TestCase):
    def test_step_preserves_backbone_buffers_and_updates_head(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(freeze_backbone=True)
        trainer.model = DriverGazeNet(
            backbone_name='mobilenet_v3_small', num_segments=2,
            temporal_mode='attn', use_tsm=True, pretrained=False,
            decoder_width=16, apply_sigmoid=False,
        )
        trainer.model.backbone.requires_grad_(False)
        trainer.frozen_backbone_stages = len(trainer.model.backbone.features)
        trainer._set_model_mode(False)
        trainer._set_model_mode(True)
        self.assertTrue(all(not block.training for block in trainer.model.backbone.features))
        self.assertTrue(trainer.model.decoder.training)
        before = {k: v.clone() for k, v in trainer.model.backbone.state_dict().items()}
        heads_before = {k: v.clone() for k, v in trainer.model.named_parameters()
                        if not k.startswith('backbone.')}
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        output = trainer.model(torch.randn(2, 2, 3, 32, 64))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in trainer.model.backbone.parameters()))
        optimizer.step()
        for key, value in trainer.model.backbone.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p, heads_before[k])
                            for k, p in trainer.model.named_parameters() if k in heads_before))
        trainer.args.freeze_backbone = False
        trainer._set_model_mode(True)
        self.assertTrue(trainer.model.backbone.training)


if __name__ == '__main__':
    unittest.main()
