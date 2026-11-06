import unittest
from unittest.mock import patch
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.mirror_roi import (contains, roi_mask, gaze_label, choose_threshold,
                              binary_metrics, pr_metrics, read_roi)
from scripts.mirror_attention import split_records


class MirrorTests(unittest.TestCase):
    roi = [[[0, .25], [.25, .25], [.25, .75], [0, .75]],
           [[.75, .25], [1, .25], [1, .75], [.75, .75]]]

    def metadata(self):
        return pd.DataFrame({"Frame": np.arange(1, 33), "gaze_x": [50.]*32,
                             "gaze_y": [50.]*32})

    def test_union_boundary_and_resize_geometry(self):
        np.testing.assert_array_equal(contains([[0, .5], [.25, .5], [.5, .5], [.9, .5]], self.roi),
                                      [True, True, False, True])
        for h, w in [(100, 200), (50, 100)]:
            mask = roi_mask(self.roi, h, w)
            self.assertTrue(mask[int(.5*h), int(.2*w)])
            self.assertFalse(mask[int(.5*h), int(.5*w)])

    def test_reviewed_camera_panel_roi_matches_inspected_coordinates(self):
        config = read_roi(Path(__file__).resolve().parents[1] / "configs/mirror_roi.json")
        points = np.array([[6008.275545, 458.956747], [381.880658, 233.154701],
                           [3195.478519, 442.551124]])
        np.testing.assert_array_equal(contains(points/[6400, 720], config["polygons"]),
                                      [True, True, False])
        mask = roi_mask(config["polygons"], 224, 1120)
        self.assertTrue(mask[100, 224])  # Inclusive seam explicitly documented.
        self.assertFalse(mask[100, 225])
        self.assertTrue(mask[100, 896])

    def test_label_uses_only_16_inputs_not_future(self):
        data = self.metadata()
        data.loc[data.Frame == 17, "gaze_x"] = 5
        self.assertEqual(gaze_label(data, list(range(1, 17)), (100, 100), self.roi), 0)
        data.loc[data.Frame == 16, "gaze_x"] = 5
        self.assertEqual(gaze_label(data, list(range(1, 17)), (100, 100), self.roi), 1)

    def test_missing_and_invalid_gaze_excludes_even_positive_clip(self):
        data = self.metadata()
        data.loc[0, "gaze_x"] = 5
        for value in (np.nan, -1., 100.):
            data.loc[1, "gaze_x"] = value
            self.assertIsNone(gaze_label(data, list(range(1, 17)), (100, 100), self.roi))
        self.assertIsNone(gaze_label(data.drop(index=1), list(range(1, 17)), (100, 100), self.roi))

    def test_threshold_ties_and_constant_baseline(self):
        # F1 ties at .8 and .4: higher precision selects .8.
        y, scores = [1, 0, 0, 1], [.8, .7, .6, .4]
        self.assertEqual(choose_threshold(y, scores), .8)
        self.assertAlmostEqual(pr_metrics([1, 0, 1, 0], [.5]*4)["ap"], .5)
        self.assertAlmostEqual(pr_metrics([1, 0], [.9, .1])["ap"], 1.)
        self.assertEqual(binary_metrics([1, 0], [0, 0], .5)["fn"], 1)
        with self.assertRaises(ValueError):
            choose_threshold([0, 0], [.1, .2])

    def test_record_split_is_stable_and_disjoint(self):
        records = ["C001/a", "C001/b", "C002/a", "C003/a"]
        self.assertEqual(split_records(records), split_records(records[::-1]))
        parts = split_records(records)
        self.assertEqual(list(parts.values()).count("calibration"), 2)
        self.assertEqual(list(parts.values()).count("evaluation"), 2)

    def test_roi_requires_review_and_nonempty_geometry(self):
        with patch.object(Path, "read_text", return_value=json.dumps({"confirmed": False, "polygons": self.roi})):
            with self.assertRaises(ValueError):
                read_roi(Path("synthetic_roi.json"))

    def test_training_prediction_does_not_apply_second_activation(self):
        from train import Trainer
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        trainer.use_amp = False
        output = torch.tensor([[[[.2, .8], [.1, .9]]]])
        trainer.model = lambda rgb: output
        mask = torch.ones_like(output)
        prediction = trainer._predict_saliency(None, mask)
        torch.testing.assert_close(prediction, output)
        self.assertNotAlmostEqual(float(prediction.sum()), 1.)

    def test_inference_loads_sigmoid_checkpoint_and_ignores_legacy_pe(self):
        from models.drivergaze_net import DriverGazeNet
        from infer import load_model
        original = DriverGazeNet(backbone_name="mobilenet_v3_small", num_segments=16,
                                 temporal_mode="attn", use_tsm=True, pretrained=False,
                                 decoder_width=16, dropout=0., apply_sigmoid=True).eval()
        state = dict(original.state_dict())
        state["temporal_pe.temporal_pe"] = torch.zeros(1, 16, 576, 1, 1)
        checkpoint = {"model": state, "output": "sigmoid", "experiment": "tsm_attn_pool",
                      "args": {"backbone": "mobilenet_v3_small", "decoder_width": 16, "dropout": 0.}}
        with patch("torch.load", return_value=checkpoint):
            restored, _, _, _, output_type = load_model("synthetic.pt", torch.device("cpu"))
        x = torch.zeros(1, 16, 3, 32, 64)
        with torch.inference_mode():
            expected, actual = original(x), restored(x)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(output_type, "sigmoid")
        self.assertTrue(bool(((actual >= 0) & (actual <= 1)).all()))

    def test_summary_requires_trained_e0_not_initial_evaluation(self):
        from scripts.summarize_experiments import COMMON, summarize
        args = {**COMMON, "freeze_backbone_stages": 0, "backbone_lr_scale": 1.,
                "freeze_backbone": False, "init_weights": "shared.pt",
                "train_path": "train", "val_path": "val"}
        checkpoint = {"args": args, "output": "sigmoid", "val_loss": 1.,
                      "epoch": 2, "val_metrics": {"KLD": 1., "CC": .5, "SIM": .4, "NSS": 2.},
                      "optimizer_updates": 123}
        # No real checkpoint, event log, or running process is used.
        with patch("torch.load", return_value=checkpoint), \
                patch.object(Path, "glob", return_value=[]), \
                patch("scripts.summarize_experiments.sha256", return_value="fixed"):
            # Hide optional receipt while keeping the checkpoint present.
            with patch.object(Path, "is_file", side_effect=[True, False]):
                trained = summarize({"E0": Path("synthetic/best.pt")})[0]
            self.assertEqual(trained["status"], "valid")
            self.assertEqual(trained["epoch"], 3)
            checkpoint["optimizer_updates"] = 0
            with patch.object(Path, "is_file", side_effect=[True, False]):
                initial = summarize({"E0": Path("synthetic/best.pt")})[0]
            self.assertEqual(initial["status"], "protocol_mismatch")


if __name__ == "__main__":
    unittest.main()
