import unittest

import torch

from utils.losses import SaliencyLoss


class SaliencyLossGradientTests(unittest.TestCase):
    def test_constant_maps_have_finite_gradients(self):
        for constant_prediction, constant_target in [(True, False), (False, True), (True, True)]:
            with self.subTest(pred=constant_prediction, target=constant_target):
                torch.manual_seed(12)
                logits = (torch.zeros(2, 1, 4, 8) if constant_prediction
                          else torch.randn(2, 1, 4, 8)).requires_grad_()
                prediction = logits.flatten(1).softmax(1).view_as(logits)
                target = torch.ones_like(logits) if constant_target else torch.rand_like(logits)
                target = target / target.sum((-2, -1), keepdim=True)
                loss = SaliencyLoss()(prediction, target)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_cc_matches_pearson_value_and_gradient_for_regular_maps(self):
        torch.manual_seed(7)
        prediction = torch.rand(2, 1, 8, 16, requires_grad=True)
        target = torch.rand_like(prediction)
        actual = SaliencyLoss(lambda_kld=0)(prediction, target)
        p = prediction.flatten(1)
        q = target.flatten(1)
        p = p - p.mean(1, keepdim=True)
        q = q - q.mean(1, keepdim=True)
        expected = -((p * q).sum(1) / (p.square().sum(1) * q.square().sum(1)).sqrt()).mean()
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad(actual, prediction, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, prediction)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_constant_valid_region_with_mask_and_weights(self):
        prediction = torch.tensor([[[[1., 1., 9., 8.]]]], requires_grad=True)
        target = torch.tensor([[[[1., 2., 4., 3.]]]])
        mask = torch.tensor([[[[True, True, False, False]]]])
        loss = SaliencyLoss()(prediction, target, mask=mask, sample_weights=torch.tensor([2.]))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(prediction.grad[..., 2:], torch.zeros_like(prediction.grad[..., 2:]))

    def test_nss_uniform_prediction_has_finite_gradient(self):
        prediction = torch.ones(1, 1, 4, 8, requires_grad=True)
        fixation = torch.zeros_like(prediction)
        fixation[..., 1, 2] = 1
        loss = SaliencyLoss(lambda_kld=0, lambda_cc=0, lambda_nss=1)(
            prediction, torch.ones_like(prediction), fixation=fixation
        )
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(loss.item(), 0.)


if __name__ == '__main__':
    unittest.main()
