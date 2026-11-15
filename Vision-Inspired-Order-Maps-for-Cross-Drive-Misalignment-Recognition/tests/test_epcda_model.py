"""Focused contract tests for the auditable EPCDA implementation."""

import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest

import torch

from epcda_model import (
    CDANDomainHead,
    EPCDAModel,
    OrderTrackedFrontEnd,
    centered_cross_covariance_loss,
    conditional_domain_features,
    grad_reverse,
)


class EPCDAModelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    @staticmethod
    def _inputs(batch_size: int = 4):
        x = torch.randn(batch_size, 3, 128)
        rpm_raw = 900.0 + 150.0 * torch.rand(batch_size, 128)
        return x, rpm_raw

    def test_full_forward_shapes_and_required_outputs(self) -> None:
        x, rpm_raw = self._inputs()
        model = EPCDAModel(dropout=0.0)
        model.eval()
        output = model(x, rpm_raw, fs=2000.0, grl_lambda=0.5)

        required = {
            "h",
            "zf",
            "zp",
            "cls",
            "dom",
            "powp",
            "powf",
            "orders",
            "sigmas",
        }
        self.assertTrue(required.issubset(output))
        self.assertEqual(output["h"].shape, (4, 128))
        self.assertEqual(output["zf"].shape, (4, 32))
        self.assertEqual(output["zp"].shape, (4, 32))
        self.assertEqual(output["cls"].shape, (4, 5))
        self.assertEqual(output["dom"].shape, (4, 2))
        self.assertEqual(output["powp"].shape, (4, 3))
        self.assertEqual(output["powf"].shape, (4, 3))
        self.assertEqual(output["orders"].shape, (16,))
        self.assertEqual(output["sigmas"].shape, (16,))
        self.assertEqual(output["front_features"].shape, (4, 144, 5))
        self.assertEqual(output["amplitude"].shape, (4, 3, 16, 5))
        self.assertEqual(output["coherence"].shape, (4, 3, 16, 5))

    def test_energy_only_and_raw_encoder_alternatives(self) -> None:
        x, rpm_raw = self._inputs(batch_size=2)
        energy_model = EPCDAModel(front_mode="energy_only", dropout=0.0).eval()
        energy_output = energy_model(x, rpm_raw, fs=torch.tensor([2000.0, 2100.0]))
        self.assertEqual(energy_output["front_features"].shape, (2, 48, 5))

        raw_model = EPCDAModel(raw_encoder=True, dropout=0.0).eval()
        raw_output = raw_model(x, rpm_raw, fs=2000.0)
        self.assertIsNone(raw_output["front_features"])
        self.assertIsNone(raw_output["coherence"])
        self.assertEqual(raw_output["h"].shape, (2, 128))
        self.assertEqual(raw_output["orders"].shape, (16,))

    def test_gaussian_kernels_are_unit_l2_and_orders_are_bounded(self) -> None:
        front = OrderTrackedFrontEnd()
        gaussian = front.gaussian_windows()
        norms = torch.linalg.vector_norm(gaussian, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.all(front.orders() >= 0.25))
        self.assertTrue(torch.all(front.orders() <= 6.0))

    def test_relative_phase_has_reference_constraint_and_nonzero_gradient(self) -> None:
        front = OrderTrackedFrontEnd(mode="full")
        x, rpm_raw = self._inputs(batch_size=2)

        result = front(x, rpm_raw, fs=2000.0)
        phase_offsets = result["phase_offsets"]
        self.assertEqual(tuple(front.phase_logits.shape), (2, 16))
        self.assertEqual(phase_offsets.shape, (3, 16))
        torch.testing.assert_close(phase_offsets[0], torch.zeros_like(phase_offsets[0]))

        # A coherence-dependent loss must update the learnable current-channel
        # phases.  The old shared phase produced only round-off-level gradients
        # because it cancelled exactly from magnitude and Z_a Z_b*.
        weights = torch.linspace(
            0.25, 1.25, result["features"].numel(), dtype=result["features"].dtype
        ).reshape_as(result["features"])
        # Different weights on the flattened real/imaginary coherence channels
        # prevent cancellation and exercise the exact feature tensor consumed
        # by the encoder rather than a side output used only for inspection.
        loss = (weights * result["features"]).mean()
        loss.backward()
        self.assertIsNotNone(front.phase_logits.grad)
        per_current_max = front.phase_logits.grad.abs().amax(dim=1)
        self.assertTrue(torch.all(per_current_max > 1e-5))
        self.assertEqual(
            int(torch.count_nonzero(front.phase_logits.grad)), front.phase_logits.numel()
        )
        self.assertTrue(torch.isfinite(front.phase_logits.grad).all())

    def test_relative_phase_changes_coherence_but_not_order_amplitude(self) -> None:
        front = OrderTrackedFrontEnd(mode="full")
        x, rpm_raw = self._inputs(batch_size=1)
        baseline = front(x, rpm_raw, fs=2000.0)
        with torch.no_grad():
            front.phase_logits[0].fill_(0.35)
            front.phase_logits[1].fill_(-0.20)
        shifted = front(x, rpm_raw, fs=2000.0)

        torch.testing.assert_close(
            shifted["amplitude"], baseline["amplitude"], atol=2e-6, rtol=2e-6
        )
        coherence_change = (
            shifted["coherence"] - baseline["coherence"]
        ).abs().max().detach()
        self.assertGreater(float(coherence_change), 1e-3)

    def test_smoothed_complex_coherence_is_not_the_single_sample_identity(self) -> None:
        front = OrderTrackedFrontEnd(mode="full")
        t = torch.linspace(0.0, 1.0, 128)
        generator = torch.Generator().manual_seed(31)
        channel_0 = torch.sin(2.0 * math.pi * (9.0 * t + 3.0 * t.square()))
        channel_1 = torch.sin(2.0 * math.pi * 9.0 * t + 0.3)
        channel_1 = channel_1 + 0.35 * (t > 0.5) * torch.sin(2.0 * math.pi * 21.0 * t)
        channel_2 = 0.7 * torch.randn(128, generator=generator)
        x = torch.stack((channel_0, channel_1, channel_2)).unsqueeze(0)
        rpm_raw = torch.full((1, 128), 900.0)
        result = front(x, rpm_raw, fs=2000.0)
        coherence = result["coherence"]
        magnitude = coherence.abs()

        self.assertEqual(coherence.shape, (1, 3, 16, 5))
        identity_deviation = (magnitude - 1.0).abs().max().detach()
        coherence_variation = coherence.real.std().detach()
        self.assertGreater(float(identity_deviation), 1e-3)
        self.assertGreater(float(coherence_variation), 1e-4)
        self.assertTrue(torch.isfinite(coherence.real).all())
        self.assertTrue(torch.isfinite(coherence.imag).all())

    def test_centered_cross_covariance_matches_manual_definition(self) -> None:
        z_f = torch.tensor([[1.0, 2.0], [3.0, 1.0], [4.0, 5.0]])
        z_p = torch.tensor([[2.0, 0.0], [1.0, 4.0], [7.0, 3.0]])
        f_centered = z_f - z_f.mean(dim=0)
        p_centered = z_p - z_p.mean(dim=0)
        covariance = f_centered.T @ p_centered / 2.0
        expected = covariance.square().sum()
        actual = centered_cross_covariance_loss(z_f, z_p)
        torch.testing.assert_close(actual, expected)

        shifted = centered_cross_covariance_loss(z_f + 13.0, z_p - 7.0)
        torch.testing.assert_close(shifted, actual)

    def test_gradient_reversal_has_the_documented_sign_once(self) -> None:
        values = torch.tensor([1.0, -2.0, 4.0], requires_grad=True)
        coefficient = 0.37
        grad_reverse(values, coefficient).sum().backward()
        expected = torch.full_like(values, -coefficient)
        torch.testing.assert_close(values.grad, expected)

    def test_phase_is_computed_from_raw_rpm_and_sampling_rate(self) -> None:
        front = OrderTrackedFrontEnd()
        rpm_raw = torch.full((1, 128), 600.0)
        phase = front.mechanical_phase(rpm_raw, fs=100.0)
        expected_step = 2.0 * math.pi * 600.0 / (60.0 * 100.0)
        expected = torch.arange(128, dtype=phase.dtype).unsqueeze(0) * expected_step
        torch.testing.assert_close(phase, expected, atol=2e-5, rtol=1e-6)

        # Identical standardized network input but different raw rpm must alter
        # the phase-locked response; x[:, 0] is never used to integrate phase.
        x = torch.randn(1, 3, 128)
        slow = front(x, torch.full((1, 128), 500.0), fs=2000.0)
        fast = front(x, torch.full((1, 128), 1500.0), fs=2000.0)
        response_difference = (slow["response"] - fast["response"]).abs().max().detach()
        self.assertGreater(float(response_difference), 1e-3)

    def test_cdan_interface(self) -> None:
        z_f = torch.randn(3, 32, requires_grad=True)
        logits = torch.randn(3, 5)
        conditional = conditional_domain_features(z_f, logits)
        self.assertEqual(conditional.shape, (3, 160))
        head = CDANDomainHead(latent_dim=32, num_classes=5)
        domain_logits = head(z_f, logits, grl_lambda=0.25)
        self.assertEqual(domain_logits.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
