#!/usr/bin/env python3
"""Model-free guards for Inkling 2 tensor selection and emitted quantization config.

Run with Python 3.10+: python3 -B tests/test_convert_inkling2.py
The MLX arithmetic is unchanged; an instrumented backend records selected widths.
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


EMBED = "model.language_model.embed_tokens"
HEAD = "lm_head"
PROJECTION = "model.language_model.layers.0.self_attn.q_proj"
EXPERT = "model.language_model.layers.0.mlp.experts.gate_up_proj"


class Tensor:
    def __init__(self, name, shape=(64, 64), dtype="mlx.bfloat16"):
        self.name, self.shape, self.dtype = name, shape, dtype

    @property
    def nbytes(self):
        size = 4 if self.dtype == "mlx.uint32" else 2
        for dimension in self.shape:
            size *= dimension
        return size


CORE = types.ModuleType("mlx.core")
SPEC = importlib.util.spec_from_file_location("convert_inkling2", Path(__file__).with_name("convert_inkling2.py"))
converter = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"mlx": types.ModuleType("mlx"), "mlx.core": CORE}):
    SPEC.loader.exec_module(converter)


class EndpointWidths(unittest.TestCase):
    def setUp(self):
        def quantize(value, *, group_size, bits):
            rows, columns = value.shape
            return (Tensor(value.name, (rows, columns * bits // 32), "mlx.uint32"),
                    Tensor(value.name + ".scales", (rows, columns // group_size)),
                    Tensor(value.name + ".biases", (rows, columns // group_size)))

        CORE.quantize = Mock(side_effect=quantize)
        CORE.load = Mock()
        CORE.eval = Mock()
        CORE.clear_cache = Mock()
        self.saved = {}
        CORE.save_safetensors = Mock(side_effect=lambda _path, values, **_kwargs: self.saved.update(values))

    def weights(self):
        return {name + ".weight": Tensor(name) for name in (
            EMBED, HEAD, PROJECTION, "model.language_model.layers.0.mlp.gate",
            "model.vision_tower.layers.0.mlp.fc1")}

    def test_dense_base_resolves_each_endpoint_width_independently(self):
        weights = self.weights()
        overrides = {}
        out = converter.quantize(weights, 16, overrides=overrides, ends={EMBED: 8, HEAD: 4})
        self.assertEqual([(call.args[0].name, call.kwargs["bits"]) for call in CORE.quantize.call_args_list],
                         [(EMBED, 8), (HEAD, 4)])
        self.assertEqual(out[EMBED + ".weight"].shape, (64, 16))
        self.assertEqual(out[HEAD + ".weight"].shape, (64, 8))
        for name in (EMBED, HEAD):
            self.assertIn(name + ".scales", out)
            self.assertIn(name + ".biases", out)
        for key, value in weights.items():
            if key not in (EMBED + ".weight", HEAD + ".weight"):
                self.assertIs(out[key], value)
        self.assertEqual(overrides, {
            EMBED: {"group_size": 64, "bits": 8, "mode": "affine"},
            HEAD: {"group_size": 64, "bits": 4, "mode": "affine"}})

    def test_dense_endpoint_override_also_applies_when_requantizing_experts(self):
        weights = self.weights()
        weights.update({EXPERT + ".weight": Tensor("packed", (2, 64, 8), "mlx.uint32"),
                        EXPERT + ".scales": Tensor("scales", (2, 64, 2)),
                        EXPERT + ".global_scale": Tensor("global", (2,))})
        bank = (Tensor("requantized"), Tensor("new-scales"), Tensor("new-biases"))
        overrides = {}
        with patch.object(converter, "requantize_bank", return_value=bank) as requantize:
            out = converter.quantize(weights, 16, [(4, 64)], overrides, {HEAD: 8})
        CORE.quantize.assert_called_once_with(weights[HEAD + ".weight"], group_size=64, bits=8)
        requantize.assert_called_once_with(weights[EXPERT + ".weight"], weights[EXPERT + ".scales"],
                                          weights[EXPERT + ".global_scale"], 4, 64)
        self.assertIs(out[EXPERT + ".weight"], bank[0])
        self.assertNotIn(EXPERT + ".global_scale", out)
        self.assertEqual(overrides[HEAD]["bits"], 8)
        self.assertEqual(overrides[EXPERT]["bits"], 4)

    def test_quantized_base_keeps_its_width_outside_endpoint_overrides(self):
        weights = self.weights()
        overrides = {}
        out = converter.quantize(weights, 8, overrides=overrides, ends={HEAD: 4})
        self.assertEqual([(call.args[0].name, call.kwargs["bits"]) for call in CORE.quantize.call_args_list],
                         [(EMBED, 8), (HEAD, 4), (PROJECTION, 8)])
        self.assertEqual(set(overrides), {HEAD})
        self.assertIs(out["model.vision_tower.layers.0.mlp.fc1.weight"],
                      weights["model.vision_tower.layers.0.mlp.fc1.weight"])

    def run_main(self, *flags):
        weights = self.weights()
        CORE.load.return_value = weights
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory, "source"), Path(directory, "target")
            source.mkdir()
            (source / "model-00001.safetensors").touch()
            (source / "config.json").write_text(json.dumps({
                "model_type": "inkling2_loop", "text_config": {"num_hidden_layers": 1}}))
            arguments = ["convert_inkling2.py", "--src", str(source), "--dst", str(target), "--bits", "16", *flags]
            with patch.object(sys, "argv", arguments), patch.object(converter, "sanitize", return_value=weights), \
                    contextlib.redirect_stdout(io.StringIO()):
                converter.main()
            config = json.loads((target / "config.json").read_text())
        return weights, config

    def test_cli_dense_base_honors_either_endpoint_flag_and_writes_its_config(self):
        for flag, endpoint in (("--head-bits", HEAD), ("--embed-bits", EMBED)):
            with self.subTest(flag=flag):
                CORE.quantize.reset_mock()
                self.saved.clear()
                weights, config = self.run_main(flag, "4")
                CORE.quantize.assert_called_once_with(weights[endpoint + ".weight"], group_size=64, bits=4)
                self.assertEqual(self.saved[endpoint + ".weight"].shape, (64, 8))
                self.assertIn(endpoint + ".scales", self.saved)
                self.assertIn(endpoint + ".biases", self.saved)
                self.assertEqual(config["quantization"][endpoint], {"bits": 4, "group_size": 64, "mode": "affine"})
                self.assertEqual(config["quantization"], config["quantization_config"])
                self.assertIs(self.saved[PROJECTION + ".weight"], weights[PROJECTION + ".weight"])

    def test_cli_dense_base_without_overrides_keeps_dense_weights_and_config(self):
        weights, config = self.run_main()
        CORE.quantize.assert_not_called()
        self.assertEqual(self.saved, weights)
        self.assertNotIn("quantization", config)
        self.assertNotIn("quantization_config", config)


if __name__ == "__main__":
    unittest.main()
