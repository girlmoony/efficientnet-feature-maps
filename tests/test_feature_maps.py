import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from feature_maps import EfficientNetLite, extract, load_labels, load_model, preprocess


class FeatureMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.assertEqual(self.root.resolve().parent, Path(__file__).resolve().parent)
        self.temp.cleanup()

    def test_model_channel_order_and_normalization(self):
        path = self.root / 'red.png'
        Image.new('RGB', (8, 8), (255, 64, 0)).save(path)
        _, preview, rgb = preprocess(path, 8, 'stretch', 'rgb', [0, 0, 0], [1, 1, 1])
        _, _, bgr = preprocess(path, 8, 'stretch', 'bgr', [0, 0, 0], [1, 1, 1])
        np.testing.assert_array_equal(rgb.numpy()[:, ::-1], bgr.numpy())
        self.assertEqual(preview.getpixel((0, 0)), (255, 64, 0))
        _, _, normalized = preprocess(path, 8, 'stretch', 'bgr', [.1, .2, .3], [.5, .5, .5])
        np.testing.assert_allclose(normalized.numpy()[0, :, 0, 0],
                                   (np.array([0, 64 / 255, 1]) - [.1, .2, .3]) / .5, atol=1e-6)

    def test_crop_is_centered_and_has_requested_shape(self):
        path = self.root / 'wide.png'
        pixels = np.zeros((200, 400, 3), dtype=np.uint8)
        pixels[:, 100:300, 0] = 255
        Image.fromarray(pixels).save(path)
        _, crop, tensor = preprocess(path, 224, 'eval', 'rgb', [.498]*3, [.502]*3)
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        self.assertTrue(np.all(np.asarray(crop)[:, :, 0] == 255))
        self.assertTrue(np.all(np.asarray(crop)[:, :, 1:] == 0))

    def test_labels_formats_and_wrong_counts(self):
        for filename, content in [('labels.txt', '猫\n犬\n'),
                                  ('labels.json', json.dumps(['猫', '犬'])),
                                  ('mapping.json', json.dumps({'1': '犬', '0': '猫'}))]:
            path = self.root / filename
            path.write_text(content, encoding='utf-8')
            self.assertEqual(load_labels(path, 2), ['猫', '犬'])
            with self.assertRaises(ValueError):
                load_labels(path, 3)

    def test_lite0_wrapped_dataparallel_weights_and_hooks(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        original = EfficientNetLite('efficientnet-lite0', 3).eval()
        path = self.root / 'lite0.pth'
        torch.save({'state_dict': {'module.' + k: v for k, v in original.state_dict().items()}}, path)
        model = load_model(path, 'efficientnet-lite0')
        tensor = torch.randn(1, 3, 224, 224)
        with torch.inference_mode():
            expected = original(tensor)[0].numpy()
        captured, logits, probabilities = extract(model, tensor)
        np.testing.assert_allclose(logits, expected, atol=1e-6)
        self.assertEqual(captured['Stem'].shape, (32, 112, 112))
        self.assertEqual(captured['Head'].shape, (1280, 7, 7))
        self.assertEqual(captured['GAP'].shape, (1280, 1, 1))
        self.assertAlmostEqual(float(probabilities.sum()), 1., places=5)
        self.assertIn('blocks.1.0._depthwise_conv', captured)
        self.assertLess(float(captured['stem.0'].min()), 0.)
        self.assertGreaterEqual(float(captured['Stem'].min()), 0.)
        with self.assertRaises(RuntimeError):
            load_model(path, 'efficientnet-lite1')

    def test_pth_tar_checkpoints_match_original_inference(self):
        torch.set_num_threads(2)
        # Both modern ZIP and legacy torch.save serialization, two variants,
        # optimizer/epoch metadata, and common wrapper keys are covered.
        cases = [('efficientnet-lite0', None, False, False),
                 ('efficientnet-lite0', 'state_dict', True, True),
                 ('efficientnet-lite1', 'model_state_dict', False, False),
                 ('efficientnet-lite1', 'model', True, False)]
        for variant, wrapper, modern, parallel in cases:
            with self.subTest(variant=variant, wrapper=wrapper, modern=modern):
                original = EfficientNetLite(variant, 3).eval()
                weights = original.state_dict()
                if parallel:
                    weights = {'module.' + k: v for k, v in weights.items()}
                checkpoint = weights if wrapper is None else {
                    wrapper: weights, 'epoch': 17, 'optimizer': {'state': {}, 'param_groups': []}}
                path = self.root / 'checkpoint.pth.tar'
                torch.save(checkpoint, path, _use_new_zipfile_serialization=modern)
                restored = load_model(path, variant)
                tensor = torch.randn(1, 3, 64, 64)
                with torch.inference_mode():
                    torch.testing.assert_close(restored(tensor), original(tensor), rtol=0, atol=0)

    def test_rejects_checkpoint_without_tensor_state_dict(self):
        path = self.root / 'invalid.pth.tar'
        torch.save({'epoch': 1, 'optimizer': {}}, path)
        with self.assertRaisesRegex(ValueError, 'Expected tensor weights'):
            load_model(path, 'efficientnet-lite1')


if __name__ == '__main__':
    unittest.main()
