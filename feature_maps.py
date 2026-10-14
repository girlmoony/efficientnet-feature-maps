"""Extract real EfficientNet-Lite0/Lite1 feature maps from a local state_dict.

Architecture follows https://github.com/RangiLyu/EfficientNet-Lite .
Only the supplied checkpoint is loaded; there are no downloads at runtime.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch import nn

STAGES = [(1, 3, 1, 1, 32, 16), (2, 3, 2, 6, 16, 24),
          (2, 5, 2, 6, 24, 40), (3, 3, 2, 6, 40, 80),
          (3, 5, 1, 6, 80, 112), (4, 5, 2, 6, 112, 192),
          (1, 3, 1, 6, 192, 320)]
VARIANTS = {'efficientnet-lite0': (1.0, 224), 'efficientnet-lite1': (1.1, 240)}
SUMMARY = ['Stem'] + [f'Stage {i}' for i in range(1, 8)] + ['Head']


class MBConv(nn.Module):
    def __init__(self, ci, co, kernel, stride, expansion):
        super().__init__()
        self.skip = stride == 1 and ci == co
        self.expansion = expansion
        ce = ci * expansion
        if expansion != 1:
            self._expand_conv = nn.Conv2d(ci, ce, 1, bias=False)
            self._bn0 = nn.BatchNorm2d(ce, eps=1e-3, momentum=.01)
        self._depthwise_conv = nn.Conv2d(ce, ce, kernel, stride, (kernel - 1) // 2,
                                         groups=ce, bias=False)
        self._bn1 = nn.BatchNorm2d(ce, eps=1e-3, momentum=.01)
        self._project_conv = nn.Conv2d(ce, co, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(co, eps=1e-3, momentum=.01)
        # Out-of-place activation preserves raw Conv/BN hook snapshots.
        self._relu = nn.ReLU6(inplace=False)

    def forward(self, x):
        identity = x
        if self.expansion != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))
        x = self._bn2(self._project_conv(x))
        return x + identity if self.skip else x


class EfficientNetLite(nn.Module):
    def __init__(self, variant, classes):
        super().__init__()
        depth, _ = VARIANTS[variant]
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(32, eps=1e-3, momentum=.01), nn.ReLU6())
        self.blocks = nn.ModuleList()
        for i, (repeat, kernel, stride, expansion, ci, co) in enumerate(STAGES):
            repeat = repeat if i in (0, 6) else math.ceil(repeat * depth)
            self.blocks.append(nn.ModuleList([
                MBConv(ci if j == 0 else co, co, kernel, stride if j == 0 else 1, expansion)
                for j in range(repeat)]))
        self.head = nn.Sequential(nn.Conv2d(320, 1280, 1, bias=False),
                                  nn.BatchNorm2d(1280, eps=1e-3, momentum=.01), nn.ReLU6())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(.2)
        self.fc = nn.Linear(1280, classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.blocks:
            for block in stage:
                x = block(x)
        x = self.head(x)
        return self.fc(self.dropout(self.avgpool(x).flatten(1)))


def load_model(path, variant):
    # torch.save files can be named .pth, .pth.tar, .pt, etc.; do not infer
    # serialization format from the filename. .pth.tar is usually not a tar archive.
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must contain a state_dict, not a pickled model object.')
    for key in ('state_dict', 'model_state_dict', 'model'):
        if key in state:
            state = state[key]
            break
    if not isinstance(state, dict) or not state:
        raise ValueError('Empty or invalid state_dict.')
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Expected tensor weights in a raw state_dict or under '
                         'state_dict, model_state_dict, or model. Model objects are unsupported.')
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    if 'fc.weight' not in state or state['fc.weight'].ndim != 2:
        raise ValueError('Expected fc.weight. This tool supports the RangiLyu state_dict format; '
                         'timm/torchvision key formats are not interchangeable.')
    model = EfficientNetLite(variant, state['fc.weight'].shape[0])
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_labels(path, classes):
    if path is None:
        return [f'class_{i}' for i in range(classes)]
    content = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.json':
        labels = json.loads(content)
        if isinstance(labels, dict):
            if set(labels) != {str(i) for i in range(classes)}:
                raise ValueError('JSON label keys must be exactly 0 through num_classes-1.')
            labels = [labels[str(i)] for i in range(classes)]
    else:
        labels = content.splitlines()
    if not isinstance(labels, list) or len(labels) != classes:
        raise ValueError(f'Labels must contain exactly {classes} entries.')
    if not all(isinstance(x, str) and x.strip() for x in labels):
        raise ValueError('Each label must be a nonempty string.')
    return [x.strip() for x in labels]


def preprocess(path, size, mode, order, mean, std):
    photo = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if mode == 'eval':
        short = size + 32
        w, h = photo.size
        dims = (short, int(short * h / w)) if w <= h else (int(short * w / h), short)
        resized = photo.resize(dims, Image.Resampling.BICUBIC)
        left = round((resized.width - size) / 2)
        top = round((resized.height - size) / 2)
        crop = resized.crop((left, top, left + size, top + size))
    else:
        crop = photo.resize((size, size), Image.Resampling.BICUBIC)
    pixels = np.asarray(crop, dtype=np.float32) / 255.
    # Files are decoded as RGB. BGR means the MODEL receives B,G,R, not a file format.
    if order == 'bgr':
        pixels = pixels[..., ::-1].copy()
    normalized = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
    return photo, crop, tensor


def extract(model, tensor, operators=True):
    captured = {}
    handles = []
    def register(name, module):
        def hook(_, inputs, output):
            captured[name] = output.detach().cpu().numpy()[0].copy()
        handles.append(module.register_forward_hook(hook))
    if operators:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, MBConv)):
                register(name, module)
    register('Stem', model.stem)
    for i, stage in enumerate(model.blocks):
        register(f'Stage {i+1}', stage[-1])
    register('Head', model.head)
    register('GAP', model.avgpool)
    try:
        with torch.inference_mode():
            logits = model(tensor)
    finally:
        for handle in handles:
            handle.remove()
    if not all(np.isfinite(x).all() for x in captured.values()) or not torch.isfinite(logits).all():
        raise ValueError('Non-finite activations or logits; check weights and preprocessing.')
    return captured, logits[0].numpy(), logits.softmax(-1)[0].numpy()


def selected_channels(a, count):
    scores = a.reshape(a.shape[0], -1).var(axis=1)
    return np.argsort(-scores, kind='stable')[:count].tolist()


def grayscale(a, size):
    lo, hi = float(a.min()), float(a.max())
    normalized = np.zeros_like(a) if hi - lo < 1e-10 else (a - lo) / (hi - lo)
    pixels = np.rint(normalized * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels).convert('RGB').resize((size, size), Image.Resampling.NEAREST)


def get_font(size, requested=None):
    candidates = [requested] if requested else []
    candidates += ['C:/Windows/Fonts/YuGothM.ttc', 'C:/Windows/Fonts/arial.ttf',
                   '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                   '/System/Library/Fonts/Supplemental/Arial.ttf']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def text(draw, xy, content, size=20, font_path=None):
    draw.text(xy, content, fill='#142536', font=get_font(size, font_path))


def montage(name, a, channels, font_path=None):
    cols = min(4, len(channels)); rows = math.ceil(len(channels) / cols)
    image = Image.new('RGB', (cols * 180 + 24, rows * 210 + 85), 'white')
    draw = ImageDraw.Draw(image)
    text(draw, (12, 8), f'{name}  {a.shape[1]}x{a.shape[2]}x{a.shape[0]} (HWC)', 21, font_path)
    text(draw, (12, 42), 'Real activations; per-channel min/max display', 16, font_path)
    for i, c in enumerate(channels):
        x, y = 12 + (i % cols) * 180, 80 + (i // cols) * 210
        image.paste(grayscale(a[c], 160), (x, y))
        text(draw, (x, y + 161), f'ch {c}', 18, font_path)
        text(draw, (x, y + 186), f'{a[c].min():.3g} .. {a[c].max():.3g}', 15, font_path)
    return image


def overview(features, crop, probabilities, variant, path, font_path):
    canvas = Image.new('RGB', (1536, 1810), '#f3f6fa'); draw = ImageDraw.Draw(canvas)
    text(draw, (20, 12), f'{variant}: actual feature maps', 32, font_path)
    text(draw, (20, 58), 'Supplied weights | top 6 channels by spatial variance | independent min/max display', 20, font_path)
    panels = ['Input'] + SUMMARY + ['GAP', 'Output']
    for i, name in enumerate(panels):
        x, y = 16 + (i % 3) * 508, 105 + (i // 3) * 410
        draw.rounded_rectangle((x, y, x + 494, y + 396), 10, fill='white', outline='#ccd6e0')
        text(draw, (x + 12, y + 8), f'{i+1:02d}  {name}', 26, font_path)
        if name == 'Input':
            canvas.paste(crop.resize((290, 290)), (x + 98, y + 55))
            text(draw, (x + 12, y + 355), f'{crop.height}x{crop.width}x3 (RGB preview)', 22, font_path)
        elif name in SUMMARY:
            a = features[name]
            for j, c in enumerate(selected_channels(a, 6)):
                px, py = x + 14 + (j % 3) * 160, y + 54 + (j // 3) * 145
                canvas.paste(grayscale(a[c], 118), (px, py))
                text(draw, (px, py + 118), f'ch {c}', 16, font_path)
            text(draw, (x + 12, y + 355), f'{a.shape[1]}x{a.shape[2]}x{a.shape[0]} (HWC)', 23, font_path)
        elif name == 'GAP':
            values = features['GAP'][:, 0, 0]
            text(draw, (x + 12, y + 55), 'Spatial mean -> 1280 values', 22, font_path)
            for j, c in enumerate(np.argsort(-values, kind='stable')[:8]):
                py = y + 105 + j * 31
                text(draw, (x + 12, py), f'ch {c}', 17, font_path)
                width = int(values[c] / max(float(values.max()), 1e-10) * 250)
                draw.rectangle((x + 115, py + 4, x + 115 + width, py + 22), fill='#2874a5')
                text(draw, (x + 380, py), f'{values[c]:.3f}', 16, font_path)
        else:
            text(draw, (x + 12, y + 55), f'{len(probabilities)} classes / top 5', 22, font_path)
            for j, c in enumerate(np.argsort(-probabilities, kind='stable')[:5]):
                text(draw, (x + 15, y + 108 + j * 47), f'ID {c}: {probabilities[c]*100:.2f} %', 23, font_path)
    text(draw, (20, 1758), 'White = higher value WITHIN one channel. These are not attention maps or Grad-CAM.', 20, font_path)
    canvas.save(path)


def write_report(output, metadata):
    options = []
    sections = []
    for i, layer in enumerate(metadata['layers']):
        label = html.escape(layer['name'])
        options.append(f'<option value="layer-{i}">{label}</option>')
        sections.append(f'<section id="layer-{i}" class="layer" hidden><h3>{label}</h3>'
                        f'<img loading="lazy" src="{layer["file"]}" alt="{label} actual activations"></section>')
    rows = ''.join(f'<tr><td>{p["class_id"]}</td><td>{html.escape(p["label"])}</td>'
                   f'<td>{p["probability"]*100:.4f}%</td></tr>' for p in metadata['top_predictions'])
    document = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>実測特徴マップ</title>
<style>body{font-family:system-ui,sans-serif;max-width:1300px;margin:24px auto;padding:0 18px;color:#182c3b}img{max-width:100%;height:auto}p{line-height:1.7}select{font:inherit;max-width:100%;padding:8px}td,th{text-align:left;padding:6px 18px 6px 0}section{margin:20px 0}</style>
<h1>EfficientNet-Lite 実際の特徴マップ</h1>
<p>指定された重みでevalモードの推論を実行した結果です。画像生成による模式図ではありません。
各マップの最小値を黒、最大値を白へ個別に変換しています。層・チャネル間で白さの絶対比較はできません。
猫への注目度・確率を表すGrad-CAMではありません。各チャネル番号は0始まりです。</p>
<p>前処理：__PREPROCESS__。学習時の設定に合わせてください。重みだけから前処理や学習履歴は確認できません。</p>
<img src="overview.png" alt="入力から出力までの実測結果">
<h2>Stem 全32チャネル</h2><img src="stem_all_channels.png" alt="Stem全32チャネルの実測出力">
<h2>層ごとの実測出力</h2><p>空間分散が大きい順にチャネルを選択。各画像下部に変換前の最小値・最大値を記載しています。</p>
<label for="layer">層：</label><select id="layer">__OPTIONS__</select>__SECTIONS__
<h2>分類出力</h2><table><thead><tr><th>ID</th><th>ラベル</th><th>softmax</th></tr></thead><tbody>__ROWS__</tbody></table>
<p>ラベル未指定時はclass_IDと表示します。これはクラス名の推定ではありません。</p>
<p><a href="metadata.json">推論条件とSHA-256</a> / <a href="layer_shapes.csv">全層のサイズ</a> / <a href="predictions.json">分類出力</a></p>
<script>const select=document.getElementById('layer');function show(){document.querySelectorAll('.layer').forEach(x=>x.hidden=x.id!==select.value)}select.addEventListener('change',show);show();</script></html>'''
    for key, value in {'__OPTIONS__': ''.join(options), '__SECTIONS__': ''.join(sections),
                       '__ROWS__': rows, '__PREPROCESS__': html.escape(json.dumps(metadata['preprocessing'], ensure_ascii=False))}.items():
        document = document.replace(key, value)
    (output / 'report.html').write_text(document, encoding='utf-8')


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True, help='Local PyTorch checkpoint (.pth / .pth.tar / .pt)')
    parser.add_argument('--image', type=Path, required=True, help='PNG/JPEG etc. decoded with Pillow')
    parser.add_argument('--labels', type=Path, help='UTF-8 TXT (one label per line) or JSON list / ID-to-name object')
    parser.add_argument('--model', choices=VARIANTS, default='efficientnet-lite1')
    parser.add_argument('--channel-order', type=str.lower, choices=('rgb', 'bgr'), default='rgb', help='Channel order supplied to MODEL')
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory')
    parser.add_argument('--size', type=positive_int, help='Input size; default lite0=224, lite1=240')
    parser.add_argument('--resize-mode', choices=('eval', 'stretch'), default='eval')
    parser.add_argument('--mean', nargs=3, type=float, default=[.498] * 3)
    parser.add_argument('--std', nargs=3, type=float, default=[.502] * 3)
    parser.add_argument('--channels', type=positive_int, default=8, help='Selected channels per layer')
    parser.add_argument('--top-k', type=positive_int, default=5)
    parser.add_argument('--scope', choices=('all', 'stages'), default='all', help='all adds Conv, BN and block outputs')
    parser.add_argument('--save-tensors', action='store_true', help='Also write all numerical activations to NPZ')
    parser.add_argument('--font', type=Path, help='Optional TTF/TTC font for PNG labels')
    parser.add_argument('--threads', type=positive_int, default=4)
    args = parser.parse_args(argv)
    if not all(math.isfinite(x) for x in args.mean + args.std) or any(x <= 0 for x in args.std):
        parser.error('--mean must be finite; --std must be finite and > 0')
    for name in ('weights', 'image', 'labels', 'font'):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            parser.error(f'--{name}: file not found: {value}')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('--output must be a new or empty directory (prevents stale results/overwriting).')
    return args


def run(args):
    torch.set_num_threads(args.threads)
    model = load_model(args.weights, args.model)
    labels = load_labels(args.labels, model.fc.out_features)
    size = args.size or VARIANTS[args.model][1]
    photo, crop, tensor = preprocess(args.image, size, args.resize_mode, args.channel_order, args.mean, args.std)
    features, logits, probabilities = extract(model, tensor, args.scope == 'all')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'feature_maps').mkdir()
    photo.save(output / 'input_original.png')
    crop.save(output / 'input_preprocessed_rgb.png')
    layers = []
    for name, a in features.items():
        if a.shape[1:] == (1, 1):
            continue  # GAP has no spatial structure; shown numerically in overview and NPZ.
        ids = selected_channels(a, args.channels)
        filename = f'feature_maps/layer_{len(layers):03d}.png'
        montage(name, a, ids, args.font).save(output / filename)
        layers.append(dict(name=name, shape_chw=list(a.shape), channels=ids, file=filename,
                           min=float(a.min()), max=float(a.max())))
    montage('Stem - all 32 channels', features['Stem'], list(range(32)), args.font).save(output / 'stem_all_channels.png')
    overview(features, crop, probabilities, args.model, output / 'overview.png', args.font)
    top = np.argsort(-probabilities, kind='stable')[:args.top_k]
    predictions = [dict(class_id=int(c), label=labels[c], logit=float(logits[c]), probability=float(probabilities[c])) for c in range(len(labels))]
    metadata = dict(model=args.model, weights_file=args.weights.name,
                    weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                    image_file=args.image.name, image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                    labels_file=args.labels.name if args.labels else None,
                    labels_sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest() if args.labels else None,
                    preprocessing=dict(size=size, resize_mode=args.resize_mode, eval_resize_short_side=size+32 if args.resize_mode=='eval' else None,
                                       channel_order=args.channel_order, mean=args.mean, std=args.std, interpolation='Pillow BICUBIC'),
                    inference=dict(mode='eval', device='cpu', strict_load=True, missing_keys=[], unexpected_keys=[], all_finite=True),
                    versions=dict(torch=torch.__version__, numpy=np.__version__, pillow=Image.__version__),
                    num_classes=len(labels), scope=args.scope, layers=layers,
                    channel_selection='descending spatial variance; zero-based indices; per-channel min-max grayscale',
                    top_predictions=[predictions[c] for c in top])
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'layer_shapes.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['layer', 'channels', 'height', 'width', 'min', 'max'])
        for layer in layers:
            writer.writerow([layer['name'], *layer['shape_chw'], layer['min'], layer['max']])
    if args.save_tensors:
        np.savez_compressed(output / 'activations.npz', **features, input_normalized=tensor.numpy(), logits=logits, probabilities=probabilities)
    write_report(output, metadata)
    print(json.dumps(dict(output=str(output.resolve()), spatial_layers=len(layers), top_predictions=metadata['top_predictions']), ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise SystemExit(f'Error: {exc}') from exc


if __name__ == '__main__':
    main()
