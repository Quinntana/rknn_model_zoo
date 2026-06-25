# Face Benchmark

Dedicated RKNN benchmark demo for edge face-verification models on RK3588 single-core NPU.

This tool does not use `RKNN_FLAG_COLLECT_PERF_MASK`. Its speed headline is wall-clock
`rknn_run + rknn_outputs_get`, measured after warmup on one selected NPU core.

## Build

```bash
./build-linux.sh -t rk3588 -a aarch64 -d face_benchmark
```

Installed demo:

```bash
cd install/rk3588_linux_aarch64/rknn_face_benchmark_demo
./face_benchmark --speed
```

## Common Commands

Speed table for every model in the registry:

```bash
./face_benchmark --models model/model_registry.yaml --speed --core 0 --warmup 20 --iters 200
```

Run only face-detection models:

```bash
./face_benchmark --speed --task face_detection
```

Write CSV:

```bash
./face_benchmark --speed --output results.csv
```

Accuracy examples:

```bash
./face_benchmark --accuracy --task face_detection --dataset internal --manifest detection_manifest.txt
./face_benchmark --accuracy --task liveness --dataset celeba_spoof --manifest liveness_manifest.txt
./face_benchmark --accuracy --task feature_extraction --dataset lfw --manifest lfw_pairs.txt
```

## Dataset Install

These datasets are **not included** in this repository. The local checkout only includes small sample
sets such as COCO subset, ImageNet samples, PPOCR samples, and demo images.

Recommended local layout:

```text
datasets/face_benchmark/
  lfw/
  wider_face/
  celeba_spoof/
  manifests/
```

### LFW

LFW is the easiest one to install directly. If the original UMass host is not reachable, the
Figshare mirror used by common dataset loaders works:

```bash
mkdir -p datasets/face_benchmark/lfw datasets/face_benchmark/manifests
wget -c https://ndownloader.figshare.com/files/5976015 \
  -O datasets/face_benchmark/lfw/lfw-funneled.tgz
wget -c https://ndownloader.figshare.com/files/5976006 \
  -O datasets/face_benchmark/lfw/pairs.txt
tar -xzf datasets/face_benchmark/lfw/lfw-funneled.tgz \
  -C datasets/face_benchmark/lfw

python3 examples/face_benchmark/tools/lfw_pairs_to_manifest.py \
  --lfw-root "$PWD/datasets/face_benchmark/lfw/lfw_funneled" \
  --pairs datasets/face_benchmark/lfw/pairs.txt \
  --output datasets/face_benchmark/manifests/lfw_pairs_abs.txt
```

Then run:

```bash
./face_benchmark --accuracy --task feature_extraction --dataset lfw \
  --manifest datasets/face_benchmark/manifests/lfw_pairs_abs.txt
```

### WIDER FACE

Download `WIDER_val.zip` and `wider_face_split.zip` from the official WIDER FACE page, then:

```bash
mkdir -p datasets/face_benchmark/wider_face
unzip WIDER_val.zip -d datasets/face_benchmark/wider_face
unzip wider_face_split.zip -d datasets/face_benchmark/wider_face

python3 examples/face_benchmark/tools/wider_val_to_manifest.py \
  --images-root datasets/face_benchmark/wider_face/WIDER_val/images \
  --annotation datasets/face_benchmark/wider_face/wider_face_split/wider_face_val_bbx_gt.txt \
  --output datasets/face_benchmark/manifests/wider_val.txt
```

Then run:

```bash
./face_benchmark --accuracy --task face_detection --dataset wider \
  --manifest datasets/face_benchmark/manifests/wider_val.txt
```

This reports internal `AP@0.5`. Official WIDER easy/medium/hard requires the official evaluation
protocol and ignore metadata.

### CelebA-Spoof

CelebA-Spoof must be downloaded through the official project links and is for non-commercial
research use. After extracting the dataset and locating a label JSON such as `test_label.json`:

```bash
python3 examples/face_benchmark/tools/celeba_spoof_to_manifest.py \
  --root datasets/face_benchmark/celeba_spoof \
  --label-json datasets/face_benchmark/celeba_spoof/metas/intra_test/test_label.json \
  --output datasets/face_benchmark/manifests/celeba_spoof_test.txt
```

If your label convention is reversed, pass `--live-value 0 --spoof-value 1`.

Then run:

```bash
./face_benchmark --accuracy --task liveness --dataset celeba_spoof \
  --manifest datasets/face_benchmark/manifests/celeba_spoof_test.txt
```

## Model Registry

Edit `model/model_registry.yaml` after install, or edit `examples/face_benchmark/model/model_registry.yaml`
before rebuilding.

Important fields:

- `task`: `face_detection`, `liveness`, or `feature_extraction`
- `path`: RKNN path, normally relative to the demo working directory
- `input_shape`: benchmark display shape, for example `[1, 3, 320, 320]`
- `input_type`: `uint8` or `float32`
- `resize`: `letterbox` or `stretch`
- `output_decoder`: `retinaface` for detection accuracy, otherwise `unsupported`
- `output_activation`: `none`, `sigmoid`, or `softmax`
- `threshold`: liveness or detection decision threshold

## Manifest Formats

Face detection internal manifest:

```text
image.jpg x1,y1,x2,y2 x1,y1,x2,y2
```

or:

```text
image.jpg x1 y1 x2 y2 x1 y1 x2 y2
```

Liveness manifest:

```text
image.jpg live
image2.jpg spoof
```

Accepted live labels: `1`, `live`, `real`, `bonafide`, `bona_fide`.
Accepted spoof labels: `0`, `spoof`, `attack`, `fake`.

Feature extraction pair manifest:

```text
image_a.jpg image_b.jpg same
image_c.jpg image_d.jpg different
```

Optional fourth column is fold index `0..9`. If omitted, pairs are assigned to folds round-robin.
Accepted same labels: `1`, `same`, `positive`, `match`.
Accepted different labels: `0`, `different`, `diff`, `negative`, `nonmatch`.

## Accuracy Metrics

- Face detection: internal `AP@0.5`, recall, precision, false positives per image.
- WIDER FACE: the tool reserves easy/medium/hard columns but does not fake official WIDER metrics without the official split/ignore metadata.
- Liveness: `APCER`, `BPCER`, `ACER`, `AUC`, `EER`.
- Feature extraction: LFW-style 10-fold mean accuracy, `EER`, `TAR@FAR=1e-3`, `TAR@FAR=1e-4`.

References:

- WIDER FACE benchmark: https://shuoyang1213.me/WIDERFACE/
- LFW verification protocol paper: https://inria.hal.science/inria-00321923v1/document
- CelebA-Spoof benchmark: https://github.com/ZhangYuanhan-AI/CelebA-Spoof
