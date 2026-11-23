import numpy as np
import pytest
import torch
from PIL import Image

from core.occupancy.classifier import CheckpointError, PreprocessingConfig, ResNet18Classifier, build_resnet18

PRE = {"resize": 64, "crop_size": 56, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
       "interpolation": "bilinear", "color_order": "RGB"}


def save_checkpoint(path, num_classes=2, seed=0, **metadata):
    torch.manual_seed(seed)
    model = build_resnet18(num_classes)
    payload = {"state_dict": model.state_dict(), **metadata}
    torch.save(payload, path)
    return model


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "model.pth"
    model = save_checkpoint(path, class_to_idx={"empty": 0, "occupied": 1}, image_size=56,
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return path, model


def rand_crop(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_output_shape_and_probabilities(ckpt):
    clf = ResNet18Classifier(ckpt[0], preprocessing=PRE, device="cpu")
    logits, probs = clf.predict_tensor(clf.preprocess_batch([rand_crop(80, 60), rand_crop(40, 90)]))
    assert logits.shape == (2, 2) and probs.shape == (2, 2)
    assert torch.allclose(probs.sum(1), torch.ones(2), atol=1e-5)
    pred = clf.predict([rand_crop(80, 60)])[0]
    assert set(pred.probabilities) == {"empty", "occupied"}
    assert pred.confidence == pytest.approx(max(pred.probabilities.values()))


def test_class_index_mapping_follows_checkpoint(tmp_path):
    path = tmp_path / "m.pth"
    model = save_checkpoint(path, class_to_idx={"occupied": 0, "empty": 1})
    clf = ResNet18Classifier(path, preprocessing=PRE, device="cpu")
    assert clf.class_names == ["occupied", "empty"]
    crop = rand_crop(70, 70)
    with torch.no_grad():
        expected = torch.softmax(model.eval()(clf.preprocess(crop)[None]), 1)[0]
    pred = clf.predict([crop])[0]
    assert pred.probabilities["occupied"] == pytest.approx(float(expected[0]), abs=1e-5)
    assert pred.probabilities["empty"] == pytest.approx(float(expected[1]), abs=1e-5)


def test_three_classes_supported(tmp_path):
    path = tmp_path / "m.pth"
    save_checkpoint(path, 3, class_names=["empty", "occupied", "standing_near"])
    clf = ResNet18Classifier(path, preprocessing=PRE, device="cpu")
    assert set(clf.predict([rand_crop(50, 50)])[0].probabilities) == {"empty", "occupied", "standing_near"}


def test_bgr_to_rgb_conversion(tmp_path):
    path = tmp_path / "plain.pth"
    save_checkpoint(path, class_names=["empty", "occupied"])
    clf = ResNet18Classifier(path, preprocessing={**PRE, "mean": [0, 0, 0], "std": [1, 1, 1]}, device="cpu")
    bgr_blue = np.zeros((64, 64, 3), dtype=np.uint8)
    bgr_blue[:, :, 0] = 255
    tensor = clf.preprocess(bgr_blue)
    assert torch.allclose(tensor[2], torch.ones_like(tensor[2]))
    assert torch.allclose(tensor[0], torch.zeros_like(tensor[0]))


def test_preprocessing_matches_torchvision_pil_pipeline(ckpt):
    clf = ResNet18Classifier(ckpt[0], preprocessing=PRE, device="cpu")
    crop = rand_crop(97, 151, seed=3)
    expected = PreprocessingConfig.from_mapping(PRE).build_transform()(Image.fromarray(crop[:, :, ::-1].copy()))
    actual = clf.preprocess(crop)
    assert actual.shape == (3, 56, 56)
    assert torch.equal(actual, expected)


def test_batch_matches_individual_and_chunking(ckpt):
    clf = ResNet18Classifier(ckpt[0], preprocessing=PRE, device="cpu", batch_size=2)
    crops = [rand_crop(60 + i * 7, 50 + i * 3, seed=i) for i in range(5)]
    batched = clf.predict(crops)
    single = [clf.predict([c])[0] for c in crops]
    for b, s in zip(batched, single):
        assert b.predicted_class == s.predicted_class
        assert b.confidence == pytest.approx(s.confidence, abs=1e-5)
    result = clf.classify({f"W{i:02d}": c for i, c in enumerate(crops)})
    assert list(result) == [f"W{i:02d}" for i in range(5)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu(ckpt):
    crops = [rand_crop(80, 80, seed=i) for i in range(3)]
    cpu = ResNet18Classifier(ckpt[0], preprocessing=PRE, device="cpu").predict(crops)
    gpu = ResNet18Classifier(ckpt[0], preprocessing=PRE, device="cuda")
    assert gpu.device.type == "cuda"
    for a, b in zip(cpu, gpu.predict(crops)):
        assert a.predicted_class == b.predicted_class
        assert a.confidence == pytest.approx(b.confidence, abs=1e-3)


def test_auto_device(ckpt):
    clf = ResNet18Classifier(ckpt[0], preprocessing=PRE, device="auto")
    assert clf.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert not clf.model.training


def test_plain_state_dict_needs_explicit_classes(tmp_path):
    path = tmp_path / "plain.pth"
    torch.save(build_resnet18(2).state_dict(), path)
    with pytest.raises(CheckpointError, match="class mapping"):
        ResNet18Classifier(path, preprocessing=PRE, device="cpu")
    assert ResNet18Classifier(path, ["empty", "occupied"], PRE, "cpu").class_names == ["empty", "occupied"]


def test_model_state_dict_key_and_dataparallel_prefix(tmp_path):
    path = tmp_path / "m.pth"
    state = {f"module.{k}": v for k, v in build_resnet18(2).state_dict().items()}
    torch.save({"model_state_dict": state, "classes": ["empty", "occupied"]}, path)
    assert ResNet18Classifier(path, preprocessing=PRE, device="cpu").num_classes == 2


def test_class_override_conflicting_with_metadata(ckpt):
    with pytest.raises(CheckpointError, match="disagrees"):
        ResNet18Classifier(ckpt[0], ["occupied", "empty"], PRE, "cpu")


def test_wrong_class_count(tmp_path):
    path = tmp_path / "m.pth"
    save_checkpoint(path, 3)
    with pytest.raises(CheckpointError, match="outputs"):
        ResNet18Classifier(path, ["empty", "occupied"], PRE, "cpu")


def test_missing_preprocessing_rejected(ckpt):
    with pytest.raises(CheckpointError, match="preprocessing"):
        ResNet18Classifier(ckpt[0], device="cpu")


def test_preprocessing_mismatch_with_checkpoint(ckpt):
    with pytest.raises(CheckpointError, match="image_size"):
        ResNet18Classifier(ckpt[0], preprocessing={**PRE, "crop_size": 48}, device="cpu")
    with pytest.raises(CheckpointError, match="mean"):
        ResNet18Classifier(ckpt[0], preprocessing={**PRE, "mean": [0.5, 0.5, 0.5]}, device="cpu")


def test_incompatible_architecture(tmp_path):
    path = tmp_path / "m.pth"
    state = build_resnet18(2).state_dict()
    del state["layer4.1.bn2.weight"]
    torch.save({"state_dict": state, "class_names": ["empty", "occupied"]}, path)
    with pytest.raises(CheckpointError, match="not compatible"):
        ResNet18Classifier(path, preprocessing=PRE, device="cpu")


def test_non_resnet_head(tmp_path):
    path = tmp_path / "m.pth"
    torch.save({"state_dict": {"weight": torch.zeros(2, 3)}, "class_names": ["a", "b"]}, path)
    with pytest.raises(CheckpointError, match="fc.weight"):
        ResNet18Classifier(path, preprocessing=PRE, device="cpu")


def test_full_model_pickle_is_not_unpickled(tmp_path):
    path = tmp_path / "full.pth"
    torch.save(build_resnet18(2), path)
    with pytest.raises(CheckpointError, match="weights_only"):
        ResNet18Classifier(path, ["empty", "occupied"], PRE, "cpu")


def test_reserved_unknown_class_rejected(tmp_path):
    path = tmp_path / "m.pth"
    save_checkpoint(path, class_names=["empty", "unknown"])
    with pytest.raises(CheckpointError, match="UNKNOWN"):
        ResNet18Classifier(path, preprocessing=PRE, device="cpu")


def test_missing_file_and_empty_crop(ckpt, tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        ResNet18Classifier(tmp_path / "nope.pth", preprocessing=PRE, device="cpu")
    clf = ResNet18Classifier(ckpt[0], preprocessing=PRE, device="cpu")
    with pytest.raises(ValueError):
        clf.preprocess(np.zeros((0, 10, 3), dtype=np.uint8))
