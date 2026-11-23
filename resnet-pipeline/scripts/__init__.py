"""Command-line utilities.

Run from the project root::

    python -m scripts.scene.select_rois --source testing/input.mp4
    python -m scripts.scene.discover_scene --from-rois config/workstations.json
    python -m scripts.scene.review_scene_map --map config/scene_map.json
    python -m scripts.occupancy.inspect_checkpoint
    python -m scripts.occupancy.test_classifier --image path/to/crop.jpg
"""
