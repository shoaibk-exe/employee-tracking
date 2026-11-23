"""Core packages for workstation occupancy and person-centric presence monitoring.

Subpackages:
  tracking/     person detect + multi-object tracker + ReID encoder
  activity/     VideoMAE desk activity (computer_interaction / neither / phone_using)
  vlm/          vision-language clothing features (Qwen / Moondream)
  identity/     same-day P## gallery and verified employee identity
  association/  person↔workstation association, fusion, time accounting
  occupancy/    ResNet-18 ROI classifier path (state, intervals, ROIs)
  scene/        workstation / scene map discovery
  io/           video I/O, overlays, CSV / Excel reports
"""
