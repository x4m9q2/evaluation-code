import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


def _resolve_local_vision_tower(vision_tower, vision_tower_cfg):
    if not isinstance(vision_tower, str):
        return vision_tower
    if os.path.exists(vision_tower) or os.path.isabs(vision_tower):
        return vision_tower

    config_root = getattr(vision_tower_cfg, "_name_or_path", None)
    if config_root:
        if os.path.isfile(config_root):
            config_root = os.path.dirname(config_root)
        candidate = os.path.normpath(os.path.join(config_root, vision_tower))
        if os.path.exists(candidate):
            return candidate
    return vision_tower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    vision_tower = _resolve_local_vision_tower(vision_tower, vision_tower_cfg)
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
