import os
from .cf_encoder import CLIPVisionTower
# from .clip_encoder import CLIPVisionTower



def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    local_clip = "/path/to/sage_repro_bundle/clip-vit-large-patch14-336"
    if vision_tower == "openai/clip-vit-large-patch14-336" and os.path.exists(local_clip):
        vision_tower = local_clip
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
