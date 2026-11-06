"""Chuyển checkpoint xám (4 kênh) sang RGB (12 kênh) để khỏi train lại từ đầu.

Ảnh xám ~ 0.299 R + 0.587 G + 0.114 B, nên lớp conv đầu của model RGB được khởi tạo sao cho cho ra gần đúng
cùng đầu ra với model xám; các lớp sau giữ nguyên. Thứ tự kênh RGB là [khung0 R, G, B, khung1 R, G, B, ...]
(xem StackToChannels trong envs/factory.py).
"""
import torch

LUMA = (0.299, 0.587, 0.114)


def _expand(w):
    """w: (out, 4, ...) -> (out, 12, ...)."""
    assert w.shape[1] == 4, f"trọng số lớp đầu có {w.shape[1]} kênh, cần 4 (xám)"
    return torch.stack([w[:, f] * c for f in range(4) for c in LUMA], dim=1)


def gray_to_rgb_state(state):
    """state_dict model xám -> state_dict model RGB (CNN, CNN+LSTM hoặc ViT)."""
    key = "patch_embed.weight" if "patch_embed.weight" in state else "cnn.0.weight"
    out = dict(state)
    out[key] = _expand(state[key])
    return out


def load_gray_as_rgb(path, algo="ppo", device="cpu"):
    """Nạp checkpoint xám từ `path`, trả về model RGB (chế độ eval) sẵn sàng train tiếp với rgb_scale=True."""
    import tempfile, os
    from ..runs import load_model
    state = torch.load(path, map_location="cpu")
    with tempfile.TemporaryDirectory() as d:          # load_model đọc từ file nên ghi tạm state RGB
        p = os.path.join(d, "rgb.pt")
        torch.save(gray_to_rgb_state(state), p)
        return load_model(p, algo, device)
