import torch
ckpt = torch.load("models/six_channel/best.pt", map_location="cpu", weights_only=False)
print("epoch:", ckpt["epoch"])
print("val_macro_f1:", ckpt["val_macro_f1"])
print("per_class_f1:", ckpt.get("per_class_f1", {}))
