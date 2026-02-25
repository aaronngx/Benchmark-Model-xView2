import torch
ckpt = torch.load('models/cv5/fold0/best.pt', map_location='cpu', weights_only=False)
print('keys:', list(ckpt.keys()))
sd = ckpt['model_state_dict']
print('model keys:', list(sd.keys())[:5])
print('epoch:', ckpt.get('epoch'))
print('val_macro_f1:', ckpt.get('val_macro_f1'))
print('model_type:', ckpt.get('model_type'))
