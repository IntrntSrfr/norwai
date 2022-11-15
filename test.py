import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([36, 64, 32, 32], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(64, 128, kernel_size=[5, 5], padding=[2, 2], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()