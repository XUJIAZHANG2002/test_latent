



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class DeepUNet(nn.Module):
    def __init__(self, in_channels=4, T=500):
        super(DeepUNet, self).__init__()
        self.T = T

        # 时间嵌入
        self.time_emb = nn.Sequential(
            nn.Embedding(T, 256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # 下采样
        self.down1 = nn.Conv2d(in_channels, 32, 3, padding=1)  # 4 -> 32
        self.down1_bn = nn.BatchNorm2d(32)
        self.down2 = nn.Conv2d(32, 64, 3, padding=1)  # 32 -> 64
        self.down2_bn = nn.BatchNorm2d(64)
        self.down3 = nn.Conv2d(64, 128, 3, padding=1)  # 64 -> 128
        self.down3_bn = nn.BatchNorm2d(128)
        self.down4 = nn.Conv2d(128, 256, 3, padding=1)  # 128 -> 256
        self.down4_bn = nn.BatchNorm2d(256)
        self.down5 = nn.Conv2d(256, 512, 3, padding=1)  # 256 -> 512
        self.down5_bn = nn.BatchNorm2d(512)

        # 瓶颈层
        self.bottleneck = nn.Conv2d(512, 512, 3, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(512)

        # 上采样
        self.up1 = nn.Conv2d(512 + 256, 256, 3, padding=1)  # 512+256 -> 256
        self.up1_bn = nn.BatchNorm2d(256)
        self.up2 = nn.Conv2d(256 + 128, 128, 3, padding=1)  # 256+128 -> 128
        self.up2_bn = nn.BatchNorm2d(128)
        self.up3 = nn.Conv2d(128 + 64, 64, 3, padding=1)  # 128+64 -> 64
        self.up3_bn = nn.BatchNorm2d(64)
        self.up4 = nn.Conv2d(64 + 32, 32, 3, padding=1)  # 64+32 -> 32
        self.up4_bn = nn.BatchNorm2d(32)
        self.up5 = nn.Conv2d(32 + in_channels, 16, 3, padding=1)  # 32+4 -> 16
        self.up5_bn = nn.BatchNorm2d(16)

        # 输出层
        self.out = nn.Conv2d(16, in_channels, 1)  # 16 -> 4

    def forward(self, x, t):
        # x: [batch, 4, 7, 7], t: [batch]
        t_emb = self.time_emb(t).view(-1, 512, 1, 1).repeat(1, 1, 7, 7)

        # 下采样
        d1 = F.relu(self.down1_bn(self.down1(x)))  # [batch, 32, 7, 7]
        d2 = F.relu(self.down2_bn(self.down2(d1)))  # [batch, 64, 7, 7]
        d3 = F.relu(self.down3_bn(self.down3(d2)))  # [batch, 128, 7, 7]
        d4 = F.relu(self.down4_bn(self.down4(d3)))  # [batch, 256, 7, 7]
        d5 = F.relu(self.down5_bn(self.down5(d4)))  # [batch, 512, 7, 7]

        # 瓶颈层
        b = F.relu(self.bottleneck_bn(self.bottleneck(d5)) + t_emb)  # [batch, 512, 7, 7]

        # 上采样
        u1 = F.relu(self.up1_bn(self.up1(torch.cat([b, d4], dim=1))))  # [batch, 256, 7, 7]
        u2 = F.relu(self.up2_bn(self.up2(torch.cat([u1, d3], dim=1))))  # [batch, 128, 7, 7]
        u3 = F.relu(self.up3_bn(self.up3(torch.cat([u2, d2], dim=1))))  # [batch, 64, 7, 7]
        u4 = F.relu(self.up4_bn(self.up4(torch.cat([u3, d1], dim=1))))  # [batch, 32, 7, 7]
        u5 = F.relu(self.up5_bn(self.up5(torch.cat([u4, x], dim=1))))  # [batch, 16, 7, 7]

        return self.out(u5)  # [batch, 4, 7, 7]

# 扩散参数
T = 500
device = 'cuda'
betas = torch.linspace(1e-4, 0.01, T, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def forward_diffusion(z_0, t):
    noise = torch.randn_like(z_0)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    z_t = torch.sqrt(alpha_t) * z_0 + torch.sqrt(1 - alpha_t) * noise
    return z_t, noise

# 加载数据和模型
latent_dataset = torch.load('latent_mnist_binarized.pt')
batch_size = 2048

unet = DeepUNet(in_channels=2, T=T).cuda()
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(160):
    for i in range(0, len(latent_dataset), batch_size):
        z_0 = latent_dataset[i:i+batch_size].cuda()
        t = torch.randint(0, T, (z_0.size(0),), device='cuda')
        z_t, noise = forward_diffusion(z_0, t)

        optimizer.zero_grad()
        pred_noise = unet(z_t, t)
        loss = F.mse_loss(pred_noise, noise)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch {epoch}, Step {i}, Loss {loss.item():.4f}, LR {scheduler.get_last_lr()[0]:.6f}')

    scheduler.step()

torch.save(unet.state_dict(), 'deep_unet_model_v6.pth')
print("Deep U-Net v6 trained and saved!")





latent_dataset = torch.load('latent_mnist_binarized.pt')
batch_size = 2048

unet = DeepUNet(in_channels=2, T=T).cuda()
unet.load_state_dict(torch.load('deep_unet_model_v6.pth'))
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(80,160):
    for i in range(0, len(latent_dataset), batch_size):
        z_0 = latent_dataset[i:i+batch_size].cuda()
        t = torch.randint(0, T, (z_0.size(0),), device='cuda')
        z_t, noise = forward_diffusion(z_0, t)

        optimizer.zero_grad()
        pred_noise = unet(z_t, t)
        loss = F.mse_loss(pred_noise, noise)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch {epoch}, Step {i}, Loss {loss.item():.4f}, LR {scheduler.get_last_lr()[0]:.6f}')

    scheduler.step()

torch.save(unet.state_dict(), 'deep_unet_model_v6.pth')





# # VAE定义（只用解码器）
# # 增强的VAE模型
class VAE(nn.Module):
    def __init__(self, latent_channels=2):
        super(VAE, self).__init__()
        self.latent_channels = latent_channels
        # 编码器：增加一层
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28x1 -> 14x14x32
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),  # 14x14x16 -> 7x7x16
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),  # 7x7x8
            nn.ReLU()
        )
        self.fc_mu = nn.Conv2d(8, latent_channels, kernel_size=1)  # 7x7x8 -> 7x7x4
        self.fc_logvar = nn.Conv2d(8, latent_channels, kernel_size=1)
        # 解码器：增加一层
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7x4 -> 14x14x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14x8 -> 28x28x8
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),  # 28x28x1
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def reverse_diffusion(z_t, unet, steps=T-1):
    z = z_t
    for step in range(steps, 0, -1):
        t = torch.full((z.size(0),), step-1, device=device)
        pred_noise = unet(z, t)
        alpha_t = alphas_cumprod[step-1].view(1).to(device)
        alpha_t_prev = alphas_cumprod[step-2].view(1).to(device) if step > 1 else torch.tensor(1.0, device=device)
        beta_t = betas[step-1].view(1).to(device)
        z = (z - (1 - alpha_t) / torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t_prev)
        if step > 1:
            z = z + torch.sqrt(beta_t) * torch.randn_like(z)
        if step % 100 == 0:
            print(f"Step {step}: z mean {z.mean().item():.4f}, std {z.std().item():.4f}")
    return z

def binarize_tensor(tensor, threshold=0.5):
    return (tensor >= threshold).float()

unet = DeepUNet(in_channels=2, T=T).cuda()
vae = VAE(latent_channels=2).cuda()
unet.load_state_dict(torch.load('deep_unet_model_v6.pth'))
vae.load_state_dict(torch.load('vae_model.pth'))
unet.eval()
vae.eval()

with torch.no_grad():
    z_t = torch.randn(1, 2, 7, 7, device='cuda')
    print(f"Initial z_t: mean {z_t.mean().item():.4f}, std {z_t.std().item():.4f}")
    z_0 = reverse_diffusion(z_t, unet)
    print(f"Final z_0: mean {z_0.mean().item():.4f}, std {z_0.std().item():.4f}")
    x_generated = vae.decode(z_0)
    x_generated_binarized = binarize_tensor(x_generated, threshold=0.5)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x_generated[0].cpu().squeeze(), cmap='gray')
    plt.title('Generated (Continuous)')
    plt.subplot(1, 2, 2)
    plt.imshow(x_generated_binarized[0].cpu().squeeze(), cmap='gray')
    plt.title('Generated (Binarized)')
    plt.show()





num_images = 100
with torch.no_grad():
    z_t = torch.randn(num_images, 2, 7, 7, device='cuda')  # [10, 4, 7, 7]
    print(f"Initial z_t: mean {z_t.mean().item():.4f}, std {z_t.std().item():.4f}")
    z_0 = reverse_diffusion(z_t, unet)
    print(f"Final z_0: mean {z_0.mean().item():.4f}, std {z_0.std().item():.4f}")
    x_generated = vae.decode(z_0)  # [10, 1, 28, 28]
    x_generated_binarized = binarize_tensor(x_generated, threshold=0.5)

    # 可视化多张图像
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        # 第一行：连续值图像
        axes[0, i].imshow(x_generated[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Continuous')

        # 第二行：二值化图像
        axes[1, i].imshow(x_generated_binarized[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Binarized')

    plt.tight_layout()
    plt.show()





def noise(z_0, t):
    noise = torch.randn_like(z_0)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    z_t = torch.sqrt(alpha_t) * z_0 + torch.sqrt(1 - alpha_t) * noise
    return z_t, noise

# 原始反向扩散
def reverse_diffusion(z_t, unet, steps=T-1):
    z = z_t
    for step in range(steps, 0, -1):
        t = torch.full((z.size(0),), step-1, device=device)
        pred_noise = unet(z, t)
        alpha_t = alphas_cumprod[step-1].view(1).to(device)
        alpha_t_prev = alphas_cumprod[step-2].view(1).to(device) if step > 1 else torch.tensor(1.0, device=device)
        beta_t = betas[step-1].view(1).to(device)
        z = (z - (1 - alpha_t) / torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t_prev)
        if step > 1:
            z = z + torch.sqrt(beta_t) * torch.randn_like(z)
        if step % 100 == 0:
            print(f"Step {step} (No Mask): z mean {z.mean().item():.4f}, std {z.std().item():.4f}")
    return z

# 新的latent inpaint方法（修正版）
def latent_inpaint(z_0, unet, steps=T-1):
    batch_size, channels, height, width = z_0.shape

    # 硬编码3x3x2 mask区域 (中心位置)
    center_h = height // 2
    center_w = width // 2
    mask_region = torch.zeros_like(z_0)
    for b in range(batch_size):
        mask_region[b, :2, center_h-1:center_h+2, center_w-1:center_w+2] = 1.0  # 3x3x2区域
        # mask_region[b, :2, center_h-0:center_h+0, center_w-0:center_w+0] = 1.0
    # 初始化z_init = z_0
    # mask_region[:, :, center_h-0:center_h+0, center_w-0:center_w+0] = 0.0
    z_init = z_0.clone()

    # 从z_0加噪到z_k (初始时间步k)
    z_k, _ = noise(z_0, torch.full((batch_size,), steps-1, device=device))

    # 反向扩散
    z = z_k
    for t in range(steps, 0, -1):
        t_tensor = torch.full((batch_size,), t-1, device=device)

        # z_fg ~ denoise(z_t, d, t)
        pred_noise_fg = unet(z, t_tensor)
        alpha_t_fg = alphas_cumprod[t-1].view(1).to(device)
        alpha_t_prev_fg = alphas_cumprod[t-2].view(1).to(device) if t > 1 else torch.tensor(1.0, device=device)
        z_fg = (z - (1 - alpha_t_fg) / torch.sqrt(1 - alpha_t_fg) * pred_noise_fg) / torch.sqrt(alpha_t_prev_fg)

        # z_bg ~ noise(z_init, t)
        z_bg, _ = noise(z_init, t_tensor)

        # 融合
        z = z_fg * mask_region + z_bg * (1 - mask_region)

        if t > 1:
            z = z + torch.sqrt(betas[t-1].view(1).to(device)) * torch.randn_like(z)
        if t % 100 == 0:
            print(f"Step {t} (With Mask): z mean {z.mean().item():.4f}, std {z.std().item():.4f}")

    return z
import numpy as np
# def latent_inpaint(z_0, unet, steps=T-1):
#     batch_size, channels, height, width = z_0.shape

#     # 硬编码3x3x2 mask区域 (中心位置)
#     # upper_left = np.random.randint(0, 5, size=2)
#     # lower_left = np.random.randint(0, 2, size=2), upper_left[1]
#     # lower_right = np.random.randio
#     # center_h = height // 2
#     # center_w = width // 2
#     # mask_region = torch.zeros_like(z_0)
#     # for b in range(batch_size):
#     #     mask_region[b, :2, center_h-1:center_h+2, center_w-1:center_w+2] = 1.0  # 3x3x2区域
#     #     # mask_region[b, :2, center_h-0:center_h+0, center_w-0:center_w+0] = 1.0
#     # # 初始化z_init = z_0
#     # # mask_region[:, :, center_h-0:center_h+0, center_w-0:center_w+0] = 0.0
#     # z_init = z_0.clone()

#     mask_region = torch.zeros_like(z_0)
#     z_init = z_0.clone()
#     for b in range(batch_size):
#         # Randomly determine mask size (between 1x1 and 4x4)
#         mask_h = np.random.randint(1, 5)
#         mask_w = np.random.randint(1, 5)

#         # Randomly determine mask position within image bounds
#         start_h = np.random.randint(0, height - mask_h + 1)
#         start_w = np.random.randint(0, width - mask_w + 1)

#         # Apply mask to first two channels
#         mask_region[b, :2, start_h:start_h + mask_h, start_w:start_w + mask_w] = 1.0

#     # 从z_0加噪到z_k (初始时间步k)
#     z_k, _ = noise(z_0, torch.full((batch_size,), steps-1, device=device))

#     # 反向扩散
#     z = z_k
#     for t in range(steps, 0, -1):
#         t_tensor = torch.full((batch_size,), t-1, device=device)

#         # z_fg ~ denoise(z_t, d, t)
#         pred_noise_fg = unet(z, t_tensor)
#         alpha_t_fg = alphas_cumprod[t-1].view(1).to(device)
#         alpha_t_prev_fg = alphas_cumprod[t-2].view(1).to(device) if t > 1 else torch.tensor(1.0, device=device)
#         z_fg = (z - (1 - alpha_t_fg) / torch.sqrt(1 - alpha_t_fg) * pred_noise_fg) / torch.sqrt(alpha_t_prev_fg)

#         # z_bg ~ noise(z_init, t)
#         z_bg, _ = noise(z_init, t_tensor)

#         # 融合
#         z = z_fg * mask_region + z_bg * (1 - mask_region)

#         if t > 1:
#             z = z + torch.sqrt(betas[t-1].view(1).to(device)) * torch.randn_like(z)
#         if t % 100 == 0:
#             print(f"Step {t} (With Mask): z mean {z.mean().item():.4f}, std {z.std().item():.4f}")

#     return z, mask_region





import torch
import matplotlib.pyplot as plt

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=10, shuffle=True)

# 加载模型
unet = DeepUNet(in_channels=2, T=T).cuda()
vae = VAE(latent_channels=2).cuda()
unet.load_state_dict(torch.load('deep_unet_model_v6.pth'))
vae.load_state_dict(torch.load('vae_model_binarized_optimized.pth'))
unet.eval()
vae.eval()

# 生成对比图并计算IoU和Mean L2 Norm
num_images = 10
with torch.no_grad():
    # 从MNIST数据集中取10张图像
    data_iter = iter(dataloader)
    x_0, _ = next(data_iter)  # [10, 1, 28, 28]
    x_0 = x_0.cuda()

    # 通过VAE编码得到z_0
    mu, logvar = vae.encode(x_0)  # mu, logvar: [10, 2*7*7]
    mu = mu.view(-1, 2, 7, 7)  # 调整形状为[10, 2, 7, 7]
    logvar = logvar.view(-1, 2, 7, 7)
    z_0 = vae.reparameterize(mu, logvar)  # 采样得到z_0: [10, 2, 7, 7]
    print(f"Initial z_0: mean {z_0.mean().item():.4f}, std {z_0.std().item():.4f}")

    # 无mask情况（直接解码z_0）
    x_generated_no_mask = vae.decode(z_0)
    x_generated_binarized_no_mask = binarize_tensor(x_generated_no_mask, threshold=0.5)

    # 有mask情况（latent inpaint）
    z_0_with_mask, mask_region = latent_inpaint(z_0, unet)
    x_generated_with_mask = vae.decode(z_0_with_mask)
    x_generated_binarized_with_mask = binarize_tensor(x_generated_with_mask, threshold=0.5)

    print(f"Final z_0 (No Mask): mean {z_0.mean().item():.4f}, std {z_0.std().item():.4f}")
    print(f"Final z_0 (With Mask): mean {z_0_with_mask.mean().item():.4f}, std {z_0_with_mask.std().item():.4f}")

    # 计算IoU
    iou_scores = []
    for i in range(num_images):
        intersection = (x_generated_binarized_no_mask[i] * x_generated_binarized_with_mask[i]).sum()
        union = (x_generated_binarized_no_mask[i] + x_generated_binarized_with_mask[i]).clamp(0, 1).sum()
        iou = intersection / union if union > 0 else 0.0
        iou_scores.append(iou.item())
        print(f"Image {i} IoU: {iou.item():.4f}")

    mean_iou = sum(iou_scores) / len(iou_scores)
    print(f"Mean IoU across {num_images} images: {mean_iou:.4f}")

    # 计算Mean L2 Norm（未二值化）
    l2_norms = []
    for i in range(num_images):
        l2_norm = torch.sqrt(torch.sum((x_generated_no_mask[i] - x_generated_with_mask[i]) ** 2))
        l2_norms.append(l2_norm.item())
        print(f"Image {i} L2 Norm: {l2_norm.item():.4f}")

    mean_l2_norm = sum(l2_norms) / len(l2_norms)
    print(f"Mean L2 Norm across {num_images} images: {mean_l2_norm:.4f}")

    # 可视化对比图
    num_display = num_images  # 显示所有10张
    fig, axes = plt.subplots(5, num_display, figsize=(num_display * 2, 10))  # 5 rows now

    for i in range(num_display):
        # 第一行：原始数据 x_0
        axes[0, i].imshow(x_0[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original Data')

        # 第二行：无mask的连续值图像
        axes[1, i].imshow(x_generated_no_mask[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('No Mask - Continuous')

        # 第三行：无mask的二值化图像
        axes[2, i].imshow(x_generated_binarized_no_mask[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('No Mask - Binarized')

        # 第四行：有mask的连续值图像
        axes[3, i].imshow(x_generated_with_mask[i].cpu().squeeze(), cmap='gray')
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_title('With Mask - Continuous')

        # 第五行：有mask的二值化图像
        axes[4, i].imshow(x_generated_binarized_with_mask[i].cpu().squeeze(), cmap='gray')
        axes[4, i].axis('off')
        if i == 0:
            axes[4, i].set_title('With Mask - Binarized')

    plt.tight_layout()
    plt.show()





import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=10, shuffle=True)

# 加载模型
unet = DeepUNet(in_channels=2, T=T).cuda()
vae = VAE(latent_channels=2).cuda()
unet.load_state_dict(torch.load('deep_unet_model_v6.pth'))
vae.load_state_dict(torch.load('vae_model_binarized_optimized.pth'))
unet.eval()
vae.eval()

# 生成对比图并计算IoU和Mean L2 Norm
num_images = 10
with torch.no_grad():
    # 从MNIST数据集中取10张图像
    data_iter = iter(dataloader)
    x_0, _ = next(data_iter)  # [10, 1, 28, 28]
    x_0 = x_0.cuda()

    # 通过VAE编码得到z_0
    mu, logvar = vae.encode(x_0)  # mu, logvar: [10, 2*7*7]
    mu = mu.view(-1, 2, 7, 7)  # 调整形状为[10, 2, 7, 7]
    logvar = logvar.view(-1, 2, 7, 7)
    z_0 = vae.reparameterize(mu, logvar)  # 采样得到z_0: [10, 2, 7, 7]
    print(f"Initial z_0: mean {z_0.mean().item():.4f}, std {z_0.std().item():.4f}")

    # 无mask情况（直接解码z_0）
    x_generated_no_mask = vae.decode(z_0)
    x_generated_binarized_no_mask = binarize_tensor(x_generated_no_mask, threshold=0.5)

    # 有mask情况（latent inpaint）
    z_0_with_mask = latent_inpaint(z_0, unet)
    x_generated_with_mask = vae.decode(z_0_with_mask)
    x_generated_binarized_with_mask = binarize_tensor(x_generated_with_mask, threshold=0.5)

    print(f"Final z_0 (No Mask): mean {z_0.mean().item():.4f}, std {z_0.std().item():.4f}")
    print(f"Final z_0 (With Mask): mean {z_0_with_mask.mean().item():.4f}, std {z_0_with_mask.std().item():.4f}")

    # 可视化对比图
    num_display = num_images  # 显示所有10张
    fig, axes = plt.subplots(5, num_display, figsize=(num_display * 2, 10))  # 5 rows now

    for i in range(num_display):
        # 第一行：原始数据 x_0
        axes[0, i].imshow(x_0[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original Data')

        # 第二行：无mask的连续值图像
        axes[1, i].imshow(x_generated_no_mask[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('No Mask - Continuous')

        # 第三行：无mask的二值化图像
        axes[2, i].imshow(x_generated_binarized_no_mask[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('No Mask - Binarized')

        # 第四行：有mask的连续值图像
        axes[3, i].imshow(x_generated_with_mask[i].cpu().squeeze(), cmap='gray')
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_title('With Mask - Continuous')

        # 第五行：有mask的二值化图像
        axes[4, i].imshow(x_generated_binarized_with_mask[i].cpu().squeeze(), cmap='gray')
        axes[4, i].axis('off')
        if i == 0:
            axes[4, i].set_title('With Mask - Binarized')

        # Add red outline around the center 12x12 region
        ax = axes[4, i]  # Choose the row that shows original images (you can apply it to other rows as well)

        # Create a Rectangle patch for the red outline
        rect = patches.Rectangle((6, 6), 12, 12, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax = axes[3, i]  # Choose the row that shows original images (you can apply it to other rows as well)
        # ax.add_patch(rect)
        # Create a Rectangle patch for the red outline
        rect = patches.Rectangle((6, 6), 12, 12, linewidth=2, edgecolor='red', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()
fig.savefig("latent_diffusion.png")



import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=10, shuffle=True)

# Generate visualization
num_images = 10
with torch.no_grad():
    # Fetch batch of images
    data_iter = iter(dataloader)
    x_0, _ = next(data_iter)  # [10, 1, 28, 28]
    x_0 = x_0.cuda()

    # Encode with VAE to get latent space
    mu, logvar = vae.encode(x_0)  # Shape: [10, 2*7*7]
    mu = mu.view(-1, 2, 7, 7)
    logvar = logvar.view(-1, 2, 7, 7)
    z_0 = vae.reparameterize(mu, logvar)  # Shape: [10, 2, 7, 7]

    # Perform inpainting with latent masking
    z_0_with_mask, mask_region = latent_inpaint(z_0, unet)

    # Decode both versions
    x_generated_with_mask = vae.decode(z_0_with_mask)
    x_generated_binarized_with_mask = binarize_tensor(x_generated_with_mask, threshold=0.5)

    # **Upsample the 7x7 masks to 28x28**
    mask_upsampled = F.interpolate(mask_region, size=(28, 28), mode='nearest')

    # Visualization setup
    fig, axes = plt.subplots(5, num_images, figsize=(num_images * 2, 10))  # 5 rows: Original, Inpainted, Binarized, Masked, Mask Overlay

    for i in range(num_images):
        # **1st Row: Original Data**
        axes[0, i].imshow(x_0[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        # **2nd Row: Inpainted Image**
        axes[1, i].imshow(x_generated_with_mask[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Inpainted')

        # **3rd Row: Binarized Inpainted**
        axes[2, i].imshow(x_generated_binarized_with_mask[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Binarized Inpainted')

        # **4th Row: Masked Region Applied**
        axes[3, i].imshow(mask_upsampled[i, 0].cpu().squeeze(), cmap='gray')
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_title('Masked Region')

        # **5th Row: Overlay Mask with Red Rectangle**
        mask_img = mask_upsampled[i, 0].cpu().numpy()
        axes[4, i].imshow(x_0[i].cpu().squeeze(), cmap='gray')
        axes[4, i].axis('off')
        if i == 0:
            axes[4, i].set_title('Mask Overlay')

        # **Find Bounding Box of the Mask**
        mask_indices = torch.nonzero(mask_upsampled[i, 0])  # Get non-zero mask coordinates
        if mask_indices.numel() > 0:
            min_h, min_w = mask_indices[:, 0].min().item(), mask_indices[:, 1].min().item()
            max_h, max_w = mask_indices[:, 0].max().item(), mask_indices[:, 1].max().item()
            width, height = max_w - min_w + 1, max_h - min_h + 1

            # Draw rectangle on masked region
            rect = patches.Rectangle((min_w, min_h), width, height, linewidth=2, edgecolor='red', facecolor='none')
            axes[4, i].add_patch(rect)

    plt.tight_layout()
    plt.show()
fig.savefig("latent_diffusion_with_masks.png")






