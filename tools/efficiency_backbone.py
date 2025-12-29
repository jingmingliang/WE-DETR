import torch
import os
from PIL import Image
import torchvision.transforms as T
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core import YAMLConfig


# 小波变换函数 - 从ResNet模型中提取出来，用于特征提取器
def multi_level_haar_wavelet_transform(X):
    """
    Perform multi-level 2D discrete Haar wavelet transform on input batch of images,
    keeping only the low frequency components at each level.
    """
    device = X.device

    # Define Haar filter
    haar_filter = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=X.dtype, device=device).unsqueeze(0).unsqueeze(0)

    def apply_wavelet_transform(x):
        # Apply filter to each channel
        B, C, H, W = x.shape
        low_pass = torch.nn.functional.conv2d(
            x.view(B * C, 1, H, W),  # Merge batch and channel
            haar_filter,
            stride=2,  # Downsample
            padding=0  # No padding
        )
        # Restore shape to (B, C, H/2, W/2)
        return low_pass.view(B, C, low_pass.shape[-2], low_pass.shape[-1])

    # Multi-level transform
    x2 = apply_wavelet_transform(X)  # Half resolution
    x4 = apply_wavelet_transform(x2)

    return x2, x4


# 2. 改进后的特征提取器
class RTDETRFeatureExtractor(torch.nn.Module):
    def __init__(self, model, target_layers, is_wavelet_model=False):
        super().__init__()
        self.model = model
        self.target_layers = target_layers
        self.is_wavelet_model = is_wavelet_model
        self.feature_maps = {}

    def forward(self, x):
        self.feature_maps.clear()

        # 保存原始输入图像以供后续处理
        original_x = x.clone()

        # 如果是小波增强模型，需要手动实现前向传播过程以捕获正确的融合特征
        if self.is_wavelet_model:
            # 计算小波变换
            x2, x4 = multi_level_haar_wavelet_transform(x)

            # 处理小波分量
            x_2 = self.model.backbone.x2(x2)
            x_4 = self.model.backbone.x4(x4)
            x_8 = self.model.backbone.x8(x4)

            # 应用空间注意力
            x_2 = self.model.backbone.SpacialAttention1(x_2)
            x_4 = self.model.backbone.SpacialAttention2(x_4)
            x_8 = self.model.backbone.SpacialAttention3(x_8)

            # 初始卷积层
            conv1 = self.model.backbone.conv1(x)

            # 应用注意力和融合第一个小波特征
            conv1_enhanced = self.model.backbone.SpacialAttention4(conv1) + x_2

            # 保存第一层融合后的特征
            if 'backbone.conv1' in self.target_layers:
                self.feature_maps['backbone.conv1'] = conv1_enhanced

            # 最大池化
            x = torch.nn.functional.max_pool2d(conv1_enhanced, kernel_size=3, stride=2, padding=1)

            # 处理res_layers各阶段
            for idx, stage in enumerate(self.model.backbone.res_layers):
                layer_name = f'backbone.res_layers.{idx}'

                # 根据索引应用不同的小波融合
                if idx == 0:
                    # 应用空间注意力和第二个小波分量
                    x = self.model.backbone.SpacialAttention5(x) + x_4
                elif idx == 1:
                    # 应用空间注意力和第三个小波分量
                    x = self.model.backbone.SpacialAttention6(x) + x_8

                # 通过当前阶段
                x = stage(x)

                # 如果是目标层，保存特征
                if layer_name in self.target_layers:
                    self.feature_maps[layer_name] = x

        else:
            # 对于原始模型，使用挂钩方式获取特征
            hooks = []

            def hook_fn(name):
                def hook(module, input, output):
                    self.feature_maps[name] = output

                return hook

            # 为每一层注册钩子
            for name in self.target_layers:
                layer = self._get_layer(name)
                hooks.append(layer.register_forward_hook(hook_fn(name)))

            # 执行前向传播
            _ = self.model(x)

            # 移除钩子
            for hook in hooks:
                hook.remove()

        # 确保正常执行模型前向传播来完成所有计算
        if not self.is_wavelet_model:
            _ = self.model(original_x)

        # 返回目标层的特征
        return [self.feature_maps[name] for name in self.target_layers]

    def _get_layer(self, name):
        layer = self.model
        for part in name.split('.'):
            layer = getattr(layer, part)
        return layer


# 3. RT-DETR模型加载
def load_rtdetr_model(config_path, weight_path, device='cpu'):
    cfg = YAMLConfig(config_path, resume=weight_path)
    if weight_path:
        checkpoint = torch.load(weight_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        cfg.model.load_state_dict(state)
    model = cfg.model.deploy().to(device)
    model.eval()
    return model


# 4. 预处理图片
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    return transform(img)[None], img


# 5. 可视化函数
def compare_backbone_features(original_features, improved_features, layer_names, save_dir, original_image):
    os.makedirs(save_dir, exist_ok=True)
    for i, (layer_name, orig_feat, improved_feat) in enumerate(zip(layer_names, original_features, improved_features)):
        def process_feature(feat):
            if hasattr(feat, 'requires_grad') and feat.requires_grad:
                feat = feat.detach()
            if hasattr(feat, 'is_cuda') and feat.is_cuda:
                feat = feat.cpu()
            if len(feat.shape) == 4:
                feat = feat[0]
            return feat.numpy()

        orig_np = process_feature(orig_feat)
        improved_np = process_feature(improved_feat)

        def generate_heatmap(feature_np):
            heatmap = np.mean(feature_np, axis=0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap

        orig_heatmap = generate_heatmap(orig_np)
        improved_heatmap = generate_heatmap(improved_np)

        # 增加子图数量到6个 (1×6布局)
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        fig.suptitle(f'Stage {i + 1}: {layer_name} - Before vs After Wavelet Enhancement', fontsize=16)

        # 获取原始图像尺寸
        orig_w, orig_h = original_image.size
        orig_array = np.array(original_image)
        
        # 将所有热图调整到与原始图像相同的尺寸
        orig_heatmap_resized = cv2.resize(orig_heatmap, (orig_w, orig_h))
        improved_heatmap_resized = cv2.resize(improved_heatmap, (orig_w, orig_h))
        diff_map = improved_heatmap - orig_heatmap
        diff_map_resized = cv2.resize(diff_map, (orig_w, orig_h))

        # 第1图：原始特征热图（调整到原图尺寸）
        im1 = axes[0].imshow(orig_heatmap_resized, cmap='jet')
        axes[0].set_title('Original Backbone')
        axes[0].axis('off')
        # plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # 第2图：改进特征热图（调整到原图尺寸）
        im2 = axes[1].imshow(improved_heatmap_resized, cmap='jet')
        axes[1].set_title('Wavelet Enhanced Backbone')
        axes[1].axis('off')
        # plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # 第3图：原始图像
        im3 = axes[2].imshow(orig_array)
        axes[2].set_title('Original Image')
        axes[2].axis('off')

        # 第4图：原始特征叠加在原图上
        heatmap_colored_orig = cv2.applyColorMap((orig_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored_orig = cv2.cvtColor(heatmap_colored_orig, cv2.COLOR_BGR2RGB)
        overlay_orig = cv2.addWeighted(orig_array, 0.4, heatmap_colored_orig, 0.6, 0)
        axes[3].imshow(overlay_orig)
        axes[3].set_title('Original Features\nOverlay')
        axes[3].axis('off')

        # 第5图：改进特征叠加在原图上
        heatmap_colored_improved = cv2.applyColorMap((improved_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored_improved = cv2.cvtColor(heatmap_colored_improved, cv2.COLOR_BGR2RGB)
        overlay_improved = cv2.addWeighted(orig_array, 0.4, heatmap_colored_improved, 0.6, 0)
        axes[4].imshow(overlay_improved)
        axes[4].set_title('Enhanced Features\nOverlay')
        axes[4].axis('off')

        # 第6图：差异图叠加在原图上
        # 创建带有颜色编码的差异热图

        # 使用RdBu_r色映射增强正负差异的可视化
        # 正值(增强)显示为红色，负值(减弱)显示为蓝色
        norm_diff_map = (diff_map_resized - diff_map_resized.min()) / (
                diff_map_resized.max() - diff_map_resized.min() + 1e-8)
        diff_heatmap = plt.cm.RdBu_r(norm_diff_map)[:, :, :3]  # 使用matplotlib的RdBu_r色图

        # 创建一个只显示差异的半透明叠加层
        alpha = 0.5  # 透明度设置
        background = orig_array.copy().astype(np.float32) / 255
        diff_overlay = background * (1 - alpha) + diff_heatmap * alpha
        diff_overlay = (diff_overlay * 255).astype(np.uint8)

        axes[5].imshow(diff_overlay)
        axes[5].set_title('Difference Overlay\n(Red=Enhanced, Blue=Reduced)')
        axes[5].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'stage_{i + 1}_{layer_name.replace(".", "_")}_comparison.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"对比图已保存: {save_path}")


def create_side_by_side_comparison(original_features, improved_features, layer_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n_stages = len(original_features)
    fig, axes = plt.subplots(2, n_stages, figsize=(4 * n_stages, 8))
    fig.suptitle('Backbone Feature Comparison: Original vs Wavelet Enhanced', fontsize=16)
    for i, (layer_name, orig_feat, improved_feat) in enumerate(zip(layer_names, original_features, improved_features)):
        def process_feature(feat):
            if hasattr(feat, 'requires_grad') and feat.requires_grad:
                feat = feat.detach()
            if hasattr(feat, 'is_cuda') and feat.is_cuda:
                feat = feat.cpu()
            if len(feat.shape) == 4:
                feat = feat[0]
            return feat.numpy()

        orig_np = process_feature(orig_feat)
        improved_np = process_feature(improved_feat)
        orig_heatmap = np.mean(orig_np, axis=0)
        orig_heatmap = (orig_heatmap - orig_heatmap.min()) / (orig_heatmap.max() - orig_heatmap.min() + 1e-8)
        improved_heatmap = np.mean(improved_np, axis=0)
        improved_heatmap = (improved_heatmap - improved_heatmap.min()) / (
                    improved_heatmap.max() - improved_heatmap.min() + 1e-8)
        im1 = axes[0, i].imshow(orig_heatmap, cmap='jet')
        axes[0, i].set_title(f'Original\n{layer_name}')
        axes[0, i].axis('off')
        im2 = axes[1, i].imshow(improved_heatmap, cmap='jet')
        axes[1, i].set_title(f'Wavelet Enhanced\n{layer_name}')
        axes[1, i].axis('off')
        mse_reduction = np.mean((orig_heatmap - improved_heatmap) ** 2)
        detail_enhancement = np.std(improved_heatmap) - np.std(orig_heatmap)
        axes[1, i].text(0.5, -0.1, f'Detail↑: {detail_enhancement:.3f}',
                        transform=axes[1, i].transAxes, ha='center', fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'backbone_comparison_overview.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"总览对比图已保存: {save_path}")


def create_improvement_metrics_chart(original_features, improved_features, layer_names, save_dir):
    metrics = []
    for orig_feat, improved_feat in zip(original_features, improved_features):
        def process_feature(feat):
            if hasattr(feat, 'requires_grad') and feat.requires_grad:
                feat = feat.detach()
            if hasattr(feat, 'is_cuda') and feat.is_cuda:
                feat = feat.cpu()
            if len(feat.shape) == 4:
                feat = feat[0]
            return feat.numpy()

        orig_np = process_feature(orig_feat)
        improved_np = process_feature(improved_feat)
        orig_richness = np.std(orig_np)
        improved_richness = np.std(improved_np)
        richness_improvement = (improved_richness - orig_richness) / (orig_richness + 1e-8) * 100
        orig_activation = np.mean(np.abs(orig_np))
        improved_activation = np.mean(np.abs(improved_np))
        activation_improvement = (improved_activation - orig_activation) / (orig_activation + 1e-8) * 100

        def calculate_entropy(feature_map):
            hist, _ = np.histogram(feature_map.flatten(), bins=256, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))

        orig_entropy = calculate_entropy(orig_np)
        improved_entropy = calculate_entropy(improved_np)
        entropy_improvement = (improved_entropy - orig_entropy) / (orig_entropy + 1e-8) * 100
        metrics.append({
            'richness': richness_improvement,
            'activation': activation_improvement,
            'entropy': entropy_improvement
        })
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Quantitative Improvement Metrics', fontsize=16)
    stages = [f'Stage {i + 1}' for i in range(len(layer_names))]
    richness_values = [m['richness'] for m in metrics]
    axes[0].bar(stages, richness_values, color='skyblue', alpha=0.7)
    axes[0].set_title('Feature Richness Improvement (%)')
    axes[0].set_ylabel('Improvement (%)')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='x', rotation=45)
    activation_values = [m['activation'] for m in metrics]
    axes[1].bar(stages, activation_values, color='lightcoral', alpha=0.7)
    axes[1].set_title('Activation Intensity Improvement (%)')
    axes[1].set_ylabel('Improvement (%)')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    entropy_values = [m['entropy'] for m in metrics]
    axes[2].bar(stages, entropy_values, color='lightgreen', alpha=0.7)
    axes[2].set_title('Information Entropy Improvement (%)')
    axes[2].set_ylabel('Improvement (%)')
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'improvement_metrics.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"改进指标图已保存: {save_path}")
    print("\n改进效果数值总结:")
    print("=" * 50)
    for i, (layer_name, metric) in enumerate(zip(layer_names, metrics)):
        print(f"Stage {i + 1} ({layer_name}):")
        print(f"  特征丰富度提升: {metric['richness']:+.1f}%")
        print(f"  激活强度提升: {metric['activation']:+.1f}%")
        print(f"  信息熵提升: {metric['entropy']:+.1f}%")
    avg_richness = np.mean(richness_values)
    avg_activation = np.mean(activation_values)
    avg_entropy = np.mean(entropy_values)
    print(f"\n平均改进效果:")
    print(f"  平均特征丰富度提升: {avg_richness:+.1f}%")
    print(f"  平均激活强度提升: {avg_activation:+.1f}%")
    print(f"  平均信息熵提升: {avg_entropy:+.1f}%")


def visualize_wavelet_improvement(original_features, improved_features, layer_names, save_dir, original_image):
    print("开始生成小波改进效果可视化...")
    
    # 原有的可视化方法
    compare_backbone_features(original_features, improved_features, layer_names, save_dir, original_image)
    create_side_by_side_comparison(original_features, improved_features, layer_names, save_dir)
    create_improvement_metrics_chart(original_features, improved_features, layer_names, save_dir)
    
    # 新增：高级特征质量分析
    print("执行高级特征质量分析...")
    analysis_results = advanced_feature_analysis(original_features, improved_features, layer_names)
    advanced_metrics = create_advanced_metrics_visualization(analysis_results, save_dir)
    
    print(f"\n所有可视化图表已保存到: {save_dir}")
    print("生成的图表包括:")
    print("1. 各stage详细对比图")
    print("2. 总览对比图")
    print("3. 量化改进指标图")
    print("4. 高级特征质量分析图 (新增)")
    
    return advanced_metrics


# 6. 主流程 - 更新为使用新的特征提取器
def main_compare(
        orig_config, orig_weight, wavelet_config, wavelet_weight, img_path, save_dir,
        target_layers=['backbone.conv1', 'backbone.res_layers.0', 'backbone.res_layers.1', 'backbone.res_layers.2',
                       'backbone.res_layers.3'],
        device='cpu'
):
    print("加载原始模型...")
    orig_model = load_rtdetr_model(orig_config, orig_weight, device)
    print("加载小波增强模型...")
    wavelet_model = load_rtdetr_model(wavelet_config, wavelet_weight, device)

    print("初始化特征提取器...")
    orig_extractor = RTDETRFeatureExtractor(orig_model, target_layers, is_wavelet_model=False)
    wavelet_extractor = RTDETRFeatureExtractor(wavelet_model, target_layers, is_wavelet_model=True)

    print(f"处理图像: {img_path}")
    img_tensor, img_pil = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)

    print("提取原始模型特征...")
    with torch.no_grad():
        orig_feats = orig_extractor(img_tensor)

    print("提取小波增强模型特征...")
    with torch.no_grad():
        wavelet_feats = wavelet_extractor(img_tensor)

    print("开始可视化对比...")
    advanced_metrics = visualize_wavelet_improvement(
        original_features=orig_feats,
        improved_features=wavelet_feats,
        layer_names=target_layers,
        save_dir=save_dir,
        original_image=img_pil
    )

    print("处理完成!")
    
    # 输出最终评估结果
    print("\n" + "="*60)
    print("最终评估结果总结:")
    print("="*60)
    print(f"小目标响应改进: {advanced_metrics['avg_small_object_improvement']:+.4f}")
    print(f"高频内容改进: {advanced_metrics['avg_frequency_improvement']:+.4f}")
    print(f"通道独立性改进: {advanced_metrics['avg_correlation_improvement']:+.4f}")
    print(f"特征判别能力改进: {advanced_metrics['avg_discriminability_improvement']:+.4f}")
    
    # 综合评分
    overall_score = (
        advanced_metrics['avg_small_object_improvement'] * 0.4 +  # 小目标响应权重最高
        advanced_metrics['avg_frequency_improvement'] * 0.3 +     # 频域改进次之
        advanced_metrics['avg_correlation_improvement'] * 0.2 +   # 通道独立性
        advanced_metrics['avg_discriminability_improvement'] * 0.1 # 判别能力
    )
    
    print(f"\n综合改进评分: {overall_score:+.4f}")
    if overall_score > 0.01:
        print("✅ 改进效果显著！")
    elif overall_score > 0:
        print("✅ 改进效果轻微但积极")
    else:
        print("❌ 改进效果不明显或负面")
    
    return advanced_metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-config', type=str,
                        default=r"D:\RT-DETR-main\rtdetr_pytorch\configs\rtdetr\rtdetr_r50vd_6x_coco.yml")
    parser.add_argument('--orig-weight', type=str,
                        default=r"D:\RT-DETR-main\rtdetr_pytorch\tools\output\rtdetr_r50vd_6x_coco\checkpoint0071.pth")
    parser.add_argument('--wavelet-config', type=str,
                        default=r"D:\RT-DETR-main\rtdetr_pytorch\configs\rtdetr\rtdetr_r50xb_6x_coco.yml")
    parser.add_argument('--wavelet-weight', type=str,
                        default=r"D:\RT-DETR-main\rtdetr_pytorch\tools\output\xb\checkpoint0124.pth")
    parser.add_argument('--img', type=str, default=r"C:\Users\dell\Downloads\visdrone\test\0000063_08000_d_0000009.jpg")
    parser.add_argument('--save-dir', type=str, default='./wavelet_comparison-improved-5', help='结果保存目录')
    parser.add_argument('--device', type=str, default='cpu', help='cpu或cuda')
    args = parser.parse_args()

    main_compare(
        orig_config=args.orig_config,
        orig_weight=args.orig_weight,
        wavelet_config=args.wavelet_config,
        wavelet_weight=args.wavelet_weight,
        img_path=args.img,
        save_dir=args.save_dir,
        device=args.device
    )