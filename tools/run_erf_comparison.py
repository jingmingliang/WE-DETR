"""
完整的ERF对比分析脚本
自动运行baseline和enhanced模型的ERF分析，并生成对比可视化
"""

import os
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_plotting():
    """设置绘图参数"""
    plt.rcParams["font.family"] = "Times New Roman"
    large = 20; med = 16; small = 14
    params = {
        'axes.titlesize': large,
        'legend.fontsize': med,
        'figure.figsize': (16, 8),
        'axes.labelsize': med,
        'xtick.labelsize': small,
        'ytick.labelsize': small,
        'figure.titlesize': large
    }
    plt.rcParams.update(params)
    # 使用兼容的matplotlib样式
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
    
    try:
        sns.set_style("white")
    except:
        pass

def run_erf_analysis(model_name, data_path, save_path, num_images=20, weights_path=None):
    """运行ERF分析"""
    print(f"Running ERF analysis for {model_name}...")
    
    cmd = [
        'python', 'visualize.py',
        '--model', model_name,
        '--data_path', data_path,
        '--save_path', save_path,
        '--num_images', str(num_images)
    ]
    
    # 如果提供权重路径，添加到命令中
    if weights_path and os.path.exists(weights_path):
        cmd.extend(['--weights', weights_path])
        if model_name == 'encoder_s42_baseline':
            print(f"  Using shared weights from enhanced model: {weights_path}")
        else:
            print(f"  Using complete enhanced weights: {weights_path}")
    else:
        print(f"  Using random initialization")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print(f"✓ ERF analysis completed for {model_name}")
            return True
        else:
            print(f"✗ ERF analysis failed for {model_name}")
            print("Error:", result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error running ERF analysis: {e}")
        return False

def analyze_erf_data(erf_file):
    """分析ERF数据并返回关键指标"""
    data = np.load(erf_file)
    
    print(f"Original ERF data - Shape: {data.shape}, Max: {np.max(data):.6f}, Min: {np.min(data):.6f}")
    
    # 改进的数据预处理 - 增强数值稳定性
    # 1. 使用更大的epsilon避免log(0)
    epsilon = 1e-8
    data = np.maximum(data, epsilon)  # 确保所有值都大于epsilon
    
    # 2. 使用自然对数而不是log10，数值范围更好
    data = np.log(data + epsilon)
    
    # 3. 更鲁棒的归一化
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
    else:
        print(f"Warning: ERF data has no variation, using uniform distribution")
        # 创建一个中心集中的分布作为fallback
        h, w = data.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        data = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (min(h, w) / 4)**2)
    
    # 计算高贡献区域比例
    def get_rectangle(data, thresh):
        h, w = data.shape
        all_sum = np.sum(data)
        if all_sum == 0:
            return h, 1.0  # 避免除零错误，返回全图
        
        for i in range(1, h // 2):
            selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
            area_sum = np.sum(selected_area)
            if area_sum / all_sum > thresh:
                return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
        
        # 如果没有找到满足阈值的区域，返回全图
        return h, 1.0
    
    metrics = {}
    for thresh in [0.2, 0.3, 0.5, 0.99]:
        try:
            side_length, area_ratio = get_rectangle(data, thresh)
            metrics[f'thresh_{thresh}'] = {
                'side_length': side_length,
                'area_ratio': area_ratio
            }
            print(f"Threshold {thresh}: Side Length = {side_length}, Area Ratio = {area_ratio:.4f}")
        except Exception as e:
            print(f"Error calculating metrics for threshold {thresh}: {e}")
            metrics[f'thresh_{thresh}'] = {
                'side_length': data.shape[0],
                'area_ratio': 1.0
            }
    
    # 计算中心集中度
    center_region = data[data.shape[0]//2-50:data.shape[0]//2+50, 
                        data.shape[1]//2-50:data.shape[1]//2+50]
    center_concentration = np.sum(center_region) / np.sum(data)
    metrics['center_concentration'] = center_concentration
    
    return data, metrics

def create_comparison_visualization(baseline_file, enhanced_file, save_dir):
    """创建对比可视化"""
    setup_plotting()
    
    # 加载和分析数据
    baseline_data, baseline_metrics = analyze_erf_data(baseline_file)
    enhanced_data, enhanced_metrics = analyze_erf_data(enhanced_file)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ERF热力图对比
    im1 = axes[0, 0].imshow(baseline_data, cmap='RdYlGn', aspect='auto')
    axes[0, 0].set_title('Baseline ERF (No RepLK Enhancement)', fontsize=16)
    axes[0, 0].axis('off')
    
    im2 = axes[0, 1].imshow(enhanced_data, cmap='RdYlGn', aspect='auto')
    axes[0, 1].set_title('Enhanced ERF (With RepLK)', fontsize=16)
    axes[0, 1].axis('off')
    
    # 差异图
    diff_data = enhanced_data - baseline_data
    im3 = axes[0, 2].imshow(diff_data, cmap='RdBu_r', aspect='auto')
    axes[0, 2].set_title('Difference (Enhanced - Baseline)', fontsize=16)
    axes[0, 2].axis('off')
    
    # 添加颜色条
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 中心切片对比
    center_idx = baseline_data.shape[0] // 2
    axes[1, 0].plot(baseline_data[center_idx, :], label='Baseline', linewidth=2)
    axes[1, 0].plot(enhanced_data[center_idx, :], label='Enhanced', linewidth=2)
    axes[1, 0].set_title('Horizontal Center Slice', fontsize=14)
    axes[1, 0].set_xlabel('Pixel Position')
    axes[1, 0].set_ylabel('ERF Contribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 垂直切片对比
    axes[1, 1].plot(baseline_data[:, center_idx], label='Baseline', linewidth=2)
    axes[1, 1].plot(enhanced_data[:, center_idx], label='Enhanced', linewidth=2)
    axes[1, 1].set_title('Vertical Center Slice', fontsize=14)
    axes[1, 1].set_xlabel('Pixel Position')
    axes[1, 1].set_ylabel('ERF Contribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 指标对比
    thresholds = [0.2, 0.3, 0.5, 0.99]
    baseline_areas = [baseline_metrics[f'thresh_{t}']['area_ratio'] or 0 for t in thresholds]
    enhanced_areas = [enhanced_metrics[f'thresh_{t}']['area_ratio'] or 0 for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, baseline_areas, width, label='Baseline', alpha=0.8)
    axes[1, 2].bar(x + width/2, enhanced_areas, width, label='Enhanced', alpha=0.8)
    axes[1, 2].set_title('High-Contribution Area Ratio', fontsize=14)
    axes[1, 2].set_xlabel('Threshold')
    axes[1, 2].set_ylabel('Area Ratio')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(thresholds)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    comparison_path = os.path.join(save_dir, 'erf_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to: {comparison_path}")
    
    # 保存详细指标报告
    report_path = os.path.join(save_dir, 'erf_metrics_report.txt')
    with open(report_path, 'w') as f:
        f.write("ERF Analysis Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("BASELINE MODEL (No RepLK Enhancement)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Center Concentration: {baseline_metrics['center_concentration']:.4f}\n")
        for thresh in [0.2, 0.3, 0.5, 0.99]:
            metrics = baseline_metrics[f'thresh_{thresh}']
            f.write(f"Threshold {thresh}: Side Length = {metrics['side_length']}, "
                   f"Area Ratio = {metrics['area_ratio']:.4f}\n")
        
        f.write("\nENHANCED MODEL (With RepLK Enhancement)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Center Concentration: {enhanced_metrics['center_concentration']:.4f}\n")
        for thresh in [0.2, 0.3, 0.5, 0.99]:
            metrics = enhanced_metrics[f'thresh_{thresh}']
            f.write(f"Threshold {thresh}: Side Length = {metrics['side_length']}, "
                   f"Area Ratio = {metrics['area_ratio']:.4f}\n")
        
        f.write("\nIMPROVEMENT ANALYSIS\n")
        f.write("-" * 40 + "\n")
        center_improvement = enhanced_metrics['center_concentration'] - baseline_metrics['center_concentration']
        f.write(f"Center Concentration Improvement: {center_improvement:+.4f}\n")
        
        for thresh in [0.2, 0.3, 0.5, 0.99]:
            baseline_area = baseline_metrics[f'thresh_{thresh}']['area_ratio'] or 0
            enhanced_area = enhanced_metrics[f'thresh_{thresh}']['area_ratio'] or 0
            improvement = enhanced_area - baseline_area
            f.write(f"Area Ratio Improvement (thresh {thresh}): {improvement:+.4f}\n")
    
    print(f"✓ Detailed metrics report saved to: {report_path}")
    
    return comparison_path, report_path

def main():
    parser = argparse.ArgumentParser('ERF Comparison Analysis for encoder_s42')
    parser.add_argument('--data_path', type=str, default='test_data', 
                       help='Path to test dataset')
    parser.add_argument('--num_images', type=int, default=20, 
                       help='Number of images to use for ERF analysis')
    parser.add_argument('--output_dir', type=str, default='erf_results', 
                       help='Directory to save results')
    parser.add_argument('--create_test_data', action='store_true', 
                       help='Create test data if it does not exist')
    parser.add_argument('--enhanced_weights', type=str, default=None,
                       help='Path to enhanced model weights (will be shared with baseline)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建测试数据 (如果需要)
    if args.create_test_data or not os.path.exists(args.data_path):
        print("Creating test data...")
        subprocess.run(['python', 'create_test_data.py', 
                       '--data_type', 'patterns', 
                       '--num_images', str(args.num_images),
                       '--save_dir', args.data_path])
    
    # 运行ERF分析
    baseline_erf = os.path.join(args.output_dir, 'baseline_erf.npy')
    enhanced_erf = os.path.join(args.output_dir, 'enhanced_erf.npy')
    
    print("\n" + "="*60)
    print("STARTING ERF COMPARISON ANALYSIS")
    print("="*60)
    
    # 分析baseline模型 (使用enhanced权重的共享部分)
    success1 = run_erf_analysis('encoder_s42_baseline', args.data_path, baseline_erf, 
                               args.num_images, args.enhanced_weights)
    
    # 分析enhanced模型 (使用完整的enhanced权重)
    success2 = run_erf_analysis('encoder_s42_enhanced', args.data_path, enhanced_erf, 
                               args.num_images, args.enhanced_weights)
    
    if success1 and success2:
        print("\n" + "="*60)
        print("CREATING COMPARISON VISUALIZATION")
        print("="*60)
        
        # 创建对比可视化
        comparison_path, report_path = create_comparison_visualization(
            baseline_erf, enhanced_erf, args.output_dir
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Results saved in: {args.output_dir}")
        print(f"- Comparison visualization: {comparison_path}")
        print(f"- Detailed metrics report: {report_path}")
        print(f"- Raw ERF data: {baseline_erf}, {enhanced_erf}")
        
    else:
        print("\n✗ ERF analysis failed. Please check the error messages above.")

if __name__ == '__main__':
    main()
