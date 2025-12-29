"""
RepLKNet ERF Analysis - Recreation of Figure 1 from 
"Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs"

This script creates publication-quality ERF visualizations and quantitative analysis
comparing baseline and RepLK-enhanced models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import ndimage
import pandas as pd

class ERFAnalyzer:
    def __init__(self, erf_dir='erf_results', colormap_style='green'):
        self.erf_dir = erf_dir
        self.colormap_style = colormap_style
        self.setup_style()
        self.create_colormap()
    
    def setup_style(self):
        """Setup publication-quality matplotlib style"""
        plt.style.use('default')
        
        # Publication parameters
        params = {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 1.8,
            'patch.linewidth': 0.8,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.spines.top': False,
            'axes.spines.right': False
        }
        plt.rcParams.update(params)
    
    def create_colormap(self):
        """Create paper-style colormap - 支持多种配色方案"""
        if self.colormap_style == 'green':
            # 论文原版绿色配色方案 (RepLKNet Figure 1) - 适配淡黄色背景
            colors = ['#ffffea', '#f0fff0', '#e6ffe6', '#ccffcc', 
                     '#b3ffb3', '#99ff99', '#66ff66', '#33ff33', 
                     '#00cc00', '#009900', '#006600', '#003300']
            self.cmap = LinearSegmentedColormap.from_list('paper_green_erf', colors, N=256)
        elif self.colormap_style == 'gray':
            # 灰度配色方案
            colors = ['#ffffff', '#f5f5f5', '#e0e0e0', '#c0c0c0', 
                     '#a0a0a0', '#808080', '#606060', '#404040', 
                     '#202020', '#000000']
            self.cmap = LinearSegmentedColormap.from_list('paper_gray_erf', colors, N=256)
        elif self.colormap_style == 'blue':
            # 蓝色配色方案
            colors = ['#ffffff', '#f0f8ff', '#e6f3ff', '#cce7ff', 
                     '#b3dbff', '#99ccff', '#66b3ff', '#3399ff', 
                     '#0080ff', '#0066cc', '#004d99', '#003366']
            self.cmap = LinearSegmentedColormap.from_list('paper_blue_erf', colors, N=256)
        else:
            # 默认使用绿色
            colors = ['#ffffff', '#f0fff0', '#e6ffe6', '#ccffcc', 
                     '#b3ffb3', '#99ff99', '#66ff66', '#33ff33', 
                     '#00cc00', '#009900', '#006600', '#003300']
            self.cmap = LinearSegmentedColormap.from_list('paper_green_erf', colors, N=256)
    
    def load_erf_data(self):
        """Load ERF data files"""
        baseline_path = os.path.join(self.erf_dir, 'baseline_erf.npy')
        enhanced_path = os.path.join(self.erf_dir, 'enhanced_erf.npy')
        
        data = {}
        if os.path.exists(baseline_path):
            data['Baseline'] = np.load(baseline_path)
            print(f"✓ Loaded baseline ERF: {data['Baseline'].shape}")
        
        if os.path.exists(enhanced_path):
            data['RepLK-Enhanced'] = np.load(enhanced_path)
            print(f"✓ Loaded enhanced ERF: {data['RepLK-Enhanced'].shape}")
        
        if not data:
            raise FileNotFoundError(f"No ERF files found in {self.erf_dir}")
        
        return data
    
    def preprocess_erf(self, erf_matrix, keep_original_size=True):
        """Preprocess ERF matrix - keep original size to show true kernel advantages"""
        # Keep original size to demonstrate real kernel size advantages
        # No resizing - analyze at actual feature map resolution
        
        # 检查输入数据的有效性
        if np.all(erf_matrix == 0) or np.all(np.isnan(erf_matrix)):
            print(f"⚠️  Warning: ERF matrix contains all zeros or NaN values")
            print(f"   Matrix shape: {erf_matrix.shape}")
            print(f"   Matrix stats: min={np.nanmin(erf_matrix):.6f}, max={np.nanmax(erf_matrix):.6f}")
            # 创建一个最小的有效ERF矩阵
            h, w = erf_matrix.shape
            center_y, center_x = h // 2, w // 2
            erf_matrix = np.zeros_like(erf_matrix)
            erf_matrix[center_y, center_x] = 1.0
        
        # Remove negative contributions (as mentioned in paper)
        erf_matrix = np.maximum(erf_matrix, 0)
        
        # 检查是否所有值都相同
        if np.max(erf_matrix) == np.min(erf_matrix):
            print(f"⚠️  Warning: ERF matrix has constant values (min=max={np.max(erf_matrix):.6f})")
            if np.max(erf_matrix) == 0:
                # 如果全为0，创建一个中心点
                h, w = erf_matrix.shape
                center_y, center_x = h // 2, w // 2
                erf_matrix[center_y, center_x] = 1.0
        
        # Apply logarithmic scaling for better visualization
        erf_matrix = np.log(erf_matrix + 1e-8)
        
        # Normalize to [0, 1] for each model independently with numerical stability
        erf_min, erf_max = erf_matrix.min(), erf_matrix.max()
        if erf_max - erf_min > 1e-10:  # 避免除零
            erf_matrix = (erf_matrix - erf_min) / (erf_max - erf_min)
        else:
            print(f"⚠️  Warning: ERF range too small (max-min={erf_max-erf_min:.2e}), using uniform distribution")
            erf_matrix = np.ones_like(erf_matrix) * 0.5
        
        return erf_matrix
    
    def calculate_quantitative_metrics(self, erf_matrix):
        """Calculate quantitative ERF metrics as in Table 10 of the paper"""
        h, w = erf_matrix.shape
        center_y, center_x = h // 2, w // 2
        
        # Thresholds for analysis (paper uses 20%, 50%, 80%, 99%)
        thresholds = [0.20, 0.50, 0.80, 0.99]
        metrics = {}
        
        for threshold in thresholds:
            # Find pixels above threshold
            mask = erf_matrix >= threshold
            
            if np.any(mask):
                # Find bounding rectangle
                rows, cols = np.where(mask)
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()
                
                # Calculate metrics
                bbox_height = max_row - min_row + 1
                bbox_width = max_col - min_col + 1
                area_ratio = (bbox_height * bbox_width) / (h * w)
                
                # Calculate effective spread
                center_distances = np.sqrt((rows - center_y)**2 + (cols - center_x)**2)
                mean_distance = np.mean(center_distances)
                max_distance = np.max(center_distances)
                
                metrics[f'{int(threshold*100)}%'] = {
                    'area_ratio': area_ratio * 100,
                    'bbox_size': f'{bbox_height}×{bbox_width}',
                    'mean_distance': mean_distance,
                    'max_distance': max_distance
                }
            else:
                metrics[f'{int(threshold*100)}%'] = {
                    'area_ratio': 0.0,
                    'bbox_size': '0×0',
                    'mean_distance': 0.0,
                    'max_distance': 0.0
                }
        
        return metrics
    
    def create_figure1_recreation(self, erf_data, save_path='figure1_recreation.png'):
        """Recreate Figure 1 from the paper with precise layout control"""
        num_plots = len(erf_data)
        fig = plt.figure(figsize=(7*num_plots, 8))
        
        processed_data = {}
        
        # 手动定位布局参数 (无主标题，无间隙)
        colorbar_height = 0.04   # 颜色条高度
        erf_bottom = 0.05        # ERF图底部边距
        erf_top_margin = 0.05    # ERF图顶部边距
        
        # 计算各部分的垂直位置 - 颜色条直接紧贴ERF图
        erf_top = 1 - erf_top_margin - colorbar_height
        erf_height = erf_top - erf_bottom
        colorbar_bottom = erf_top  # 颜色条底部就是ERF图顶部，无间隙
        
        for idx, (model_name, erf_matrix) in enumerate(erf_data.items()):
            # Preprocess ERF data
            processed_erf = self.preprocess_erf(erf_matrix)
            processed_data[model_name] = processed_erf
            
            # 计算水平位置
            plot_width = 0.8 / num_plots  # 总宽度的80%用于图形
            plot_left = 0.1 + idx * (0.8 / num_plots)  # 左边距10%
            
            # 创建ERF图 - 手动定位
            ax = fig.add_axes([plot_left, erf_bottom, plot_width * 0.9, erf_height])
            ax.set_facecolor('#ffffea')
            im = ax.imshow(processed_erf, cmap=self.cmap, aspect='equal')
            
            # Add center point marker (red cross as in paper) - 可选显示
            center_y, center_x = processed_erf.shape[0] // 2, processed_erf.shape[1] // 2
            # ax.plot(center_x, center_y, 'r+', markersize=12, markeredgewidth=3)  # 注释掉以隐藏十字
            
            ax.axis('off')
            
            # 创建颜色条 - 手动定位，紧贴ERF图上方
            cbar_ax = fig.add_axes([plot_left, colorbar_bottom, plot_width * 0.9, colorbar_height])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Contribution Score (Normalized)', fontsize=9)
            
            # 设置刻度和标签在颜色条上方
            cbar_ax.xaxis.set_ticks_position('top')
            cbar_ax.xaxis.set_label_position('top')
            
            # 调整颜色条的外观 - 刻度在上方
            cbar.outline.set_linewidth(0.5)
            cbar.ax.tick_params(
                top=True,           # 显示顶部刻度
                bottom=False,       # 隐藏底部刻度
                labeltop=True,      # 显示顶部标签
                labelbottom=False,  # 隐藏底部标签
                labelsize=8, 
                width=0.5, 
                length=3,
                pad=2               # 标签与刻度的距离
            )
        
        # 不添加主标题
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"✓ Figure 1 recreation saved to: {save_path}")
        return fig, processed_data
    
    def create_quantitative_table(self, erf_data, processed_data, save_path='table10_recreation.png'):
        """Recreate Table 10 quantitative analysis"""
        # Calculate metrics for all models
        all_metrics = {}
        for model_name, processed_erf in processed_data.items():
            all_metrics[model_name] = self.calculate_quantitative_metrics(processed_erf)
        
        # Create comparison table
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Prepare data for table
        thresholds = ['20%', '50%', '80%', '99%']
        table_data = []
        
        for model_name, metrics in all_metrics.items():
            row = [model_name]
            for threshold in thresholds:
                area_ratio = metrics[threshold]['area_ratio']
                bbox_size = metrics[threshold]['bbox_size']
                row.append(f'{area_ratio:.3f}%\n({bbox_size})')
            table_data.append(row)
        
        # Create table
        headers = ['Model'] + [f'Area Ratio @ {t}' for t in thresholds]
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Header styling
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Row styling
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f8f8')
        
        ax.set_title('Quantitative ERF Analysis (Original Resolution)\n' +
                    'Area ratio at actual feature map size - showing true kernel advantages',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"✓ Table 10 recreation saved to: {save_path}")
        return fig
    
    def create_radial_analysis(self, processed_data, save_path='radial_analysis.png'):
        """Create radial profile analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Radial profile plot
        for idx, (model_name, erf_matrix) in enumerate(processed_data.items()):
            h, w = erf_matrix.shape
            center_y, center_x = h // 2, w // 2
            
            # Calculate radial profile
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            max_radius = int(min(center_x, center_y, w - center_x, h - center_y))
            radii = np.arange(0, max_radius, 3)
            radial_profile = []
            
            for r in radii:
                mask = (distances >= r) & (distances < r + 3)
                if np.any(mask):
                    radial_profile.append(np.mean(erf_matrix[mask]))
                else:
                    radial_profile.append(0)
            
            ax1.plot(radii, radial_profile, label=model_name, 
                    color=colors[idx], linewidth=2.5, marker='o', markersize=4)
        
        ax1.set_xlabel('Distance from Center (pixels)')
        ax1.set_ylabel('Average Contribution Score')
        ax1.set_title('Radial ERF Profile', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative contribution plot
        for idx, (model_name, erf_matrix) in enumerate(processed_data.items()):
            h, w = erf_matrix.shape
            center_y, center_x = h // 2, w // 2
            
            # Calculate cumulative contribution
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            max_radius = int(min(center_x, center_y, w - center_x, h - center_y))
            radii = np.arange(0, max_radius, 5)
            cumulative_contrib = []
            
            for r in radii:
                mask = distances <= r
                cumulative_contrib.append(np.sum(erf_matrix[mask]) / np.sum(erf_matrix))
            
            ax2.plot(radii, cumulative_contrib, label=model_name,
                    color=colors[idx], linewidth=2.5, marker='s', markersize=4)
        
        ax2.set_xlabel('Radius from Center (pixels)')
        ax2.set_ylabel('Cumulative Contribution Ratio')
        ax2.set_title('Cumulative ERF Analysis', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"✓ Radial analysis saved to: {save_path}")
        return fig
    
    def run_complete_analysis(self):
        """Run complete ERF analysis"""
        print("🔍 Starting RepLKNet ERF Analysis...")
        print("=" * 50)
        
        # Load data
        erf_data = self.load_erf_data()
        
        # Create Figure 1 recreation
        fig1, processed_data = self.create_figure1_recreation(erf_data)
        
        # Create quantitative table
        fig2 = self.create_quantitative_table(erf_data, processed_data)
        
        # Create radial analysis
        fig3 = self.create_radial_analysis(processed_data)
        
        # Print summary metrics
        print("\n📊 Summary Metrics:")
        print("-" * 30)
        
        for model_name, processed_erf in processed_data.items():
            metrics = self.calculate_quantitative_metrics(processed_erf)
            print(f"\n{model_name}:")
            for threshold, data in metrics.items():
                print(f"  {threshold} threshold: {data['area_ratio']:.1f}% area ({data['bbox_size']})")
        
        print(f"\n✅ Complete analysis finished!")
        print(f"📁 All files saved in current directory")
        
        return fig1, fig2, fig3

def main():
    """Main function with colormap selection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RepLKNet ERF Analysis with Custom Colormaps')
    parser.add_argument('--colormap', type=str, default='green', 
                       choices=['green', 'gray', 'blue'],
                       help='Colormap style: green (paper original), gray, or blue')
    parser.add_argument('--erf_dir', type=str, default='erf_results',
                       help='Directory containing ERF data files')
    
    args = parser.parse_args()
    
    print(f"🎨 Using colormap: {args.colormap}")
    analyzer = ERFAnalyzer(erf_dir=args.erf_dir, colormap_style=args.colormap)
    analyzer.run_complete_analysis()

if __name__ == '__main__':
    main()
