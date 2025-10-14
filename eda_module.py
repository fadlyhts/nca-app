import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class EDAGenerator:
    def __init__(self, df: pd.DataFrame, session_id: str, plots_dir: str = "plots"):
        self.df = df
        self.session_id = session_id
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        self.generated_plots = {}
        
    def get_dataset_info(self):
        memory_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        return {
            "n_rows": len(self.df),
            "n_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "memory_usage": f"{memory_mb:.2f} MB"
        }
    
    def get_summary_statistics(self):
        return self.df.describe().to_dict()
    
    def get_missing_values(self):
        missing = []
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                missing.append({
                    "column": col,
                    "missing_count": int(missing_count),
                    "missing_percentage": float(missing_count / len(self.df) * 100)
                })
        return missing
    
    def get_column_types(self):
        numerical = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numerical, categorical
    
    def generate_correlation_heatmap(self):
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return None
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = self.plots_dir / f"{self.session_id}_correlation.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.generated_plots['correlation_heatmap'] = str(plot_path)
        return str(plot_path)
    
    def generate_distribution_plots(self, max_cols=10):
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            return None
        
        cols_to_plot = numerical_cols[:max_cols]
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(cols_to_plot):
            axes[idx].hist(self.df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        for idx in range(len(cols_to_plot), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{self.session_id}_distributions.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.generated_plots['distributions'] = str(plot_path)
        return str(plot_path)
    
    def generate_boxplots(self, max_cols=10):
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            return None
        
        cols_to_plot = numerical_cols[:max_cols]
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(cols_to_plot):
            axes[idx].boxplot(self.df[col].dropna(), vert=True)
            axes[idx].set_title(f'Boxplot of {col}', fontweight='bold')
            axes[idx].set_ylabel(col)
            axes[idx].grid(True, alpha=0.3)
        
        for idx in range(len(cols_to_plot), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{self.session_id}_boxplots.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.generated_plots['boxplots'] = str(plot_path)
        return str(plot_path)
    
    def generate_categorical_plots(self, max_cols=6):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not categorical_cols:
            return None
        
        cols_to_plot = [col for col in categorical_cols if self.df[col].nunique() < 20][:max_cols]
        if not cols_to_plot:
            return None
        
        n_cols = min(2, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(cols_to_plot):
            value_counts = self.df[col].value_counts().head(10)
            axes[idx].bar(range(len(value_counts)), value_counts.values, color='coral', edgecolor='black')
            axes[idx].set_xticks(range(len(value_counts)))
            axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[idx].set_title(f'Count Plot of {col}', fontweight='bold')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        for idx in range(len(cols_to_plot), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{self.session_id}_categorical.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.generated_plots['categorical_plots'] = str(plot_path)
        return str(plot_path)
    
    def generate_all_plots(self):
        self.generate_correlation_heatmap()
        self.generate_distribution_plots()
        self.generate_boxplots()
        self.generate_categorical_plots()
        return self.generated_plots
    
    def get_complete_eda(self):
        dataset_info = self.get_dataset_info()
        summary_stats = self.get_summary_statistics()
        missing_values = self.get_missing_values()
        numerical_cols, categorical_cols = self.get_column_types()
        plots = self.generate_all_plots()
        
        return {
            "session_id": self.session_id,
            "dataset_info": dataset_info,
            "summary_statistics": summary_stats,
            "missing_values": missing_values,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "plots": plots,
            "message": "EDA generated successfully"
        }
