import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict, Any
import logging

class Visualizer:
    def __init__(self, plots_dir: str = "plots"):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_line_plot(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "",
        filename: Optional[str] = None
    ) -> str:
        """Create a line plot."""
        try:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x=x_column, y=y_column)
            plt.title(title)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
            if filename:
                filepath = self.plots_dir / f"{filename}.png"
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            return ""
            
        except Exception as e:
            self.logger.error(f"Error creating line plot: {str(e)}")
            return ""
    
    def create_bar_plot(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "",
        filename: Optional[str] = None
    ) -> str:
        """Create a bar plot."""
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=data, x=x_column, y=y_column)
            plt.title(title)
            plt.xticks(rotation=45)
            
            if filename:
                filepath = self.plots_dir / f"{filename}.png"
                plt.savefig(filepath, bbox_inches='tight')
                plt.close()
                return str(filepath)
            return ""
            
        except Exception as e:
            self.logger.error(f"Error creating bar plot: {str(e)}")
            return ""
    
    def create_scatter_plot(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        color_column: Optional[str] = None,
        title: str = "",
        filename: Optional[str] = None
    ) -> str:
        """Create a scatter plot."""
        try:
            plt.figure(figsize=(10, 6))
            if color_column:
                sns.scatterplot(data=data, x=x_column, y=y_column, hue=color_column)
            else:
                sns.scatterplot(data=data, x=x_column, y=y_column)
            plt.title(title)
            
            if filename:
                filepath = self.plots_dir / f"{filename}.png"
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            return ""
            
        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {str(e)}")
            return ""
    
    def create_interactive_plot(
        self,
        data: pd.DataFrame,
        plot_type: str,
        x_column: str,
        y_column: str,
        color_column: Optional[str] = None,
        title: str = "",
        filename: Optional[str] = None
    ) -> Optional[str]:
        """Create an interactive plot using Plotly."""
        try:
            if plot_type == "scatter":
                fig = px.scatter(data, x=x_column, y=y_column, color=color_column, title=title)
            elif plot_type == "line":
                fig = px.line(data, x=x_column, y=y_column, color=color_column, title=title)
            elif plot_type == "bar":
                fig = px.bar(data, x=x_column, y=y_column, color=color_column, title=title)
            else:
                self.logger.error(f"Unsupported plot type: {plot_type}")
                return None
            
            if filename:
                filepath = self.plots_dir / f"{filename}.html"
                fig.write_html(str(filepath))
                return str(filepath)
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating interactive plot: {str(e)}")
            return None 