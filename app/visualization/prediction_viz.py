"""
Visualization components for forecast results.

This module provides visualization utilities for forecast data,
creating interactive charts and plots.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64


class ForecastVisualizer:
    """Creates visualizations for forecast data."""
    
    def __init__(self, color_scheme=None):
        """Initialize the ForecastVisualizer.

        Args:
            color_scheme: Optional custom color scheme
        """
        # Default color scheme
        self.color_scheme = color_scheme or {
            'historical': '#1f77b4',  # Blue
            'forecast': '#ff7f0e',    # Orange
            'optimized': '#2ca02c',   # Green
            'bounds': 'rgba(200, 200, 200, 0.2)'  # Light gray for confidence intervals
        }
    
    def create_time_series_plot(self, historical_data=None, forecast_data=None, date_col='date',
                               value_col='value', title=None, confidence_intervals=None):
        """Create a time series plot with historical and forecast data.

        Args:
            historical_data: Optional pandas.DataFrame with historical data
            forecast_data: pandas.DataFrame with forecast data
            date_col: Name of the date column
            value_col: Name of the value column
            title: Plot title
            confidence_intervals: Optional dict with confidence intervals

        Returns:
            plotly.graph_objects.Figure: Time series plot
        """
        fig = go.Figure()
        
        # Add historical data if provided
        if historical_data is not None and date_col in historical_data.columns and value_col in historical_data.columns:
            # Ensure date column is datetime
            hist_data = historical_data.copy()
            if not pd.api.types.is_datetime64_any_dtype(hist_data[date_col]):
                hist_data[date_col] = pd.to_datetime(hist_data[date_col])
            
            # Sort by date
            hist_data = hist_data.sort_values(date_col)
            
            # Add historical trace
            fig.add_trace(go.Scatter(
                x=hist_data[date_col], 
                y=hist_data[value_col],
                mode='lines+markers',
                name='Historical',
                line=dict(color=self.color_scheme['historical'], width=2, dash='solid'),
                marker=dict(size=6, color=self.color_scheme['historical'])
            ))
        
        # Add forecast data
        if forecast_data is not None and date_col in forecast_data.columns and value_col in forecast_data.columns:
            # Ensure date column is datetime
            fore_data = forecast_data.copy()
            if not pd.api.types.is_datetime64_any_dtype(fore_data[date_col]):
                fore_data[date_col] = pd.to_datetime(fore_data[date_col])
            
            # Sort by date
            fore_data = fore_data.sort_values(date_col)
            
            # Add forecast trace
            fig.add_trace(go.Scatter(
                x=fore_data[date_col], 
                y=fore_data[value_col],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=self.color_scheme['forecast'], width=2, dash='solid'),
                marker=dict(size=8, color=self.color_scheme['forecast'])
            ))
            
            # Add confidence intervals if provided
            if confidence_intervals is not None and 'lower' in confidence_intervals and 'upper' in confidence_intervals:
                # Add lower bound
                fig.add_trace(go.Scatter(
                    x=fore_data[date_col],
                    y=confidence_intervals['lower'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                # Add upper bound
                fig.add_trace(go.Scatter(
                    x=fore_data[date_col],
                    y=confidence_intervals['upper'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=self.color_scheme['bounds'],
                    name='Confidence Interval'
                ))
        
        # Add vertical line separating historical and forecast data if both are present
        if historical_data is not None and forecast_data is not None:
            # Find the last historical date
            last_hist_date = historical_data[date_col].max()
            
            # Add vertical line
            fig.add_vline(
                x=last_hist_date, 
                line_dash="dash", 
                line_width=1, 
                line_color="gray",
                annotation_text="Forecast Start", 
                annotation_position="top right"
            )
        
        # Set plot title and labels
        fig.update_layout(
            title=title or f"{value_col.capitalize()} Forecast",
            xaxis_title="Date",
            yaxis_title=value_col.capitalize(),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            hovermode="x unified"
        )
        
        return fig
    
    def create_comparative_plot(self, forecast_data, group_col, date_col='date', 
                               value_col='value', title=None, max_groups=10):
        """Create a comparative plot showing forecasts for different groups.

        Args:
            forecast_data: pandas.DataFrame with forecast data
            group_col: Column to group by
            date_col: Name of the date column
            value_col: Name of the value column
            title: Plot title
            max_groups: Maximum number of groups to show

        Returns:
            plotly.graph_objects.Figure: Comparative plot
        """
        if group_col not in forecast_data.columns:
            return None
        
        # Ensure date column is datetime
        data = forecast_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Get top groups by average value
        top_groups = data.groupby(group_col)[value_col].mean().nlargest(max_groups).index.tolist()
        
        # Filter to top groups
        data = data[data[group_col].isin(top_groups)]
        
        # Create line plot
        fig = px.line(
            data, 
            x=date_col, 
            y=value_col, 
            color=group_col,
            title=title or f"{value_col.capitalize()} Forecast by {group_col}",
            markers=True
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=value_col.capitalize(),
            legend_title=group_col.capitalize(),
            template="plotly_white",
            hovermode="x unified"
        )
        
        return fig
    
    def create_heatmap(self, forecast_data, x_col, y_col, value_col='value', 
                      title=None, colorscale='Viridis'):
        """Create a heatmap visualization.

        Args:
            forecast_data: pandas.DataFrame with forecast data
            x_col: Column for x-axis
            y_col: Column for y-axis
            value_col: Column for cell values
            title: Plot title
            colorscale: Colorscale for the heatmap

        Returns:
            plotly.graph_objects.Figure: Heatmap visualization
        """
        if x_col not in forecast_data.columns or y_col not in forecast_data.columns or value_col not in forecast_data.columns:
            return None
        
        # Aggregate data
        pivot_data = forecast_data.pivot_table(
            values=value_col,
            index=y_col,
            columns=x_col,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = px.imshow(
            pivot_data,
            labels=dict(x=x_col, y=y_col, color=value_col),
            title=title or f"{value_col.capitalize()} Heatmap by {x_col} and {y_col}",
            color_continuous_scale=colorscale
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=x_col.capitalize(),
            yaxis_title=y_col.capitalize(),
            coloraxis_colorbar=dict(
                title=value_col.capitalize()
            ),
            template="plotly_white"
        )
        
        return fig
    
    def create_bar_chart(self, forecast_data, x_col, value_col='value', 
                        title=None, color_col=None, orientation='v'):
        """Create a bar chart visualization.

        Args:
            forecast_data: pandas.DataFrame with forecast data
            x_col: Column for x-axis
            value_col: Column for bar heights
            title: Plot title
            color_col: Optional column for color coding
            orientation: 'v' for vertical, 'h' for horizontal

        Returns:
            plotly.graph_objects.Figure: Bar chart visualization
        """
        if x_col not in forecast_data.columns or value_col not in forecast_data.columns:
            return None
        
        # Create bar chart
        if orientation == 'v':
            fig = px.bar(
                forecast_data, 
                x=x_col, 
                y=value_col,
                color=color_col,
                title=title or f"{value_col.capitalize()} by {x_col}",
                barmode='group' if color_col else None
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_col.capitalize(),
                yaxis_title=value_col.capitalize(),
                template="plotly_white"
            )
        else:  # horizontal
            fig = px.bar(
                forecast_data, 
                y=x_col, 
                x=value_col,
                color=color_col,
                title=title or f"{value_col.capitalize()} by {x_col}",
                barmode='group' if color_col else None
            )
            
            # Update layout
            fig.update_layout(
                yaxis_title=x_col.capitalize(),
                xaxis_title=value_col.capitalize(),
                template="plotly_white"
            )
        
        return fig
    
    def create_scatter_plot(self, data, x_col, y_col, color_col=None, 
                          size_col=None, title=None, add_trendline=True):
        """Create a scatter plot visualization.

        Args:
            data: pandas.DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Optional column for color coding
            size_col: Optional column for point size
            title: Plot title
            add_trendline: Whether to add a trendline

        Returns:
            plotly.graph_objects.Figure: Scatter plot visualization
        """
        if x_col not in data.columns or y_col not in data.columns:
            return None
        
        # Create scatter plot
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col,
            size=size_col,
            title=title or f"{y_col.capitalize()} vs {x_col.capitalize()}",
            trendline='ols' if add_trendline else None,
            opacity=0.7
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=x_col.capitalize(),
            yaxis_title=y_col.capitalize(),
            template="plotly_white"
        )
        
        return fig
    
    def create_combined_view(self, price_data, leadtime_data, price_col='price', 
                           leadtime_col='lead_time', date_col='date', title=None):
        """Create a combined view with price and lead time on the same chart.

        Args:
            price_data: pandas.DataFrame with price data
            leadtime_data: pandas.DataFrame with lead time data
            price_col: Name of the price column
            leadtime_col: Name of the lead time column
            date_col: Name of the date column
            title: Plot title

        Returns:
            plotly.graph_objects.Figure: Combined visualization
        """
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price trace
        if price_data is not None and date_col in price_data.columns and price_col in price_data.columns:
            # Ensure date column is datetime
            price_df = price_data.copy()
            if not pd.api.types.is_datetime64_any_dtype(price_df[date_col]):
                price_df[date_col] = pd.to_datetime(price_df[date_col])
            
            # Sort by date
            price_df = price_df.sort_values(date_col)
            
            # Add price trace
            fig.add_trace(
                go.Scatter(
                    x=price_df[date_col], 
                    y=price_df[price_col],
                    name=price_col.capitalize(),
                    mode='lines+markers',
                    line=dict(color=self.color_scheme['historical'], width=2),
                    marker=dict(size=6, color=self.color_scheme['historical'])
                ),
                secondary_y=False
            )
        
        # Add lead time trace
        if leadtime_data is not None and date_col in leadtime_data.columns and leadtime_col in leadtime_data.columns:
            # Ensure date column is datetime
            leadtime_df = leadtime_data.copy()
            if not pd.api.types.is_datetime64_any_dtype(leadtime_df[date_col]):
                leadtime_df[date_col] = pd.to_datetime(leadtime_df[date_col])
            
            # Sort by date
            leadtime_df = leadtime_df.sort_values(date_col)
            
            # Add lead time trace
            fig.add_trace(
                go.Scatter(
                    x=leadtime_df[date_col], 
                    y=leadtime_df[leadtime_col],
                    name=leadtime_col.capitalize(),
                    mode='lines+markers',
                    line=dict(color=self.color_scheme['forecast'], width=2),
                    marker=dict(size=6, color=self.color_scheme['forecast'])
                ),
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title=title or "Price and Lead Time Trends",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            hovermode="x unified"
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text=price_col.capitalize(), secondary_y=False)
        fig.update_yaxes(title_text=leadtime_col.capitalize(), secondary_y=True)
        
        return fig
    
    def create_distribution_plot(self, data, value_col, group_col=None, title=None):
        """Create a distribution plot (histogram with box plot).

        Args:
            data: pandas.DataFrame with data
            value_col: Column containing values to plot
            group_col: Optional column for grouping
            title: Plot title

        Returns:
            plotly.graph_objects.Figure: Distribution plot
        """
        if value_col not in data.columns:
            return None
        
        if group_col is not None and group_col in data.columns:
            # Create box plot by group
            fig = px.box(
                data, 
                x=group_col if group_col else None, 
                y=value_col,
                color=group_col,
                title=title or f"{value_col.capitalize()} Distribution by {group_col}"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=group_col.capitalize() if group_col else None,
                yaxis_title=value_col.capitalize(),
                template="plotly_white"
            )
        else:
            # Create histogram with box plot
            fig = px.histogram(
                data, 
                x=value_col,
                marginal="box", 
                title=title or f"{value_col.capitalize()} Distribution",
                histnorm='probability density',
                opacity=0.7
            )
            
            # Add mean and median lines
            mean_val = data[value_col].mean()
            median_val = data[value_col].median()
            
            fig.add_vline(x=mean_val, line_dash="solid", line_width=2, line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
            
            fig.add_vline(x=median_val, line_dash="dash", line_width=2, line_color="green", 
                         annotation_text=f"Median: {median_val:.2f}", annotation_position="top left")
            
            # Update layout
            fig.update_layout(
                xaxis_title=value_col.capitalize(),
                yaxis_title="Density",
                template="plotly_white"
            )
        
        return fig
    
    def export_figure(self, fig, format_type='png', width=800, height=600, scale=2):
        """Export a figure to an image format.

        Args:
            fig: plotly.graph_objects.Figure to export
            format_type: Output format ('png', 'jpeg', 'svg', 'pdf')
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for the image

        Returns:
            bytes: Image data
        """
        # Create a buffer to save the image
        buf = io.BytesIO()
        
        # Convert format type to kaleido format
        kaleido_format = format_type.lower()
        if kaleido_format == 'jpeg':
            kaleido_format = 'jpg'
        
        # Write the figure to the buffer
        fig.write_image(buf, format=kaleido_format, width=width, height=height, scale=scale)
        
        # Reset buffer position
        buf.seek(0)
        
        return buf.getvalue()
    
    def get_download_link(self, fig, filename, format_type='png', width=800, height=600, scale=2):
        """Create an HTML download link for a figure.

        Args:
            fig: plotly.graph_objects.Figure to export
            filename: Output filename (without extension)
            format_type: Output format ('png', 'jpeg', 'svg', 'pdf')
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for the image

        Returns:
            str: HTML download link
        """
        # Export the figure
        img_bytes = self.export_figure(fig, format_type, width, height, scale)
        
        # Encode to base64
        b64 = base64.b64encode(img_bytes).decode()
        
        # Determine MIME type
        mime_map = {
            'png': 'image/png',
            'jpeg': 'image/jpeg',
            'jpg': 'image/jpeg',
            'svg': 'image/svg+xml',
            'pdf': 'application/pdf'
        }
        mime_type = mime_map.get(format_type.lower(), 'application/octet-stream')
        
        # Create HTML link
        href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}.{format_type.lower()}">Download {format_type.upper()}</a>'
        
        return href