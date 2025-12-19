"""
Optimization module for applying business constraints to forecasts.

This module uses Google OR-Tools to optimize forecasts based on
business constraints such as smoothness, bounds, and trends.
"""
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Base class for all optimizers."""
    
    @abstractmethod
    def optimize_forecast(self, forecast, historical_data=None, **kwargs):
        """Optimize the forecast based on constraints."""
        pass


class CPSatOptimizer(BaseOptimizer):
    """Optimizer using CP-SAT solver."""
    
    def __init__(self, scaling_factor=1000):
        """Initialize the CPSatOptimizer.

        Args:
            scaling_factor: Factor to scale values for integer optimization
        """
        self.scaling_factor = scaling_factor
    
    def optimize_forecast(self, forecast, historical_data=None, custom_constraints=None, **kwargs):
        """Optimize the forecast based on business constraints.

        Args:
            forecast: pandas.Series with forecast values
            historical_data: Optional historical data for context
            custom_constraints: Custom constraints to apply (dict)
            **kwargs: Additional parameters

        Returns:
            pandas.Series: Optimized forecast
        """
        if forecast is None or len(forecast) == 0:
            return forecast
        
        # Extract values and scale them for integer optimization
        forecast_values = forecast.values
        scaled_values = (forecast_values * self.scaling_factor).astype(int)
        
        # Create CP-SAT model
        model = cp_model.CpModel()
        
        # Create variables for each forecast period
        horizon = len(forecast_values)
        variables = []
        for i in range(horizon):
            # Determine reasonable bounds for the variable
            if custom_constraints and 'min_value' in custom_constraints:
                min_val = int(custom_constraints['min_value'] * self.scaling_factor)
            else:
                min_val = max(0, int(forecast_values.min() * self.scaling_factor * 0.5))
            
            if custom_constraints and 'max_value' in custom_constraints:
                max_val = int(custom_constraints['max_value'] * self.scaling_factor)
            else:
                max_val = int(forecast_values.max() * self.scaling_factor * 1.5)
            
            # Create integer variable with bounds
            variables.append(model.NewIntVar(min_val, max_val, f'forecast_{i}'))
        
        # Objective: Minimize deviation from original forecast
        # We use squared deviation to penalize larger deviations
        obj_terms = []
        for i in range(horizon):
            # Create auxiliary variable for absolute difference
            diff = model.NewIntVar(0, max(abs(min_val), abs(max_val)) * 2, f'diff_{i}')
            
            # |x - y| = z can be modeled as:
            # z >= x - y and z >= y - x
            model.Add(diff >= variables[i] - scaled_values[i])
            model.Add(diff >= scaled_values[i] - variables[i])
            
            # Add to objective
            obj_terms.append(diff)
        
        # Set objective to minimize sum of diffs
        model.Minimize(sum(obj_terms))
        
        # Apply smoothness constraints
        if custom_constraints and 'smoothness_factor' in custom_constraints:
            smoothness_factor = int(custom_constraints['smoothness_factor'] * self.scaling_factor)
            
            # Limit changes between consecutive periods
            for i in range(1, horizon):
                diff = model.NewIntVar(-smoothness_factor, smoothness_factor, f'change_{i}')
                model.Add(diff == variables[i] - variables[i-1])
        
        # Apply monotonicity constraints if requested
        if custom_constraints and 'monotonic' in custom_constraints:
            direction = custom_constraints['monotonic']
            
            if direction == 'increasing':
                # Each value must be >= the previous value
                for i in range(1, horizon):
                    model.Add(variables[i] >= variables[i-1])
                    
            elif direction == 'decreasing':
                # Each value must be <= the previous value
                for i in range(1, horizon):
                    model.Add(variables[i] <= variables[i-1])
        
        # Apply seasonality preservation if requested and possible
        if custom_constraints and 'preserve_seasonality' in custom_constraints and custom_constraints['preserve_seasonality']:
            # Check if we have enough historical data to determine seasonal pattern
            if historical_data is not None and len(historical_data) >= 12:
                try:
                    # Calculate seasonal pattern from historical data (simplified)
                    # This assumes monthly data with annual seasonality
                    monthly_avg = historical_data.groupby(historical_data.index.month).mean()
                    
                    # Calculate month-to-month differences
                    monthly_diffs = []
                    for i in range(1, 12):
                        monthly_diffs.append(monthly_avg.iloc[i] - monthly_avg.iloc[i-1])
                    monthly_diffs.append(monthly_avg.iloc[0] - monthly_avg.iloc[11])
                    
                    # Force similar month-to-month pattern in forecast
                    for i in range(1, horizon):
                        month_i = forecast.index[i].month - 1  # 0-based index
                        month_i_prev = forecast.index[i-1].month - 1  # 0-based index
                        
                        # If moving to the next month
                        if month_i != month_i_prev:
                            # Get expected difference direction
                            expected_diff = monthly_diffs[month_i_prev]
                            
                            # If positive difference expected
                            if expected_diff > 0:
                                # Force current value to be >= previous
                                model.Add(variables[i] >= variables[i-1])
                            # If negative difference expected
                            elif expected_diff < 0:
                                # Force current value to be <= previous
                                model.Add(variables[i] <= variables[i-1])
                except Exception as e:
                    print(f"Error applying seasonality preservation: {str(e)}")
        
        # Solve the model
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract solution
            optimized_values = [solver.Value(var) / self.scaling_factor for var in variables]
            
            # Create a new series with the optimized values
            optimized_forecast = pd.Series(
                data=optimized_values,
                index=forecast.index
            )
            
            return optimized_forecast
        else:
            print(f"Optimization failed: {solver.StatusName(status)}")
            return forecast  # Return original forecast if optimization fails


class LinearProgrammingOptimizer(BaseOptimizer):
    """Optimizer using Linear Programming."""
    
    def optimize_forecast(self, forecast, historical_data=None, custom_constraints=None, **kwargs):
        """Optimize the forecast based on business constraints.

        Args:
            forecast: pandas.Series with forecast values
            historical_data: Optional historical data for context
            custom_constraints: Custom constraints to apply (dict)
            **kwargs: Additional parameters

        Returns:
            pandas.Series: Optimized forecast
        """
        # This is a simplified version for when ortools is not available
        # or for scenarios where CP-SAT is too complex
        
        import scipy.optimize as opt
        
        if forecast is None or len(forecast) == 0:
            return forecast
        
        # Extract values
        forecast_values = forecast.values
        horizon = len(forecast_values)
        
        # Define bounds
        if custom_constraints and 'min_value' in custom_constraints:
            lb = custom_constraints['min_value'] * np.ones(horizon)
        else:
            lb = np.zeros(horizon)  # Default lower bound is 0
            
        if custom_constraints and 'max_value' in custom_constraints:
            ub = custom_constraints['max_value'] * np.ones(horizon)
        else:
            ub = forecast_values.max() * 1.5 * np.ones(horizon)
        
        bounds = opt.Bounds(lb, ub)
        
        # Define objective: minimize squared deviation from original forecast
        def objective(x):
            return np.sum((x - forecast_values)**2)
        
        # Define constraints
        constraints = []
        
        # Smoothness constraint
        if custom_constraints and 'smoothness_factor' in custom_constraints:
            smoothness_factor = custom_constraints['smoothness_factor']
            
            # Constraint matrix for consecutive differences
            A_smooth = np.zeros((2*(horizon-1), horizon))
            for i in range(horizon-1):
                # x[i+1] - x[i] <= smoothness_factor
                A_smooth[2*i, i] = -1
                A_smooth[2*i, i+1] = 1
                
                # x[i+1] - x[i] >= -smoothness_factor
                A_smooth[2*i+1, i] = 1
                A_smooth[2*i+1, i+1] = -1
                
            # Bounds for smoothness constraints
            lb_smooth = -np.inf * np.ones(2*(horizon-1))
            ub_smooth = np.concatenate([
                smoothness_factor * np.ones(horizon-1),
                smoothness_factor * np.ones(horizon-1)
            ])
            
            constraints.append(opt.LinearConstraint(A_smooth, lb_smooth, ub_smooth))
        
        # Monotonicity constraint
        if custom_constraints and 'monotonic' in custom_constraints:
            direction = custom_constraints['monotonic']
            
            # Constraint matrix for consecutive values
            A_mono = np.zeros((horizon-1, horizon))
            for i in range(horizon-1):
                A_mono[i, i] = -1
                A_mono[i, i+1] = 1
            
            if direction == 'increasing':
                # x[i+1] - x[i] >= 0
                lb_mono = np.zeros(horizon-1)
                ub_mono = np.inf * np.ones(horizon-1)
                
            elif direction == 'decreasing':
                # x[i+1] - x[i] <= 0
                lb_mono = -np.inf * np.ones(horizon-1)
                ub_mono = np.zeros(horizon-1)
            else:
                lb_mono = -np.inf * np.ones(horizon-1)
                ub_mono = np.inf * np.ones(horizon-1)
            
            constraints.append(opt.LinearConstraint(A_mono, lb_mono, ub_mono))
        
        # Solve optimization problem
        result = opt.minimize(
            objective,
            forecast_values,  # Initial guess is the original forecast
            method='SLSQP',
            bounds=bounds,
            constraints=constraints if constraints else None
        )
        
        if result.success:
            # Create a new series with the optimized values
            optimized_forecast = pd.Series(
                data=result.x,
                index=forecast.index
            )
            
            return optimized_forecast
        else:
            print(f"Optimization failed: {result.message}")
            return forecast  # Return original forecast if optimization fails


class OptimizerFactory:
    """Factory for creating optimizer instances."""
    
    @staticmethod
    def create_optimizer(optimizer_type, **kwargs):
        """Create an optimizer instance.

        Args:
            optimizer_type: Type of optimizer to create ('cp_sat' or 'lp')
            **kwargs: Parameters for the optimizer

        Returns:
            BaseOptimizer: An instance of the requested optimizer
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'cp_sat':
            return CPSatOptimizer(**kwargs)
        
        elif optimizer_type == 'lp':
            return LinearProgrammingOptimizer(**kwargs)
        
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class BusinessConstraintGenerator:
    """Generates business constraints for optimization."""
    
    @staticmethod
    def generate_constraints(data, column, constraint_type='auto'):
        """Generate business constraints based on historical data.

        Args:
            data: pandas.DataFrame containing historical data
            column: Column name to analyze
            constraint_type: Type of constraint to generate

        Returns:
            dict: Generated constraints
        """
        constraints = {}
        
        if constraint_type in ['auto', 'bounds']:
            # Generate min/max bounds
            if pd.api.types.is_numeric_dtype(data[column]):
                # Calculate bounds based on historical data
                min_val = data[column].min()
                max_val = data[column].max()
                
                # Add some margin
                margin = (max_val - min_val) * 0.1
                
                constraints['min_value'] = max(0, min_val - margin)
                constraints['max_value'] = max_val + margin
        
        if constraint_type in ['auto', 'smoothness']:
            # Generate smoothness factor
            if pd.api.types.is_numeric_dtype(data[column]):
                # Calculate typical month-to-month changes
                if 'date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['date']):
                    # Calculate average absolute change
                    data = data.sort_values('date')
                    changes = data[column].diff().abs().dropna()
                    
                    if not changes.empty:
                        # Use 95th percentile of changes for smoothness constraint
                        smoothness = changes.quantile(0.95)
                        constraints['smoothness_factor'] = smoothness
        
        if constraint_type in ['auto', 'trend']:
            # Detect if there's a trend
            if pd.api.types.is_numeric_dtype(data[column]) and len(data) >= 6:
                # Linear regression slope
                x = np.arange(len(data))
                y = data[column].values
                
                try:
                    slope, _ = np.polyfit(x, y, 1)
                    
                    # Check if the trend is significant
                    if abs(slope) > 0.01 * data[column].mean():
                        if slope > 0:
                            constraints['monotonic'] = 'increasing'
                        else:
                            constraints['monotonic'] = 'decreasing'
                except:
                    pass
        
        return constraints