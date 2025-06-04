import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Read the Excel files
articles = pd.read_excel("data/articles.xlsx")
materials = pd.read_excel("data/materials.xlsx")
production_orders = pd.read_excel("data/production_orders.xlsx")
work_orders = pd.read_excel("data/work_orders.xlsx")

# Print column names for debugging
print("Materials columns:", materials.columns.tolist())
print("Production orders columns:", production_orders.columns.tolist())

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. Production Analysis
def analyze_production_metrics():
    try:
        # Production quantity distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=production_orders, x='productionquantity', bins=30)
        plt.title('Distribution of Production Quantities')
        plt.xlabel('Production Quantity')
        plt.ylabel('Count')
        plt.savefig('production_quantity_distribution.png')
        plt.close()

        # Price analysis
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=production_orders, y='price')
        plt.title('Price Distribution')
        plt.ylabel('Price')
        plt.savefig('price_distribution.png')
        plt.close()
    except Exception as e:
        print(f"Error in production metrics analysis: {str(e)}")

# 2. Material Analysis
def analyze_material_metrics():
    try:
        # Stock quantity analysis
        plt.figure(figsize=(12, 6))
        sns.histplot(data=materials, x='stockquantity', bins=30)
        plt.title('Distribution of Stock Quantities')
        plt.xlabel('Stock Quantity')
        plt.ylabel('Count')
        plt.savefig('stock_quantity_distribution.png')
        plt.close()

        # Material types frequency (using materialname instead of material)
        material_counts = materials['materialname'].value_counts().head(10)
        plt.figure(figsize=(12, 6))
        material_counts.plot(kind='bar')
        plt.title('Top 10 Most Common Materials')
        plt.xlabel('Material Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('material_types_frequency.png')
        plt.close()
    except Exception as e:
        print(f"Error in material metrics analysis: {str(e)}")

# 3. Time Series Analysis
def analyze_time_series():
    try:
        # Convert date columns to datetime
        production_orders['deliverydate'] = pd.to_datetime(production_orders['deliverydate'])
        
        # Orders over time
        orders_over_time = production_orders.groupby(production_orders['deliverydate'].dt.date).size()
        plt.figure(figsize=(15, 6))
        orders_over_time.plot(kind='line')
        plt.title('Production Orders Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Orders')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('orders_over_time.png')
        plt.close()

        # Average production quantity over time
        avg_quantity = production_orders.groupby(production_orders['deliverydate'].dt.date)['productionquantity'].mean()
        plt.figure(figsize=(15, 6))
        avg_quantity.plot(kind='line')
        plt.title('Average Production Quantity Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Quantity')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('avg_quantity_over_time.png')
        plt.close()
    except Exception as e:
        print(f"Error in time series analysis: {str(e)}")

# 4. Correlation Analysis
def analyze_correlations():
    try:
        # Select numeric columns for correlation
        numeric_cols = ['productionquantity', 'price']
        corr_matrix = production_orders[numeric_cols].corr()
        
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation between Production Quantity and Price')
        plt.tight_layout()
        plt.savefig('quantity_price_correlation.png')
        plt.close()
    except Exception as e:
        print(f"Error in correlation analysis: {str(e)}")

# 5. Material Usage Analysis
def analyze_material_usage():
    try:
        # Merge production orders with materials
        merged_data = pd.merge(production_orders, materials, on='orderid', how='left')
        
        # Material usage by quantity (using materialname instead of material)
        material_usage = merged_data.groupby('materialname')['productionquantity'].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        material_usage.plot(kind='bar')
        plt.title('Top 10 Materials by Production Quantity')
        plt.xlabel('Material')
        plt.ylabel('Total Production Quantity')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('material_usage.png')
        plt.close()
    except Exception as e:
        print(f"Error in material usage analysis: {str(e)}")

# Data Quality Analysis Function
def analyze_data_quality():
    try:
        print("\n=== Data Quality Analysis Report ===\n")
        
        # 1. Missing Value Analysis
        print("1. Missing Value Analysis:")
        print("-" * 50)
        
        for df_name, df in [("Articles", articles), 
                          ("Materials", materials), 
                          ("Production Orders", production_orders),
                          ("Work Orders", work_orders)]:
            print(f"\n{df_name} Missing Values:")
            missing_data = df.isnull().sum()
            missing_percentage = (missing_data / len(df)) * 100
            missing_info = pd.DataFrame({
                'Missing Values': missing_data,
                'Percentage': missing_percentage
            })
            print(missing_info[missing_info['Missing Values'] > 0].sort_values('Percentage', ascending=False))
        
        # 2. Duplicate Analysis
        print("\n2. Duplicate Analysis:")
        print("-" * 50)
        
        for df_name, df in [("Articles", articles), 
                          ("Materials", materials), 
                          ("Production Orders", production_orders),
                          ("Work Orders", work_orders)]:
            duplicates = df.duplicated().sum()
            print(f"\n{df_name} Duplicates: {duplicates} rows")
            
            # Check for duplicates in key columns
            if 'orderid' in df.columns:
                order_duplicates = df.duplicated(subset=['orderid']).sum()
                print(f"Duplicates in orderid: {order_duplicates}")
            if 'workorderid' in df.columns:
                work_duplicates = df.duplicated(subset=['workorderid']).sum()
                print(f"Duplicates in workorderid: {work_duplicates}")
        
        # 3. Date Format Analysis
        print("\n3. Date Format Analysis:")
        print("-" * 50)
        
        date_columns = {
            "Production Orders": ['deliverydate', 'createdate', 'reportingtime'],
            "Work Orders": ['createdate', 'reportingtime']
        }
        
        for df_name, date_cols in date_columns.items():
            df = production_orders if df_name == "Production Orders" else work_orders
            print(f"\n{df_name} Date Format Analysis:")
            for col in date_cols:
                if col in df.columns:
                    # Check if column contains datetime values
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        print(f"{col}: Consistent datetime format")
                    else:
                        print(f"{col}: Inconsistent format - needs conversion")
        
        # 4. Key Column Analysis
        print("\n4. Key Column Analysis:")
        print("-" * 50)
        
        # Check for missing values in key columns
        key_columns = {
            "Production Orders": ['orderid', 'articlenumber', 'drawno'],
            "Materials": ['materialid', 'materialname'],
            "Work Orders": ['workorderid', 'orderid']
        }
        
        for df_name, key_cols in key_columns.items():
            df = production_orders if df_name == "Production Orders" else (materials if df_name == "Materials" else work_orders)
            print(f"\n{df_name} Key Columns:")
            for col in key_cols:
                if col in df.columns:
                    missing = df[col].isnull().sum()
                    percentage = (missing / len(df)) * 100
                    print(f"{col}: {missing} missing values ({percentage:.2f}%)")
        
        # 5. Data Type Analysis
        print("\n5. Data Type Analysis:")
        print("-" * 50)
        
        for df_name, df in [("Articles", articles), 
                          ("Materials", materials), 
                          ("Production Orders", production_orders),
                          ("Work Orders", work_orders)]:
            print(f"\n{df_name} Data Types:")
            print(df.dtypes)
            
    except Exception as e:
        print(f"Error in data quality analysis: {str(e)}")

def analyze_descriptive_statistics():
    try:
        print("\n=== Descriptive Statistics and Distribution Analysis ===\n")
        
        # 1. Basic Summary Statistics
        print("1. Basic Summary Statistics:")
        print("-" * 50)
        
        # Define ID columns to exclude from numeric analysis
        id_columns = ['orderid', 'workorderid', 'materialid', 'articleid']
        
        for df_name, df in [("Articles", articles), 
                          ("Materials", materials), 
                          ("Production Orders", production_orders),
                          ("Work Orders", work_orders)]:
            print(f"\n{df_name} Summary Statistics:")
            # Get numeric columns excluding ID columns
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if not any(id_col in col.lower() for id_col in id_columns)]
            
            if len(numeric_cols) > 0:
                summary = df[numeric_cols].describe()
                print(summary)
                
                # Create box plots for numeric columns
                plt.figure(figsize=(15, 6))
                df[numeric_cols].boxplot()
                plt.title(f'{df_name} - Numeric Columns Distribution')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{df_name.lower().replace(" ", "_")}_numeric_distribution.png')
                plt.close()
        
        # 2. Categorical and ID Analysis
        print("\n2. Categorical and ID Analysis:")
        print("-" * 50)
        
        categorical_columns = {
            "Materials": ['materialname', 'materialid'],
            "Production Orders": ['articlenumber', 'drawno', 'orderid'],
            "Work Orders": ['workorderid', 'orderid']
        }
        
        for df_name, cat_cols in categorical_columns.items():
            df = materials if df_name == "Materials" else (production_orders if df_name == "Production Orders" else work_orders)
            print(f"\n{df_name} Categorical Frequencies:")
            for col in cat_cols:
                if col in df.columns:
                    value_counts = df[col].value_counts().head(10)
                    print(f"\nTop 10 {col}:")
                    print(value_counts)
                    
                    # Create bar plot for top categories
                    plt.figure(figsize=(12, 6))
                    value_counts.plot(kind='bar')
                    plt.title(f'Top 10 {col} in {df_name}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f'{df_name.lower().replace(" ", "_")}_{col}_frequency.png')
                    plt.close()
                    
                    # For ID columns, also show basic statistics
                    if any(id_col in col.lower() for id_col in id_columns):
                        print(f"\n{col} Statistics:")
                        print(f"Total unique values: {df[col].nunique()}")
                        print(f"Most frequent value: {df[col].mode().iloc[0]}")
                        print(f"Least frequent value: {df[col].value_counts().index[-1]}")
        
        # 3. Time Series Analysis
        print("\n3. Time Series Analysis:")
        print("-" * 50)
        
        # Convert date columns to datetime
        if 'deliverydate' in production_orders.columns:
            production_orders['deliverydate'] = pd.to_datetime(production_orders['deliverydate'])
            
            # Monthly production order trends
            monthly_orders = production_orders.groupby(production_orders['deliverydate'].dt.to_period('M')).size()
            plt.figure(figsize=(15, 6))
            monthly_orders.plot(kind='line', marker='o')
            plt.title('Monthly Production Order Trends')
            plt.xlabel('Month')
            plt.ylabel('Number of Orders')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('monthly_production_trends.png')
            plt.close()
            
            print("\nMonthly Production Order Statistics:")
            print(monthly_orders.describe())
        
        # Analyze reporting time patterns
        if 'reportingtime' in work_orders.columns:
            work_orders['reportingtime'] = pd.to_datetime(work_orders['reportingtime'])
            
            # Hourly distribution of reporting times
            hourly_distribution = work_orders['reportingtime'].dt.hour.value_counts().sort_index()
            plt.figure(figsize=(15, 6))
            hourly_distribution.plot(kind='bar')
            plt.title('Distribution of Reporting Times by Hour')
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Reports')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('reporting_time_distribution.png')
            plt.close()
            
            print("\nReporting Time Distribution by Hour:")
            print(hourly_distribution)
            
            # Weekly patterns
            weekly_distribution = work_orders['reportingtime'].dt.dayofweek.value_counts().sort_index()
            plt.figure(figsize=(12, 6))
            weekly_distribution.plot(kind='bar')
            plt.title('Distribution of Reporting Times by Day of Week')
            plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
            plt.ylabel('Number of Reports')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('weekly_reporting_patterns.png')
            plt.close()
            
            print("\nReporting Time Distribution by Day of Week:")
            print(weekly_distribution)
        
    except Exception as e:
        print(f"Error in descriptive statistics analysis: {str(e)}")

def analyze_stock_and_material_usage():
    try:
        print("\n=== Stock and Material Usage Analysis ===\n")
        
        # 1. Critical Stock Level Analysis
        print("1. Critical Stock Level Analysis:")
        print("-" * 50)
        
        # Define critical stock threshold (örnek olarak 10 birim)
        CRITICAL_STOCK_THRESHOLD = 10
        
        # Analyze current stock levels
        low_stock_materials = materials[materials['stockquantity'] <= CRITICAL_STOCK_THRESHOLD]
        print(f"\nMaterials below critical threshold ({CRITICAL_STOCK_THRESHOLD}):")
        print(f"Total count: {len(low_stock_materials)}")
        
        if len(low_stock_materials) > 0:
            print("\nTop 10 materials with lowest stock:")
            print(low_stock_materials[['materialname', 'stockquantity']].sort_values('stockquantity').head(10))
            
            # Visualize low stock materials
            plt.figure(figsize=(12, 6))
            sns.barplot(data=low_stock_materials.head(10), x='stockquantity', y='materialname')
            plt.title('Top 10 Materials with Lowest Stock')
            plt.xlabel('Stock Quantity')
            plt.ylabel('Material Name')
            plt.tight_layout()
            plt.savefig('low_stock_materials.png')
            plt.close()
        
        # 2. Consumption Rate Analysis
        print("\n2. Consumption Rate Analysis:")
        print("-" * 50)
        
        # Merge production orders with materials to analyze consumption
        if 'orderid' in materials.columns and 'orderid' in production_orders.columns:
            # Calculate material consumption from production orders
            material_consumption = production_orders.groupby('material')['productionquantity'].sum().reset_index()
            material_consumption = material_consumption.merge(
                materials[['materialname', 'stockquantity']], 
                left_on='material', 
                right_on='materialname', 
                how='left'
            )
            
            # Calculate turnover rate (consumption / current stock)
            material_consumption['turnover_rate'] = material_consumption['productionquantity'] / material_consumption['stockquantity']
            material_consumption['turnover_rate'] = material_consumption['turnover_rate'].replace([np.inf, -np.inf], np.nan)
            
            print("\nTop 10 materials by consumption:")
            print(material_consumption.sort_values('productionquantity', ascending=False).head(10))
            
            print("\nTop 10 materials by turnover rate:")
            print(material_consumption.sort_values('turnover_rate', ascending=False).head(10))
            
            # Visualize consumption patterns
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=material_consumption, x='stockquantity', y='productionquantity')
            plt.title('Stock Quantity vs Production Quantity')
            plt.xlabel('Current Stock')
            plt.ylabel('Total Production Quantity')
            plt.tight_layout()
            plt.savefig('stock_vs_consumption.png')
            plt.close()
        
        # 3. Fragile Materials Analysis
        print("\n3. Fragile Materials Analysis:")
        print("-" * 50)
        
        if 'orderid' in materials.columns and 'orderid' in production_orders.columns:
            # Calculate days between first and last order for each material
            production_orders['deliverydate'] = pd.to_datetime(production_orders['deliverydate'])
            material_time_span = production_orders.groupby('material').agg({
                'deliverydate': ['min', 'max'],
                'productionquantity': 'sum'
            }).reset_index()
            
            material_time_span['days_span'] = (material_time_span[('deliverydate', 'max')] - 
                                             material_time_span[('deliverydate', 'min')]).dt.days
            
            # Calculate daily consumption rate
            material_time_span['daily_consumption'] = material_time_span[('productionquantity', 'sum')] / material_time_span['days_span']
            
            # Identify fragile materials (high consumption rate relative to stock)
            material_time_span = material_time_span.merge(
                materials[['materialname', 'stockquantity']], 
                left_on='material', 
                right_on='materialname', 
                how='left'
            )
            
            material_time_span['days_until_stockout'] = material_time_span['stockquantity'] / material_time_span['daily_consumption']
            
            print("\nMaterials at risk of stockout (less than 7 days):")
            fragile_materials = material_time_span[material_time_span['days_until_stockout'] < 7]
            print(fragile_materials[['materialname', 'stockquantity', 'daily_consumption', 'days_until_stockout']].sort_values('days_until_stockout'))
            
            print("\nOverstocked materials (more than 90 days of stock):")
            overstocked_materials = material_time_span[material_time_span['days_until_stockout'] > 90]
            print(overstocked_materials[['materialname', 'stockquantity', 'daily_consumption', 'days_until_stockout']].sort_values('days_until_stockout', ascending=False))
            
            # Visualize fragile materials
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=material_time_span, x='stockquantity', y='daily_consumption')
            plt.title('Stock Quantity vs Daily Consumption Rate')
            plt.xlabel('Current Stock')
            plt.ylabel('Daily Consumption Rate')
            plt.tight_layout()
            plt.savefig('fragile_materials_analysis.png')
            plt.close()
        
    except Exception as e:
        print(f"Error in stock and material usage analysis: {str(e)}")

# Run all analyses
print("Starting analysis...")
analyze_data_quality()
analyze_descriptive_statistics()
analyze_stock_and_material_usage()
print("Analysis complete! Check the generated PNG files for results.") 