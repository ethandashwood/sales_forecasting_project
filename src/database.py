import mysql.connector
import pandas as pd

# Database configuration (derived from your config.php)
DB_CONFIG = {
    "host": "localhost",
    "user": "ed906_fedbu1",
    "password": "[qxUi3v&o}Ci",
    "database": "ed906_fedb1"
}

def get_db_connection():
    """Establishes and returns a MySQL connection object."""
    return mysql.connector.connect(**DB_CONFIG)

def fetch_user_sales_df():
    """
    Fetches sales data from the MySQL 'weekly_sales' table and transforms
    it into the standardized format used by the training scripts.
    """
    try:
        conn = get_db_connection()
        query = "SELECT user_id, sale_week, mon, tue, wed, thu, fri, sat, sun FROM weekly_sales"
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()

        # Map day columns to row-based dates
        days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        day_offsets = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6}

        # Melt the wide format (days as columns) into long format (days as rows)
        df_long = df.melt(
            id_vars=['user_id', 'sale_week'], 
            value_vars=days, 
            var_name='day_name', 
            value_name='sales'
        )

        # Calculate the actual date for each entry
        df_long['date'] = pd.to_datetime(df_long['sale_week'])
        df_long['date'] = df_long.apply(
            lambda r: r['date'] + pd.Timedelta(days=day_offsets[r['day_name']]), 
            axis=1
        )

        # Standardize features to match the unified schema (unify_schema.py)
        df_long['business_id'] = 'user_' + df_long['user_id'].astype(str)
        df_long['store_id'] = df_long['user_id'].astype(str)
        df_long['business_type'] = 'custom_entry'
        df_long['store_type'] = 'user_logged'
        df_long['store_size_sqft'] = 2000
        df_long['region'] = 'local'
        df_long['product_category'] = 'general'
        df_long['is_holiday'] = 0

        return df_long[['date', 'business_id', 'sales', 'store_id', 'business_type', 'store_type', 'store_size_sqft', 'region', 'product_category', 'is_holiday']]
    
    except Exception as e:
        print(f"Error connecting to MySQL or fetching data: {e}")
        return pd.DataFrame()

def get_latest_user_features(user_id: int):
    """
    Retrieves the most recent sales data for a specific user and calculates
    lag features required for the prediction models.
    """
    try:
        conn = get_db_connection()
        # Fetch up to 8 weeks to satisfy lag_8_mean requirement
        query = """
            SELECT (mon + tue + wed + thu + fri + sat + sun) as weekly_total, sale_week 
            FROM weekly_sales 
            WHERE user_id = %s 
            ORDER BY sale_week DESC 
            LIMIT 8
        """
        df = pd.read_sql(query, conn, params=(user_id,))
        conn.close()

        if df.empty:
            return None

        # Convert to list (index 0 is most recent week)
        totals = df['weekly_total'].astype(float).tolist()
        
        # Determine the "next" period for which we are forecasting
        last_date = pd.to_datetime(df['sale_week'].iloc[0])
        next_period = last_date + pd.Timedelta(weeks=1)

        return {
            "business_type": "custom_entry",
            "store_type": "user_logged",
            "store_size_sqft": 2000.0,
            "region": "local",
            "product_category": "general",
            "year": int(next_period.year),
            "month": int(next_period.month),
            "week_of_year": int(next_period.isocalendar().week),
            "lag_1": totals[0] if len(totals) >= 1 else 0.0,
            "lag_2": totals[1] if len(totals) >= 2 else 0.0,
            "lag_4_mean": sum(totals[:4]) / len(totals[:4]) if len(totals) >= 1 else 0.0,
            "lag_8_mean": sum(totals[:8]) / len(totals[:8]) if len(totals) >= 1 else 0.0
        }
    except Exception as e:
        print(f"Error fetching user features: {e}")
        return None