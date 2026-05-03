import os
import mysql.connector


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", "3306")),
    )


def save_or_update_user_predictions(
    user_id: int,
    weekly_prediction: float,
    monthly_prediction: float,
    yearly_prediction: float,
):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO user_predictions (
            user_id,
            weekly_prediction,
            monthly_prediction,
            yearly_prediction
        )
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            weekly_prediction = VALUES(weekly_prediction),
            monthly_prediction = VALUES(monthly_prediction),
            yearly_prediction = VALUES(yearly_prediction),
            updated_at = CURRENT_TIMESTAMP
    """

    cursor.execute(
        query,
        (
            user_id,
            weekly_prediction,
            monthly_prediction,
            yearly_prediction,
        ),
    )

    conn.commit()
    cursor.close()
    conn.close()


def get_user_predictions(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT
            user_id,
            weekly_prediction,
            monthly_prediction,
            yearly_prediction,
            updated_at
        FROM user_predictions
        WHERE user_id = %s
    """

    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return result