from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import requests

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

dag = DAG(
    '04_api_data_extraction',
    default_args=default_args,
    description='DAG to extract data from API and store in logs',
    schedule_interval=None,  # Set to None for manual triggering
    catchup=False,
)


def fetch_data_and_log(*args, **kwargs):
    api_url = "http://localhost:9000/users/get"

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx, 5xx)

        data = response.json()

        # Logging the result
        for item in data:
            print(item)

        # You can customize the logging method as per your requirements
        # For example, use Airflow log
        # kwargs['ti'].xcom_push(key='api_data', value=data)
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        raise


task_fetch_data = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_and_log,
    provide_context=True,
    dag=dag,
)


def post_data_and_log(*args, **kwargs):
    api_url = "http://localhost:9000/users/post"
    request_body = {
        'key1': 'value1',
        'key2': 'value2',
        # Add other key-value pairs as needed
    }

    try:
        response = requests.post(api_url, json=request_body)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx, 5xx)

        data = response.json()

        # Logging the result
        for item in data:
            print(item)

        # You can customize the logging method as per your requirements
        # For example, use Airflow log
        # kwargs['ti'].xcom_push(key='api_data', value=data)
    except Exception as e:
        print(f"Error making POST request to API: {e}")
        raise


task_post_data = PythonOperator(
    task_id='post_data',
    python_callable=post_data_and_log,
    provide_context=True,
    dag=dag,
)


task_fetch_data>task_post_data  # Set up task dependencies as needed

# Delete
# base_api_url = "http://localhost:9000/users/delete"
# user_id = 123  # Replace with the actual user ID or provide it dynamically
# api_url = f"{base_api_url}/{user_id}"