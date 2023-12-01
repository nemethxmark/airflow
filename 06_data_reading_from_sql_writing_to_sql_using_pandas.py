from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import pandas as pd

# Define default_args dictionary to specify the default parameters of the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 11, 28),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 10,
    'retry_delay': timedelta(minutes=5),
    'schedule_interval': None,
    'max_active_runs': 1,
    'catchup': False,
}

# Define the DAG
dag = DAG(
    '3_pandas_sql_file_processing_with_logging',
    default_args=default_args,
    description='A DAG for SQL file processing using Pandas with logging',
)

def read_from_sqlite():
    dag_directory = os.path.dirname(os.path.realpath(__file__))
    input_file_path = os.path.join(dag_directory, 'files', 'mydatabase_chinook.db')

    # Use Pandas to read data from SQLite and convert to DataFrame
    df = pd.read_sql_query('SELECT * FROM Album', sqlite3.connect(input_file_path))

    return df

def log_first_row(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='read_from_sqlite')

    # Log the first row of the DataFrame
    first_row = data.head(1)
    print(f'Logging first row:\n{first_row}')

def write_to_sqlite(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='read_from_sqlite')
    dag_directory = os.path.dirname(os.path.realpath(__file__))
    output_file_path = os.path.join(dag_directory, 'files', 'mydatabase_chinook_out.db')

    # Use Pandas to write DataFrame back to SQLite
    data.to_sql('output_table', sqlite3.connect(output_file_path), if_exists='replace', index=False)

# Define the tasks
read_task = PythonOperator(
    task_id='read_from_sqlite',
    python_callable=read_from_sqlite,
    dag=dag,
)

log_task = PythonOperator(
    task_id='log_first_row',
    python_callable=log_first_row,
    provide_context=True,
    dag=dag,
)

write_task = PythonOperator(
    task_id='write_to_sqlite',
    python_callable=write_to_sqlite,
    provide_context=True,
    dag=dag,
)

# Define the task dependencies
read_task >> log_task >> write_task
