from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os

# Define default_args dictionary to specify the default parameters of the DAG
default_args = {
    'owner': 'airflow', # sb need to be logged in
    'depends_on_past': False,
    'start_date': datetime(2023, 11, 28),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 10,
    'retry_delay': timedelta(minutes=5), # if fails, when to retry
    'schedule_interval': None,  # Set schedule_interval to None for manual triggering, @daily or * * * * * - min hour dayMonth Month dayweek
    # 0 2 0 0 0 every day 2AM, 0 0 * * 1-5 weekdays midnight, 0/15 * * * * every 15mins, 0 3,6,9 * * * 3AM,6AM, 9AM
    'max_active_runs': 1,  # Limit the number of active DAG runs to 1, how many DAG instance can run at the same time (Same DAG)
    'catchup': False,  # Do not catch up on historical DAG runs
}

# Define the DAG
dag = DAG(
    'simple_file_processing_3',
    default_args=default_args,
    description='A simple DAG for file processing',
)

# Define the function that will be executed by the PythonOperator
def process_file():
    dag_directory = os.path.dirname(os.path.realpath(__file__))
    input_file_path = os.path.join(dag_directory, 'files', 'input.txt')
    output_file_path = os.path.join(dag_directory, 'files', 'output.txt')

    with open(input_file_path, 'r') as input_file:
        content = input_file.read()

    with open(output_file_path, 'w') as output_file:
        output_file.write(content)

# Define the PythonOperator
process_file_task = PythonOperator(
    task_id='process_file',
    python_callable=process_file,
    dag=dag,
)

# Set the task dependencies
process_file_task

if __name__ == "__main__":
    dag.cli()
