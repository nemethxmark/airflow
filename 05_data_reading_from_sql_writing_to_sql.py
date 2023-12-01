from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import sqlite3

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
    '0_simple_sql_file_processing',
    default_args=default_args,
    description='A simple DAG for SQL file processing',
)

def read_from_sqlite():
    dag_directory = os.path.dirname(os.path.realpath(__file__))
    input_file_path = os.path.join(dag_directory, 'files', 'mydatabase_chinook.db')

    conn = sqlite3.connect(input_file_path)  # Replace with your SQLite input file path
    cursor = conn.cursor()

    # Example: Read data from a table
    cursor.execute('''SELECT * FROM Album''')
    data = cursor.fetchall()

    conn.close()
    return data

def write_to_sqlite(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='read_from_sqlite')
    dag_directory = os.path.dirname(os.path.realpath(__file__))
    output_file_path = os.path.join(dag_directory, 'files', 'mydatabase_chinook_out.db')

    conn = sqlite3.connect(output_file_path)  # Replace with your SQLite output file path
    cursor = conn.cursor()

    # Example: Create a new table and insert data
    cursor.execute('CREATE TABLE IF NOT EXISTS output_table (id INTEGER PRIMARY KEY, name TEXT)')
    cursor.executemany('INSERT INTO output_table (id, name) VALUES (?, ?)', data)

    cursor.execute(f"CREATE TABLE IF NOT EXISTS Album (\
                        AlbumId INTEGER PRIMARY KEY,\
                        Title TEXT,\
                        ArtistId INTEGER\
                    )")
    for album in data:
        cursor.execute(f"INSERT INTO Album (AlbumId,Title,ArtistId) VALUES (?, ?, ?)", album)

    conn.commit()
    conn.close()

# Define the tasks
read_task = PythonOperator(
    task_id='read_from_sqlite',
    python_callable=read_from_sqlite,
    dag=dag,
)

write_task = PythonOperator(
    task_id='write_to_sqlite',
    python_callable=write_to_sqlite,
    provide_context=True,
    dag=dag,
)

# Define the task dependencies
read_task >> write_task


# Operator (Python, Bash, Docker, SQL, HTTP listen on a path)
# ti just for small data - use external storage, or databases (or caching at iterative, compress, streaming (can be streamed in bulks)

# Use cases to check
# SQL to pandas to SQL - log few rows, modify something, log it again


# Hadoop - SQL - pandas - SQL - Hadoop

# Create an API

# API - HTTP - AIRFLOW (extraxting data from 3rd party API, Trigerr external process by an API call, ALerting, Notification by messaging API)
