import datetime

import gitlab
import tabulate

# GitLab configurations
gitlab_url = "https://yyz-gitlab.local.tenstorrent.com/"
private_token = "[YOUR_GITLAB_PRIVATE_TOKEN]"
project_id = "tenstorrent/pybuda"

# Other configurations
gitlab_datetime_format = "%Y-%m-%dT%H:%M:%S.%f%z"
print_datetime_format = "%d-%m-%Y %H:%M:%S"

# Filter conditions
schedule_name = "PyBuda Dev Models"    # Name of the scheduled pipeline to reference
job_name = "[job name]"             # Job name, e.g. silicon-nlp-pytorch-xglm-wh-b0-n150
pipeline_limit_num = 9              # History limit for specified job


def filter_pipeline_condition(pipeline):
    condition = True

    # Reference only "main" branch
    if pipeline.ref.lower() != "main":
        condition = False

    # Reference only scheduled pipelines (nightlies, weeklies)
    if pipeline.source.lower() != "schedule":
        condition = False
        
    if pipeline.status.lower() == "canceled":
        condition = False
        
    return condition


def collect_pipeline_details(pipeline):
    details = {}

    details["id"] = pipeline.id
    details["status"] = pipeline.status
    
    started_at = datetime.datetime.strptime(pipeline.started_at, gitlab_datetime_format) if pipeline.started_at else None
    details["started_at"] = started_at.strftime(print_datetime_format) if started_at else "N/A"
    finished_at = datetime.datetime.strptime(pipeline.finished_at, gitlab_datetime_format) if pipeline.finished_at else None
    details["finished_at"] = finished_at.strftime(print_datetime_format) if started_at and finished_at else "N/A"
    details["duration"] = pipeline.duration

    details["ref"] = pipeline.ref
    details["sha"] = pipeline.sha
    details["source"] = pipeline.source
    details["web_url"] = pipeline.web_url

    return details


def filter_job_condition(job, job_name=None):
    condition = True
        
    # Reference only specific job
    if job_name != "" and job.name.lower() != job_name.lower():
        condition = False

    return condition


def collect_job_details(job):
    details = {}
    
    details["name"] = job.name
    details["ref"] = job.ref
    details["stage"] = job.stage
    details["status"] = "❌ " + job.status if job.status == "failed" else "✅ " + job.status
    details["web_url"] = job.web_url
    
    started_at = datetime.datetime.strptime(job.started_at, gitlab_datetime_format) if job.started_at else None
    details["started_at"] = started_at.strftime(print_datetime_format) if started_at else "N/A"
    finished_at = datetime.datetime.strptime(job.finished_at, gitlab_datetime_format) if job.finished_at else None
    details["finished_at"] = finished_at.strftime(print_datetime_format) if started_at else "N/A"
    details["duration"] = job.duration
    
    details["short_commit"] = job.commit["short_id"]

    return details


def print_job_history_table(table_rows):
    job_heading = f"| History for: {job_name} |"
    table_headers = ["#", "Pipeline ID", "Job Status", "Job Duration", "Job Started At", "Job Finished At", "Job Short Commit", "Job Web URL"]
    
    print()
    print("-" * len(job_heading))
    print(job_heading)
    print(tabulate.tabulate(table_rows, headers=table_headers, tablefmt="grid"))


# Function to filter and print scheduled jobs
def list_scheduled_jobs(project_id, job_name=""):
    # Create a GitLab client
    gl = gitlab.Gitlab(url=gitlab_url, private_token=private_token)
    
    project = gl.projects.get(project_id)
    schedules = project.pipelineschedules.list(all=True)
    filtered_schedules = [schedule for schedule in schedules if schedule.description == schedule_name]
    assert len(filtered_schedules) == 1, f"Found {len(filtered_schedules)} schedules with name {schedule_name}"

    # Fetch detailed pipeline information
    table_rows = []
    pipelines = filtered_schedules[0].pipelines.list(all=True)[::-1]
    
    table_rows_num = pipeline_limit_num if len(pipelines) > pipeline_limit_num else len(pipelines)
    
    for i, pipeline in enumerate(pipelines):
        if i >= pipeline_limit_num:
            break
        pipeline = project.pipelines.get(pipeline.id)
        
        if filter_pipeline_condition(pipeline):
            print(f"Processing {i + 1}/{table_rows_num}")
            pipeline_details = collect_pipeline_details(pipeline)            

            # List jobs for the scheduled pipeline
            jobs = pipeline.jobs.list(all=True)
            for job in jobs:
                if filter_job_condition(job, job_name):
                    job_details = collect_job_details(job)
                    
                    table_row = [i + 1, pipeline_details["id"], job_details["status"], job_details["duration"], job_details["started_at"], job_details["finished_at"], job_details["short_commit"], job_details["web_url"]]
                    table_rows.append(table_row)
                    
    print_job_history_table(table_rows)


list_scheduled_jobs(project_id, job_name)
