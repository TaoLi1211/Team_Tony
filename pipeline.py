from clearml import PipelineController

pipe = PipelineController(
    name="Predictive Maintenance Pipeline",
    project="Predictive Maintenance",
    version="1.0"
)

pipe.add_step(
    name='data_ingestion',
    base_task_project="Predictive Maintenance",
    base_task_name="Data Ingestion"
)

pipe.add_step(
    name='data_preprocessing',
    base_task_project="Predictive Maintenance",
    base_task_name="Data Preprocessing",
    parents=['data_ingestion']
)

pipe.add_step(
    name='model_training',
    base_task_project="Predictive Maintenance",
    base_task_name="Model Training",
    parents=['data_preprocessing']
)

pipe.add_step(
    name='model_evaluation',
    base_task_project="Predictive Maintenance",
    base_task_name="Model Evaluation",
    parents=['model_training']
)

if __name__ == '__main__':
    pipe.start_locally(run_pipeline_steps_locally=True)
