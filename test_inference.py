from core.inference import predict_project_type
from core.evaluator import generate_evaluation
from core.report_generator import generate_report, save_report

sample_code = """
from sklearn.linear_model import LinearRegression
model = LinearRegression()
"""

project_type = predict_project_type(sample_code)
evaluation = generate_evaluation(project_type)

report = generate_report(project_type, evaluation)
save_report(report)

print("Evaluation report generated successfully!")
from core.visualizer import plot_regression_example

plot_regression_example()
