from integration_tools.utils.parser import parser
from integration_tools import data_integration
import argparse
parser_of_fatty_liver_project = argparse.ArgumentParser(description='Initial tool for data integration')
parameters=parser.parse_parameters(parser_of_fatty_liver_project)
integrator=data_integration.Data_inegrator(parameters)
integrator.find_best_configuration()
integrator.project_all_data()