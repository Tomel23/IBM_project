from operator import itemgetter

import pandas as pd
from collections import namedtuple
import docplex.cp.utils_visu as visu
from docplex.mp.model import Model
from docplex.cp.model import CpoModel, CpoSegmentedFunction

# utility to convert a weekday string to an index in 0..6
from datetime import datetime


# -----------------------------------------------------------------------------
# Initialize the problem data/inicjalizacja danych
# -----------------------------------------------------------------------------

# Project task descriptor/klasa opisująca zadanie
class ProjectTask(
    namedtuple("Task", ["id", "title", "days", "start", "end", "min_requirement", "max_requirement", "prev"])):
    def __str__(self):
        title = self.title
        days = self.days
        return '%s_%s_%02d' % (title, days, self.start)

    def overlaps(self, task):
        if not isinstance(task, ProjectTask):
            return False
        else:
            return task.end > self.start and task.start < self.end


# Employer descriptor/klasa opisująca pracownika
class TEmployer(namedtuple("Employer", ["name", "pay_rate", "scale"])):
    def __str__(self):
        return self.name


# usuwanie dni weekendowych
def remove_weekends(time):
    return time - (int(time / 7) * 2)


# -----------------------------------------------------------------------------
# Prepare the data for modeling
# -----------------------------------------------------------------------------

# Load data form cvs /table |ładowanie tablez z pliku
task_xls_file = pd.ExcelFile('Lista_zadan_do_opytmalizacji.xls')
# parsowanie arkuszy na pandas dataframe
TaskTable = task_xls_file.parse('Tasks')
EmployerTable = task_xls_file.parse('Employers')

# tworzenie listy pracownikow
employers = [TEmployer(EmployerTable["name"][i], EmployerTable["pay_rate"][i], EmployerTable["scale"][i]) for i in
             range(len(EmployerTable))]

# tworzenie listy zadan
tasks = [ProjectTask(*task_row) for task_row in TaskTable.itertuples(index=False)]

PROJECT = {'duration': 17}

# Max work time
MAX_WORK_TIME = PROJECT['duration']

# Project work time
PROJECT_WORK_TIME = remove_weekends(MAX_WORK_TIME)

# -----------------------------------------------------------------------------
# Build the model
# -----------------------------------------------------------------------------

# Create model
mdl = Model("employers")

# -----------------------------------------------------------------------------
# Initialize model variable sets/inicjalizacja zmiennych dla modelu
# -----------------------------------------------------------------------------


# One binary variable for each pair (employer, task) equal to 1 if employer e is assigned to task/ macierz pracownik x zadanie
employer_assignment_vars = mdl.binary_var_matrix(employers, tasks, 'assign')

print(employer_assignment_vars)

# Time variables. For each employer, allocate one variable for worktime
employer_work_time_vars = mdl.continuous_var_dict(employers, lb=0, name='EmployerWorkTime')
employer_over_average_time_vars = mdl.continuous_var_dict(employers, lb=0, name='EmployerOverAverageWorkTime')
employer_under_average_time_vars = mdl.continuous_var_dict(employers, lb=0, name='EmployerUnderAverageWorkTime')
# Finally the global average work time
average_employer_work_time = mdl.continuous_var(lb=0, name='AverageWorkTime')

# -----------------------------------------------------------------------------
# Add constraint
# -----------------------------------------------------------------------------

# Fourth constraint: a employer cannot be assigned overlapping tasks¶
# Post only one constraint per couple(task1, task2)
number_of_overlaps = 0
nb_tasks = len(tasks)
for i1 in range(nb_tasks):
    for i2 in range(i1 + 1, nb_tasks):
        T1 = tasks[i1]
        T2 = tasks[i2]
        if T1.overlaps(task=T2):
            number_of_overlaps += 1
            for n in employers:
                mdl.add_constraint(employer_assignment_vars[n, T1] + employer_assignment_vars[n, T2] <= 1,
                                   "high_overlapping_{0!s}_{1!s}_{2!s}".format(T1, T2, n))
print("# overlapping tasks: {}".format(number_of_overlaps))

# Number of employer per task
for t in tasks:
    demand_min = t.min_requirement
    demand_max = t.max_requirement
    total_assigned = mdl.sum(employer_assignment_vars[e, t] for e in employers)
    mdl.add_constraint(total_assigned >= demand_min,
                       "high_req_min_{0!s}_{1}".format(t, demand_min))
    mdl.add_constraint(total_assigned <= demand_max,
                       "medium_req_max_{0!s}_{1}".format(t, demand_max))

# suma wszystkich przypisanych do zadań pracowników
total_number_of_assignments = mdl.sum(employer_assignment_vars[e, t] for e in employers for t in tasks)

# Single empolyers work time
for e in employers:
    work_time_var = employer_work_time_vars[e]  # czas pracy kazdego pracownika
    mdl.add_constraint(
        work_time_var == mdl.sum(employer_assignment_vars[e, t] * t.days for t in tasks),
        "work_time_{0!s}".format(e))
    mdl.add_constraint(
        work_time_var == average_employer_work_time + employer_over_average_time_vars[e] -
        employer_under_average_time_vars[e],
        "average_work_time_{0!s}".format(e))
    # State the maximum work time as a constraint, so that it can be relaxed,
    # should the problem become infeasible.
    mdl.add_constraint(work_time_var <= MAX_WORK_TIME, "max_time_{0!s}".format(e))

# czas pracy pracownikow
total_over_average_worktime = mdl.sum(employer_over_average_time_vars[e] for e in employers)
total_under_average_worktime = mdl.sum(employer_under_average_time_vars[e] for e in employers)
total_fairness = total_over_average_worktime + total_under_average_worktime

# koszt pracy pracownikow
employer_costs = [employer_assignment_vars[e, t] * e.pay_rate * e.scale * t.days for e in employers for t in tasks]
total_salary_cost = mdl.sum(employer_costs)
# -----------------------------------------------------------------------------
# KPI
# -----------------------------------------------------------------------------
mdl.add_kpi(total_number_of_assignments, "Total number of assignments")
mdl.add_kpi(total_salary_cost, "Total salary cost")
mdl.add_kpi(total_over_average_worktime, "Total over-average worktime")
mdl.add_kpi(total_under_average_worktime, "Total under-average worktime")
mdl.add_kpi(total_fairness, "Total fairness")

mdl.print_information()

mdl.minimize(total_salary_cost + total_fairness + total_number_of_assignments)

# Set Cplex mipgap to 1e-5 to enforce precision to be of the order of a unit (objective value magnitude is ~1e+5).
mdl.parameters.mip.tolerances.mipgap = 1e-5
# -----------------------------------------------------------------------------
# Solve the model and display the result
# -----------------------------------------------------------------------------
url = "https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/"  # ENTER YOUR URL HERE
key = "api_16e103e1-cd2c-41f0-b6ec-d3da832a59d4"  # ENTER YOUR KEY HERE
t = mdl.solve(url=url, key=key, log_output=True)
# t=mdl.solve(FailLimit=100000, TimeLimit=10)
assert t, "solve failed"
mdl.report()
mdl.print_solution()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Build set of all task index and assign a color to each of them to be used in figures
task_by_id = {t.id for t in tasks}
colors = range(1, len(task_by_id))
colorByTask = {}
for d, c in zip(task_by_id, ['r', 'm', 'b', 'g', 'y', 'c', 'k', 'r', 'm', 'b', 'g', 'y', 'c']):
    colorByTask[d] = c

# Build dictionary with number of assigned nurses for each shift
nbAssignmentsByTask = {}
result = []
sol = []
for e in employers:
    for t in tasks:
        if employer_assignment_vars[e, t].solution_value > 0:
            sol.append({'id': t.id, 'employer_name': e.name})
            nbAssignmentsByTask[t] = nbAssignmentsByTask.get(t, 0) + 1

print(sol)
import itertools

print(nbAssignmentsByTask)
result = sorted(sol, key=itemgetter('id'))

for key, value in itertools.groupby(sol, key=itemgetter('id')):
    emp = []
    print('------------------')
    print('task id ' + str(key))
    print('assigned employers')
    for i in value:
        print(i.get('employer_name'))

        emp.append(i)
    result.append((key, emp))
    del emp

print(result)
