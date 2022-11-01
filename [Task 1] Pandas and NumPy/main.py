# -*- coding: utf-8 -*-
import pandas as pd


def main():
    salary = pd.read_csv('ROSSTAT_SALARY_RU_2019_normalized.csv', index_col='REGION_NAME')
    salary = salary.drop(['Смоленская область', 'Чувашская Республика', 'Костромская область', 'Саратовская область'], axis = 0)
    salary = salary.sort_values(by='SALARY')
    print("Variation series (X):\n", salary)
    print("X(9): ", salary.SALARY[8], "\nX(53): ", salary.SALARY[52], "\nX(56): ", salary.SALARY[55])
    print("Sample mean: ", round(salary.SALARY.mean(), 2))
    print("Sample median: ", salary.SALARY.median())


if __name__ == '__main__':
   main()

