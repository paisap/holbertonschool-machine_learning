#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

p1 = plt.bar(student_grades, student_grades)
plt.legend(p1[0])
plt.show()