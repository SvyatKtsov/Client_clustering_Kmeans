import sys, os
# add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Second_Part_of_Main_Module.Pearson_corr_and_LinReg_Task2 import *


all_features=list(an_cpy.columns[i] for i in range(len(an_cpy.columns)) if i not in [0,1,2,3,4,5,20,21,22,24]) #churn=20 index
print(all_features)




ax = plt.figure().add_subplot(projection='3d')
ax1 = plt.figure().add_subplot(projection='3d')

x1 = an_cpy['total_eve_minutes']
x2 = an_cpy['total_eve_charge']
x3 = an_cpy['number_customer_service_calls']
x4 = an_cpy.iloc[:,22] ##[:,22]
x_has_intern_plan = an_cpy['Has_intern_plan']

ax.scatter(x1,x2,x3,color='b')
ax1.scatter(x3,x4,color='r')
# ax.scatter(x1, x2, x3, color='r', label='total_eve_minutes')
# ax.scatter(x2, x3, x1, color='g', label='total_eve_charge')
# ax.scatter(x3, x1, x2, color='b', label='number_customer_service_calls')


axNew=plt.figure().add_subplot(projection='3d')
axNew.scatter(an_cpy['total_day_minutes'],an_cpy['total_day_charge'],x3,color='g')


colors = []
for val in x3:
    if val <= 3:
        colors.append('red')
    elif 3 < val < 7:
        colors.append('blue')
    else:
        colors.append('green')

# creating 3d-scatter plot
ax2 = plt.figure().add_subplot(projection='3d')
ax2.scatter(x1, x2, x3, c=colors); plt.show()