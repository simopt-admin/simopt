
**Emergency Department Health Care**
======================================

Description:

In this particular problem, an emergency department in a hospital is simulated. There are two types of patients who enter
the hospital, walk-in and ambulance, those that come in through an ambulance bypass the nurse at reception while walk-ins
must see the receptionist first. All patients must go through the examination room where a doctor assesses whether further
are necessary. If extra test are needed, they are sent to a lab technician who performs the tests and are then sent back
to the examination room. If the patient does not need any tests, the doctor determines if any treatments are needed. Those
who do not need treatment are given medication and can leave. Those who do need treatment, but are not in critical condition
are routed to the treatment room where they are attended to by a treatment room nurse. Critical patients are directed to the 
emergency room where an emergency room nurse provides service. All patients who are treated are then dismissed from the
hospital, and all patients are served in a first-in, first-out basis.

The receptionists, doctors, lab technicians, and nurses earn 40,000 120,000, 50,000, and 35,000, respectively. 

Sources of Randomness:
There are six sources of randomness in this problem and are used to calculate the service time at a specific location. 
Reception is exponentially distributed with a rate of 1/7.5. Lab tests are triangularly distributed with a minimum of 10, 
maximum of 30, and mode of 20. The examination room has an exponential distribution with a rate of 1/15. The reexamination
process is exponentially distributed with a rate of 1/9. The treatment room is triangularly distributed with a minimum of
20, mode of 28, and maximum of 30. Finally, the emergency room waiting time is exponentially distributed with a rate of 
1/90.

Model Factors: 

 *Routing Probabilities* (:math:`p`) - The probability a person will need extra tests, treatment is needed and whether it is major or minor.

 *Walk-In Arrivals* (:math:`s`) - The arrivals of walk in patients follow a non-stationary Poisson process.

 *Ambulance Arrivals* (:math:`a`) - Ambulance arrivals follow a Poisson processith a rate of 2 per hour.

 *Rate of Exponential Distributions* (:math:`r`) - The rate of the exponential distribution used to calculate a certain wait time.

 *Minimum of Triangular Distributions* (:math:`min`) - The minimum parameter of the triangular distribution used to calculate a certain wait time.

 *Mode of Triangular Distributions* (:math:`mode`) - The mode parameter of the triangular distribution used to calculate a certain wait time.

 *Max of Triangular Distributions* (:math:`max`)- The maximum parameter of the triangular distribution used to calculate a certain wait time.


Responses:

 *Total Cost* - We want to minimize the total amount spent on hiring receptionists, doctors, nurses, and technicians while also satisfying all constraints.

References: 

This model is adapted from Dr. Eckman's SimOpt Problems Library, EmergencyRoom Folder.

**Optimization Problem: Continuous Newsvendor**

Decision Variables:

*Number of Employees Hired* (:math:`x`) - This model will determine how many of a certain employee, :math:`x`, to hire. 

Objectives: 

The purpose of this model is to minimize the cost to the hospital when hiring employees, while also meeting the demands
of the hospital.

Constraints: 
Within the problem, there are certain restrictions. The maximum amount of employees that can be hired by the hospital are 
specific for each kind of employee due to cost considerations. At most, there can be 3 receptionists, 6 doctors, 5 lab 
technicians, 6 treatment nurses and 12 ER nurses. The average total waiting time for critical patients must not exceed 
two hours. Nonnegativity for all factors

Problem Factors:
Empty?

Fixed Model Factors:
Empty

Starting Solution = 1 receptionist, 4 doctors, 3 technicians, 2 treatment nurses, and 5 emergency nurses

Model Default Factors:
*Employee Salary* - The price that it costs to hire an employee is 40,000, 120,000, 50,000, 35,000, for receptionists, doctors,
lab technicians, and nurses, respectively.
*Reception Wait Time* - The rate used to calculate the wait time at reception is 1/7.5.
*Salvage Price* - The price that the liquid can be salvaged at is 1 dollar.
*Burr_c* - The alpha constant for the Burr random distribution is set to 2
*Burr_k* - The beta constant for the Burr random distribution is set to 20

Optimal Solution:
Unknown

Optimal Objective Function Value:
Unknown

