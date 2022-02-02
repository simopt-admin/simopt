Contamination Control Problem
==================================================

**Model Description**

Consider a food supply chain consisting :math:`n` stages. Suppose there exists a possibility that pathogenic microorganisms and other poisonous elements contaminate some fraction of the food supply at each stage. The contamination has a growth rate :math:`Λ_i`, which follows a Beta distribution (:math:`0 <= Λ_i <= 1`), at stage :math:`i`. A prevention effort can be made at stage :math:`i`, which deceases the contamination by a rate :math:`Γ_i`, following a Beta distribution (:math:`0 <= Γ_i <= 1`). Whether the prevention measure is executed depends on the prevention decision factor, :math:`u_i`. 
The model will determine the contamination levels in each state, given the initial contamination fraction :math:`X_1`, which follows a Beta distribution.

The contamination level at each stage is given by the following equation:

:math:`X_i = Λ_i * (1 - u_i) * (1 - X_i-1) + (1 - Γ_i * u_i) * X_i-1`

Factor
    - Contamination Rate Alpha:
    		Description: Alpha parameter of beta distribution for growth rate of contamination at each stage.
    		
		Default value: 1.0
    - Contamination Rate Beta:  
    		Description: Beta parameter of beta distribution for growth rate of contamination at each stage. 
    		
		Default value: 17/3
    - Restore Rate Alpha:  
    		Description: Alpha parameter of beta distribution for rate that contamination decreases by after prevention effort.
    		
		Default value: 1.0
    - Restore Rate Beta:  
    		Description: Beta parameter of beta distribution for rate that contamination decreases by after prevention effort.
    		
		Default value: 3/7
    - Initial Rate Alpha:  
    		Description: Alpha parameter of beta distribution for initial contamination fraction.
    		
		Default value:  1.0
    - Initial Rate Beta:  
    		Description: Beta parameter of beta distribution for initial contamination fraction.
    		
		Default value: 30.0
    - Stages: 
    		Description: Stage of food supply chain.
    		
		Default value: 5 
    - Prevention Decision:  
    		Description: Prevention decision. :math:`u_i = 1` if a prevention measure is executed at the stage :math:`i`, :math:`u_i = 0` otherwise.		
    		
		Default value: (0, 0, 0, 0, 0) 

Response
	- Level (:math:`X_i`): List of contamination levels over time


References: 
Contamination Control Problem. Shin, Kaeyoung., Pasupathy, Raghu. Virginia Tech. (December 18, 2010).
This example is adapted from the article by Y. Hu et al. [1]


**Optimization Problem**
description in words and summary of properties, e.g., 3-dimensional problem with stochastic constraints

Objective: Minimize the overall cost of the preventive measures and efficiently control contamination in the food supply chain
(:math:`Σ (c_i * u_i)`)

Constraints: < in words, but can refer to relevant response names >

Decision Variables: < in words, but can refer to relevant factor names >

Fixed factor values: (could be the default values listed above)

Starting solution:

Optimal solution:

Optimal objective function value:
