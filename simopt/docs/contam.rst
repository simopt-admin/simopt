Contamination Control Problem
==================================================

**Model Description**

Consider a food supply chain consisting :math:`n` stages. Suppose there exists a possibility that pathogenic microorganisms and other poisonous elements contaminate some fraction of the food supply at each stage. The contamination has a growth rate :math:`Λ_i`, which follows a Beta distribution (:math:`0 <= Λ_i <= 1`), at stage :math:`i`. A prevention effort can be made at stage :math:`i`, which deceases the contamination by a rate :math:`Γ_i`, following a Beta distribution (:math:`0 <= Γ_i <= 1`). Whether the prevention measure is executed depends on the prevention decision factor, :math:`u_i`. 
The model will determine the contamination levels in each state, given the initial contamination fraction :math:`X_1`, which follows a Beta distribution.

The contamination level at each stage is given by the following equation:

:math:`X_i = Λ_i * (1 - u_i) * (1 - X_i-1) + (1 - Γ_i * u_i) * X_i-1`

Factor
    - Contamination Rate Alpha:  
    		Description: 
    		
		Default value: 1.0
    - Contamination Rate Beta:  
    		Description: 
    		
		Default value: 17/3
    - Restore Rate Alpha:  
    		Description: 
    		
		Default value: 1.0
    - Restore Rate Beta:  
    		Description: 
    		
		Default value: 3/7
    - Initial Rate Alpha:  
    		Description: 
    		
		Default value:  1.0
    - Initial Rate Beta:  
    		Description: 
    		
		Default value: 30.0
    - Stages: 
    		Description: Stage of food supply chain.
    		
		Default value: 5 
    - Prevention Decision:  
    		Description: Prevention decision. :math:`u_i = 1` if a prevention measure is executed at the stage :math:`i`, :math:`u_i = 0` otherwise.		
    		
		Default value: (0, 0, 0, 0, 0) 

Response
	- Level (:math:`X_i`): List of contamination levels over time


References: Contamination Control Problem. 
