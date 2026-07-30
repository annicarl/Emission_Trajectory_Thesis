# Sensitivity analysis summary

Scenario: `baseline`. Baseline cumulative emissions: 4,820,964,231 t CO₂e.

## Five most influential parameters

- `lca:Consumption:ICEVs-Petrol_large`: maximum change 137,552,137 t CO₂e; span 275,104,275 t CO₂e; maximum elasticity 0.285.
- `lca:TTW:ICEVs-Petrol_large`: maximum change 105,519,448 t CO₂e; span 211,038,896 t CO₂e; maximum elasticity 0.219.
- `lca:Consumption:ICEVs-Petrol_medium`: maximum change 79,250,277 t CO₂e; span 158,500,554 t CO₂e; maximum elasticity 0.164.
- `lca:TTW:ICEVs-Petrol_medium`: maximum change 60,794,733 t CO₂e; span 121,589,466 t CO₂e; maximum elasticity 0.126.
- `lca:Consumption:ICEVs-Petrol_small`: maximum change 54,534,964 t CO₂e; span 109,069,929 t CO₂e; maximum elasticity 0.113.

## Production emissions

- `lca:Vehicle_production:ICEVs-Petrol_large`: maximum change 28,548,851 t CO₂e.
- `lca:Vehicle_production:ICEVs-Petrol_medium`: maximum change 11,813,949 t CO₂e.
- `config:fleet_target.percent_change`: maximum change 8,745,913 t CO₂e.
- `lca:Vehicle_production:BEV_large`: maximum change 8,673,915 t CO₂e.
- `lca:Vehicle_production:ICEVs-Diesel_large`: maximum change 7,650,228 t CO₂e.

## Use emissions

- `lca:Consumption:ICEVs-Petrol_large`: maximum change 105,519,448 t CO₂e.
- `lca:TTW:ICEVs-Petrol_large`: maximum change 105,519,448 t CO₂e.
- `lca:Consumption:ICEVs-Petrol_medium`: maximum change 60,794,733 t CO₂e.
- `lca:TTW:ICEVs-Petrol_medium`: maximum change 60,794,733 t CO₂e.
- `lca:Consumption:ICEVs-Petrol_small`: maximum change 41,835,041 t CO₂e.

## End of life emissions

- `lca:End_of_life:ICEVs-Petrol_large`: maximum change 9,992,899 t CO₂e.
- `lca:End_of_life:ICEVs-Petrol_medium`: maximum change 6,129,438 t CO₂e.
- `lca:End_of_life:ICEVs-Diesel_large`: maximum change 4,047,210 t CO₂e.
- `lca:End_of_life:ICEVs-Petrol_small`: maximum change 4,016,052 t CO₂e.
- `lca:End_of_life:BEV_large`: maximum change 3,687,404 t CO₂e.

## Low-priority parameters

6 parameter(s) change cumulative emissions by less than 0.01%.

## Interpretation limits

Only the centrally configured reference scenario is evaluated in this first version. Cross-scenario robustness and scenario dependence therefore cannot yet be inferred.
Battery-production emissions are zero because the supplied LCA file contains battery capacity but no battery-production intensity (t CO₂e/kWh). No value was invented.
Emission-saving uncertainty versus another policy scenario cannot be calculated from a baseline-only analysis; the output schema remains scenario-aware for later runs.