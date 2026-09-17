# ADF Study — Analysis Summary

Generated from results/grid/runs.jsonl

## TSR (Task Success Rate) by cell

Cell                                                     ADF    N     TSR    [95% CI]
-------------------------------------------------------------------------------------
  finance_ecl   nova-micro-v1:0   L0          0.381    50  96.0%  [90.0%, 100.0%]
  finance_ecl   nova-micro-v1:0   L0prime     0.000    55  96.4%  [90.9%, 100.0%]
  finance_ecl   nova-micro-v1:0   L1          0.857    50  96.0%  [90.0%, 100.0%]
  finance_ecl   nova-micro-v1:0   L2          1.159    50  96.0%  [90.0%, 100.0%]
  finance_ecl   nova-micro-v1:0   L3          1.921    50  100.0%  [100.0%, 100.0%]
  finance_ecl   nova-micro-v1:0   L4          2.363    50  100.0%  [100.0%, 100.0%]
  finance_ecl   nova-micro-v1:0   L5          3.567    50  92.0%  [84.0%, 98.0%]
  finance_ecl   nova-micro-v1:0   L5_codegen  3.567    50  0.0%  [0.0%, 0.0%]
  finance_ecl   nova-micro-v1:0   L6          3.758    50  100.0%  [100.0%, 100.0%]
  finance_ecl   gemma-3-27b-it    L0prime     0.000    21  90.5%  [76.2%, 100.0%]

## DI (within-instance) by cell

Cell                                                     DI     PS     RR   
----------------------------------------------------------------------
  finance_ecl   nova-micro-v1:0   L0          1.000  1.000  1.000
  finance_ecl   nova-micro-v1:0   L0prime     1.000  1.000  1.000
  finance_ecl   nova-micro-v1:0   L1          0.982  0.927  1.000
  finance_ecl   nova-micro-v1:0   L2          0.964  0.854  1.000
  finance_ecl   nova-micro-v1:0   L3          0.969  0.875  1.000
  finance_ecl   nova-micro-v1:0   L4          0.971  0.885  1.000
  finance_ecl   nova-micro-v1:0   L5          0.987  0.948  1.000
  finance_ecl   nova-micro-v1:0   L5_codegen  0.984  0.938  0.000
  finance_ecl   nova-micro-v1:0   L6          0.961  0.844  1.000

## Hypothesis test results

  H1 (DI~ADF monotone): 0/1 pairs support the hypothesis
  H2 (non-monotone success): 0/1 pairs support the hypothesis
  H3 (task-dep. optimum): 0/0 pairs support the hypothesis
  H4 (model-dep. optimum, primary): 1/1 pairs support the hypothesis
  H5 (substrate equiv.) DI: 1/1 | TSR: 0/1