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
  finance_ecl   nova-pro-v1:0     L0          0.381    50  92.0%  [84.0%, 98.0%]
  finance_ecl   nova-pro-v1:0     L0prime     0.000    50  100.0%  [100.0%, 100.0%]
  finance_ecl   nova-pro-v1:0     L1          0.857    50  92.0%  [84.0%, 98.0%]
  finance_ecl   nova-pro-v1:0     L2          1.159    46  89.1%  [80.4%, 97.8%]
  finance_ecl   gemma-3-27b-it    L0          0.381    50  100.0%  [100.0%, 100.0%]
  finance_ecl   gemma-3-27b-it    L0prime     0.000    50  96.0%  [90.0%, 100.0%]
  finance_ecl   gemma-3-27b-it    L1          0.857    50  100.0%  [100.0%, 100.0%]
  finance_ecl   gemma-3-27b-it    L2          1.159    50  100.0%  [100.0%, 100.0%]
  finance_ecl   gemma-3-27b-it    L3          1.921    50  0.0%  [0.0%, 0.0%]
  finance_ecl   gemma-3-27b-it    L4          2.363    50  0.0%  [0.0%, 0.0%]
  finance_ecl   gemma-3-27b-it    L5          3.567    50  0.0%  [0.0%, 0.0%]
  finance_ecl   gemma-3-27b-it    L5_codegen  3.567    50  0.0%  [0.0%, 0.0%]
  finance_ecl   gemma-3-27b-it    L6          3.758    50  0.0%  [0.0%, 0.0%]
  finance_ecl   qwen3-32b-v1:0    L0          0.381    50  100.0%  [100.0%, 100.0%]
  finance_ecl   qwen3-32b-v1:0    L0prime     0.000    50  100.0%  [100.0%, 100.0%]
  finance_ecl   qwen3-32b-v1:0    L1          0.857    50  100.0%  [100.0%, 100.0%]
  finance_ecl   qwen3-32b-v1:0    L2          1.159    50  100.0%  [100.0%, 100.0%]
  finance_ecl   qwen3-32b-v1:0    L3          1.921    50  100.0%  [100.0%, 100.0%]
  finance_ecl   qwen3-32b-v1:0    L4          2.363    50  100.0%  [100.0%, 100.0%]
  finance_ecl   qwen3-32b-v1:0    L5          3.567    50  100.0%  [100.0%, 100.0%]
  finance_ecl   qwen3-32b-v1:0    L5_codegen  3.567    50  0.0%  [0.0%, 0.0%]
  finance_ecl   qwen3-32b-v1:0    L6          3.758    50  100.0%  [100.0%, 100.0%]

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
  finance_ecl   nova-pro-v1:0     L0          0.953  1.000  0.917
  finance_ecl   nova-pro-v1:0     L0prime     1.000  1.000  1.000
  finance_ecl   nova-pro-v1:0     L1          0.977  0.906  1.000
  finance_ecl   nova-pro-v1:0     L2          0.753  0.438  0.800
  finance_ecl   gemma-3-27b-it    L0          1.000  1.000  0.042
  finance_ecl   gemma-3-27b-it    L0prime     0.958  1.000  0.917
  finance_ecl   gemma-3-27b-it    L1          0.977  0.906  0.250
  finance_ecl   gemma-3-27b-it    L2          0.984  0.938  0.167
  finance_ecl   gemma-3-27b-it    L3          0.974  0.896  1.000
  finance_ecl   gemma-3-27b-it    L4          0.974  0.896  1.000
  finance_ecl   gemma-3-27b-it    L5          0.969  0.875  1.000
  finance_ecl   gemma-3-27b-it    L5_codegen  0.977  0.906  0.583
  finance_ecl   gemma-3-27b-it    L6          0.979  0.917  0.417
  finance_ecl   qwen3-32b-v1:0    L0          1.000  1.000  0.708
  finance_ecl   qwen3-32b-v1:0    L0prime     1.000  1.000  1.000
  finance_ecl   qwen3-32b-v1:0    L1          0.990  0.958  0.708
  finance_ecl   qwen3-32b-v1:0    L2          0.969  0.875  0.750
  finance_ecl   qwen3-32b-v1:0    L3          0.997  0.990  0.167
  finance_ecl   qwen3-32b-v1:0    L4          0.979  0.917  0.000
  finance_ecl   qwen3-32b-v1:0    L5          0.979  0.917  0.208
  finance_ecl   qwen3-32b-v1:0    L5_codegen  1.000  1.000  0.000
  finance_ecl   qwen3-32b-v1:0    L6          0.990  0.958  0.167

## Hypothesis test results

  H1 (DI~ADF monotone): 0/4 pairs support the hypothesis
  H2 (non-monotone success): 0/4 pairs support the hypothesis
  H3 (task-dep. optimum): 0/0 pairs support the hypothesis
  H4 (model-dep. optimum, primary): 1/1 pairs support the hypothesis
  H5 (substrate equiv.) DI: 3/3 | TSR: 0/3