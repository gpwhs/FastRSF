# FastRSF
A fast implementation of Random Survival Forests inspired by Scikit-Survival but without Sklearn's tree implementation.
Very much WIP so far, but currently [example_use.py](example_use.py) shows how to use it, and [compare.py](compare.py) shows how basic
performance compares to scikit-survival. Memory usage seems much lower, but speed is not yet as good, at least for this toy example.



'''
❯ uv run compare.py
Loading data...

Training scikit-survival RandomSurvivalForest...
Training time: 0.06 seconds
Prediction time: 0.01 seconds
Concordance index: 0.7575

Training FastRandomSurvivalForest...
Training time: 1.29 seconds
Prediction time: 0.04 seconds
Concordance index: 0.7419

Performance comparison:
Training speedup: 0.05x
Prediction speedup: 0.34x
Concordance index difference: -0.015671
Correlation between predictions: 0.843660
RMSE between predictions: 51.779950
Saved prediction comparison plot to 'prediction_comparison.png'

Comparing survival function predictions...
Saved survival function comparison plot to 'survival_function_comparison.png'
Saved side-by-side survival function comparison to 'survival_function_side_by_side.png'

Training smaller models for memory comparison...

Approximate memory usage:
scikit-survival RSF: 0.86 MB
FastRandomSurvivalForest: 0.19 MB
Memory reduction: 78.18%

==================================================
SUMMARY
==================================================
Training speed: SLOWER by 95.41%
Prediction speed: SLOWER by 66.11%
Concordance index: WORSE by 0.015671
Prediction correlation: 0.843660
Memory usage: LESS by 78.18%
==================================================''''
![[prediction_comparison.png]]
