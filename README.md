# RI-MML

Реализация резольвентно-индуцированного Muntz--Mittag--Leffler спектрального метода Petrov--Galerkin из [article.tex](/c:/Users/danii/Desktop/Diss/ri-mml/article.tex).

## Состав

- ООП-модель задачи `FractionalProblem`
- полугруппа показателей `Lambda_alpha`
- ортонормированный Müntz-базис
- operator-induced trial-базис `Phi_j = R_c M_j`
- сборка и решение дискретной схемы
- вычисление функции Миттаг--Леффлера через `pymittagleffler` с внутренним fallback

## Минимальный пример

```python
import numpy as np

from rimml import FractionalProblem, SolverConfig, SpectralPetrovGalerkinSolver


problem = FractionalProblem(
    alpha=0.5,
    epsilon=0.1,
    interval_end=1.0,
    u0=1.0,
    coefficient=lambda x: 2.0 + 0.25 * x,
    forcing=lambda x: np.ones_like(x),
    reference_coefficient=2.0,
)

solver = SpectralPetrovGalerkinSolver(problem, SolverConfig(basis_size=6))
solution = solver.solve()

grid = np.linspace(0.0, 1.0, 200)
u_values = solution.evaluate(grid)
```
