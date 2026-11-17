# Theory: why the data's statistics should predict which method wins

This note gives the derivations behind the hypotheses in `results/hypotheses.md`. Numbers quoted for GEMAS come from `results/spatial_stats/gemas_summary.json` (all cleaned samples, descriptive) unless marked "training folds".

Notation: $Z(s)$ is topsoil pH (CaCl$_2$) at location $s$; distances $h, d$ are in km in ETRS89-LAEA (EPSG:3035).

---

## 1. The model

We treat pH as

$$Z(s) = m(s) + S(s) + \varepsilon(s),$$

- $m(s)$: a smooth trend (constant for Ordinary Kriging; a function of coordinates or covariates for Universal and Regression Kriging),
- $S(s)$: a zero-mean, second-order stationary process with variance $C_1$ (the *partial sill*) and correlation function $r(h)$, with $r(0)=1$ and $r(h)\to 0$ as $h$ grows,
- $\varepsilon(s)$: independent noise with variance $C_0$ (the *nugget*): laboratory error plus soil variation at scales shorter than the sampling spacing.

This is the standard geostatistical model (Matheron, 1963; Cressie, 1993). The idea that "nearby things are more related than distant things" (Tobler, 1970) is the statement that $r(h)$ decreases with $h$.

## 2. The variogram and the nugget ratio

The semivariogram is

$$\gamma(h) = \tfrac12\,\mathbb E\big[(Z(s)-Z(s+h))^2\big] = C_0 + C_1\,\big(1-r(h)\big), \quad h>0.$$

It is estimated with Matheron's estimator, $\hat\gamma(h) = \frac{1}{2|N(h)|}\sum_{(i,j)\in N(h)}(z_i-z_j)^2$, and fitted with exponential, spherical and Matérn models by weighted least squares with weights $N(h)/\hat\gamma(h)^2$ (Cressie, 1985; Webster & Oliver, 2007).

Three numbers summarise it:

| Quantity | Meaning | GEMAS (all data) |
|---|---|---|
| Nugget $C_0$ | variance no neighbour can explain | 0.39 |
| Sill $C_0+C_1$ | total variance at large distances | 1.10 |
| Practical range $a$ | distance where $r(h)=0.05$ | 517 km (exponential fit) |
| Nugget ratio $\eta = C_0/(C_0+C_1)$ | share of variance that is pure noise at the sampling scale | 0.36 |

## 3. Ordinary Kriging and its error

Ordinary Kriging (OK) predicts $\hat Z(s_0)=\sum_i \lambda_i Z(s_i)$ with $\sum_i\lambda_i=1$, choosing $\lambda$ to minimise $\mathbb E[(\hat Z(s_0)-Z(s_0))^2]$. With $C$ the covariance matrix of the training data and $c_0$ the vector of covariances between the data and $s_0$, a Lagrange multiplier $\mu$ gives the system

$$\begin{bmatrix} C & \mathbf 1\\ \mathbf 1^\top & 0\end{bmatrix}\begin{bmatrix}\lambda\\ \mu\end{bmatrix} = \begin{bmatrix} c_0\\ 1\end{bmatrix}, \qquad \sigma^2_{OK}(s_0) = C(0) - \lambda^\top c_0 - \mu .$$

Equivalently,

$$\hat Z(s_0) = \hat\mu + c_0^\top C^{-1}(z-\hat\mu\mathbf 1),\quad \hat\mu=\frac{\mathbf 1^\top C^{-1}z}{\mathbf 1^\top C^{-1}\mathbf 1},\quad \sigma^2_{OK}=C(0)-c_0^\top C^{-1}c_0+\frac{(1-\mathbf 1^\top C^{-1}c_0)^2}{\mathbf 1^\top C^{-1}\mathbf 1}.$$

Here $C(0)=C_0+C_1$ because we predict a new *measurement*, which carries its own nugget.

**The key property:** $\sigma^2_{OK}(s_0)$ depends on the variogram and on *where* the training points are, but not on the measured values at the test point. So it can be computed before any test pH value is seen.

### Result 1: beyond the range, kriging becomes the mean

If every training point is farther than the range from $s_0$, then $c_0=0$. Solving the system gives $\lambda = C^{-1}\mathbf 1/(\mathbf 1^\top C^{-1}\mathbf 1)$ and $\mu=-1/(\mathbf 1^\top C^{-1}\mathbf 1)$, so

$$\hat Z(s_0)=\hat\mu, \qquad \sigma^2_{OK}(s_0)=C(0)+\frac{1}{\mathbf 1^\top C^{-1}\mathbf 1}.$$

OK then predicts the (generalised least squares) mean of the samples in the system, so it cannot beat that mean in regions that are far from all of them.

**In our implementation** the system uses only the 16–64 nearest training samples (`src/models.py: krige_local`), so $\hat\mu$ is a *local* mean, not the overall training mean used by the Mean baseline. Because pH varies gradually across Europe, the local mean can beat the overall mean far from the data. This is what we observe: on the spatial split, OK's skill is 0.17 for test samples more than 250 km from any training sample (n = 82) and 0.13 at 150–250 km (`results/errors/gemas_error_by_distance.csv`). Result 1 therefore bounds the gain over the *local* mean, not over the overall mean.

### Result 2: the nugget caps the gain

Take simple kriging (known mean) with one neighbour at distance $d$. The best weight is $\lambda = C_1 r(d)/(C_0+C_1)$ and

$$\frac{\text{MSE}_{\text{kriging}}}{\text{MSE}_{\text{mean}}} = 1-(1-\eta)^2\,r(d)^2 .$$

More generally, the noise $\varepsilon(s_0)$ is independent of every other observation, so for **any** predictor built from measurements at other locations

$$\mathbb E[(\hat Z(s_0)-Z(s_0))^2] = \mathbb E[(\hat Z(s_0)-m(s_0)-S(s_0))^2] + C_0 \;\ge\; C_0 .$$

Dividing by the mean-baseline error, which is roughly the sill, gives

$$\frac{\text{RMSE}_{\text{OK}}}{\text{RMSE}_{\text{mean}}} \gtrsim \sqrt{\eta}.$$

For GEMAS, $\sqrt{\eta}\approx 0.60$, so no coordinate-only method can reduce the RMSE by more than about 40%. Covariates can do better only if they explain part of what looks like nugget at the sampling scale.

**Worked numbers (GEMAS exponential fit, one neighbour):**

| Distance to nearest training point | $r(d)$ | MSE ratio $1-(1-\eta)^2 r^2$ | RMSE ratio |
|---|---|---|---|
| 43 km (typical under a random split) | 0.78 | 0.75 | 0.87 |
| 100 km (median under our spatial split) | 0.56 | 0.87 | 0.93 |
| 250 km (90th percentile, spatial split) | 0.24 | 0.98 | 0.99 |
| 517 km (practical range) | 0.05 | 1.00 | 1.00 |

So the same method can look useful or useless depending only on how far the test points are from the training points.

## 4. Trends: Universal and Regression Kriging

Universal Kriging (UK) uses $m(s)=\beta^\top x(s)$ with $x$ = coordinates; Regression Kriging (RK) fits the trend with covariates first, then kriges the residuals (Hengl et al., 2007). In our code both are implemented in two steps: fit the trend, then krige its out-of-fold residuals. For UK this approximates the one-step UK system. Splitting the error into a trend part and a residual part gives

$$\text{MSE}_{\text{UK/RK}} \approx \text{MSE}_{\text{trend}}\times \underbrace{\frac{\overline{\sigma^2_{OK,R}}}{C_{0,R}+C_{1,R}}}_{\text{residual kriging factor}},$$

where the residual kriging factor is Result 2 applied to the residual variogram. Two consequences:

1. The trend part does **not** depend on the distance to training points, so UK and RK keep some advantage beyond the range, where OK has none.
2. That advantage holds only if the trend is still valid where it is applied. A polynomial in coordinates fitted on one region can extrapolate badly into a held-out region. We therefore measure $\text{MSE}_{\text{trend}}$ by **inner spatial cross-validation** on the training data, not by the in-sample $R^2$.

GEMAS (all data): coordinates explain $R^2=0.20$ of pH variance, coordinates + climate/elevation explain $0.28$, and the residuals still have a nugget ratio of about 0.45, so some spatial structure remains after the trend.

## 5. Gaussian Process Regression is kriging

With a constant mean $m$, a covariance kernel $k$ and noise variance $\sigma_n^2$, the GP posterior mean is

$$\bar f(s_0) = m + k_0^\top(K+\sigma_n^2 I)^{-1}(y-m\mathbf 1).$$

Setting $k(h)=C_1 r(h)$ and $\sigma_n^2=C_0$ gives exactly the simple kriging predictor; estimating $m$ by generalised least squares gives Ordinary Kriging (Rasmussen & Williams, 2006, §2.7; Stein, 1999). So in theory GPR and OK differ only in how the covariance is chosen and fitted: GPR maximises the marginal likelihood, kriging fits the empirical variogram. Our implementations add two practical differences: GPR uses all ~1,400 training points and subtracts the plain training mean (`normalize_y`), while OK uses the 16–64 nearest points and re-estimates the mean locally. We therefore expect their accuracy to be close, and any gap tells us which fitting procedure suits the data better.

## 6. Why a random split can flatter spatial methods

Under a random split, test points sit among the training points, at roughly the sampling spacing. Under a spatial block split, whole regions are held out, and test points are farther away. For GEMAS (design: about one site per 50 × 50 km cell):

| Split | Median distance, test point → nearest training point | 90th percentile |
|---|---|---|
| Random (5-fold) | 43–44 km | 51–52 km |
| Spatial blocks (500 km, 5-fold) | 98–110 km | 203–247 km |

By Result 2, the random split tests kriging where $r(d)\approx0.78$, and the spatial split where $r(d)$ is between 0.56 and 0.24. The two splits answer different questions: the random split measures interpolation, the spatial split measures prediction into unseen regions (Roberts et al., 2017; Ploton et al., 2020). Wadoux et al. (2021) argue that spatial cross-validation can be *pessimistic* about map accuracy when the goal is a map of the whole region. Our goal is prediction in held-out regions, so the spatial split is the relevant test, and the random split is reported as a reference.

A side note on spacing: if GEMAS sites were placed at random with the same density ($\lambda\approx0.52$ per 1,000 km²), the expected nearest-neighbour distance would be $1/(2\sqrt\lambda)\approx 22$ km. The observed median is 42 km because GEMAS used a regular grid. Sites therefore never sit very close together, which limits how well the short-distance part of the variogram, and so the nugget, can be estimated.

## 7. From theory to a prediction before testing

For each outer fold we use **only training pH values and test locations** to compute

$$\widehat{\frac{\text{MSE}}{\text{MSE}_{\text{mean}}}} = \begin{cases}\overline{\sigma^2_{OK}(s_0)}/\text{sill} & \text{OK, GPR}\\[3pt] \dfrac{\text{CV-MSE}_{\text{trend}}}{\text{CV-MSE}_{\text{mean}}}\times\overline{\sigma^2_{OK,R}(s_0)}/\text{sill}_R & \text{UK, RK}\\[6pt] \dfrac{\text{CV-MSE}_{\text{model}}}{\text{CV-MSE}_{\text{mean}}} & \text{LinReg, RF}\end{cases}$$

using a local neighbourhood of the 64 nearest training points (code: `src/kriging_theory.py`, `scripts/04_hypotheses.py`). This turns "which method should win?" into a number per method, written down before the test set is used. Section 8 of the website and `results/verdicts.md` compare these numbers with the observed test errors. On GEMAS, the random split gave higher skill than the spatial split for every kriging method on average, but not consistently enough across folds for H6 to count as supported.

**Assumptions that can break the prediction:**
- *Stationarity:* one variogram for all of Europe. If the spatial structure differs between, say, Scandinavia and the Mediterranean, the kriging variance is wrong locally.
- *Gaussian-like errors:* kriging variance describes squared error well only if errors are roughly symmetric.
- *Trend transfer:* inner CV estimates how well the trend extrapolates between the training blocks. The held-out blocks may be harder or easier.
- *Independence of trend and residual errors* in the product formula of Section 4.

## References

- Cressie, N. (1985). Fitting variogram models by weighted least squares. *Mathematical Geology*, 17(5), 563–586. doi:10.1007/BF01032109
- Cressie, N. (1993). *Statistics for Spatial Data* (rev. ed.). Wiley. doi:10.1002/9781119115151
- Hengl, T., Heuvelink, G. B. M., & Rossiter, D. G. (2007). About regression-kriging: From equations to case studies. *Computers & Geosciences*, 33(10), 1301–1315. doi:10.1016/j.cageo.2007.05.001
- Matheron, G. (1963). Principles of geostatistics. *Economic Geology*, 58(8), 1246–1266. doi:10.2113/gsecongeo.58.8.1246
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11, 4540. doi:10.1038/s41467-020-18321-y
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. https://gaussianprocess.org/gpml/
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. doi:10.1111/ecog.02881
- Stein, M. L. (1999). *Interpolation of Spatial Data: Some Theory for Kriging*. Springer. doi:10.1007/978-1-4612-1494-6
- Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. *Economic Geography*, 46(sup1), 234–240. doi:10.2307/143141
- Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S., & Brus, D. J. (2021). Spatial cross-validation is not the right way to evaluate map accuracy. *Ecological Modelling*, 457, 109692. doi:10.1016/j.ecolmodel.2021.109692
- Webster, R., & Oliver, M. A. (2007). *Geostatistics for Environmental Scientists* (2nd ed.). Wiley. doi:10.1002/9780470517277
