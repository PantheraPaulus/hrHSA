# Theory

hrHSA is built around a simple ecological principle: selection is inferred from how observed use differs from a biologically defined distribution of alternatives. An observed location is evidence of use, but selection can only be defined relative to what could have been used instead. The meaning of a habitat-selection coefficient therefore depends as much on the definition of availability as on the environmental predictor itself.

This principle provides a common theoretical foundation for resource-selection functions (RSFs), step-selection functions (SSFs), and integrated step-selection functions (iSSFs). The models differ principally in the alternatives against which observed use is compared. RSFs define availability over a broader spatial domain, SSFs restrict alternatives to endpoints reachable from the current movement state, and iSSFs additionally estimate components of the movement process that generates those alternatives.

A useful generic representation is

$$
p_U(z \mid \mathcal A)
=
\frac{
a(z)\,w(z)
}{
\int_{\mathcal A} a(r)\,w(r)\,dr
},
$$

where $a(z)$ describes the distribution of available alternatives and

$$
w(z)
=
\exp\{\eta(z)\}
$$

describes their relative weighting under the fitted ecological model. Habitat-selection analysis can therefore be understood as reweighting availability.

The scale and interpretation of $z$, $a(z)$, and $\eta(z)$ change across model classes:

| Model | Ecological question | Availability | Statistical comparison |
|---|---|---|---|
| RSF | Which environmental conditions or areas are used disproportionately? | Broad spatial domain | Used locations versus sampled availability |
| SSF | Which reachable endpoint is chosen next? | Movement-constrained alternatives from the current state | One observed endpoint versus local alternatives |
| iSSF | How do habitat and environmental conditions affect both choice and movement? | Movement-generated alternatives with proposal correction | Local choice with jointly estimated selection and movement effects |

Frequentist and Bayesian inference do not represent different ecological theories. They provide alternative inferential frameworks for estimating the same selection and movement relationships. Bayesian formulations are particularly useful when individual heterogeneity, hierarchical structure, regularization, or propagation of uncertainty into derived predictions are central to the scientific question.

## Part I — Selection, availability, and scale

### 1. Selection is use relative to availability

Animals do not merely occur in environments. Through movement, settlement, foraging, territoriality, avoidance, and social interaction they continually redistribute themselves among alternatives. The resulting spatial pattern is therefore the outcome of behavioural decisions acting within environmental and social constraints (Fretwell & Lucas, 1969; Rosenzweig, 1981; Johnson, 1980).

This makes habitat organism-specific. A land-cover category or raster cell is not intrinsically selected, avoided, favourable, or poor. Its ecological meaning depends on the resources, conditions, risks, competitors, and movement constraints experienced by the focal organism, and these relationships may change with season, life-history stage, behavioural state, or individual identity.

A telemetry relocation provides evidence of use. It does not by itself demonstrate selection, preference, or habitat quality. Selection is disproportionate use relative to availability. In the strict sense, preference describes what would be chosen if alternatives were equally available, whereas observational telemetry studies generally estimate selection under unequal and constrained availability (Johnson, 1980; Manly et al., 2002; Lele et al., 2013).

For discrete habitat classes $h=1,\ldots,H$, let $u_h$ denote the proportion of observed use and $a_h$ the corresponding proportion available. A simple selection ratio is

$$
r_h
=
\frac{u_h}{a_h}.
$$

Values greater than one indicate use in excess of availability, whereas values below one indicate use below availability. A common habitat can consequently contain many animal locations while still being avoided relative to its abundance, and a rare habitat can contain few relocations while being strongly selected.

The same logic extends to continuous environmental gradients. A selection model estimates how the observed distribution is reweighted relative to the available distribution. Availability therefore determines the reference measure against which every fitted effect is interpreted.

### 2. Scale, accessibility, and the observation process

Johnson's (1980) hierarchy of selection made the scale dependence of habitat selection explicit. He distinguished four nested orders:

1. first-order selection of the geographical range of a species;
2. second-order placement of an individual or social group's home range within that range;
3. third-order use of habitat components within the home range; and
4. fourth-order selection of particular resources or sites during specific activities.

These orders are best understood as an ecological hierarchy rather than a rigid statistical classification. Modern telemetry resolves movement decisions at temporal scales far finer than those originally envisaged, but the central insight remains unchanged: selection is only meaningful relative to the alternatives appropriate to the decision scale.

The same environmental feature can therefore appear selected at one scale and avoided at another. A mountain system may be selected during home-range establishment while steep slopes within the established range are avoided during routine movement. Apparent contradictions among studies can arise simply because availability was defined at different spatial, temporal, or behavioural scales.

At broad scales, accessibility may be approximated by a home range, population range, or study domain. At finer scales, the current location, elapsed time, movement capacity, landscape permeability, and recent movement direction strongly constrain what can be reached next. The progression from RSF to SSF and iSSF is therefore not merely an increase in statistical complexity. It represents an increasingly explicit description of the alternatives available to an animal.

Telemetry introduces an additional observation layer. Let the true continuous trajectory of individual $i$ be $S_i(t)$, while the recorded telemetry consists of observations

$$
D_i
=
\left\{
\tilde{s}_{ij},
t_{ij}
\right\}_{j=1}^{n_i}.
$$

The observed trajectory is a filtered representation of the underlying movement process. Fix interval, positional error, missed fixes, habitat-dependent acquisition success, residence time, and temporal aggregation can all influence the recorded data.

Conceptually,

```text
ecological process
        ↓
continuous movement and space use
        ↓
telemetry observation process
        ↓
recorded relocations
        ↓
availability and statistical model
```

A habitat-selection model is therefore conditional not only on the ecological definition of availability but also on the spatial and temporal resolution at which movement was observed. Increasing model complexity cannot automatically recover ecological processes that were not resolved by the observation design.

### 3. Selection, density, competition, and fitness

Observed space use does not translate directly into habitat quality. Classical habitat-selection theory makes clear that environmental productivity, competition, territoriality, density, and access can jointly determine where animals occur.

Under the Ideal Free Distribution of Fretwell and Lucas (1969), let $N_h$ denote density in habitat $h$ and let

$$
\Phi_h(N_h)
$$

represent expected per-capita fitness. Competition implies

$$
\frac{\partial\Phi_h(N_h)}{\partial N_h}
<
0.
$$

At equilibrium, occupied habitats provide equal expected payoffs,

$$
\Phi_h(N_h^\ast)
=
\lambda,
$$

so a productive habitat can contain a greater number of individuals without yielding greater realized fitness per animal. High use or high density is therefore not equivalent to high individual performance.

The Ideal Despotic Distribution further relaxes the assumption of free access. Dominant individuals can monopolize profitable territories and exclude subordinates, such that observed distribution reflects both environmental quality and socially mediated accessibility. Similar complications arise from territoriality, source-sink dynamics, density dependence, predation risk, and behavioural specialization.

This distinction motivated Van Horne's (1983) warning that density can be a misleading indicator of habitat quality. Habitat-selection coefficients likewise quantify redistribution relative to the specified alternatives; they do not establish that selected environments improve survival or reproduction. Linking selection to demographic quality requires survival, reproduction, population growth, or another explicitly modelled fitness outcome. hrHSA therefore treats environmental selection as a behavioural or spatial relationship conditional on the ecological and social context represented by the data rather than as an intrinsic property of the landscape.

## Part II — Models of spatial choice

### 4. Resource-selection functions

RSFs describe relative use across a broader availability domain. Let $\Omega$ denote that domain, $f_A(s)$ the spatial distribution of available locations, and $x(s)$ the environmental predictor vector at location $s$. An inhomogeneous point-process representation is

$$
\lambda(s)
=
c\,f_A(s)\,
w\{x(s)\},
$$

where

$$
w(x)
=
\exp(x^\mathsf{T}\beta)
$$

is the resource-selection function and $c$ controls total intensity.

Conditional on the number of observed relocations, the implied distribution of use is

$$
f_U(s)
=
\frac{
f_A(s)\,
w\{x(s)\}
}{
\int_{\Omega}
f_A(r)\,
w\{x(r)\}\,dr
}.
$$

This expression contains the central interpretation of an RSF: the fitted selection function **reweights the distribution of available space**. Its absolute scale is arbitrary; relative contrasts are the primary inferential quantity.

#### Use-availability approximation

In practice, the denominator is commonly approximated by drawing locations from the available distribution and combining them with observed locations,

$$
Y_j
=
\begin{cases}
1, & \text{used location},\\
0, & \text{sampled available location}.
\end{cases}
$$

A logistic model can then be fitted,

$$
\operatorname{logit}
\left[
P(Y_j=1\mid x_j)
\right]
=
\alpha
+
x_j^\mathsf{T}\beta.
$$

The sampled available locations are not biological absences. Their number is chosen by the analyst to approximate the environmental distribution of availability. Consequently, the intercept depends on the used-to-available sampling ratio and generally has no interpretation as an absolute probability of occurrence. With sufficiently dense availability sampling, the slope estimates approximate those of the underlying point-process formulation (Johnson et al., 2006; Warton & Shepherd, 2010; Aarts et al., 2012).

#### Relative selection strength

For two environmental conditions $x_1$ and $x_0$,

$$
\operatorname{RSS}(x_1,x_0)
=
\frac{w(x_1)}{w(x_0)}
=
\exp
\left[
(x_1-x_0)^\mathsf{T}\beta
\right].
$$

For a one-unit difference in a single linear predictor,

$$
\operatorname{RSS}
=
\exp(\beta_j),
$$

holding all remaining model terms constant.

When models contain quadratic terms or interactions, individual coefficients no longer describe a constant marginal response. Explicit contrasts or predicted response curves are then generally more interpretable than reading coefficients independently.

A spatial RSF prediction,

$$
\hat{w}\{x(s)\}
=
\exp
\left[
x(s)^\mathsf{T}\hat{\beta}
\right],
$$

is therefore a map of relative selection under the fitted availability design. It is not automatically a probability of occupancy, residence, survival, or habitat quality.

### 5. Step-selection functions

At sufficiently fine temporal scales, broad spatial availability becomes biologically implausible. A location may fall within an animal's home range while remaining impossible to reach during the interval between two telemetry fixes. SSFs therefore redefine availability locally.

An observed step joins consecutive relocations,

$$
s_t
\longrightarrow
s_{t+1},
$$

with step length

$$
L_t
=
\left\|
s_{t+1}-s_t
\right\|
$$

and turning angle $\theta_t$ relative to the previous movement direction.

For each observed endpoint, alternatives are generated from the same starting location using a movement-informed proposal. The resulting choice set is

$$
C_t
=
\left\{
s_{t+1}^{(0)},
s_{t+1}^{(1)},
\ldots,
s_{t+1}^{(K)}
\right\},
$$

where candidate $0$ denotes the observed endpoint.

For candidate-specific predictors $x_{tj}$,

$$
P(j\mid C_t)
=
\frac{
\exp(x_{tj}^{\mathsf T}\beta)
}{
\sum_k
\exp(x_{tk}^{\mathsf T}\beta)
}.
$$

This is the local analogue of the RSF formulation. Instead of reweighting a broad spatial availability density, the model reweights the alternatives in the current movement-constrained choice set.

No stratum-specific intercept is required because adding the same constant to all candidate utilities cancels from the softmax.

#### Environmental conditions and movement state

Candidate endpoints can be annotated with static environmental conditions as well as fields that vary through time. The ecological timing must remain explicit. Endpoint conditions describe the environment into which a candidate moves, whereas departure conditions describe the state from which movement begins.

Directional environmental fields require particular care. For example, a wind vector can be projected onto each candidate's movement bearing. A positive along-track component represents tailwind support, a negative component headwind, and a perpendicular component crosswind. Even when the same wind vector is measured at the beginning of a step, its projection can vary among candidate bearings and therefore contribute identifiable choice information.

#### Selection opportunity

A strong biological response cannot be estimated precisely when the animal rarely encountered contrasting alternatives. For predictor $k$, the information supplied by stratum $s$ is related to

$$
I_{s,k}
=
\operatorname{Var}_{p_s}
\left(
x_{s,j,k}
\right),
$$

where the variance is weighted by the fitted choice probabilities.

Low variation among alternatives therefore limits information about selection. An uncertain coefficient can arise because an animal encountered little contrast, not because the underlying preference was necessarily weak.

This distinction is particularly important when comparing individuals. Differences in estimated selection can arise from

$$
\text{response}
\quad+\quad
\text{opportunity}
\quad+\quad
\text{sampling uncertainty}.
$$

Selection opportunity is thus the local analogue of availability: animals cannot reveal choices among environmental conditions they did not encounter as meaningful alternatives.

### 6. Integrated step-selection functions

A conventional SSF uses an estimated movement distribution to generate alternatives and then estimates environmental selection conditional on those alternatives. However, the observed step-length and turning distributions used to construct availability have themselves already been shaped by habitat and behaviour.

Integrated step-selection analysis addresses this circularity by modelling movement and habitat selection jointly while retaining the distribution used to generate candidate steps as a proposal mechanism (Avgar et al., 2016).

#### Proposal and ecological movement model

Let

$$
q_{sj}
$$

denote the density under the proposal used to generate candidate $j$ in stratum $s$. A proposal-corrected utility can be written as

$$
\eta_{sj}
=
x_{sj}^{\mathsf T}\beta
-
\log q_{sj}.
$$

The term

$$
-\log q_{sj}
$$

is an importance-sampling correction rather than an ecological coefficient. It separates the distribution used computationally to draw alternatives from the movement process estimated by the model.

This distinction is fundamental:

```text
proposal distribution
        ↓
generates candidate alternatives

ecological model
        ↓
reweights those alternatives
        ↓
selection + movement process
```

The proposal can consequently be chosen for efficient and representative sampling without being mistaken for the final ecological movement kernel.

#### Movement basis

A commonly useful movement basis contains

$$
L,
\qquad
\log L,
\qquad
\cos\theta.
$$

Consider the step-length contribution

$$
\gamma_L L
+
\gamma_{\log L}\log L.
$$

Combined with the proposal correction, these terms can imply a Gamma-like step-length kernel with shape and rate related to

$$
k
=
1+\gamma_{\log L},
\qquad
\lambda
=
-\gamma_L,
$$

under the corresponding parameterization and provided that the resulting kernel is proper. Expected displacement is then

$$
E[L]
=
\frac{k}{\lambda}.
$$

The turning term $\cos\theta$ controls directional persistence, with increasing positive coefficients favouring continuation in the previous direction.

These quantities describe displacement between telemetry fixes. They should not be interpreted as the complete distance travelled along the unobserved continuous path between fixes.

#### Environmental modification of movement

Environmental conditions can modify the movement kernel through interactions with step length, log step length, or turning angle. Heat, terrain ruggedness, snow depth, wind support, or behavioural state may therefore affect how far or how directionally an animal moves.

A predictor that is constant for all candidates in a stratum cannot contribute an identifiable standalone effect because it cancels from the conditional likelihood. It can nevertheless be estimated through interactions with candidate-varying movement terms. Conversely, directional quantities such as wind support can vary among candidate headings even when measured from a common departure location.

The ecological consequences are often nonlinear. If an environmental predictor modifies both $L$ and $\log L$, a simple additive interaction on the utility scale can imply a complex change in the complete step-length distribution. Derived quantities such as expected displacement, turning persistence, and movement-response curves are therefore usually more meaningful than interpreting isolated interaction coefficients.

## Part III — Population inference, uncertainty, and prediction

### 7. Individuals, hierarchical models, and Bayesian inference

Telemetry datasets typically contain many observations from comparatively few animals. Thousands of relocations from one individual do not provide thousands of independent population replicates. Population inference must therefore distinguish repeated observations within animals from variation among animals.

A hierarchical representation can express individual-specific coefficients as

$$
\beta_i
=
\mu_\beta
+
b_i,
\qquad
b_i
\sim
\mathcal N(0,\Sigma_\beta).
$$

Here,

$$
\mu_\beta
$$

describes the population-average response, while

$$
\Sigma_\beta
$$

describes variation and covariance among individual responses.

This distinction is scientifically important. Sex, age, reproductive state, experience, dominance, behavioural specialization, local environmental exposure, or population membership can all produce persistent differences among individuals. Between-individual variation should therefore not automatically be treated as statistical noise.

#### Complete pooling, no pooling, and partial pooling

Three limiting approaches illustrate the advantage of hierarchical modelling.

Under complete pooling, all animals share one coefficient,

$$
\beta_i=\mu_\beta.
$$

This provides a population estimate but assumes that animals are exchangeable with no meaningful heterogeneity.

Under no pooling, each animal is estimated independently,

$$
\beta_i
\;\text{independent across }i.
$$

This allows heterogeneity but discards information shared across animals and can produce unstable estimates for poorly sampled individuals.

Hierarchical models provide partial pooling. Individual effects are estimated jointly from a population distribution,

$$
\beta_i
\sim
\mathcal N(\mu_\beta,\Sigma_\beta),
$$

so the degree of pooling is determined by the information in the data. Well-informed individuals can deviate strongly from the population mean, whereas weakly informed animals are shrunk toward the population distribution.

This shrinkage is not an arbitrary penalty. It follows from the hierarchical probability model and reflects uncertainty about individual effects.

Conceptually,

```text
individual data
      ↘
       individual effect
      ↗        ↑
population distribution
```

The population informs individuals, while the individual data jointly inform the population distribution.

#### Bayesian hierarchical inference

Bayesian inference represents uncertainty through the joint posterior distribution. For observations $D$, latent individual effects $Z$, and population parameters $\Theta$,

$$
p(Z,\Theta\mid D)
\propto
p(D\mid Z,\Theta)
\,
p(Z\mid\Theta)
\,
p(\Theta).
$$

In a hierarchical selection model, this posterior can simultaneously describe

- population-average selection or movement responses;
- individual-specific coefficients;
- between-individual heterogeneity;
- covariance among individual responses;
- uncertainty in every level of the hierarchy; and
- uncertainty propagated into predictions and derived ecological quantities.

For a non-centred hierarchical coefficient,

$$
\beta_{ip}
=
\mu_p
+
\sigma_p z_{ip},
\qquad
z_{ip}\sim\mathcal N(0,1),
$$

$\mu_p$ describes the average effect of predictor $p$, $sigma_p$ describes heterogeneity among individuals, and $beta_{ip}$ describes the partially pooled response of individual $i$.

These quantities answer different ecological questions and should not be collapsed into a single coefficient table.

A credible population effect does not imply that all animals respond similarly. Conversely, substantial $\sigma_p$ can indicate meaningful ecological heterogeneity even when the population mean is close to zero.

#### Shrinkage and regularization

Partial pooling is one form of Bayesian shrinkage: weakly informed individual effects are drawn toward the population distribution.

A second form operates across predictors. Large environmental datasets can contain many correlated or weakly identified candidate covariates. Weakly informative priors can stabilize estimation, while explicit shrinkage priors can concentrate weak effects toward zero without imposing a hard inclusion/exclusion decision.

For example, the horseshoe family can be represented schematically as

$$
\beta_j
\sim
\mathcal N
\left(
0,
\tau^2\lambda_j^2
\right),
$$

where $\tau$ is a global shrinkage scale and $\lambda_j$ is a predictor-specific local scale. Most coefficients can be strongly regularized through the global scale while large local scales allow strongly supported effects to remain comparatively unshrunk.

The regularized horseshoe modifies the extreme tails to improve stability in finite data settings (Piironen & Vehtari, 2017).

Such priors are particularly useful when the candidate set is larger than a tightly pre-specified ecological model. They should not, however, be interpreted as automatic causal variable selection. Local shrinkage parameters are not posterior inclusion probabilities, and correlated predictors can trade against one another while jointly representing an ecological signal.

The role of shrinkage is therefore to stabilize estimation and express prior information about plausible model complexity, not to replace ecological reasoning.

#### Why Bayesian inference is useful in spatial ecology

The principal advantages of Bayesian inference in hrHSA are not specific to any one model class.

First, hierarchical structure is represented directly. Population effects, individual effects, and heterogeneity can be estimated jointly rather than through separate two-stage analyses.

Second, uncertainty propagates naturally. A derived movement quantity such as expected displacement can be computed for every posterior draw,

$$
E[L]^{(m)}
=
g\left(
\beta^{(m)}
\right),
$$

yielding a posterior distribution on the ecological scale rather than relying only on an approximation around a point estimate.

Third, nonlinear prediction respects parameter uncertainty. Because

$$
E_i
\left[
\exp(x^\mathsf T\beta_i)
\right]
\neq
\exp
\left[
x^\mathsf T E_i(\beta_i)
\right],
$$

a prediction from the population-average coefficient vector is not generally equivalent to the average prediction across heterogeneous individuals.

Fourth, posterior distributions allow direct probability statements about ecological quantities, subject to the model and prior assumptions. Examples include the posterior probability that a population response is positive, that heterogeneity exceeds an ecologically relevant threshold, or that expected displacement differs between environmental conditions.

Finally, prior distributions provide explicit regularization. This is particularly valuable for hierarchical models, correlated environmental predictors, and limited numbers of individuals, where unconstrained estimation can become unstable.

Bayesian inference does not remove the need for careful availability definitions, sufficient environmental contrast, representative individuals, or model checking. A posterior distribution quantifies uncertainty conditional on the specified model; it does not automatically account for ecological mechanisms omitted from that model.

### 8. Dependence, validation, and predictive uncertainty

Successive observations from a tracked animal are serially dependent. That dependence is not merely a statistical nuisance. Residence, movement persistence, site fidelity, and repeated return are components of the biological process being studied.

Arbitrarily thinning telemetry until observations appear independent can therefore discard ecologically relevant information. Validation should instead reflect the level at which prediction is intended.

Different validation questions correspond to different estimands:

$$
\text{new observation}
\neq
\text{new time period}
\neq
\text{new stratum}
\neq
\text{new individual}
\neq
\text{new population}.
$$

A method that evaluates one of these targets does not automatically validate the others.

#### Transfer among individuals

Leave-one-individual-out validation asks whether a relationship estimated from the remaining animals predicts a new individual. It is therefore particularly relevant when the scientific goal is population-level generalization.

For hierarchical models, this prediction problem differs fundamentally from prediction for an individual already represented in the hierarchy. A known animal can use its partially pooled posterior effect, whereas a genuinely new animal must be predicted from the population distribution.

#### Temporal validation

Contiguous temporal blocks address stability through time. They can reveal seasonal change, behavioural transitions, environmental non-stationarity, or periods for which the fitted relationship performs poorly.

Randomly mixing temporally adjacent observations between training and validation data can produce optimistic estimates because nearby observations share both environmental conditions and movement history.

#### RSF validation

For RSFs, the continuous Boyce index compares held-out use with available conditions along the predicted selection gradient. If $P_k$ is the proportion of observed held-out locations in prediction interval $k$ and $E_k$ the corresponding available proportion,

$$
R_k
=
\frac{P_k}{E_k},
$$

and the Boyce index measures the rank association between prediction and $R_k$.

A high value indicates that high predicted selection corresponds to disproportionately high held-out use. The complete predicted-to-expected curve remains important because a single correlation coefficient can conceal the parts of the prediction range in which a model fails.

#### SSF and iSSF validation

For a conditional choice model, validation naturally occurs on the choice scale. If the observed candidate in stratum $s$ has fitted probability $p_{\mathrm{model}}(y_s)$, a useful gain relative to uniform choice is

$$
G_s
=
\log
p_{\mathrm{model}}(y_s)
-
\log
\left(
\frac{1}{J_s}
\right),
$$

where $J_s$ is the number of alternatives.

Positive gain indicates performance better than uniform selection among the candidate set.

Held-out strata, held-out time blocks, held-out individuals, and approximate leave-one-out diagnostics on already-fitted observations answer different questions and should not be treated as interchangeable measures of model quality.

#### Posterior predictive assessment

Bayesian models additionally permit posterior predictive checking. For posterior draw $m$,

$$
\theta^{(m)}
\sim
p(\theta\mid D),
$$

a replicated dataset can be generated from

$$
D_{\mathrm{rep}}^{(m)}
\sim
p(D\mid \theta^{(m)}).
$$

The resulting replicated distributions can be compared with the observed data through quantities relevant to the ecological model: choice probabilities, environmental distributions of selected endpoints, movement lengths, turning angles, spatial-use patterns, or other summary statistics.

Posterior predictive agreement is not evidence that a model is uniquely correct. It asks whether the fitted model can reproduce important features of the observations. Systematic discrepancies indicate aspects of the process not adequately represented by the current formulation.

#### Sources of uncertainty

Validation uncertainty and parameter uncertainty should remain conceptually distinct.

In a frequentist blocked bootstrap with the fitted model held fixed, variation among resamples describes uncertainty due to the finite validation sample conditional on the fitted coefficients.

In a Bayesian analysis, posterior draws can additionally be propagated through each validation replicate. The resulting distribution then incorporates both

$$
\text{parameter uncertainty}
+
\text{finite validation-sample uncertainty}.
$$

Neither automatically accounts for structural uncertainty caused by an incorrect availability definition, omitted predictor, observation bias, or incorrect model class.

### 9. Prediction, interpretation, and choosing an analytical route

The natural prediction target differs among RSF, SSF, and iSSF models.

| Model | Natural prediction | Conditional on | Not automatically |
|---|---|---|---|
| RSF | Relative selection across space | Defined availability domain and environmental model | Absolute occurrence probability |
| SSF | Relative probability of choosing a local endpoint | Current state and candidate choice set | Long-term space-use distribution |
| iSSF | Local choice and environmentally modified movement kernel | Current state, proposal correction, and fitted movement-selection model | Habitat quality or fitness |

For an RSF, predictions compare relative selection across a defined environmental domain. They can be normalized within that domain, but the resulting values remain conditional on the chosen availability distribution.

For an SSF, probabilities are local to one choice set. A candidate can receive high probability because it is favourable relative to alternatives reachable from the current location, even if the corresponding environment is not globally rare or valuable.

For an iSSF, the fitted model additionally describes how environmental conditions change movement. Derived quantities such as expected step length, turning persistence, or a complete step-length distribution often provide more direct ecological interpretation than raw movement-interaction coefficients.

Longer-term quantities such as utilization distributions, residence times, crossing probabilities, connectivity, and home-range geometry emerge from repeated movement decisions rather than from one transition in isolation. In principle they are obtained by propagating or simulating the fitted transition process over time.

None of these predictions is automatically a measure of habitat quality or fitness. Demonstrating that selected conditions improve survival, reproduction, or population growth requires demographic outcomes or an explicit fitness model.

A practical analytical sequence is therefore:

1. **Define the ecological decision scale.** What spatial or temporal choice is being represented?
2. **Define availability before fitting.** Selection has no interpretation without its reference distribution.
3. **Use an RSF** when a broader spatial availability domain is defensible and relative space use is the target.
4. **Use an SSF** when local reachability between consecutive observations is central.
5. **Use an iSSF** when movement itself is part of the ecological question or environmental conditions are hypothesized to modify displacement or turning.
6. **Use hierarchical inference** when population-level response and between-individual variation are both scientific targets.
7. **Use Bayesian inference** when partial pooling, regularization, posterior uncertainty, or uncertainty propagation into nonlinear predictions is important.
8. **Validate according to the intended prediction target.** Prediction for a new stratum, new time period, new individual, and new population are different problems.
9. **Interpret results on their natural scale.** Relative selection, local choice probability, expected displacement, density, habitat quality, and fitness are not interchangeable.

The central theoretical progression can therefore be summarized as

```text
animals encounter alternatives
            ↓
availability defines the comparison
            ↓
environment and movement reweight those alternatives
            ↓
individual responses vary
            ↓
inference estimates population and individual structure
            ↓
validation tests the intended level of generalization
            ↓
prediction is interpreted on the scale of the fitted process
```

## Principal references

Aarts, G., Fieberg, J. & Matthiopoulos, J. (2012). Comparative interpretation of count, presence-absence and point methods for species distribution models. *Methods in Ecology and Evolution*, **3**, 177–187.

Avgar, T., Potts, J. R., Lewis, M. A. & Boyce, M. S. (2016). Integrated step selection analysis: bridging the gap between resource selection and animal movement. *Methods in Ecology and Evolution*, **7**, 619–630.

Charnov, E. L. (1976). Optimal foraging, the marginal value theorem. *Theoretical Population Biology*, **9**, 129–136.

Fretwell, S. D. & Lucas, H. L. (1969). On territorial behavior and other factors influencing habitat distribution in birds. *Acta Biotheoretica*, **19**, 16–36.

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2013). *Bayesian Data Analysis*. 3rd ed. CRC Press.

Johnson, C. J., Nielsen, S. E., Merrill, E. H., McDonald, T. L. & Boyce, M. S. (2006). Resource selection functions based on use-availability data: theoretical motivation and evaluation methods. *Journal of Wildlife Management*, **70**, 347–357.

Johnson, D. H. (1980). The comparison of usage and availability measurements for evaluating resource preference. *Ecology*, **61**, 65–71.

Lele, S. R., Merrill, E. H., Keim, J. & Boyce, M. S. (2013). Selection, use, choice and occupancy: clarifying concepts in resource selection studies. *Journal of Animal Ecology*, **82**, 1183–1191.

MacArthur, R. H. & Pianka, E. R. (1966). On optimal use of a patchy environment. *American Naturalist*, **100**, 603–609.

Manly, B. F. J., McDonald, L. L., Thomas, D. L., McDonald, T. L. & Erickson, W. P. (2002). *Resource Selection by Animals: Statistical Design and Analysis for Field Studies*. 2nd ed. Kluwer Academic Publishers.

Morris, D. W. (2003). Toward an ecological synthesis: a case for habitat selection. *Oecologia*, **136**, 1–13.

Muff, S., Signer, J. & Fieberg, J. (2020). Accounting for individual-specific variation in habitat-selection studies: efficient estimation of mixed-effects models using Bayesian or frequentist computation. *Journal of Animal Ecology*, **89**, 80–92.

Northrup, J. M. et al. (2022). Conceptual and methodological advances in habitat-selection modeling: guidelines for ecology and evolution. *Ecological Applications*, **32**, e02470.

Piironen, J. & Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, **11**, 5018–5051.

Rosenzweig, M. L. (1981). A theory of habitat selection. *Ecology*, **62**, 327–335.

Van Horne, B. (1983). Density as a misleading indicator of habitat quality. *Journal of Wildlife Management*, **47**, 893–901.

Vehtari, A., Gelman, A. & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, **27**, 1413–1432.

Warton, D. I. & Shepherd, L. C. (2010). Poisson point process models solve the “pseudo-absence problem” for presence-only data in ecology. *Annals of Applied Statistics*, **4**, 1383–1402.