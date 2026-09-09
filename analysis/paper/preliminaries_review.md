# Preliminaries review

## Outline and paragraph roles

1. Setup: general behavior cloning, followed by the direct MSE regression
   baseline and its equation before the first subsection, as in the user's revision.
2. Background: group diffusion and Flow under denoising-based policy learning,
   and restore the common multimodality explanation: generative policies
   represent alternative actions, whereas direct MSE can average them.
3. Established evidence: describe the effectively unimodal action distributions
   and mean-action evaluation reported in Pan et al.'s Flow-policy probes,
   followed by their deterministic-expert dataset comparison.
4. Established evidence: introduce deterministic MIP's performance, including
   its squared-error supervision, noisy training, and two-evaluation inference.
5. Quantitative support and takeaway: replace the table with the published
   seven-task aggregate from Pan et al., Figure 1. Conclude that regression
   can match Flow without modeling multiple action modes, without deriving
   our central research question from MIP.
6. M-estimation and robust regression: define a residual penalty, connect
   negative log-likelihood to MSE, and explain robustness through the
   derivative with respect to the predicted action.
7. Heteroscedastic regression: define input-dependent scale and Gaussian
   likelihood, distinguish scale adaptation from the residual penalty, and
   transition to the data section.

The three subsections are: (1) multimodality and deterministic policies;
(2) M-estimation and robust regression; and (3) heteroscedastic
regression. Paragraph-role comments are included in the LaTeX source.
Student-t and the proposed HT objective remain in the later method section.

Following the user's positioning instruction, subsection 1.1 answers the
multimodality explanation and establishes the viability of regression-based
control; it is not the origin story for our method.
The research-question paragraph and detailed MIP training narrative are
omitted. Our central question and contribution framing belong in the
Introduction; they are not newly drafted or relocated by this scoped edit.
The skill's claim-evidence review keeps the two findings separate: the
existence of a successful deterministic policy is established by published
task success, not inferred from a Gaussian fit to another policy's output.
The former standalone multimodality section is superseded; its files and raw
probe results are retained.

## Source and numerical verification

Source: Pan et al., *Much Ado About Noising: Dispelling the Myths of Generative
Robotic Control*, [arXiv v3](https://arxiv.org/html/2512.01809v3).

- The current excerpt uses the user's `pan2026much` key and ICLR 2026
  metadata, verified against the
  [official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9299ca9ed58731945e934adb5b71728c-Abstract-Conference.html).
  The old `pan2025muchado` bibliography entry remains available for older drafts.
- `robomimic2021` supplies behavior-cloning background. Appendix D.4 of the
  [study](https://arxiv.org/html/2108.03298v1) describes its direct-prediction
  ablation. It is not cited as a comparison against diffusion or Flow.
- `chi2023diffusionpolicy` supplies denoising-versus-conventional-BC evidence, using the RSS
  citation from the [project page](https://diffusion-policy.cs.columbia.edu/).
  It is not presented as a controlled direct-MSE comparison. RoboMimic's 2021
  arXiv citation follows its project-provided citation and the user's key.
  Chi et al.'s multimodal-action motivation supports the restored explanation;
  it is introduced as an explanation in the literature, not our data finding.
- Section 3.2, Figure 3 reports single clusters in action-sample visualizations
  at symmetry-critical or ambiguous states in Push-T, Kitchen, and Tool-Hang.
  Its mean-action evaluation gives success rates 0.95, 0.97, and 0.76, versus
  0.97, 0.99, and 0.80 for stochastic sampling. The manuscript describes
  these as findings from the authors' probes, not a universal unimodality
  theorem or a Gaussian fit. The longer quotation from Section 4.3 is removed.
- Section 3.2 also compares the original data with demonstrations re-collected
  from a Flow policy executed deterministically with zero initial noise.
  The Flow/regression success rates are 0.78/0.58 on the original data and
  0.72/0.64 on the deterministic-expert data. This is additional evidence
  against multimodality as the explanation, not an implication drawn solely
  from visualizing sampled actions.
- MIP's noisy training input, supervised second prediction, and deterministic
  two-evaluation inference are specified in Section 4.1, Equations 4.4–4.5.
- Current inline comparison: Figure 1 (right) reports average relative success
  across the authors' seven most challenging tasks: MIP 1.02, Flow 1.00,
  and Regression 0.74. Verified against the text and rendered first page of
  the [v3 PDF](https://arxiv.org/pdf/2512.01809v3). These are Flow-normalized
  averages, not absolute success rates, percentage-point differences, or
  an average over all 28 benchmarks. The text explicitly states the task
  scope and relative normalization. This is the authors' published aggregate,
  not a new aggregate calculated from the previously selected table cells.
- Historical audit of the removed table: every value came from Table 12,
  using best-checkpoint success multiplied by 100 and averaged across three
  seeds. The original cells are retained below for recovery.
- Removed Table 12 excerpt, in MSE, MIP, Flow order, as percentages:
  Chi-Transformer Transport-mh (28, 42, 44), Transport-ph (68, 80, 88),
  Tool-Hang (40, 76, 68); Chi-UNet Transport-mh (22, 62, 52),
  Transport-ph (64, 80, 80), Tool-Hang (68, 80, 84).
  All 18 entries were checked against the HTML table and PDF text on page 35.
- Transport-ph is an additional dataset setting, not a third distinct task.
  Section B.1 defines PH as proficient-human demonstrations and MH as mixed
  proficient/non-proficient demonstrations. The removed table illustrated
  settings where direct regression lags, not a representative benchmark average.
- MSE and MIP are deterministic at inference. MIP also uses squared-error
  supervision, so the paper explicitly defines "plain MSE regression" before
  comparing it with MIP. MIP is not described as a different loss family.
- Regression/MIP/Flow use 1/2/9 network evaluations, respectively (Sections
  2 and 4.1). The condensed paragraph retains MIP's two evaluations to avoid
  conflating deterministic inference with single-pass prediction.
- This comparison establishes a performance gap between learning procedures;
  it does not isolate objective choice from iterative computation. Subsection
  1.1 reports this comparison without introducing our research question.

Statistical background:

- Huber (1964), [*Robust Estimation of a Location Parameter*](https://doi.org/10.1214/aoms/1177703732):
  pp. 74–75 introduce the residual-penalty definition of M-estimation, its
  likelihood special case, and the quadratic/linear Huber penalty. The
  original paper was read through a
  [scan transcription](https://www.scribd.com/document/861209215/Huber-RobustEstimationLocation-1964)
  because the publisher endpoint did not expose the article text. The
  bibliography links to the original article DOI, not the mirror.
- Nix and Weigend (1994),
  [*Estimating the Mean and Variance of the Target Probability Distribution*](https://doi.org/10.1109/ICNN.1994.374138):
  Section III.A derives Gaussian negative log-likelihood with input-dependent
  variance and shows how constant variance recovers squared-error training.
  Section IV.A discusses the resulting inverse-variance weighting. Verified
  against the [original paper PDF](https://francesco215.github.io/autoregressive_diffusion/website/nix1994.pdf).
- Seitzer et al. (2022),
  [*On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks*](https://arxiv.org/html/2203.09168):
  Section 2 gives the conditional Gaussian model, its NLL, and its mean
  gradient. This is cited for the heteroscedastic formulation, not as
  evidence that Student-t resolves its optimization pitfalls.

## Claim–evidence map

- Claim: Direct action regression is a behavior-cloning baseline. | Evidence:
  RoboMimic's algorithm description and direct-prediction ablation.
  | Status: supported.
- Claim: Diffusion policies have demonstrated strong behavior-cloning
  performance. | Evidence: Chi et al.'s simulation and real-robot results.
  | Status: supported; no direct-MSE causal comparison is attributed to them.
- Claim: Denoising-based policies have outperformed conventional BC baselines
  on challenging manipulation tasks. | Evidence: Chi et al.'s comparisons
  with LSTM-GMM, BeT, and IBC, and Pan et al.'s direct-MSE/Flow comparisons.
  | Status: supported as an empirical policy comparison, not proof that the
  training objective alone causes the gap. The opening uses the broader
  baseline category and does not introduce our research question.
- Claim: Pan et al. report effectively unimodal distributions in their
  Flow-policy probes. | Evidence: Section 3.2, Figure 3 and the mean-action
  evaluation. | Status: supported as an attributed empirical finding, not
  a universal statement about all Flow policies or demonstration datasets.
- Claim: The multimodality explanation does not account for the demonstrated
  Flow/regression gap. | Evidence: Pan et al.'s sampled-action probes,
  mean-action intervention, and persistent gap with deterministic-expert
  demonstrations (Section 3.2). | Status: supported for the cited experiments;
  not a claim that multimodal data never occur in robot learning.
- Claim: A successful deterministic policy can be learned. | Evidence: MIP's
  deterministic definition and published task success. | Status: supported.
- Claim: Deterministic MIP is competitive with Flow, while direct regression
  has lower aggregate performance on the cited challenging-task subset.
  | Evidence: Figure 1's published average relative success rates of 1.02
  (MIP), 1.00 (Flow), and 0.74 (Regression), across seven challenging tasks.
  | Status: supported as a qualitative performance comparison, not a
  statistical equivalence claim or a claim that MIP wins every setting.
- Claim: Regression-based policies can achieve Flow-level performance without
  modeling multiple action modes. | Evidence: MIP's deterministic regression
  construction and published task success. | Status: supported as an existence
  result; it does not guarantee that plain MSE or a single-pass policy learns
  equally well. The prose says "can match Flow," not "MSE must work."
- Claim: The gap is caused by data scale or tails. | Evidence: later dataset
  measurements and controlled interventions are required. | Status: needs
  evidence; not asserted in the preliminaries. The section ends by indicating
  which data properties will be examined next.
- Claim: M-estimation includes likelihood-based residual objectives and
  least squares. | Evidence: Huber (1964), pp. 74–75; substituting the Gaussian
  log-density gives the displayed MSE objective up to fixed scale/constants.
  | Status: supported.
- Claim: Squared error has a residual derivative that grows linearly, whereas
  coordinate-wise Huber penalties have bounded derivatives. | Evidence:
  direct differentiation and Huber's quadratic/linear definition.
  | Status: supported for the gradient with respect to the prediction.
- Claim: A scalar input-dependent Gaussian scale gives inverse-variance
  weighting plus a log-scale term. | Evidence: Nix and Weigend (1994),
  Section III.A; Seitzer et al. (2022), Section 2; the d-dimensional isotropic
  Gaussian normalization gives the factor d in Eq. 3. | Status: supported.
- Claim: Heteroscedastic Gaussian regression retains a quadratic penalty
  in the standardized residual. | Evidence: Eq. 3 written in terms of
  r/sigma. | Status: supported for a given predicted scale; this is not a
  claim about a likelihood profiled over scale.
- Claim: These training objectives permit deterministic inference.
  | Evidence: the defined predictor f_theta maps an input directly to an
  action; sampling is not required by the training loss. | Status: supported
  by the policy construction, without a new task-success claim.

## Mathematical and implementation consistency

- The new density discussion explicitly concerns continuous channels.
  Binary-like grippers are not put into a Gaussian model. Continuous GR1
  hand joints remain in scope.
- The Gaussian interpretation of MSE is a likelihood equivalence, not a
  claim that Gaussian noise is necessary for learning a conditional mean.
  Non-Gaussianity alone does not establish that MSE cannot learn.
- The residual is prediction minus target throughout. Consequently the
  derivative of the squared norm is 2r, consistent with Eq. 1 (which has
  no factor of one half).
- For a d-dimensional Gaussian with covariance sigma^2 I, the log-determinant
  term is d log(sigma), not log(sigma) or d log(sigma^2). The remaining
  constant is d/2 log(2 pi). Differentiating with respect to the mean gives
  r/sigma^2; differentiating with respect to log(sigma) gives
  d - ||r||^2/sigma^2.
- Sigma is one scalar per input/chunk, consistent with the local HT
  implementation guide. This background does not introduce a separate
  per-coordinate variance head or claim to model full covariance.
- A bounded residual derivative is not a proof of a globally bounded neural
  parameter gradient: the network Jacobian also contributes. The draft
  therefore says "prediction gradient," not "influence function" or a
  guarantee of stable training.
- The log-scale term penalizes scale inflation; it does not guarantee
  freedom from variance collapse or optimization pathologies.
- The distinction between scale and tails is conceptual background. To
  motivate HT later, tail evidence should remain after local centering and
  scale normalization, and method benefit requires experimental evidence.

## Five-dimension self-review

- Contribution: pass for positioning. Prior work is presented as background,
  with no "MIP therefore motivates our central question" transition. The
  Introduction must establish the paper's own data-centered contribution.
- Writing clarity: pass. The section moves from setup to established
  evidence, statistical background, and then the data analysis. Subsection
  1.1 answers the restored explanation and ends with a regression-viability
  conclusion, not a research question. The robust-gradient explanation and
  closing transition are shortened without changing their definitions.
  The data transition discusses extreme residuals without declaring all
  large errors to be noise. No Gaussianity premise is inferred from Flow.
- Experimental strength: pass for attribution and reporting. The inline
  numbers reproduce a published aggregate, with its relative normalization
  and seven-task scope stated explicitly.
- Evaluation completeness: pass for this motivating comparison. New evidence
  is required later to establish the mechanism and the proposed objective.
- Method soundness: pass. Determinism is distinguished from single-pass
  inference; MIP's squared-error supervision is explicit. The statistical
  background separates the residual penalty from input-dependent scale,
  defines their likelihood connection, and does not presuppose HT's benefit.

### Adversarial self-review questions

1. Contribution: does the background make our work read as a MIP extension,
   or present established regression tools as our novelty? No. The MIP-based
   research-question transition is removed, both tools are attributed, and
   our HT construction and empirical contribution remain in later sections.
2. Writing clarity: can a reader identify the three subsection roles and
   distinguish scale adaptation from robustness? Yes. Each has a direct
   opening sentence, and the final paragraph explicitly connects both
   statistical choices to the next data section.
3. Experimental strength: are new empirical claims made without evidence?
   No. The added Flow summary is checked against Pan et al., Section 3.2;
   the table is replaced by their Figure 1 aggregate. No new findings about
   our data are claimed, and no selective table-cell average is used.
4. Evaluation completeness: does the preliminary establish that HT closes
   the control-performance gap? No, and it does not claim to. Data probes
   motivate the model; controlled main experiments must demonstrate benefit.
5. Method design soundness: are normalization, residual sign, dimension,
   channel scope, and gradient interpretation consistent? Yes, as checked
   above. No implication of "non-Gaussian therefore unlearnable" is used.

Reverse outlining checked flow, terminology, and the source of each claim.
The table and its LaTeX label are removed from the current section as
requested; its numerical contents remain in the historical audit above.
The inline comparison cites the source's Figure 1, not an internal figure
or table number. No font shrinking, negative spacing, or template changes
are used.
The preview was compiled with the unmodified ICLR 2027 style and
visually inspected. No overfull boxes, undefined references, or font
substitutions remain. With all three subsections, the excerpt now occupies
one page of text, with references on a separate second preview page.
The wrapper forces the
reference-page break for inspection; the section source adds no forced
page breaks or spacing overrides. Tectonic reports its existing bibliography
rerun-consistency warning, but all citations resolve in the final PDF.
