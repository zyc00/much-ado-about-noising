# Paper Outline

**Structural principles**

1. Claims and evidence are interleaved. There is no standalone "Experiments" section. Every mechanism claim is immediately followed by the experiment that tests it.
2. The mechanism carries the argument. The method is an instantiation of the mechanism, not the contribution.
3. Section 2 and Section 7 use the same constructed task family. Give it a name in Section 2 and reuse that name in Section 7.
4. Section 5 states one mechanism (the effective gradient weight profile), not two co-equal ingredients.
5. **Escalating impact via a prediction ladder.** The paper's five predictions form an ascending sequence of stakes: 5.5 contamination (self-designed synthetic), 6.3 ph/mh (public benchmark), 6.5 late decline (our own weakness as the prediction target), 6.6 scale (external data we do not control), 7.1 our own failure (direction included). At each one, state the stake — what it would mean if the result came out the other way — *before* revealing the result. Five bets, five payoffs, monotonically rising. Never disguise a post-hoc observation as a member of this ladder; one exposed fake collapses the whole sequence. If something was only observed, label it an observation.
6. **Facts become inferences.** What Section 2 presents as a brute fact (a dataset where MSE wins) is re-presented in Section 7 as a derivation the reader can perform themselves. The same object, seen twice with different eyes, is what makes the closing loop land — the name and the back-reference are only the signposts.

**Reader test before drafting.** Show the abstract to someone unfamiliar with the area and ask them to state the contribution in one sentence. If they say "the authors propose a new loss," the framing has failed. If they say "the authors argue generative modeling is not necessary for robot policy learning," it has succeeded.

---

## Title and Abstract

**Working title:** *Multimodality Hurts Training, Not Execution: An Optimization Account of Behavior Cloning Objectives*

- Main title is the claim, word-for-word matching the paper's spine; "execution" answers MIP's "execute well" directly. Subtitle carries the searchable keywords (behavior cloning, objectives).
- Backup: *The Cost of Multimodality Is Paid at Training Time* (more vivid, loses the execution counterpoint).
- The title must not contain the method name.
- **Title-sequence test:** reading only the section and subsection titles in order should reproduce the full argument. If a title does not read naturally in that sequence, fix the title, not the story.
- The abstract ends with three or four numbered findings. The first one is a mechanism claim, not a method claim.
- Suggested finding set:
  1. Multimodality in human demonstrations does not prevent regression policies from executing well; it prevents them from training well.
  2. The harm is an allocation effect: large-loss multimodal regions suppress the fitting of precision-critical segments.
  3. Correcting the effective gradient weight profile is sufficient. No iterative denoising, no anchor, no distribution model.
  4. Which objective is optimal is a property of the data, not of the method. We give constructions where each of MSE, MIP, and ours wins.

---

## 1. Introduction

- Generative policies win on robot manipulation. The community explanation is that they model multimodality.
- Prior work (Much Ado About Noising) argues multimodality is not the reason, and supports this with an inference-time argument. It does not provide a pure regression solution.
- Our position, stated as the paper's spine: **multimodality does affect policy learning, but through optimization rather than expressivity. It is not why regression cannot execute well; it is why regression cannot train well.**
- Keep the execute/train contrast in its explicit form. It is the sharpest sentence in the paper.
- End the introduction by announcing the reframing, so Section 2 arrives expected rather than abrupt. One sentence, e.g.: *"Before asking why regression fails, we first establish that 'which objective is best' has no data-independent answer, and must be asked conditionally on the data."*
- Numbered contribution list matching the abstract.

---

## 2. No Free Lunch in Objective Choice

*(Header in paper: "No Free Lunch in Objective Choice". Alt: "The Optimal Objective Depends on the Data".)*

**Purpose.** Establish existence, not a criterion. This section buys the license used in Section 7.

- Introduce the 2D funnel to dock construction. Name it here (e.g. *nuisance-jitter docking*) and reuse the name in Section 7.
- Result: there exist data-generating processes on which plain MSE outperforms Diffusion Policy and Flow.
- State explicitly that this is not an adversarial construction. Zero-mean, asymmetric, time-varying operator jitter is an intrinsic component of teleoperated demonstration data.
- State explicitly that a full criterion for predicting the winner from data properties is an open problem, and that Section 7 returns to it.
- Do not introduce HT here. It has not been defined yet, and its failure case belongs in Section 7 where the reader can interpret it.

**Entering the section.** Open by hooking the community's default assumption, not by presenting the construction: *"Debates over policy objectives are usually conducted as rankings. We first show that no data-independent ranking exists."* The reader is told their belief is at stake before seeing the evidence.

**Name the decisive variable.** State explicitly that the switch flipping the outcome is the structure of label variation: whether it is nuisance or strategy, equivalently whether the conditional mean is itself a valid action. Section 4 opens by pointing back at exactly this variable, which is what carries the thread across the Preliminaries section: *"Section 2 showed that the outcome hinges on the structure of label variation. We now measure that structure where it matters: in human demonstrations."*

**Preempt the reflexive objection in-section.** A reader finishing this section will ask: then aren't your robomimic results also just a product of your data family? Answer it here, not in Section 6: *"Consequently, this paper does not argue that any objective is universally best. It gives an account of which data properties decide the outcome, and of what they imply in the regime that matters in practice: human-collected demonstrations."* This turns NFL from self-inflicted into the paper's registered position.

**Leaving the section: forward map.** Close with three or four sentences, each pointing at a later section:

> The construction shows that the winner is decided by a property of the data, not of the objective. This turns the question "which objective is best" into "which objective does this data demand." The remainder of the paper answers it for human demonstrations: Section 4 measures what this data actually looks like, Section 5 identifies the mechanism through which its structure interacts with the training objective, and Section 7 returns to the construction introduced here, this time with our own objective included.

The last sentence is a promissory note. The reader knows the construction will return with our own method inside it, and Section 7 closes a loop the reader has been holding open.

**Size cap.** Three-quarters of a page plus one figure. This section buys a license; it is not a main result, and it must not compete with Section 5. The more ceremonially the forward map promises, the more heavily Sections 4 and 5 must deliver.

---

## 3. Preliminaries

*(Not "Problem Setup" — this section also carries the paper-wide experimental setup, and "Preliminaries" accommodates that.)*

This section carries the reproducibility load for the whole paper, since there is no separate experiments section.

- Chunked behavior cloning notation.
- MSE as the conditional mean, equivalently isotropic-Gaussian MLE.
- Benchmarks, backbones, evaluation protocol, training budget.
- State the protocol lineage: the main table follows the DP and MIP evaluation protocols so that cited baseline numbers remain comparable, and baselines are taken from published results rather than retuned.

---

## 4. The Structure of Multimodality in Human Demonstrations

Open with the back-reference that picks up Section 2's thread across the Preliminaries valley: *"Section 2 showed that the outcome hinges on the structure of label variation. We now measure that structure where it matters: in human demonstrations."*

### 4.1 Multimodality is constrained by the structure of the action space

- An action sequence is a trajectory in SE(3) x gripper, parameterized by time. Variation can therefore only enter through two channels: **path geometry** and **time parameterization**.
- Path geometry: direction, orientation, amplitude.
- Time parameterization: rate along the path (speed) and phase of discrete events (timing).
- This is a constructive decomposition of the channels. The relative dominance of components within each channel is measured, not derived. Say this. Do not claim the component list is exhaustive.
- Measurement result: speed is the most common and most pronounced, followed by direction and timing.

### 4.2 Multimodality concentrates in large-magnitude actions

- Stratify the conditional action distribution by action magnitude.
- This subsection exists to supply the premise Section 5 consumes. If it does not deliver "large-loss region," Section 4 is not load-bearing and should be cut.
- Note the internal structure carefully: directional multimodality is sharpest in the slow tercile while the fast tercile is nearly unimodal directionally. Reconcile this with the speed-multimodality claim explicitly, or a careful reader will find the tension.

---

## 5. Why Regression Fails to Train: An Optimization Account

*(Subsection titles below are deliberately chained: 5.1's object is 5.2's subject; 5.3 and 5.4 read as one sentence across two lines — "mode averaging is not the failure mechanism / misallocated gradient weight is". If 5.4's fragment form feels risky, fall back to "The failure is misallocated gradient weight".)*

**Open this section with a three or four sentence roadmap linking 5.1 through 5.5.** Reviewers writing their summary will draw on this paragraph. It is the cheapest insurance against readers losing the thread inside an interleaved structure.

### 5.1 Multimodal regions carry irreducible loss

- Loss distribution over the dataset.
- Gradient magnitude distribution, stratified by region.

### 5.2 Irreducible loss crowds out precision-critical segments

- This is the causal core of the paper and needs the strongest evidence in it.
- Toy demonstration of the effect in a controlled setting.
- Fitting analysis on real data: precision-segment residuals as a function of the loss mass carried by multimodal regions.

### 5.3 Mode averaging is not the failure mechanism

- Bimodal-state probe on human tool-hang: at 279 of 500 bimodal states, MSE, HT, and MIP all sit near a mode rather than the midpoint, with no separation between objectives.
- This is a negative result and should be presented as one. It removes the explanation most readers will assume, which is what makes the rest of the section necessary.

### 5.4 Misallocated gradient weight is

- **State a single mechanism.** Different objectives induce different effective per-sample gradient weights; the profile, not the likelihood family, determines the outcome.
- Ladder, presented as evidence for the axis rather than as a component ablation:
  - MSE: flat weights, pays everywhere.
  - HG: corrects across-sample scale, still flat within a sample. Improves but is insufficient.
  - HT: adds residual-dependent decay. Profile complete.
- One sentence closing the "is this just a robust loss" attack surface:
  > To verify that the weight profile rather than the distributional model is responsible, we replace the entire likelihood with a fixed detached weight proportional to the inverse residual, with no scale head and no Student-t. This reaches 0.84 / 0.78 on human tool-hang, comparable to HT (Appendix X).
- If this sentence is omitted, add a statement in the discussion that Student-t is not claimed to be the only realization, otherwise the question is left open.

### 5.5 The account predicts, and the prediction holds

- Post-hoc action-label contamination benchmark. Chunk-coherent lateral residual: 0 with 70 percent, +0.2 with 20 percent, +1.6 with 10 percent, so the mean is +0.2.
- **State the stake before the reveal** (first rung of the prediction ladder): if the objectives did not separate here as the weight-profile account requires, the account would be indistinguishable from a generic robustness story.
- Predicted and observed: MSE and HG converge to approximately +0.2; MIP step 1 lands at +0.2 and step 2 reduces the bias but does not remove it; HT rejects both positive branches and recovers approximately 0.
- This is the paper's independence from MIP. It must appear in the main text, not the appendix.

---

## 6. A Minimal Instantiation and Its Validation

Keep the tone restrained throughout. The objective is an instantiation of Section 5, not a proposal.

### 6.1 A single-evaluation objective

- Single forward pass at t=0 with zero action input. Two heads: action and raw scale. One network evaluation per action versus one hundred.
- Loss definition, the role of the log-scale normalizer, and the one hyperparameter setting used everywhere.
- State that a single fixed configuration is used across all tasks and backbones.

### 6.2 Benchmarks under standard protocols

- Main table over robomimic and Push-T, state and image.
- Write the caption in terms of what the table tests, not what it shows. For example: if the account in Section 5 is correct, reweighting objectives should help more where multimodality is stronger.

### 6.3 A directional prediction: mixed versus proficient operators

- The theory predicts a larger relative advantage on multi-human data than on single-proficient-human data.
- **State the stake before the reveal:** *"Had the advantage been larger on single-operator data, the account in Section 5 would be wrong at its core."* The reader is told this shot could miss, then sees that it did not.
- Report the ph/mh delta explicitly rather than leaving it implicit in the table.
- This also converts any single-task deficit on clean ph data from a weakness into an expected outcome.

### 6.4 Controlled comparison at matched capacity

- Controlled comparison at matched model size, batch size, and hyperparameters. Distinguish it clearly from the main table, which uses published numbers under each method's own tuning.
- State plainly that baselines here fall below their published numbers because they are not retuned, and that this is expected.
- Report epochs to reach a fixed fraction of final success rate as a scalar, alongside the curves.
- Clarify that the horizontal axis is training cost and the network-evaluation counts in the legend are inference cost, and that the two are independent.
- Note that per-epoch training cost is comparable across arms, so epochs are a fair proxy.

### 6.5 Late-training decline follows from the same mechanism

- On some tasks the objective peaks and then declines, while MIP does not.
- Present this as following from Claim 5.4 rather than as an unexplained artifact: an objective that down-weights large residuals progressively narrows its own effective training set as residuals shrink. Fast fitting and late narrowing are two faces of the same weight profile. MIP restructures via the anchor and has no equivalent self-reinforcing loop.
- Supporting measurement, if available: entropy of the effective sample-weight distribution over training, and whether its inflection coincides with the onset of the decline.
- With this account in place, the curves can be shown to the full budget rather than truncated.

### 6.6 Prediction at scale: heterogeneity amplifies the gap

- Frame this as a prediction test, not a confirmation that the method also works at scale.
- Prediction, written down before reporting: as demonstration data becomes larger and more heterogeneous, the relative advantage should grow. This places ph, mh, and large-scale heterogeneous data on one monotone sequence.
- Report the margin against the sequence, not only the absolute success rate.
- Known limit to state: only the proposed arm is trained here, with baselines taken from published numbers, so this leg establishes effectiveness at scale but not a controlled convergence comparison. Consider a short matched baseline run covering the early part of the curve.

### 6.7 Real-robot validation

- Task, platform, number of trials, and what it is intended to establish.

---

## 7. Where Our Objective Fails, and Why

Do not name this section "Limitations." It is the final prediction being tested, not a concession.

Subsection titles:
- **7.1 The same construction, on scripted robot data** (the comma is deliberate — it marks the return to Section 2)
- **7.2 Objectives as positions on a mean-to-mode axis** (the knob framing; this is the interface left for the controllable-generator follow-up)
- **7.3 Choosing the position from the data** (the open criterion problem, stated as a direction rather than a bare "Outlook")

- Open by pointing back to Section 2: we argued at the outset that no objective is universally optimal, and we now apply the same construction to our own.

**7.1 must be written predict-then-reveal, not result-then-explain.** Do not open with the outcome table. First, a paragraph in which the reader computes the result themselves from Section 5's account:

> Before running anything, the account in Section 5 tells us what must happen here. The label distribution places 80 percent of its mass on one branch; a mode-seeking objective will lock onto it and inherit its bias, while the mean is, by construction, the correct action. Our objective is mode-seeking. It should fail here, and it should fail toward the dominant branch.

Only then the table, with results matched to the prediction item by item. This is the peak of the paper's impact curve: not "the authors admit their method loses," but "I, the reader, can already derive where it loses." The theory running inside the reader's own head is worth more than any number. In Section 2 this construction was a brute fact; here it is a derivation. Same object, different eyes — that conversion, not the back-reference, is what closes the loop.

- Transfer the named construction from Section 2 to scripted robot data, and add HT and MIP to the comparison.
- Result and account: where the label distribution is nuisance jitter with a valid convex mean, mean-seeking objectives are correct and mode-seeking objectives, including ours, are biased. The failure is predicted by the mechanism, not discovered after the fact.
- Note that the construction holding at both the 2D toy level and the scripted robot level is itself a result: the phenomenon is not an artifact of low dimensionality.

**Fire the planted guns by name.** When 7.2 introduces the axis, point at where it was planted: *"the variable identified in Section 2 — whether label variation is nuisance or strategy — is precisely the axis on which every objective in this paper sits."* Readers are acutely sensitive to "that earlier detail was preparation for this," but only if reminded it is the same object. The other planted guns: MIP's "execute well" (fired by the title and 5.3), and the late-training decline (fired by 5.4 via 6.5).

**Outlook**

- Every objective in this comparison carries a knob that positions it along the mean-seeking to mode-seeking axis: the anchor noise scale for MIP, the degrees of freedom for Student-t, the noise schedule for diffusion. The open question is not which method to pick but where on that axis the data says to sit.
- Extending this to model and data scaling: as data grows more heterogeneous, the appropriate position on the axis should shift, and this is measurable.
- Keep the stance sentence (also a candidate for the abstract's last line): understand the properties of human demonstration data and choose backbone and objective accordingly, rather than scaling on noisy data with an unsuitable backbone and objective and no analysis. Place it just before the closing paragraph, not after it.

**Closing paragraph: a sentence-level mirror of the introduction, four moves, then stop.** Recall the community assumption verbatim, replace it, compress the account, flip the question:

> We began with the assumption this field has worked under: that generative policies win because they model multimodality. The account given here replaces it. Multimodality decides how gradient weight is allocated during training; objectives differ in where they sit on a single axis; and where to sit is a property of the data. The question is no longer which objective is best. It is what your data demands.

The last sentence doubles as the interface to the controllable-generator follow-up.

**Two anti-patterns that kill the impact curve:**

1. No recap before or inside Section 7. "As we have shown..." summaries do the reader's inference for them at exactly the moment they should be doing it themselves. Refer back with pointers (*Section 5's account*), never with restatements.
2. Stop immediately after the flip sentence. No limitation list after it (Section 7 has already absorbed limitations — that is the point of the structure), no miscellaneous discussion. The paper's final period lands on the flipped question; one more paragraph and the resonance is gone.

---

## Appendix

- Hyperparameter sweeps: degrees of freedom, scale bias.
- Inverse-residual reweighting control: full setup and results.
- Table aligned to the MIP 50-episode protocol.
- Full training configurations and per-cell budgets.
- Additional benchmark results.

---

## Pre-drafting checklist

- [ ] Claim 5.4 is written as a single mechanism, not two co-equal ingredients. Nothing should be drafted before this is settled, since every paragraph in Section 5 depends on it.
- [ ] The Section 6.6 prediction is written down before the results are reported.
- [ ] Section 4 contains an explicit causal link to Section 5's premise. If Section 4 can be deleted without breaking Section 5, it is not yet load-bearing.
- [ ] The construction has one name, used in both Section 2 and Section 7.
- [ ] The abstract passes the one-sentence reader test.
- [ ] Any phenomenon the mechanism does not explain is flagged as an open observation rather than left for a reviewer to find.
