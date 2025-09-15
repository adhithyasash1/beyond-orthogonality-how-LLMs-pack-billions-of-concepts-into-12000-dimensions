# beyond orthogonality: how language models pack billions of concepts into 12,000 dimensions

this repo contains a simple python demo and explanation inspired by nicholas yoder’s article *“beyond orthogonality: how language models pack billions of concepts into 12,000 dimensions”* (feb 2025) (https://nickyoder.com/johnson-lindenstrauss/). the goal is to make the ideas accessible both **numerically** and **visually**.

---

## background

large language models like gpt-3 use embeddings with around **12,000 dimensions**. at first glance, that might seem too small to encode the billions of nuanced concepts that humans deal with.

but high-dimensional geometry has surprising properties:

* vectors don’t need to be perfectly orthogonal (90° apart).
* *quasi-orthogonality* (angles close to 85–95°) allows many more vectors to coexist.
* the **johnson–lindenstrauss lemma (jl lemma)** shows we can compress high-dimensional points into much smaller spaces while approximately preserving distances.

together, these insights explain how embeddings can represent so much meaning in relatively modest dimensions.

---

## what this repo demonstrates

the included notebook/code illustrates three main ideas:

### 1. quasi-orthogonality in high dimensions

* we generate random vectors in a 100-dimensional space.
* by measuring pairwise angles, we see that most vectors naturally fall close to 90°, even though they are not perfectly orthogonal.
* visualization: a histogram of angles with a reference line at 90°.

### 2. the johnson–lindenstrauss lemma in action

* we create 500 random vectors in 1,000 dimensions.
* we project them down to 50 dimensions using a random gaussian projection.
* we then compare pairwise distances before vs. after projection.
* visualization: a histogram of distance ratios, showing that most distances are well-preserved.

### 3. embedding capacity calculator

* based on the formula from the article:

`vectors ≈ 10^(k * f² / 1500)` 

where:  
- `k` = embedding dimension (e.g., 12,288 for gpt-3)
- `f` = degrees of “freedom” from orthogonality (90° – angle)
- the calculator estimates how many concepts can fit at different tolerances.

* examples for gpt-3 (12,288d):

  * at 89° (f=1): \~10^8 concepts
  * at 88° (f=2): \~10^32 concepts
  * at 87° (f=3): \~10^73 concepts
  * at 85° (f=5): \~10^200 concepts
* even conservative cases exceed the estimated number of atoms in the observable universe (\~10^80).

---

## why this matters

* **capacity is not the bottleneck.** embedding spaces already have room for far more concepts than humans could ever need. the real challenge is learning **optimal arrangements** of these concepts.
* **random projections are powerful.** they provide simple, efficient ways to reduce dimensionality without complex optimization.
* **language models thrive on structured overlap.** embeddings don’t require perfect independence; instead, concepts share space in nuanced, useful ways.

---

## how to run

1. clone the repo and install dependencies:

`git clone <your-repo-url>\ncd <repo>\npip install -r requirements.txt`

2. run the script or open the notebook:

`python embedding_geometry_demo.py`

3. you’ll see:

* a histogram of vector angles in 100d.
* a histogram of distance distortions after jl projection.
* numeric output showing average angle, distance distortions, and estimated embedding capacities.

---

## dependencies

* numpy
* matplotlib
* scikit-learn

---

## references

* nicholas yoder, *beyond orthogonality: how language models pack billions of concepts into 12,000 dimensions* (2025).
* johnson & lindenstrauss, *extensions of lipschitz mappings into a hilbert space* (1984).
* 3blue1brown, *transformers* video series (2025).
