# CART, Ensemble Learning, and Boosting

This guide explains the supervised-learning methods available in the
Radiology Correlation System's algorithm lab. The examples use a binary
prediction: whether a target finding is **PRESENT (1)** or **ABSENT (0)**.

## 1. CART decision trees

**CART** means *Classification and Regression Trees*. The algorithm lab uses
its classification form: a tree asks feature-based yes/no questions and
predicts PRESENT or ABSENT at a final leaf.

### How it works

1. Start with all training reports in one group (the root).
2. Try splitting a feature at candidate thresholds. For a numeric feature,
   candidates are the midpoints between its distinct sorted values.
3. Measure how mixed the labels are before and after each split.
4. Choose the split that reduces the mixture the most.
5. Repeat for each child group until the maximum depth is reached or a group
   contains only one class.

The tree predicts the majority class in a leaf. In a tie, this implementation
predicts PRESENT.

### Gini impurity and split calculation

Gini impurity measures how mixed a group is:

```text
Gini = 1 - (p_present^2 + p_absent^2)
```

For two classes, this is also `2 × p_present × p_absent`. A group containing
only one class has impurity 0; a half-and-half group has impurity 0.5.

To score a split, calculate the weighted average impurity of its two child
groups, then subtract that from the parent impurity:

```text
weighted child impurity =
    (left group size / parent size) × left Gini
  + (right group size / parent size) × right Gini

Gini reduction = parent Gini - weighted child impurity
```

**Example:** Suppose a parent has 3 PRESENT and 3 ABSENT reports. Its Gini is
`1 - (3/6)^2 - (3/6)^2 = 0.5`. A candidate split puts all 3 ABSENT reports on
the left and all 3 PRESENT reports on the right. Each child has Gini 0, so the
weighted child impurity is 0 and the Gini reduction is `0.5 - 0 = 0.5`.
That is a perfect split.

In the implementation, sample weights are included in these class proportions
and group sizes. This matters when CART is being used as an AdaBoost learner.

### Reading a tree rule

A rule such as `feature <= threshold` sends a report left when it is true and
right otherwise. Continue following the questions until reaching a leaf such
as `Predict PRESENT`. Limiting tree depth makes the rules shorter and can
reduce overfitting, though a tree that is too shallow may miss useful patterns.

## 2. Ensemble learning

**Ensemble learning** combines several models to make one prediction. The
intuition is that multiple models can be more reliable than one model when
their errors are not all the same.

The algorithm lab offers two ensemble approaches:

- **Random Forest** uses bagging: trees are trained independently on
  bootstrapped samples, then their votes are combined.
- **AdaBoost** uses boosting: small trees are trained in sequence, with later
  trees paying more attention to examples earlier trees got wrong.

### Random Forest: bagging and voting

For each tree, the implementation:

1. Draws a training set of the same size as the original training set, with
   replacement. This is a **bootstrap sample**, so some reports may appear
   more than once and others may be left out.
2. Builds a CART tree up to the selected maximum depth.
3. At each split, considers a random subset of features (approximately the
   square root of the total feature count).

The trees predict independently. For a binary task, their predictions are
combined by majority vote:

```text
PRESENT votes = 7; ABSENT votes = 3  ->  predict PRESENT
```

This implementation predicts PRESENT if at least half the trees vote
PRESENT, so an exact tie also predicts PRESENT. The reported feature
importance is the average of the trees' importance scores.

**Why it can help:** Each tree sees a slightly different sample and selection
of features. Averaging their votes can make the result less sensitive to one
particular training sample or tree.

## 3. Boosting: AdaBoost

**Boosting** builds models one after another. Each new model is encouraged to
focus on training reports that the existing models find difficult. In this
algorithm lab, AdaBoost uses shallow CART trees called **decision stumps**
(maximum depth 1).

### How AdaBoost works here

1. Give every training report the same initial weight.
2. Fit a decision stump using those weights.
3. Calculate the stump's weighted error: add the weights of the reports it
   classified incorrectly.
4. Give the stump a vote strength, or **alpha**:

   ```text
   alpha = 0.5 × ln((1 - error) / error)
   ```

   A lower error gives a larger alpha, so a more accurate stump gets a
   stronger vote.
5. Increase the relative weights of reports the stump got wrong and decrease
   the relative weights of reports it got right. Normalize the weights so
   they sum to 1.
6. Repeat, up to the requested number of stumps. Stop early if a stump is
   perfect or if no stump performs better than chance.
7. Combine the stumps' predictions using their alpha-weighted votes.

For the update, labels and predictions are represented as `+1` for PRESENT
and `-1` for ABSENT:

```text
new weight = old weight × exp(-alpha × true_label × prediction)
normalized weight = new weight / sum of all new weights
```

The product `true_label × prediction` is `+1` for a correct prediction and
`-1` for an incorrect one. Therefore, the update reduces a correct example's
weight and increases an incorrect example's weight.

### Small AdaBoost calculation

Suppose there are 4 equally weighted reports, so each starts with weight
`1/4 = 0.25`. A stump gets 1 report wrong:

```text
weighted error = 0.25
alpha = 0.5 × ln((1 - 0.25) / 0.25)
      = 0.5 × ln(3)
      ≈ 0.549
```

Before normalization, each of the 3 correctly classified reports gets weight
`0.25 × exp(-0.549) ≈ 0.144`; the incorrectly classified report gets
`0.25 × exp(0.549) ≈ 0.433`. The total is about `0.866`. After dividing by
that total, each correct report has weight about `0.167` and the incorrect one
has weight `0.5`. The next stump therefore pays more attention to the
misclassified report.

At prediction time, a PRESENT stump contributes `+alpha` and an ABSENT stump
contributes `-alpha`:

```text
score = sum(alpha × stump prediction)
score >= 0  ->  PRESENT
score < 0   ->  ABSENT
```

If no stump beats chance on the training split, the implementation instead
predicts the training majority class. The number of estimators reported is
the number of stumps actually used, which can be smaller than requested.

## 4. How to choose between them

| Method | Best understood as | Main advantage | Trade-off |
|---|---|---|---|
| CART | One sequence of decision rules | Easy to inspect and explain | A single tree can be sensitive to the training data |
| Random Forest | Many varied trees vote together | Often steadier than one tree | Harder to explain as one simple rule set |
| AdaBoost | A sequence of weighted, simple trees | Focuses successive learners on difficult examples | Can be affected by noisy or mislabeled examples |

## 5. K-means clustering in this project

The app's **K-means & PCA** option groups reports with similar finding
patterns. It reads the bundled
`data/processed/iu_xray_nlp_labels.csv` file and uses the selected finding
names as features. It does not use a target label or train/test split:
K-means is an unsupervised method, so it groups by similarity rather than
learning a PRESENT/ABSENT answer.

### How report rows become feature vectors

The feature builder makes one row per report ID:

1. Rows without a report ID are dropped. If the CSV has multiple rows for one
   report, each finding is represented by its most frequent state for that
   report.
2. Missing finding labels are filled with `NOT_MENTIONED`; labels are cleaned
   and checked to be `PRESENT`, `ABSENT`, or `NOT_MENTIONED`.
3. Each selected finding becomes three numeric columns, one for each state.
   The matching state gets `1`, and the other two get `0`.

For example, if the selected findings are `A` and `B`, a report with `A =
PRESENT` and `B = NOT_MENTIONED` becomes:

```text
A_present  A_absent  A_not_mentioned | B_present  B_absent  B_not_mentioned
    1           0             0       |     0          0              1
```

So each point is a report's full encoded finding profile. Reports with the
same profile have identical vectors. The app shows the number of reports and
the number of distinct profiles separately.

### How the clusters are formed

The user chooses `k`, the number of clusters (between 2 and the smaller of 8
or the number of distinct profiles). The implementation then:

1. Chooses initial centers using **k-means++**. It first picks one distinct
   profile at random (with a fixed seed of 42 for repeatable results). Each
   next center is more likely to be a profile far from the centers already
   chosen.
2. Assigns every report to its nearest center using squared Euclidean
   distance:

   ```text
   squared distance(report, center) =
       sum over features of (report feature - center feature)^2
   ```

3. Recalculates each center as the average feature vector of the reports
   assigned to it.
4. Repeats assignment and averaging until assignments and centers stop
   changing (within a `0.0001` center tolerance), or it reaches 100 iterations.

**Small distance example:** To keep the arithmetic short, imagine each report
has just two numeric features and the centers are `(0, 0)` and `(4, 4)`.
Report `(0, 2)` is at squared distance
`(0 - 0)^2 + (2 - 0)^2 = 4` from the first center, and
`(0 - 4)^2 + (2 - 4)^2 = 20` from the second. It is assigned to the first
cluster. Actual project vectors have three one-hot columns per finding, but
the same distance calculation is used.

**Center and inertia example:** If reports `(0, 0)` and `(0, 2)` are in one
cluster, their center is the feature-by-feature average `(0, 1)`. Their
squared distances to that center are `1` and `1`. If a second cluster has
report `(4, 4)` at its own center, its distance is `0`, so total **inertia**
is `1 + 1 + 0 = 2`.

Inertia is the sum of all reports' squared distances to their assigned
centers. Smaller inertia means tighter clusters for that run, but it usually
falls as `k` grows; it should not be used by itself to claim that one chosen
cluster count is objectively best. If a cluster becomes empty during fitting,
the implementation gives it the currently farthest report, then recalculates
the centers. The returned cluster numbers are zero-based internally; the app
shows them as Cluster 1, Cluster 2, and so on.

## 6. PCA and the cluster plot

**PCA** (Principal Component Analysis) takes the same feature matrix and
projects each report into two coordinates so the app can draw a 2D scatter
plot. The important detail is that the app runs K-means on the original
one-hot feature vectors, then runs PCA separately for display. PCA does not
create the clusters and does not change the cluster assignments.

### How PCA calculates the two coordinates

1. For each feature column, calculate its average and subtract that average
   from every report's value. This centers each column around zero.
2. Use singular value decomposition (SVD) to find directions (principal
   components) that capture the largest remaining variation in the data.
3. Project each centered report onto the first two directions. Those two
   projection scores are its `PC1` and `PC2` chart coordinates.
4. Calculate the fraction of total variation captured by each direction:

   ```text
   variance for component = singular value^2
   explained fraction = component variance / total variance
   ```

For example, if the first two singular values were `3` and `1` and there were
no other components, their squared values would be `9` and `1`. The first
component would explain `9 / (9 + 1) = 90%` of the variation, and the second
would explain `1 / 10 = 10%`. These are sample numbers to illustrate the
calculation; the app displays the fractions calculated from its cohort.

The app colors each PCA point by the K-means assignment and labels it with
its report ID. Nearby points often have similar finding patterns, but because
the 2D plot can omit variation from other dimensions, visual distance on the
plot is not exactly the distance K-means used. The average finding profile
table shows each cluster's center back in the original feature columns.

PCA centers columns but does not standardize them by their standard
deviations. In this app all encoded columns are 0/1 indicators. PCA requires
at least two reports and at least one feature, and it reports an error if the
cohort has no variation. K-means also requires finite feature values and no
more clusters than distinct profiles.

These are exploratory analysis tools, not clinical decision rules. Cluster
patterns and visualizations depend on the selected findings and available
reports; validate interpretations with appropriate clinical expertise.
