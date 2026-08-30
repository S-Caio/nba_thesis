#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

#set heading(numbering: "1.1")

= Appendix A: Selection of a distance function

== Motivation

The main paper uses a distance function as a general metric to indicate progress and goodness-of-fit in distributions between simulated and real-life NBA seasons. I argue here that stock versions of distribution distance functions are not fit for the purposes of the paper, so they must be modified.

In particular, the distance function is used as a yardstick with which to measure progress during training and for later diagnoses. It is often applied to the distribution of win percentage within a season. This distribution is crucial because it allows the identification of competitive dynamics. We can see whether there are clusters of bad, mid-tier, and stellar teams --- and in what proportion they appear relative to each other. 

Thus a distance function should, first and foremost, reward simulated seasons of the correct _shape_. For example, if the real-life reference season is bi-modal and the simulations produced two shapes: one also bi-modal, but with peaks that are slightly misplaced relative to the reference season; and another that is uni-modal, but whose singular peak matches well with one of the reference season's peaks. The distance function should, provided that the difference in fit is not dramatically different, prefer the former season over the latter. The former, even if it does not match up exactly with the reference's modes, accurately represents the competitive balance within a season. There are well defined clusters characterising the competitive balance of the league which are not present in the latter season.

Other goodness-of-fit concerns also enter the equation. Questions such as: how well peaks match, how kurtotic the distributions are, or how extreme the data is cannot be completely forgotten in favour of counting modes. Therefore, the distance function must balance all of these properties together; I merely wish that multi-modality be given special --- but not undivided --- attention.

As a last criterion, the distance function must be computable even when the amount of data is low, as NBA seasons contain only 30 teams. Unfortunately, this rules out metrics such as Kullback-Leibler (KL) or Jensen-Shannon Divergence (JSD). Computing these on low amounts of data usually requires binning which reduces the resolution of comparison if the bins are too large, or risks numerical issues when the bins are too small. More resilient statistics, and the ones I will focus on, are Wasserstein distance, Energy distance, and the Kolmogorov-Smirnoff statistic. 

To further build intuition as to which elements are preferrable, I display the winning percentage distributions of real NBA seasons against the KDE of simulations early in the research process in @kdes. After performing Monte-Carlo (MC) simulation of 1000 simulated seasons, I extract all first (season 0) and last seasons (season 9) to display. The seasons had meaningfully different shapes at that point in the process, so they serve as interesting comparison points.

From looking at the KDEs we get a feel for the competing demands on the distance function. The 2012-13 season, for example, exhibits some bi-modality, but the peaks are quite close together and not as well delineated as in later seasons. Season 0, which exhibits strong bi-modality, does not fit the peaks or the reference distribution very well. Season 9 achieves a much better fit here, but it is uni-modal#footnote("This is a clear case of the secondary aims becoming more important than multi-modality."). In this scenario, given how bad the fit of season 0 is, a suitable distance function should most likely prefer season 9. The case flips for the 2017-18 season. While season 0 has more extreme peaks than the real-life distribution, the overall shape is a remarkably close. Season 9 is a decent fit, but its shape does not pass muster. An interesting case takes place in season 2020-21. At face-value it seems similar to the 2017-18 season, but season 0's fit is not as tight. There is some room for argumentation that season 9 is actually a better fit, or at least as competitive. Thus this is left undecided. Then, for the 2025-26 season it is clear that season 0 is the best fit once we take shape into account. It is the best overall fit of all distributions, even if season 9 matches the largest peak very well.

This is ultimately a subjective exercise, and the reader may disagree with my judgement. It is my wish to take the reader through my decision process, so that they may understand how specific decisions were made. 




#figure(
  image("../free_agency_refactor/eval_scripts/generated_plots/season_kdes_30_08_26_small.pdf", width : 90%),
  caption: "KDEs of sampled NBA seasons againt the first season in every trajectory (season 0) and the last (season 9). The simulated KDEs represent the aggregated, mean KDE over 1000 trajectories. Columns are real-life NBA seasons. Rows are ",
  placement: auto
) <kdes>

== Selecting a distance function

We can now calculate distance functions and see which are closest to the ideal set out earlier.
#figure(
table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Type of calculation*],
    [*Description*],
    [*Formula*],
  ),
  [Raw],
  [Simple use of the function; no transformation to the data.],
  [$d = f(x, y)$],
  [Standardised],
  [Strips out mean and scale effect.],
  [$d = f((x - overline(x)) / sigma_x, (y - overline(y)) / sigma_y)$],
  [Centred],
  [Controls only for the effect of the mean],
  [$d = f(x - overline(x), y - overline(y))$]
),
caption: "Definitions of the transformations applied to the data before calculating distance functions",
placement: auto
)<defining_metrics>

We can start by computing the distance function between our aggregate Monte-Carlo KDEs and the actual seasons we saw in @kdes. @table_agg_metrics shows that, in general, distance metrics do not evaluate shape that well, especially when the data is in its raw format. Season 9 routinely outperforms season 0 even when the reference point is season 2017-18 or 2025-26. It may also be noted that the distance value when data was centred are virtually identical to the raw version. It is only when we standardise the values that significant change takes place, suggesting spread was harming season 0's performance according to nearly all distance metrics.

After adjusting for spread, the metrics all converge into a similar ranking and that, at least at first glance, agrees with the desired intuition. They collectively rate season 0's distribution as closer to seasons 2017-18, 2020-21, and 2025-26 than season 9. Only the 2012-13 season is closer to season 9.


#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: (
      left, left,
      center, center, center,
      center, center, center,
      center, center, center,
    ),

    // Top rule
    // table.hline(stroke: 0.8pt),
    toprule(),

    // Main header
    table.cell(colspan: 2)[*Season*],
    table.cell(colspan: 3)[*Wasserstein*],
    table.cell(colspan: 3)[*Energy distance*],
    table.cell(colspan: 3)[*KS*],

    // Header separator
    // toprule(),
    midrule(),

    // Sub-header
    [], [],
    [Raw], [Centred], [Stand.],
    [Raw], [Centred], [Stand.],
    [Raw], [Centred], [Stand.],

    // Bottom of header
    // table.hline(stroke: 0.5pt),
    midrule(),


    // Data
    [2012-13], [0], [0.0416], [0.0417], [0.1096],
    [0.0825], [0.0825], [0.0999],
    [0.1546], [0.1546], [0.1047],

    [2012-13], [9], [0.0169], [0.0169], [0.1054],
    [0.0336], [0.0336], [0.0881],
    [0.0767], [0.0719], [0.0926],

    [2017-18], [0], [0.0468], [0.0468], [0.1142],
    [0.0937], [0.0937], [0.1047],
    [0.1814], [0.1814], [0.1156],

    [2017-18], [9], [0.0276], [0.0276], [0.1662],
    [0.0552], [0.0552], [0.1409],
    [0.1155], [0.1155], [0.1259],

    [2020-21], [0], [0.0555], [0.0555], [0.1115],
    [0.1030], [0.1030], [0.1044],
    [0.1814], [0.1814], [0.1179],

    [2020-21], [9], [0.0238], [0.0238], [0.1204],
    [0.0475], [0.0475], [0.1019],
    [0.1175], [0.1175], [0.0953],

    [2025-26], [0], [0.0326], [0.0326], [0.1022],
    [0.0602], [0.0602], [0.0943],
    [0.1156], [0.1156], [0.1124],

    [2025-26], [9], [0.0258], [0.0258], [0.1509],
    [0.0562], [0.0562], [0.1308],
    [0.1317], [0.1317], [0.1317],

    // Bottom rule
    // table.hline(stroke: 0.8pt),
    bottomrule(),

    // Spacing
    inset: (x: 6pt, y: 4pt),
  ),
  caption: "Distance metrics between reference seasons and aggregate KDES"
) <table_agg_metrics>

Thus, given that standardising the data leads to such dramatically better results, the other versions will be discarded from further analysis. Standardising seems to be a cheap way to direct importance more to shape rather than how well individual peaks match up. The next step is to pick a singular distance function. We can start by plotting the distributions of distances we attain if we compute the distance function for every simulated season against all reference seasons. The can be seen in @dist_of_distances_plot, which shows the right-skewed distribution of Energy and Wasserstein alongside the jagged, multi-modal distribution of KS. While the KS statistic is easy and cheap to compute, its erratic behaviour suggests some instability, which is par for the course, since it is simply the maximum distance between two empirical CDFs.




#figure(
  image("../free_agency_refactor/eval_scripts/generated_plots/SHAPE_distribution_of_distances_30_08_26_small.pdf"),
  caption: "Distribution distance between all MC-simulated seasons and reference seasons."
) <dist_of_distances_plot>

When it comes to Wasserstein and Energy, both seem to share strikingly similar distributions, with very little disagreement between them. There is also significant amounts of overlap between the distributions, which points to both the heterogeneity of the simulations and perhaps a difficulty in separating which simulated season is closest to real seasons. If we direct our attention towards @mean_dist_plot, however, we see that the mean distance is well defined and well separated. This imparts confidence that the distance function is able to pick out and higlight real differences.

#figure(
  image("../free_agency_refactor/eval_scripts/generated_plots/bootstrapped_mean_dist_30_08_26_small.pdf"),
  caption: "Bootstrapped distribution of the mean distance between seasons"
) <mean_dist_plot>