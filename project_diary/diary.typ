#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot, chart


#set text(
  font: "Libertinus Serif",
  size: 12pt,
)
#set document(title: [Master's thesis diary])


#show title: set text(size: 30pt)
#show title: set align(center)
#show heading.where(level: 1): set align(center)

// #title("Master's thesis diary")
// #align(center)[
//   #set text(size: 18pt,
//   font: "TeX Gyre Pagella",)
//   _Reinforcement Learning for Policy Prediction_ \
//   Caio Simon
// ]

#let title-page(title:[], subtitle:[], author:[], fill: yellow) = {
  set page(fill: rgb("#D0F0C0"), margin: (top: 1.5in, rest: 2in))
  set text(font: "Libertinus Serif", size: 18pt)
  set heading(numbering: "1.1.1")
  line(start: (0%, 0%), end: (8.5in, 0%), stroke: (thickness: 2pt))
  align(horizon + left)[
    #text(size: 30pt, title)\
    #v(1em)
    #text(size: 18pt, style: "italic", subtitle)
    #v(0.01em)
    #text(size: 16pt, author)
  ]
  
  align(bottom + left)[#datetime.today().display()]
}

// #show: body => title-page(
//   title: [Master's Thesis Diary],
//   author: "Caio Simon",
//   subtitle: "Reinforcement Learning for Policy Prediction"

// )

#title-page(
  title: [Master's Thesis Diary],
  author: [Caio Simon],
  subtitle: [Reinforcement Learning for Policy Prediction],
)

#pagebreak()


= Chronological Order


== June 20, 2026

Currently, I have managed to build a small simulation of basketball season games. It is an exceedingly simple simulation, but one that provides a strong basis for future work and small reinforcement learning experiments. The main features are the following.

=== Player Attributes

Players are described by a single number, which I currently call their *impact* score. These scores are drawn from a lognormal distribution with parameters

$
  "Lognormal"(mu = 0, sigma = 1)
$


I also experimented with a Pareto (power-law) distribution, but it proved too extreme for my taste. It generated many below-average players alongside a few with astronomically high ratings. The lognormal distribution retains the desirable heavy-tailed property without producing such exaggerated differences. Typically, the best player in a draw is roughly 10 to 15 times stronger than the average player, which already feels sufficiently realistic.

=== Talent Distribution

At present, players are simply sampled from this lognormal distribution and assigned randomly to teams. This is obviously *unrealistic*, as talented players may prefer to play for successful teams or for teams with salary-cap flexibility. Implementing those preferences is something I will have to tackle later.

For now, teams cannot trade for or sign players. I expect that once those mechanics are implemented, talent will naturally become distributed in a more realistic fashion.

=== Game Simulation

Games are currently simulated using a deliberately simple model. Each team has ten players, each with an impact score, denoted by $s_p$. The total team strength of team $i$ is

$
S_i = sum_(p in i) s_p.
$

Here, $p in i$ indicates that player $p$ belongs to team $i$.

The winner of a game is determined according to

$
P("Team" i "wins") = S_i - S_j + epsilon.
$

where

$
epsilon ~ "Logistic"(0, sigma_"noise").
$

If

$
S_i - S_j + epsilon > 0,
$

then team $i$ is declared the winner.

There are several shortcomings with this approach.

First, every player contributes equally to the team's strength. In reality, an NBA team's tenth-best player is unlikely to receive nearly as many minutes as its star player. Ideally, player contributions should depend on playing time, or—better yet—I should allow the reinforcement learning agent to determine the team's rotation subject to realistic constraints (for example, no player can exceed 48 minutes while the team must allocate exactly 240 minutes in total). To prevent agents from exploiting a single superstar, I may also need to model fatigue or injuries, or simply cap playing time at around 40 minutes.

Second, players clearly cannot be described by a single scalar skill value. Real basketball players differ in shooting, passing, perimeter defence, interior defence, slashing, post play, mid-range scoring, and countless other abilities. Although modelling games at that level of detail would be fascinating, the primary focus of this project is team behaviour—particularly behaviour related to tanking. Consequently, I will continue using a single impact score for now. As the simulator develops, I may split this into offensive and defensive ratings, $s_"off"$ and $s_"def",$ respectively.

Building on this point, the current impact distribution is simply chosen by hand. In an ideal world, I would calibrate player ratings using a latent-factor model estimated from real player data. Alternatively, I could incorporate existing impact metrics such as PIPM, xRAPM, LEBRON, or DARKO. I could even build a statistical model of basketball games, although that would probably move the project away from its central research question.

Finally, there is the logistic noise term, $sigma_"noise"$, which controls the balance between skill and luck. Choosing this variance appropriately will be important when calibrating the realism of the simulation.

I simulated sixteen seasons while following the NBA scheduling structure (more games within divisions and conferences) and chose the parameters through informed guesswork. Despite its simplicity, the resulting distribution of win percentages is already reasonably convincing.

#figure(
  image("figs/win_pct_plot.png", width: 75%),
  caption: [
    First comparison between simulated and real win percentages. The simulated distribution does not yet exhibit the slight bimodality observed in the NBA, but I expect this to emerge once teams begin making strategic decisions.
  ],
)

== June 29, 2026

It has been a little while since I last wrote about the project, but some progress has been made in the meantime. First, a few administrative updates:

- I found an existing basketball general manager simulator called `ZenGM`. It is very good, very deep, and written in TypeScript. I could adapt it to serve as my simulator, which would reduce the number of modelling decisions I have to make and defend in the thesis. On the other hand, I genuinely enjoy thinking through the modelling problem myself. I texted Jeremy, the creator of ZenGM, and he mentioned that realism was not one of his primary goals. I doubt that anything I build would approach ZenGM in terms of quality, given that it is already a mature project, but writing my own simulator would give me much more control over how the world works.
- Anders is now my supervisor.

Now onto the project status.

=== Player Signing

Basic free agency functionality has now been implemented. Teams sign players by offering them a non-zero salary. At the moment, players do not have preferences; the first team to bid on a player signs them. This will be improved in later iterations.

=== Initial Player Distribution

With player signing now implemented, there is no longer any reason to randomly allocate players to teams. Instead, teams begin under a salary cap (currently set to 100 for simplicity) and sequentially construct their rosters by offering salaries and contract lengths to available players.

The major weakness is still that players do not choose between competing offers. The first bid they receive is automatically accepted.

=== Player Evolution

Players now follow a predetermined ageing curve, shown below.
#figure(
  cetz.canvas({
    import cetz.draw: *
    // import cetz-plot: *

    let f = x => -0.005 * calc.pow(x - 27, 3)
    let l = x => 0 * x

    plot.plot(
      size: (10, 6),
      axis-style: "scientific",

      x-min: 19,
      x-max: 40,
      y-min: -11,
      y-max: 3,

      x-label: [Age],
      y-label: [$Delta$ rating],

      {
        plot.add(domain: (19, 40), f)
        plot.add(domain: (19, 40), l)
      }
      
    )
  }),
  caption: [Expected change in player rating as a function of age.]
)

The evolution function is

$
delta = -0.005 ("age" - 27)^3 + epsilon
$

where $epsilon$ represents random noise.

For now I simply hard-code $epsilon = 0.5$, although I have considered making the variance depend on player quality—for example, allowing elite players to improve more quickly or decline more sharply. I have not yet decided whether that is desirable.

=== RL Training

I am happy to report that I successfully trained an RLlib model on this environment for the first time.

Getting everything to work was somewhat painful because of dependency issues and missing packages (RLlib is not particularly Windows-friendly), but after enough troubleshooting the training finally ran successfully.

For rewards I use the exponential decay function

$
r = e^(-k ("position" - 1))
$

with $k = 0.3.$

#figure(
  cetz.canvas({
    import cetz.draw: *

    set-style(
      axes: (
        grid: (stroke: gray + 0.4pt),
      ),
    )

    plot.plot(
      size: (12, 7),

      x-min: 1,
      x-max: 16,

      y-min: 0,
      y-max: 1.05,

      x-label: [League position],
      y-label: [Reward],

      x-ticks: (1, 5, 10, 15),
      x-tick-step: none,
      y-ticks: (0, 0.2, 0.4, 0.6, 0.8, 1.0),
      y-tick-step: none,

      x-grid: "major",
      y-grid: "major",

      legend: "inner-north-east",

      {
        plot.add(
          domain: (1, 16),
          x => calc.exp(-0.3 * (x - 1)),
          label: [$R(p) = e^(-0.3(p-1))$],
        )
      }
    )
  }),
  caption: [Reward curve for teams in positions 1–16.]
)

This reward function applies to every playoff team (currently positions 1–16). Teams that miss the playoffs also receive rewards according to the same function, but only after passing through the draft lottery under the current NBA lottery rules. In other words, they can still receive rewards corresponding to high draft positions, but the outcome is probabilistic rather than deterministic.

My next priorities are implementing player retirement and introducing new players through either a draft or some form of bidding process.

== July 1, 2026

Today was mostly spent tracking down bugs related to the player draft order while making sure the drafting procedure is correct. I am now confident that teams receive the correct draft order. Agents can also observe their current win percentage.

While working on the environment, however, I realized that the main class has become *huge*. Even this relatively simple environment already spans hundreds of lines of code and is becoming increasingly difficult to maintain.

To address this, I discussed the design with Claude and Gemini. Claude suggested splitting the `PettingZoo` environment into several separate files, each responsible for a different part of the simulation. This modular approach separates the reinforcement learning logic from the simulator itself, making both development and unit testing considerably easier. Claude even generated a proposed file structure.

Gemini suggested implementing action masking and replacing turn counters (`self.num_moves`) with a state machine that explicitly tracks the current phase of the environment and the valid transitions between phases. I do not think the latter is necessary at this point, but it may become useful as the simulator grows more complex.

Tomorrow I plan to:

- Thoroughly test the scripts generated by Claude and make sure I understand what every component does.
- Write my own unit tests for practice.
- Run a complete simulated season, even if the agents choose random actions.
- Attempt to train agents using this new modular structure.


== July 2, 2026

I have become slightly more acquainted with the environment built by Claude, but it is still a bit daunting, even at this early stage. I built some unit tests myself to get a feel for things, but I still need to try running everything using Ray RLlib.

As for my to-do list, I added action masking (which should make learning faster) and modified the observation space so that teams can observe their position in the standings, their strength relative to other teams (perhaps through a Z-score), and other league-wide information.

Now I am struggling with the following question: how do I actually evaluate whether tanking is occurring?

My first idea was to use regression. The plan was to predict the probability of signing a player based on their attributes (such as skill and age), together with a dummy variable for the team's position in the standings. Holding player characteristics fixed, I could then examine how a team's willingness to sign a player changes with its competitive position.

However, I do not particularly like this idea because it is perfectly rational for a team to sign a younger player who is currently slightly worse than the best available player if it expects that player to develop into a superior long-term asset.

Another possibility, which I like more, is to train a second model with the discount factor set to zero (or another similarly small value), or simply construct a handwritten policy that greedily signs the best available player. These agents would optimize only immediate reward and therefore next-season success. I could then compare this *greedy* policy with my learned policy and investigate where they converge and diverge.

My expectation is that successful teams would behave similarly under both policies, whereas middle-tier and weaker teams would differ substantially. The remaining question is *how* to measure this divergence. Claude suggested comparing the probability mass that the learned policy assigns to the best available player against the greedy policy. I could then regress this divergence on predictors such as team-season fixed effects, league standing, or a dummy indicating whether a team is in championship contention (for example, the top $k$ teams).

#block(
  fill: rgb("#fff3f3"),
  stroke: rgb("#cc4444"),
  inset: 12pt,
  radius: 4pt,
)[
*Short note.*

As an addendum, Claude suggested training several policies using different discount factors,

$
gamma in {0, 0.3, 0.6, 0.9},
$

creating a kind of treatment "dose". I could then examine how behaviour changes as future rewards become more important.
]

Another appealing idea is to run a regression predicting the total change in impact score between the beginning and the end of the off-season. I like this approach because it is simple and intuitive. Draft position or league standing could serve as the primary explanatory variable, alongside other fixed effects.

The thought of switching to an environment that models bank failures is still in the back of my mind. Before investing much more time in this project, I may spend some time reading about that literature. The main attraction is that banking already has many well-established simulation models and models of bank runs. That would reduce the number of modelling assumptions I need to make and defend, allowing me to focus more on reinforcement learning and policy interventions.


== July 4, 2026

I have been thinking more about how to measure tanking, and I think I now have a much better idea of the direction I want to take. This goes back to the initial experiment where players were distributed randomly across teams and then played against one another. I'll include that figure again here.

#figure(
  image("figs/win_pct_plot.png", width: 75%),
  caption: [
    First look at simulated versus real win percentages. I still do not obtain the slightly bimodal distribution, but I hope this will emerge once teams begin making strategic decisions.
  ],
)

Previously, I was being too myopic. I was trying to determine whether *individual* teams were tanking, but that level of analysis is not what I ultimately care about in this project. Instead, I should be able to detect tanking at the league level. By plotting the distribution of win percentages under different league rules, I can assess how bimodal the distributions become—a potential signature of tanking. I can also compare these distributions using metrics such as the Wasserstein distance or KL divergence to quantify how close they are to an ideal reference distribution.

For example, suppose I construct an environment with no draft lottery. Young talent is either allocated randomly or enters the league through free agency. I then train an agent whose sole objective is to maximize championships.#footnote[
  I could also experiment with different discount factors if necessary.
]

This produces a *win-maximizing* league in which no rewards are given for losing, and teams must rely entirely on intelligent roster management.#footnote[
  I have not thought this through completely yet, but I could also allow some heterogeneity in objectives—for example, some teams may value championships more highly than regular-season wins.
]

The important point is that this league provides no structural incentive to lose games, except perhaps temporarily to clear salary-cap space before attempting to contend again.

I can then simulate many seasons of this league and obtain a reference distribution of win percentages. I expect it to resemble a roughly normal distribution, perhaps with a slight skew toward stronger teams. I would then generate comparable distributions for alternative league rules: the current NBA lottery, COLA, Nate Silver's Arcs proposal, the old lottery system, the new proposal, and so on.

My expectation is that these alternative leagues will exhibit greater bimodality because losing remains strategically valuable. Treating the "winner's league" as a reference distribution, I can compare all other leagues using Wasserstein distance or KL divergence and then perform permutation tests to assess statistical significance. Overall, I think this is a fairly solid research design.

The argument still depends on the *win-maximizing* league serving as a credible benchmark. Reviewers may question that assumption, particularly because such a league resembles European football, where wealth concentration and long-term dominance often reduce competitive balance.

While discussing this with ChatGPT, I came up with several alternative approaches, all based on the distribution of win percentages.

=== Idea 1

Suppose the observed distribution is a mixture of two latent populations: tanking teams and winning teams (or perhaps three populations if "buying" teams are added later). For simplicity, consider only two groups. I fit the model on a league where I know tanking occurs—for example, historical NBA data—which estimates

$
f(x) = pi_T f_T(x) + (1 - pi_T) f_W(x).
$

Here, $pi_T$ is the probability that a team is tanking, while $f_T(x)$ and $f_W(x)$ represent the win-percentage distributions of tanking and winning teams, respectively.

I can then apply this model to another league while holding $f_T(x)$ and $f_W(x)$ fixed. The estimated value of $pi_T$ then serves as the probability that teams in the new league are tanking.

The main weakness is that the distributions themselves are unlikely to remain constant across different league structures. Their means and variances will almost certainly differ, which motivates a hierarchical extension.

=== Option 2 — Hierarchical Model

Instead of assuming common distributions across leagues, I allow them to vary. Let team $i$ in league $j$ have win percentage $x_(i j)$. Then

$
x_(i j) ~ p_(i j) f_T + (1 - p_(i j)) f_W.
$

Next,

$
p_(i j) ~ "Beta"(alpha, beta)
$

This directly estimates the probability that each team is tanking.

Since $f_T$ and $f_W$ are themselves distributions (likely Gaussian or perhaps Beta), their parameters could also be drawn from league-specific hyperpriors. This would allow the distributions themselves to differ between leagues, capturing the possibility that teams "tank differently" under different competitive environments.

The downside is that, with enough flexibility, the latent distributions may lose their interpretation as clean "tanking" and "winning" groups. My current intuition is therefore to begin with shared distributions across leagues and only introduce league-specific variation if predictive performance proves inadequate.


== July 17, 2026
It has been a long time since I last wrote here, unfortunately. I have the sense that I only want to put very "official-sounding" developments here in this diary, but I don't think this is productive. Writing even short entries helps to keep me on track.

Well, here is a brief summary of all that has happened meanwhile:
- I attended the ABM course at the Barcelona School of Economics. I learnt some interesting things about ABMs, frontier techniques, and I was especially interested in how they calibrate parameters. They presented two approaches that I found especially interesting: optimising a surrogate model, and approximate bayesian computation. Both are easier to do with traditional ABMs though. I wonder if I can pass in environment parameters to my RL agents as observations, such that they could generalise to different parameter settings. Then doing approximate Bayesian computation would be a lot easier.
- I added some new features. The action space is now discrete, which was needed for _action masking_. And I now added the "pass" action, which allows the player to skip player signing if need be. This is only available once the agent has $"max number of players" - 1 $ on their team. Otherwise they are forced to sign players.
- I put the scripts on the Arnes cluster and set up the environment (took a while to get everything working), but now I officially ran my training script there with a GPU! Very happy about this.

The next order of business is to get players to actually care about money. For now they just go to the first bidder, but I will start having teams compete for signings. For simplicity, players will just be money-motivated --- they will simply pick the team that offers the most money (not even discounted; just $"years" dot "salary"$). Later on I can add some preference for winning teams, but that is a later development.

I will go to the summer school in London soon, so I might not work much until then. Still, I am enjoying this project. I just need to take it more seriously moving forward.

== July 18, 2026
Little update! I ran the training script for 100 iterations just to see if I could spot any patterns. Seems like the model hasn't exactly learnt though! Look at the Wasserstein distance between a 10-season episode and the actual NBA here:

#figure(
  image("figs/wasserstein_first_try.pdf.png", width : 75%)
)

Bear in mind this is just 1 episode. Maybe a few simulations would make the signal slightly clearer. Still, must do more work. I will run it for thousands of iterations and see what we get.

If that doesn't work, I still have a few tricks up my sleeve:
- A "league" sort of training where I can put frozen versions of my model + some heuristic policies.
- Gotta make salaries matter more; players will go to the team that offers the highest salary



== July 28, 2026

Ok, the UCL summer school is done and I am back to work. Yesterday I added some important features. First, the agent can now observe its aggregate team strength ($sum_i s_i$), and also where it is within the episode --- what percentage of the episode has passed. I first wanted to give as much information as possible to the system, but I think I might take it away, as the agents probably shift their behaviour to more win-now towards the end of the episode, whereas I want their behaviour to be stable irrespective of where we are in the episode.

And I also now added a bidding system, so agents place bids on players every round and players go to where they are offered the highest salary. Some cool things have come from this.

In the plot below I show the distribution of win percentages from teams in the league in the last 100 updates of the network. After every update, I run a little episode and record it. And now I facet the plot according to the season we are in.

#figure(
  image("figs/last_updates_win_pct_after_salaries.png", width : 75%)
)

So the first season is especially interesting. There seems to be a strong case of bi-modality here, perhaps with many teams wanting to set themselves up with future talent. In fact, it seems like almost the exact mirror of the real NBA distribution. 

Three thoughts about this: the thing where I track the season percentage might be harming things here, as teams know which point of the simualtion they are in. At certain seasons they might switch strategy because they know the end is near. I will probably deactivate that. 

Second, I question a little whether I should keep this structure where teams start with a blank slate; it might be more realistic to randomly allocate players, discard the first season, and see what teams do afterwards. The advantage here is realism. GMs come in with teams that already have strengths and faults. The path dependence means that GMS have to adapt their plans according to their current team structure. However, real-life GMs have more tools at their disposal than just signing players. They can trade for picks/players or rest members of their team. This hasn't been implemented (yet), so maybe the current approach is still defensible.

Third, I have to check my conditions to move onto the next season. If the condition is that teams must have at least 9 players, then one team with 10 would have an advantage, even if a team with 9 players would in fact prefer to sign somebody else. I will add one extra bidding round.
// So this is kind of cool, right? Win percentages are very even in the beginning, but then it mophs into the bell curve as time progresses. What I especially like is that, in early seasons, there seems to be a bi-modality thing going on. It is the reverse of the NBA pattern, which has more winning teams than losing teams, but this is an encouraging development nonetheless!

I think I will remove the season tracking. A more useful gauge would probably be some historical metrics, like win percentage last two seasons and team strength last two seasons.

Here is another nice plot. I ran a sample episode after every single update to the neural network to get an idea of how development was going. Then I mark the worst team at the start of each episode and show how it develops throughout the seasons:

#figure(
  image("figs/traj_worst_teams.png", width : 75%)
)

So the trajectory indicates to me that there is a significant improvement after the team goes through the draft, but that is still not enough to win big. Instead, teams remain around the 50% win mark. Maybe I need to adjust how good I make my rookies.

Looking at the same plot, but in the scale of rank within a season, we find that it is quite possible for these teams to reach the upper echelon of wins, but that the range itself is quite wide:

#figure(
  image("figs/traj_worst_teams_rank.png", width : 75%)
)

And I also plotted the mean rank of teams after their initial season, seen in the Figure below. What is chiefly important here is the decay that teams face after finishing first --- do they retain an advantage after finishing well, or does that trickle away? And does it change slowly or quickly? We see that there is some persistence from one season to the next. Teams that performed well in the first season tend to still be amongst the best in season 2, but that success quickly fades away. By seasons 3 and 4 the team now has an average finishing position. It is important, of course, to remember that this is just the average rank, which is hiding a lot of heterogeneity. Some teams might have become the worst in the league, some remained at the top for ages. The contention period for a team is around 2 or 3 seasons then, which I think is decently ok. I might extend it a little. I should also make this plot for the real NBA as well.


#figure(
  image("figs/decay_after_position.png", width : 75%)
)




== July 30, 2026
As I write this, I have just finished sending a job to run in the cluster. I made a few changes to the network architecture and agent observables. First, I removed the season tracking. It probably induced some strategic behaviour, where teams tanked in the beginning and later on tried to go full strength. I don't have solid evidence of this apart from the density plots I shared last time, but I find it logical to expect such a thing.

Now for the bigger stuff. Replacing the season tracking are a few lagged variables, which were added to allow the agent to get a sense of history. I now share the winning percentage and team strength of the past 3 seasons. I chose 3 because I expect that to be around the length of a team's phase cycle; Looking at 4 or 5 seasons ago doesn't seem very informative. And extending the number of lags would become a de-facto season tracking again.

Then I also added a _future_ observation --- the projected cap hit, which is attained by looking at the players signed by the agent for the next few seasons. This allows it to really understand what offering a 3 vs 5 year contract looks like. Since 5 years is the maximum contract length, I make the projection window also 5 years.

And finally, I changed the network substantially. I will summarise the current architecture in bullet points:
- Convolutional layer for player embeddings (kernel size = 3 is back, with a padding of 1)
- The player market matrix first gets passed through the convolutional layer, then a "market features" layer that compresses all of this information. It is so that the player embeddings don't take up all of the observation space
- In between the two steps above I compute the *mean embedding* for each of the 32 player embedding dimensions. The mean is supposed to represent more or less the market state. This gets appended to the final trunk module in order to make market decisions.
  - This gets me close already to a Deep Sets sort of framework. For now I haven't fully transitioned, but the option remains open.
- Now instead of having separate modules for the team vector, scalars, and history, I conslidate them all into one beautiful vector.
- Oh, and LayerNorm is also applied in a lot of places!


I am thinking about whether I should use transformers instead of (or in addition to) the mean player embedding. I could create a sort of team-conditioned attention layer, where, depending on the state of the team, certain player types are highlighted. That would be cool. 




== August 1, 2026
The 20-hour job I sent last time is done. It was quite a while and I think I may have abused the GPU slightly, but some useful things came from that. First is that I started to see more traces of bi-modality in my win percentage distributions (shown below)

#figure(
  image("figs/traj_worst_teams_aug1.png", width : 75%)
)

Admittedly, this is on less data than I had before, as I reduced the frequency of logs. This brings me to the main difficulty right now, which is speed. Simply speaking, training is incredibly slow. In 20 hours I only performed around 2000 updates to the network. Ideally I would be able to run through it much quicker. To rectify this I now did two things:
- First, I moved the environment from a sequential framework (`AECEnv`) to a `ParallelEnv` style, which managed to speed things up quite significantly. Second, I created more "slack" in the player market. Before, I created just as many players as were needed by teams ($"player per team" dot "teams"$), but now I add 30% on top of that. This gives teams more options in terms of who to sign. In small tests I ran locally (no neural network, just randomly picking allowed actions), there was a meaningful speed improvement from this --- even when using the sequential environment. That is because before, once the pool of available players got small, teams were forced either to pick some bad players or hit the "pass" action. Now they can credibly pick decent/mediocre players at the end of rounds, instead of outright awful ones.

But the speed issue may not be solved still. There is still the Neural Network to content with, so I might have to look at strategies to speed that up (or make it smaller).

And now there is also another point. I don't know whether my current setup can replicate what NBA teams do. Past the first signing round, where the entire roster must be recruited, subsequent offseasons are quite tame. There are only 2 or 3 rounds in each of them. For more realism, I will probably have to add player rest (thus allowing teams to operate within the season) and trading of picks and players. Trading will be complicated, so I might start with affecting rotations.

Before I plunge into this type of work though, I want to see what I can still get out of this environment. I'll run some speed tests to look at how the new `ParallelEnv` performs and then train it in earnest to see how far I can get. 

Also, something else to think about: from the last few density plots by season I made, I notice that the first few seasons seem different from the ones towards the end. The density mostly concentrates on teams being below .500, probably because they want to set themselves up with young talent in the future. One way to solve this (which would also shave off some time in the beginning of the simulation) is to start out not with a completely blank slate, but with maybe 3 or 4 players already randomly allocated to each team. This would induce a form of path dependence, where teams would have to adjust based on the roster they are given, instead of starting from a complete blank slate. I like the idea but first I want to keep going with what I have.


== August 4, 2026

The training script had to be modified to account for the new `ParallelEnv`. Before, one training iteration was of size 4000 steps in the environment. With the `AECEnv` each step served up the observations of just 1 agent, so every step was a little impoverished. In contrast, with `ParallelEnv` every step needs to provide observations for 30 players, which goes into the gigabytes. This required more RAM than is available in the system, so it crashes.

One small solution is to simply reduce the batch size for training. 256 worked well and now I am trying 1000. This will work if I want to keep my environment unchanged. 

But after thinking about it some more, I quite like the idea of randomly allocating some players to each team already. That would generate some path dependence simulation-wise (heterogenerous endowments) and it would allow me to significantly speed up training. Also, what I could do to still use the full 4000 batch size is to expose not the entire player market to the agent, but only the available free agents. This would necessitate a change to action masking mechanics as well, but it is doable in a day.

Importantly, I still want to keep the action space a fixed size, so what I would restrict the matrix to some fixed size. Say, for example, that I have 390 players in total. I allocate 5 players randomly to each team. That would give me 240 free agents in the market. I would then always expose a 240-player array then. If there are fewer free agents than 240, I will simply mask the action.



== August 6, 2026

Added some new features. I decided to move ahead with the initial endowments for teams. I think it will aid in realism, especially as I was seeing in the previous density plots that the first season tended to be woefully different from the rest. Now there is some more consistency. Teams receive 5 players only in expectation. In actuality they are randomly given out, so you may end up with fully formed teams or those with only 2 or 3 players. Initial contract lengths are short, and salaries are probabilistically assigned in proportion to the player's ability.

I also added some more contextual variables. Specifically, the number of players in a team, the relative number of players in a team (a z-score), relative team strength (also a z-score), and average age. Hopefully this makes learning simpler.

All that is left is to test everything out. Arnes is having some GPU problems (a lot of them are getting drained), so we'll see.

I am also looking to pull some more levers. Specifically, I am thinking of changing the reward function (maybe multiply the reward by 2, so that being a winner is even more valued; teams are way too happy losing, it seems), and/or changing the player distribution. A $"lognormal"(0, 1)$ might leave too little space (and a lot of concentration) for mediocre players. Something like a $"lognormal"(1, 0.5)$ would still retain the long tail but create more separation in terms of truly awful players and such.

Then, of course, is the calibration idea, where I let the network see some key parameters (noise scale, shape or mean for lognormal distribution of rookies) and then vary it during episodes so that I can see which combinations yield the smallest distance to the real NBA distribution.

== August 7, 2026

Tests went through. Little improvement. I made this plot of the Wasserstein distance per season. It still isn't going very well. The first season still retains the bi-modality, but it is in the opposite direction of what I want. Later seasons are decent, but they don't have that distinctive bi-modal shape; more of a bell-curve (though I will argue the top does have some interesting features)

#figure(
  image("figs/wasserstein_per_season_7_august.png", width : 75%)
)

#figure(
  image("figs/win_pct_7_august.png", width : 75%)
)

Though, on the bright side, I did manage to reduce iteration time somewhat. For better results I will reduce the number of players in total (maybe 330 instead of 390) and I will remove the 5-year contract length. That should save some space in the action mask, which is my heaviest observation.

Furthermore, I will also apply the multiplier to the reward function, as discussed last time. I want teams to want to win. Let's see what that does. If that doesn't provoke a change, I will mess with the lognormal again, as discussed last time.


== August 10, 2026

I am back after doing some work the last few days. I am also on holiday with the family so I have less time to dedicate to this unfortunately. But there have been quite a few developments since last time.

First, I tried scaling the reward function by 5, but that had no effect. Instead, I changed the shape of the function by reducing $k$ from $0.3$ to $0.15$. That makes the descent in reward shallower, giving some significant weight to finishing in 16th, for example. I am also considering giving $0$ reward to teams that don't make the playoffs, but I am not ready to make that step just yet. It could lead to some weird behaviour around the edges of the playoff bracket. I suppose this edge behaviour does exist, but I won't include it just yet. I want teams to still have some gradient if they finish badly. $0$ reward would kill that.

#figure(
  cetz.canvas({
    import cetz.draw: *

    set-style(
      axes: (
        grid: (stroke: gray + 0.4pt),
      ),
    )

    plot.plot(
      size: (12, 7),

      x-min: 1,
      x-max: 30,

      y-min: 0,
      y-max: 1.05,

      x-label: [League position],
      y-label: [Reward],

      x-ticks: (1, 5, 10, 16, 25, 30),
      x-tick-step: none,
      y-ticks: (0, 0.2, 0.4, 0.6, 0.8, 1.0),
      y-tick-step: none,

      x-grid: "major",
      y-grid: "major",

      legend: "inner-north-east",

      {
        plot.add(
          domain: (1, 30),
          x => calc.exp(-0.15 * (x - 1)),
          label: [$R(p) = e^(-0.15(p-1))$],
        )
        plot.add(
          domain: (1, 30),
          x => calc.exp(-0.3 * (x - 1)),
          label: [$R(p) = e^(-0.3(p-1))$]
        )
      }
    )
  }),
  caption: [Old (red) and new (blue) reward curves]
)

The most significant development was that I started looking into my player development system. I found out that my original curves were wildly miscalibrated, with young players experiencing rapid, sustained improvement which made it clearly advantageous to tank in the beginning in order to get the best talent. So I worked (with considerable help from Claude) on two improved systems.  

The system `v2` was based on multiplicative growth instead of additive. This was easier to calibrate, as a player of rating 0.4 wouldn't suddenly get a $+2$ just by being 19 years old. Instead, it adapted to the player's original rating, which made it so that the rating structure was kept more stable.

Namely, we define a drift function $f("age", "potential")$, which looks like this:

$
  f("age", "potential") = cases(
    0.015 * (27 - "age") + 0.03 * ("potential" - 1.0)  & "if" "age" <= 27,
  -0.02 * ("age" - 27) & "otherwise",
)
$

Potential is a new attribute which I draw from a lognormal, such that I get long-tail outcomes. It has the power to speed up development.

Then we add some normal noise to the function and exponentiate the result. The product of this operation multiplies the current rating $x_t$ to arrive at $x_(t+1)$:

$
  x_(t+1) = x_(t) dot exp(f("age", "potential") + epsilon)
$
$
  epsilon tilde "N"(0, sigma)
$

So it is basically a random walk with drift depending on age and potential. This worked decently well, but ratings still increased over the season, or at least the influence of the best players relative to your median player.

Thus, I tried another candidate, which I called `v3`. I wanted to keep the distribution of talent more or less the same throughout an episode. This one is a bit more complicated. We start by defining and age curve, $g("age")$:
$
  g("age") = cases(
    2 dot exp(-0.03 (27 - "age")) & "if" "age" <= 27,
    2 dot exp(-0.025 ("age" - 27)) & "otherwise"
  )
$ 

The "otherwise" expression is just there to be make the descent a little slower. 

After we get the age curve, we take rating and potential into log-space. Remember that a lognormal is simply just a normal curve exponentiated, so taking it to a log makes it normal again. I will express logged version of a variable with $x'$ and $z'$ for $log(x)$ and $log("potential")$, respectively.

$
  u = log(g("age")) + log("potential") \
  x'_(t+1) = x'_t + kappa (u - x'_t) + epsilon \
  x_(t + 1) = exp(x'_(t+1))
$

So it is important to break things down here. In the target we see the addition of the log of the age factor and the log of the potential factor, but remember that $log(x) + log(y) = log(x y)$. Basically the potential modifer tells us how above and beyond (or below) a player goes in terms of their development relative to their age. If a player has potential 1, then the player only gets a potential boost equal to the age factor #footnote($u = log(g("age") dot "potential")$). If potential is 2, then that player develops twice as quickly as the age factor would imply. In fact, the development is always twice as powerful.

So the target is this new addition of age and potential. The new rating, still in log-space, goes in the direction of the difference between the target and the current rating, $u - x'$. $kappa$ controls how tightly players go towards their target. And we have some normal noise, of course.

I implemented these three and tracked the percentiles of talent throughout Monte-Carlo simulations of seasons, which can be seen below.

#figure(
  image("figs/rating_distribution_over_time_august_10.pdf", width : 75%)
)

The light ribbons represent the 90th and 10th percentiles, while the darker grey is the 75th and 25th percentile. The black line is the median.


There was a clear explosion in talent during the duration of an episode. I think that is why tanking was so prevalent previously. The other two are much more stable. This can also be shown by the ratio of players in the 90th percentile over the ratings of the 50th percentile. 

#figure(
  image("figs/ratio_plot_august_10.pdf", width : 75%)
)

So `v3` keeps the most stable ratings. It does seems to suffer from a slight decline throughout the episode for most players, though the 90th percentile remains more or less stable. I want to play around some more to stem this loss, but now the ratings are much more well behaved.

It is important to note that I also changed the distributions of players. The original cohort is drawn from a $"lognormal(1, 0.5)"$ distribution, while the cohort of rookies comes from a $"lognormal"(0.5, 0.5)$. I'm not sure these numbers are what I want though. I think I may be underrating how important star players are. Maybe players at the 90th percentile should be around 2 times as good as the median. And maybe I also need to adjust the potential draws to keep a sort of stable ratings population. Or increase the shape parameter of the lognormal. Make extreme values more likely.

There are many more ideas as well! One is to vary the location parameter of the rookie distribution. I can make it be drawn from a normal, such that a rookie class is unpredictable in its quality. Another is to make teams draft not just automatically from rookie rating, but rather from some noisy combination of potential and current rating. I think that could be cool!

To end things, I have two pragmatic goals I have to think of. The first is that I want the talent distribution to remain pretty much constant throughout an episode. With the current parameters of `v3` there is a gentle downgrade in talent quality. I think I can fix this by increasing the location parameter in the rookie class, so that they start out closer to the original class. The second issue is that I want stars to really be stars and be a lot better than the median player. Right now the 90th percentile starts at around 1.5 times the median talent. I don't know if I have to look higher (95th percentile, maybe) but there may not be enough demand for stars in this league. I can change this be increasing the scale/shape parameter of the lognormals. 

== August 14, 2026

I have some positive news! I worked on making training faster, which involved a lot of messing around with hyperparameters. I don't think I have the final combination yet, but I was actually able to train for around 6000 iterations this time around. And the results are encouraging! I plotted the Wasserstein metric between real NBA seasons and my simulations, and we can see a nice downward slope. It is *crucial* to note that I changed the way I calculate the Wasserstein metric here. Instead of pooling all real seasons together and using that as a reference, I calculate the metric against every individual real season. This is somewhat more honest#footnote("And usually larger than the pooled version"), as it doesn't muddle together a sort of "average season shape," but instead looks at all seasons combined. Unfortunately it still uses that sort of average shape on the simulation side. It would be too computationally expensive to look at all trajectory-iteration-season shapes.

So the shape of the curve is encouraging. We are getting closer and closer to an NBA shape. I should also note that when I initially calculated with the pooled Wasserstein distance I found results as low as 0.02. What is concernng though, when we look at the plot below, is that my simulations have more similarity with the old NBA system than the current one. So while I see a significant improvement on the Wasserstein metric, it seems that on net I am getting closer to approximating the old NBA system than the current one. I am not extremely worried about this yet, but it is something to keep in mind.

#figure(
  image("figs/wassertein_two_comparison_14_08.pdf", width : 75%)
)


On that topic, I also wonder whether Wasserstein is the best metric to watch given that I am operating on a region of bounded support. The distance statistic will invariably have a very small range. Does a 0.02 to mean a bad fit, whereas 0.01 is great? Moreover, since the distance between two systems is only 0.0116, can this metric accurately measure the distance between dstributions? I could try to answer other questions, such as how persistently does an NBA team remain on top or at the bottom? How fast does a team rise, and so on? These could be incorporated into loss function for my calibration.

Anyway, getting back to the real topic. If I look at which seasons drive the discrepancy in Wasserstein metric, I find that 2022-23 and 2023-24 are the root cause of this spike. Otherwise, it seems like the current and old systems are more or less on-par with each other. Earlier seasons of the old system are the most similar to my current system, it seems.

#figure(
  image("figs/wasserstein_time_plot_14_08.pdf", width : 75%),
  caption: "Computed using the last 100 simulated update trajectories."
)

This is interesting. The 2023-24 season is when the new CBA came into being, but the spike in Wasserstein distance happens one season earlier, in 2023-24. So in the interest of investigation, here are the density plots for all seasons:

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    image("figs/all_season_densities.pdf", width: 100%),
    image("figs/all_season_densities_all_together.pdf", width: 100%),
  ),
  caption: [NBA win percentage densities]
)

The most distinctive feature of the two seasons is how concentrated they are. You can see in the right panel that 2022-23 and 2023-24 have some of the highest spikes, while the simulated seasons most similar to the real NBA seem to be those that have a flatter, less concentrated shape, such as 2010-11 or 2012-13. I wonder if this isn't because I am not disambiguating between trajectories and individual seasons when computing the Wasserstein distance. It could be that the combination of many seasons gives that sort of muddled flat shape, while real individual variation washes away. Worth checking for sure. If it gets computationally hard I can restrict to just the last few training iterations perhaps.

Looking at the actual shape of the seasons, it does seem like the desired shape is coming together. Season 4, for example, has a beautiful shape, with some winning teams and some clear tanking teams. It looks like a mixture of distributions. Seasons 5-7 have a sort of muddled shape, it seems, though it hides some significant heterogeneity in terms of individual trajectories.

#figure(
  image("figs/last_iter_densities_14_08.pdf", width : 75%)
)

And for a really crazy plot I compared every density that you see above to every density of the current system:

#figure(
  image("figs/every_season_vs_every_season_curr_14_august.png", width : 100%)
)

This really drives home the fact that the 2022-23 and 2023-24 seasons were relative outliers, as none of the simulated seasons seem to fit them very well, whereas the other seasons are fit much better. This might be a call to look at the *median Wasserstein distance* instead of the mean, since it parses out these relative effects.

It also seems to me like the first simmed season doesn't fit a lot of real seasons very well. I could maybe use it as a throaway season at some point. I will have to decide later though.

In reality, I am still doing this based off of not so much data, so here is what I am going to do. First, I will look into calculating the per-season Wasserstein distance instead of pooling across all seasons. Then, I will also try to collect more data in a new script using the latest saved version of my agents. That will solidify my conclusions. And I will port the current plots onto this new script, obviously. I could produce plots like the distribution of wasserstein over different seasons to see if I really should use the first season as a throaway.

And, on the environment side, I have to update the drafting logic to take into account projected potential, not just current rating. Maybe I will also update the player retirement logic as well.


== August 19, 2026

I need to do some thinking here for my evaluation script. Let me describe the issue. Actually, it can probably be separated into two issues. The first is which statistic to use in order to quantify the differences between real and simulated distributions. I started out using Wasserstein as my default metric, but it wasn't quite agreeing with me. If we go back to the previous entry, on the plot above where I have the KDEs of every simulated season against every real season, we see that Season 0 resembles the 2025-26 season quite a lot, but Wasserstein grades similarity as lower than seasons 4-7. I can accept that season 4 is a good fit, but seasons 7 and 8 don't seem like such a great fit. I could compare large, aggregate distributions using KL or JSD, but once I am looking at individual seasons then the KDE estimate is unreliable and the metric explodes, so not so great. From my tests, KS seems like a metric that tells me what I am looking for, but even then it isn't perfect.

Second issue: do I compare to aggregate NBA seasons, or individual seasons? I currently favour the latter because an average plot could make a distribution that isn't characteristic of any real true NBA season. At the same time, maybe it is a good estimator? And there is quite a lot of change between NBA seasons. 

*Next steps.* I want to make a function that does the following:
- Plots an NBA season of my choice, together with the aggregate (pooled) KDE density plot for a specific season.
- Shows the distribution of a statistic (KS, Wasserstein, Energy, JSD) between that selected NBA season and all of the season $n$ simulations. 
- Then, create another plot to look at the estimator of that value, with bootstrapped intervals. So maybe I can make a grid of both the mean and the median.
- Possibly distinguish between making comparisons on pooled distributions and individual distributions. 


== August 31, 2026
I am back from Greece#footnote("I had a great time!") and I have been working at the evaluation script. In particular, I wanted to find the best distance function for my purposes. I think the work is now done in that regard (and it was so long!). The conclusion is basically that between Wasserstein, Energy, and KS, neither comes out ahead as the clear winner. The most important thing is not which function to use, but how to preprocess the data. Standardising the data prior to computation has a much larger effect. With it, the overall _shape_ of the distributions becomes a much more important attribute. I wrote this up in more detail as a supplementary material which is also included on the GitHub, so I will leave it at that. But to summarise, my basic process was to look at simulated distributions versus real-life NBA seasons, decide for myself how I would rate them (which is closest), and then see what the distance functions themselves said. Energy distance + standardisation came out on top.

Now I need to work on more diagnostics. One that would be good is to compute the distances between real NBA seasons. Then after I get that distribution I can look at the distances between my simulated seasons themselves. If the distributions overlap significantly then I know that the variability between simulated seasons is within a normal range.

I also need to show whether my seasons are closer to the old draft rule system or closer to the current paradigm. A plot showing the mean distance (with bootstrapped CI) of my seasons from the different systems would be good. 


#pagebreak()


= To-Do (short-term)

- $checkmark$ Action masking
- $checkmark$ Allow teams to see their place in the standings, as well as their strength score alongside other teams.
- (At least worth thinking about.) Instead of having a fixed player cohort, maybe I can use Deep Sets.
  - Gemini suggested a useful idea: build a player encoder that every player passes through. Then take the mean or maximum embedding as a "global context" variable, which is appended to the matrix.
  - Update on this: I currently use a convolutional layer to pass through all players. The cohort is still a fixed size, but the weight-sharing should help speed up training.
- $checkmark$ Allow teams to sign players (free agency).
  - Initially, agents will be able to observe player ratings to make the simulation simpler.
  - Add player preferences based on a team's past success and the salary they are offering.
- $checkmark$ Player evolution (easier)
- $checkmark$ Player retirement
- $checkmark$ New players entering the league
- $checkmark$ Add lagged observations of team strength and team win percentage
- Add one extra simulation round
- Create heuristic policy
- Create league of opponents
- $checkmark$ Make players care about salary $dot$ years, not just salary
- $checkmark$ Add future cap hits into list of observables: salaries that we already know will be in the next few season's cap.
- Make decay plot for real NBA
- Use reward function to look at a decay plot x average reward during episode sort of plot.
- To think about: instead of using mean player embeddings, use some sort of team-conditioned attention that, given a team's current position in the market, highlights the most important players.


= To-Do (long-term)

- `PettingZoo` environment where agents can sign players during the off-season.
- `PettingZoo` environment where agents can trade players.
- `PettingZoo` environment where agents can draft young players.
- Calibration of parameters (very long-term).
- Survival analysis for ageing players.


= Weaknesses

- Initial talent distribution.
- Network currently is primed to learn simply the ordering of players --- as in player with id 1 is great, player with id 300 is terrible. Perhaps a Deep Sets/Attention approach might be better. 
- Parameters are largely guesstimated.
- The game simulation is too simplistic.
- Better players should contribute more than end-of-bench players.
- Retirement should take into account player ability.