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

#pagebreak()
= Chronological Order

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



== July 18, 2026
Little update! I ran the training script for 100 iterations just to see if I could spot any patterns. Seems like the model hasn't exactly learnt though! Look at the Wasserstein distance between a 10-season episode and the actual NBA here:

#figure(
  image("figs/wasserstein_first_try.pdf.png", width : 75%)
)

Bear in mind this is just 1 episode. Maybe a few simulations would make the signal slightly clearer. Still, must do more work. I will run it for thousands of iterations and see what we get.

If that doesn't work, I still have a few tricks up my sleeve:
- A "league" sort of training where I can put frozen versions of my model + some heuristic policies.
- Gotta make salaries matter more; players will go to the team that offers the highest salary



== July 17, 2026
It has been a long time since I last wrote here, unfortunately. I have the sense that I only want to put very "official-sounding" developments here in this diary, but I don't think this is productive. Writing even short entries helps to keep me on track.

Well, here is a brief summary of all that has happened meanwhile:
- I attended the ABM course at the Barcelona School of Economics. I learnt some interesting things about ABMs, frontier techniques, and I was especially interested in how they calibrate parameters. They presented two approaches that I found especially interesting: optimising a surrogate model, and approximate bayesian computation. Both are easier to do with traditional ABMs though. I wonder if I can pass in environment parameters to my RL agents as observations, such that they could generalise to different parameter settings. Then doing approximate Bayesian computation would be a lot easier.
- I added some new features. The action space is now discrete, which was needed for _action masking_. And I now added the "pass" action, which allows the player to skip player signing if need be. This is only available once the agent has $"max number of players" - 1 $ on their team. Otherwise they are forced to sign players.
- I put the scripts on the Arnes cluster and set up the environment (took a while to get everything working), but now I officially ran my training script there with a GPU! Very happy about this.

The next order of business is to get players to actually care about money. For now they just go to the first bidder, but I will start having teams compete for signings. For simplicity, players will just be money-motivated --- they will simply pick the team that offers the most money (not even discounted; just $"years" dot "salary"$). Later on I can add some preference for winning teams, but that is a later development.

I will go to the summer school in London soon, so I might not work much until then. Still, I am enjoying this project. I just need to take it more seriously moving forward.

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