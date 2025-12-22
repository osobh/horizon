Most of today’s AI systems are constrained by human-designed, fixed architectures
and cannot autonomously and continuously improve themselves. The scientific
method, on the other hand, provides a cumulative and open-ended system, where
each innovation builds upon previous artifacts, enabling future discoveries. There
is growing hope that the current manual process of advancing AI could itself be
automated. If done safely, such automation would accelerate AI development
and allow us to reap its benefits much sooner. This prospect raises the question
of how AI systems can endlessly improve themselves while getting better at
solving relevant problems. Previous approaches, such as meta-learning, provide
a toolset for automating the discovery of novel algorithms but are limited by
the human design of a suitable search space and first-order improvements. The
Gödel machine [116], on the other hand, introduced a theoretical approach to a
self-improving AI, capable of modifying itself in a provably beneficial manner.
Unfortunately, this original formulation is in practice impossible to create due to the
inability to prove the impact of most self-modifications. To address this limitation,
we propose the Darwin Gödel Machine (DGM), a novel self-improving system that
iteratively modifies its own code (thereby also improving its ability to modify its
own codebase) and empirically validates each change using coding benchmarks.
In this paper, the DGM aims to optimize the design of coding agents, powered by
frozen foundation models, which enable the ability to read, write, and execute code
via tool use. Inspired by biological evolution and open-endedness research, the
DGM maintains an archive of generated coding agents. It then samples from this
archive and tries to create a new, interesting, improved version of the sampled agent.
This open-ended exploration forms a growing tree of diverse, high-quality agents
and allows the parallel exploration of many different paths through the search space.
Empirically, the DGM automatically improves its coding capabilities (e.g., better
code editing tools, long-context window management, peer-review mechanisms),
producing performance increases on SWE-bench from 20.0% to 50.0%, and on
Polyglot from 14.2% to 30.7%. Furthermore, the DGM significantly outperforms
baselines without self-improvement or open-ended exploration. All experiments
were done with safety precautions (e.g., sandboxing, human oversight). Overall, the
DGM represents a significant step toward self-improving AI, capable of gathering
its own stepping stones along a path that unfolds into endless innovation. All code
is open-sourced at https://github.com/jennyzzt/dgm.
1 Introduction
Scientific progress is cumulative and open-ended, with each breakthrough standing on the shoulders
of countless prior insights. In the same way, our most advanced AI systems are built upon a long
\*co-authors
†co-senior authors
Archive
Self-modify
select
Coding agent’s
own repo

- Self-improve
  instruction
  Coding
  agent
  Code Diff:
  Feature to
  improve itself
  parent
  Coding
  agent
  New
  coding
  agent
  child
  add
  Evaluate on
  benchmark
  Task repo
  (e.g., GitHub repo)
- Task instruction
  (e.g., GitHub issue)
  New
  coding
  agent
  Code Diff:
  Solve task
  Figure 1: Darwin Gödel Machine. The DGM iteratively builds a growing archive of agents by
  interleaving self-modification with downstream task evaluation. Agents in the archive are selected for
  self-modification through open-ended exploration.
  lineage of innovations. For instance, transformers [131], the backbone of current large language
  models (LLMs) [14], did not emerge in isolation but were built upon years of past innovations, such
  as recurrent neural networks [51, 110] and attention mechanisms [7, 64, 101]. However, most of
  today’s AI systems remain bound by fixed, human-designed architectures that learn within predefined
  boundaries, without the capacity to autonomously rewrite their own source code to self-improve. As
  a result, each advancement in AI development still leans heavily on human interventions, tethering
  the pace of progress. This paper investigates the intriguing possibility of safely automating the search
  for ever-better AI. One can imagine an AI system that, like scientific discovery itself, becomes an
  engine of its own advancement: building upon its past, recursively improving, and propelling itself
  toward more advanced capabilities.
  Schmidhuber [116] presented a class of mathematically rigorous, self-referential, self-improving
  problem solvers. It relies on formal proofs to justify code rewrites, ensuring that any self-modification
  is provably beneficial. However, in practice and without restrictive assumptions about the system,
  it is impossible to formally prove whether a modification to an AI system will be beneficial. For
  example, while it may seem that an LLM-based coding agent would benefit from access to more
  tools (e.g., code search, test runners), the actual impact depends heavily on the model’s training and
  task context (e.g., a testing tool that is optimized for one setup may confuse the agent when working
  with others). Instead of requiring formal proofs, we empirically validate self-modifications against a
  benchmark, allowing the system to improve and explore based on observed results. This approach
  mirrors biological evolution, where mutations and adaptations are not verified in advance but are
  produced, trialed, and then selected via natural selection. We also take inspiration from Darwinian
  evolution [25] and investigate the effectiveness of maintaining a library of previously discovered
  agents to serve as stepping stones for future generations.
  We propose the Darwin Gödel Machine (DGM), a self-referential, self-improving system that
  writes and modifies its own code to become a better coding agent. Each self-modification requires
  the DGM to edit its own codebase. We use Python, which is Turing-complete, giving the DGM
  the potential to build any computable machine. Our framework envisions agents that can rewrite
  their own training scripts (including training a new foundation model (FM)). However, we do not
  show that in this paper, as training FMs is computationally intensive and would introduce substantial
  additional complexity, which we leave as future work. Instead, this paper focuses on improving the
  design of coding agents with frozen pretrained FMs (e.g., tool use, workflows). The DGM alternates
  between self-modification and evaluation phases. During the self-modification phase, selected coding
  agents from the archive generate modified versions of themselves. During the evaluation phase, each
  modified agent is tested on a coding benchmark, estimating the agent’s coding capabilities, and then
  added to the archive. By improving its own capabilities through this loop, the DGM becomes better
  at both solving coding tasks and making future self-improvements. A key assumption is that an
  increase in performance on coding benchmarks indicates better coding capabilities, and hence better
  ability to self-modify and self-improve. Furthermore, the DGM maintains an archive of generated
  coding agents, initialized with only one agent, and continuously accumulates all generated variants
  over time. To support continual self-improvement, the DGM draws inspiration from open-endedness
  research [35, 36, 134], accumulating diverse stepping stones (i.e., interesting yet suboptimal solutions
  or features that may enable future breakthroughs). This open-ended exploration encourages the
  discovery of novel and potentially useful self-modifications beyond immediate performance gains.
  2
  We present results on two coding benchmarks: SWE-bench [60] and Polyglot [104]. The DGM
  automatically improves itself from 20.0% to 50.0% on SWE-bench, and from 14.2% to 30.7% on
  Polyglot. We show that self-improvement enables continued progress, as the DGM outperforms the
  baseline where the same initial agent is repeatedly used to modify and generate new agents without
  self-improvement. We also show that open-ended exploration and keeping an archive of all previously
  generated agents lead to the discovery of better coding agents. The DGM outperforms the baseline
  of not having open-ended exploration (i.e., a baseline without the accumulation of an archive of
  interestingly different stepping stones), where the coding agent always builds off the most recent
  version of itself. Overall, the DGM represents a step toward AI systems that can build upon their
  own prior innovations and improve recursively. We consider and discuss safety aspects extensively,
  including sandboxing and traceability of self-modifications, to ensure responsible experimentation
  (Section 5). By advancing the possibility of safe, self-referential, self-improving models, the DGM
  moves us closer to AI that not only learns but evolves in an open-ended, self-accelerating trajectory,
  much like science itself.
  2 Related Work
  Open-Endedness. A grand challenge for driving unbounded innovation is designing open-ended AI
  systems that continuously generate novel and learnable artifacts [126]. Building on this, Hughes et al.
  [56] characterized open-endedness as a system’s capacity to generate sequences of artifacts that are
  both novel and learnable from an observer’s perspective. A central difficulty lies in structuring and
  exploring vast search spaces to consistently produce artifacts that are interesting to humans [20, 59].
  Early efforts addressed this through quality-diversity algorithms [17, 90, 94, 105], goal-directed
  exploration methods [2, 30, 32, 33, 113], intrinsic motivation [72, 75, 100, 103], or learning progress
  frameworks [9, 21, 23, 27, 40, 58, 61, 117, 118]. More recently, large-scale foundation models
  (FMs) [14, 106] have emerged as powerful proxies for human notions of interestingness [35, 112, 148]
  and effective mutation operators to propose novel solutions in code [35, 53, 73, 97, 108]. FMs can
  guide autotelic agents [22–24], model human preferences for quality and diversity [13, 29, 47, 66,
  67, 78, 111, 133], design reward functions [35, 85, 132], create simulated environments [1, 15, 92,
  93, 102, 129], drive ever-evolving multi-agent dynamics [28, 153], search diverse ambulating robot
  morphologies [73], and search expansive solution spaces for benchmark or objective optimization [35,
  36, 53, 62, 69, 79, 82–84, 97, 108, 148]. However, these approaches have yet to close the self-
  improvement loop, meaning improvements on downstream tasks do not translate into enhanced
  capabilities for self-modification or the acceleration of further innovations. We aim to mimic the
  acceleration of science and technology, where new tools and discoveries catalyze the creation of even
  more discoveries. Similarly, how can we emulate nature’s arc of evolution, which bends not only
  toward complexity but also an ever greater capacity to evolve [26, 41, 49]?
  Meta-Learning FM Agents. Many FM-based agents are handcrafted. Some building blocks include
  prompt engineering [18, 119], chain-of-thought [45, 52, 77, 91, 136, 138, 144], self-reflection
  [86, 121, 138], multi-agent debate [62, 76], memory [80, 89, 152], temperature sampling [155], and
  retrieval augmented generation [74]. The manual composition of these components limits the system’s
  abilities to the ingenuity of its human designer. More recently, several meta-learning approaches
  have emerged that leverage FM to automatically optimize prompts [19, 34, 36, 63, 141, 143] and
  design agentic modules [38, 95, 96, 109, 128, 139, 140, 147, 149, 150, 154, 156]. The Automated
  Design of Agentic Systems [ADAS, 53] iteratively generates downstream agents with a fixed meta-
  agent, evaluates them against a target benchmark, and incorporates feedback to refine subsequent
  generations. In contrast, the DGM is a single system that both solves downstream tasks (i.e., coding
  problems) and refines its own implementation (i.e., its codebase), removing the need for a fixed,
  handcrafted meta-agent and enabling self-referential improvements.
  Self-Improving AI. Early on, various researchers outlined theoretical and conceptual approaches
  to self-improvement [42, 115, 116]. Some practical approaches to automated self-improvement
  include systems defined by neural network weight parameterizations [46, 48, 50, 65, 81]. Metz et al.
  [88] developed a gradient-based optimizer that is self-referentially meta-trained using a variant of
  population-based training [57]. Lange et al. [68] extended this approach to gradient-free learning.
  Silver et al. [122] used self-play to continuously evolve agents, achieving superhuman performance in
  challenging domains such as chess and Go. More closely related to the DGM are recent approaches
  that leverage FM-based agents for self-improvement [54, 55, 107, 123, 140, 145]. Zelikman et al.
  3
  [145] use a meta-agent to generate downstream agents, updating the meta-agent based on the meta-
  utility derived from the generated solutions. Yin et al. [140] use a single system to both solve
  downstream tasks and recursively modify itself. However, the downstream tasks or the meta-utility
  do not always align with the capabilities required for self-improvement. In the DGM, improvement
  in downstream tasks directly reflects an increase in self-improvement ability, enabling the potential
  for self-accelerating progress. Most similar is concurrent work by Robeyns et al. [107], which
  also has a single agent recursively solving coding problems and modifying its own codebase. The
  main difference between the DGM and Robeyns et al. [107] is that the DGM has an open-ended
  exploration loop, encouraging self-modifications beyond immediate performance gains and thus
  avoiding stagnation in suboptimal self-modifications.
  3 Darwin Gödel Machine
  A Gödel Machine is a theoretical idea of an AI that searches for ways that provably improve
  itself [116]. In this paper, we propose Darwin Gödel Machine (DGM), an attempt to realize the
  long-held dream of creating a Gödel Machine. The DGM relaxes the Gödel Machine’s impractical
  requirement of theoretically proving that a change will improve the system, instead requiring empirical
  evidence from experiments to demonstrate that a proposed new version enhances performance.
  Additionally, since the DGM relies on empirical evidence of improvement, it may get stuck in a
  local optimum within the vast search space of possible systems (i.e., all computable algorithms). To
  address this, the DGM maintains an archive of discovered solutions during the search, facilitating
  open-ended exploration rather than relying on evolving a single solution. Since the principles echo
  Darwinian evolution [25], where new innovations emerge by selecting an entity from an archive of
  previously discovered solutions, modifying it, and keeping it if it is interestingly new [35, 125, 148],
  we call our algorithm a Darwin Gödel Machine.
  Self-referential Self-improvement of Coding Agents. The DGM is initialized with only one coding
  agent, and its progression is evaluated on coding benchmarks. A coding agent is defined as a single
  system, implemented with a code repository and powered by frozen pretrained foundation models
  (FMs), capable of reading, writing, and executing code. Recent works [53, 149] demonstrate that
  such agents can be improved through meta-learning of their designs (e.g., prompts, workflows, and
  tools), which are implemented in their code repository. Therefore, we define self-improvement as a
  coding task that involves modifying the design of an agent’s own components (i.e., its own code). The
  key motivation is that the empirical evidence must reflect the system’s ability to both self-improve
  and solve downstream tasks. By configuring the DGM as a coding agent and testing its coding
  capabilities, the observed improvements demonstrate not only enhanced performance in downstream
  tasks but also the capacity for further self-improvement, as self-improvement is fundamentally a
  coding task that modifies the coding agent’s own code repository.
  Population-based Open-ended Exploration. Starting from a single initial coding agent, the DGM
  builds an archive of all discovered agents. In each iteration, the DGM selects parent agents to self-
  modify and branch off to produce new agents. Parent selection is roughly proportional to each agent’s
  performance score and the number of its children with codebase-editing functionality (Appendix A.2).
  Each selected parent analyzes its own benchmark evaluation logs, proposes the next feature to
  implement, and receives this proposal as a problem statement to execute (Appendix A.3). The parent
  then implements the suggested feature into its own codebase, generating a new coding agent. Each
  newly generated agent is quantitatively evaluated on a chosen coding benchmark to estimate its
  coding abilities. Only agents that compile successfully and retain the ability to edit a given codebase
  are added to the DGM archive, as only they can continue self-modification. All others are discarded.
  The cycle of parent selection, self-modification, and evaluation continues, progressively growing the
  archive of solutions. Importantly, we note that archived solutions can serve as stepping stones that
  result in improvements much later than their original discovery, making our approach substantially
  different from hill-climbing agentic design approaches [107]. Currently, the open-ended exploration
  process (i.e., archive maintenance, parent selection) is fixed and not modifiable by the DGM, which
  we leave as an avenue for future work. Appendix A.4 shows the pseudocode for the DGM algorithm.
  4 Experiments
  Section 4.1 describes the experimental setup, including the initial coding agent that initializes the
  DGM. We conduct experiments on two coding benchmarks: SWE-bench [60] and Polyglot [104]
  4
  (Section 4.2). For each benchmark, we compare the DGM against two baselines: DGM without
  self-improvement and DGM without open-ended exploration (Section 4.3). Across all experiments,
  we find that the DGM outperforms both baselines, showing that the self-improvement mechanism
  and open-ended exploration are essential for sustained performance gains (Section 4.4). Furthermore,
  we show that the features discovered by the DGM transfer across LLMs (Section 4.4).
  4.1 Experiment Setup
  The DGM is initialized with a single coding agent. This initial agent is built around an FM and is
  augmented with tool use capabilities [4, 114]. The FM autonomously determines its action at each
  step, using whichever tool it deems appropriate. The FM has access to two tools in the initial agent: a
  Bash tool for executing bash commands, and an edit tool for viewing and editing directories and entire
  files (Appendix A.1). The agent receives a single input prompt that specifies the location of the target
  repository, the problem statement, and the repository’s testing framework (Appendix A.1). We run the
  DGM for 80 iterations (generating one new agent per iteration), with two iterations running in parallel
  for SWE-bench and four for Polyglot (Appendix A.2). During self-modifications, coding agents
  are powered by Claude 3.5 Sonnet (New) [3] both SWE-bench and Polyglot experiments. During
  benchmark evaluation, coding agents are powered by Claude 3.5 Sonnet (New) for SWE-bench and
  o3-mini [99] for Polyglot.
  4.2 Benchmarks
  Both SWE-bench and Polyglot are benchmarks that evaluate how well AI agents solve coding
  tasks automatically. Given a code repository and a task instruction, the agent is expected to make
  changes to the repository in order to fulfill the task. Both SWE-bench and Polyglot are widely
  used benchmarks [16, 39, 43, 137, 146, 151] that require the AI agent to navigate a code repository,
  understand the interplay between functions in different files, and spot small errors in convoluted code.
  SWE-bench only has Python tasks, while Polyglot has tasks in multiple programming languages.
  Another difference is that each SWE-bench task may require edits to multiple files, whereas each
  Polyglot task primarily involves implementing a solution from scratch in a single file (although the
  agent still needs to examine other files to understand what changes are necessary), resulting in fewer
  file edits overall. We evaluate the DGM on two popular benchmarks that assess different aspects of
  coding tasks to validate the algorithm’s effectiveness across various use cases.
  SWE-bench. To avoid wasting compute on unsolvable tasks, we use SWE-bench Verified [98],
  a subset of SWE-bench [60] filtered by human annotators to ensure that every task is solvable.
  Throughout this paper, the term SWE-bench refers by default to the SWE-bench Verified subset.
  Polyglot. Polyglot includes tasks in multiple programming languages (C++, Rust, Python, etc.) [104].
  Compared to SWE-bench, one of the most widely used coding benchmarks and likely included in the
  training sets of FMs, Polyglot is more niche and less likely to be included in FMs’ post-training data.
  Additionally, Polyglot is primarily used for evaluating Aider [39] by its developer. This provides
  an opportunity to compare automatically designed agents with a representative coding agent in the
  community, which has been continuously developed and tested against the Polyglot benchmark over
  a year by human developers. Finally, Polyglot enables testing of the DGM’s ability to generalize
  across multiple programming languages beyond just Python. We adopt a similar setup to the Polyglot
  leaderboard [104], with one key difference: the leaderboard reports pass@2 performance, where the
  agent can view feedback from ground-truth tests once. In contrast, we use a pass@1 setting, where
  the agent never sees the results of ground-truth tests.
  Since it is expensive to evaluate on the full benchmarks (Appendix B.1), we use a staged evaluation
  strategy to estimate the coding agent’s performance. In general, we evaluate the coding agent on more
  tasks when we are more confident that it is a relatively strong performer. We first evaluate each coding
  agent on a set of 10 tasks to verify basic codebase-editing functionality. Only agents that retain the
  ability to edit code can solve downstream coding tasks or perform further self-modifications. Agents
  that pass this initial stage are then evaluated on an expanded set of 50 tasks. For SWE-bench, tasks
  are selected based on SWE-bench-verified-mini [50], which is designed to be a representative sample
  of the full benchmark (Appendix B.2). For Polyglot, tasks are chosen as a random subset of the full
  benchmark (Appendix B.3). Because the LLMs we use are inherently stochastic, performance can be
  noisy. We use smaller subset evaluations to estimate general effectiveness and identify top candidates.
  5
  For SWE-bench, if a coding agent achieves over 40% success (a heuristic chosen based on the noise
  observed in preliminary runs) on the 60-task subset and ranks among the top two performers in the
  archive, it is then evaluated on 200 tasks (140 plus the previous 60) to more accurately assess its true
  coding capability (Appendix B.2). For Polyglot, if a coding agent achieves over 40% success on the
  10-task subset, it is evaluated on an expanded 50-task subset (Appendix B.3).
  4.3 Baselines
  To isolate and evaluate the contribution of the DGM’s core components, we compare the DGM
  against two baselines: DGM without self-improvement (DGM w/o self-improve) and DGM without
  open-ended exploration (DGM w/o open-ended exploration). DGM w/o self-improve replicates the
  approach of ADAS [53] in this setting, wherein the meta agent responsible for modifying the coding
  agents remains fixed as the initial agent throughout the experiment. This baseline allows us to assess
  the role of iterative self-improvement in accelerating the evolution of better coding agents. DGM w/o
  open-ended exploration eliminates the use of an archive and always self-modifies the latest stored
  version of itself. If a coding agent self-modifies to the point where it loses the basic functionality
  required to edit a codebase, it can no longer modify itself or solve any coding task. Therefore, DGM
  w/o open-ended exploration retains the latest version of itself that still maintains the basic functionality
  for codebase editing. This baseline allows us to evaluate the impact of having an archive and the
  well-documented beneficial principles of open-ended exploration [20, 23, 36, 70, 111, 125, 148] in
  guiding the agent’s evolution.
  In addition to the learned baselines, we compare the DGM against handcrafted, open-source solutions.
  For SWE-bench, we take the state-of-the-art (SoTA) open-source solution that has been checked
  (i.e., the SWE-bench team was able to reproduce the results) (Appendix B.4). For Polyglot, we
  take the representative agent (Aider) [39], which is open-sourced and designed to support multiple
  programming languages and large codebase editing (Appendix B.5). For a fair comparison, we
  measure the percentage of solved tasks on the same benchmark subsets used to evaluate the DGM
  (Appendix B.2, Appendix B.3). These values are shown as dotted horizontal lines in Figure 2.
  4.4 Results
  SWE-bench Score of Best Agent
  0.50
  0.45
  0.40
  0.35
  0.30
  0.25
  0.20
  Checked Open-sourced SoTA
  DGM w/o Open-ended exploration
  DGM w/o Self-improve
  Representative Agent Baseline (Aider)
  DGM
  Polyglot Score of Best Agent
  0.35
  0.30
  0.25
  0.20
  0.15
  0 10 20 30 40 50 60 70 80
  0 10 20 30 40 50 60 70 80
  Iterations
  Iterations
  Figure 2: Self-improvement and open-ended exploration enable the DGM to continue making
  progress and improve its performance. The DGM automatically discovers increasingly better
  coding agents and performs better on both (Left) SWE-bench and (Right) Polyglot. It outperforms
  baselines that lack either self-improvement or open-ended exploration, showing that both components
  are essential for continual self-improvement. These scores are obtained from evaluating on the
  benchmark subsets detailed in Section 4.2.
  After 80 iterations of the DGM, the coding agent’s performance increases from 20.0% to 50.0%
  on SWE-bench, and from 14.0% to 38.0% on Polyglot (Figure 2). Since the DGM is evaluated
  on only 50 tasks in the Polyglot experiment setup (Section 4.2), we additionally evaluate both the
  initial agent and the best DGM-discovered agent on the full Polyglot benchmark to more accurately
  estimate the improvement. On the full Polyglot benchmark, the DGM improves the coding agent
  from 14.2% to 30.7%. This shows that the DGM can automatically self-improve to create a better
  coding agent. Moreover, the performance of the best DGM-discovered agent is comparable to that of
  6
  DGM Progress
  DGM Archive Tree
  SWE-bench
  Score
  History-aware
  patch generations
  and ranking
  More granular file
  editing via string
  replacement
  10 tasks (no basic functionality)
  60 tasks
  200 tasks (score ≥ 0.4 and
  top two of archive then)
  Final best agent
  Auto-summarize
  on context limit
  Non-empty patch
  validation and retry
  Multiple patch
  generations and
  ranking
  More granular file
  viewing via lines
  Figure 3: The DGM automatically self-improves to become a better coding agent. (Left) Archive
  of coding agents generated during the DGM run on SWE-bench. Each node represents a coding agent,
  with node 0 corresponding to the initial agent. Node color indicates performance on SWE-bench
  (percentage of solved tasks), while border color reflects the number of tasks for which the agent
  was evaluated. Edges show which agents self-modified to produce the offsprings. Many paths to
  innovation traverse lower-performing nodes, and key innovations (like node 24) lead to an explosion
  of innovations built on top of them. Both properties underscore the benefits of open-ended search.
  (Right) Progress plot of the DGM on SWE-bench. The light blue line shows the average score of all
  agents possessing basic codebase-editing functionality. The blue line tracks the best score achieved by
  any agent in the archive at each iteration. The dark line shows the lineage of the final best-discovered
  agent and its precursor nodes, which includes two performance dips. This illustrates the benefits of
  open-ended search, which explores a diverse set of interesting stepping stones instead of focusing
  only on branching off the best solution found so far.
  the checked, open-source, human-designed SoTA on SWE-bench (Figure 2). On Polyglot, although
  the DGM starts with an initial agent whose performance is lower than that of Aider, it discovers
  an agent that far surpasses Aider (Figure 2). The DGM-discovered agents are comparable to or
  outperform handcrafted agents on both benchmarks. While the SoTA SWE-bench agent and Aider
  were painstakingly shaped by human efforts, the DGM hints at a future in which such ingenuity is
  automated, evolving through self-referential cycles of continuous self-improvements.
  The DGM automatically improves both the tools and the workflow of how FMs are utilized (Figure 3).
  For example, the DGM enhanced the edit tool to allow more granular file viewing (by lines) and
  more precise file editing (by string replacement), instead of always viewing or replacing the entire
  file. Workflow improvements include making multiple attempts to solve a task and using another FM
  to evaluate and select the best solution. Other workflow improvements include considering previous
  attempts when generating subsequent ones. Appendix C.1 and Appendix C.2 show all modifications
  leading up to the final best-discovered agents on SWE-bench and Polyglot respectively.
  Because open-ended exploration allows branching from any agent in the archive with non-zero
  probability, the DGM can get out of deceptive dips or peaks in performance. For example, at
  iterations 4 and 56 of the experiment on SWE-bench, although the agent’s score temporarily fell
  below that of its parent, the DGM was still able to explore innovations along that path and create a
  new agent that outperformed all of its predecessors (Figure 3). Furthermore, open-ended exploration
  allows different implementations of the same target functionality to be attempted. For example, while
  the goal is to provide finer-grained editing tools, the specific implementation of this feature can vary
  greatly and hence lead to very different performance (Appendix D). The DGM can explore multiple
  implementations to find the most suitable one and avoid getting trapped in a suboptimal one.
  The DGM outperforms the baselines of DGM w/o self-improve and DGM w/o open-ended exploration
  on both benchmarks (Figure 2). Without updating the meta agent that modifies coding agents, DGM
  w/o self-improve improves the agents in early iterations, but its gains taper off quickly (Appendix E.1).
  In DGM w/o open-ended exploration, only the most recent agent is retained, so a poorly performing
  self-modification makes subsequent improvements harder to achieve (Appendix E.1).
  To evaluate the generality of the improvements from the DGM, we tested the initial agent (Section 4.1)
  and the best agent discovered during the DGM run (Figure 2) with different FMs than those used
  7
  Model Transfer on SWE-bench
  Model Transfer on Polyglot
  Task Transfer on Polyglot
  Success Rate on SWE-bench (%)
  59.0%
  60
  50
  40
  30
  20
  10
  0
  50.0%
  33.0%
  23.0%
  14.2%
  20.0%
  19.0%
  o3-mini Success Rate on Polyglot (%)
  Claude 3.5 Sonnet (New) Claude 3.7 Sonnet
  40
  35
  30
  25
  20
  15
  10
  5
  0
  Initial Agent
  Best Agent transfer to other FMs
  35.6% 36.8%
  32.0% 33.3%
  30.7%
  o3-mini Claude 3.5 Sonnet (New) Claude 3.7 Sonnet
  Best Agent from DGM
  Aider
  Success Rate on Polyglot (%)
  40
  35
  30
  25
  20
  15
  10
  5
  0
  DGM search on all language
  DGM search only on python
  33.0%
  30.8%
  20.6%
  17.6%
  17.3%
  15.2%
  11.8%
  8.8%
  python task non-python task
  Figure 4: Transfer between Models and Tasks. (Left and Middle) The superior performance of
  DGM-discovered agents can be transferred across different models and (Right) different task domains,
  such as from Python tasks to tasks in other languages like Rust, C++, Go, and others.
  during optimization. For SWE-bench, where the DGM was run using Claude 3.5 Sonnet (New), we
  replaced the FM with Claude 3.7 Sonnet [5] or o3-mini, and evaluated on 200 tasks (Figure 4, Left).
  With o3-mini, the initial agent achieved 23.0% and the DGM-discovered agent 33.0%. With Claude
  3.7 Sonnet, the initial agent achieved 19.0% and the DGM-discovered agent 59.5%. For Polyglot,
  where the DGM was run with o3-mini, we replaced the FM with Claude 3.5 Sonnet (New) or Claude
  3.7 Sonnet, and evaluated on the full benchmark (Figure 4, Middle). With Claude 3.5 Sonnet (New),
  the initial agent achieved 32.0% and the DGM-discovered agent 33.3%. With Claude 3.7 Sonnet,
  the initial agent achieved 35.6% and the DGM-discovered agent 36.8%. These results suggest that
  the DGM yields improvements that generalize across FMs, rather than being tightly coupled to the
  specific FM used during its run (Figure 4).
  Furthermore, to evaluate the transferability of the DGM-discovered agent across programming lan-
  guages, we experiment with a version of the DGM trained exclusively on Python tasks from Polyglot
  and then transfer the discovered agent to tasks in other languages. Focusing primarily on Python
  tasks slightly improves performance on Python tasks but reduces performance on non-Python tasks
  compared to the DGM trained on all languages (Figure 4, Right). However, after being transferred
  from Python to other unseen languages during the search, the agent still achieves performance
  comparable to that of the DGM trained on all languages and substantially outperforms both the initial
  agent and Aider. These results demonstrate the robustness of the discovered improvements, showing
  that they do not overfit to a specific programming language.
  5 Safety Discussion
  Systems capable of self-improvement, such as the DGM, represent a step toward more autonomous AI
  development, aligning with long-standing goals in the field of making capable AI that can benefit hu-
  manity [20, 71, 87, 115]. However, this capability introduces unique safety considerations stemming
  from the system’s ability to autonomously modify its own code. Modifications optimized solely for
  benchmark performance might inadvertently introduce vulnerabilities or behaviors misaligned with
  human intentions, even if they improve the target metric [12]. In particular, if evaluation benchmarks
  do not fully capture all desired agent properties (e.g., safety and robustness), the self-improvement
  loop could amplify misalignment over successive generations. Iterative self-modification could also
  lead to increasingly complex and uninterpretable internal logic, hindering human understanding,
  oversight, and control [6, 37, 44, 120].
  Recognizing these challenges, the current implementation and experimental setup of the DGM
  incorporates several safeguards. All agent execution and self-modification processes are conducted
  within isolated sandboxed environments, limiting their ability to affect the host system, and thereby
  mitigating the risk of unintended actions. Each execution within the sandbox is subjected to a strict
  time limit, reducing the risk of resource exhaustion or unbounded behavior. The self-improvement
  process is currently confined to the well-defined domain of enhancing performance on specific coding
  benchmarks by modifying the agent’s own Python codebase, thus limiting the scope of potential
  modifications. Additionally, we actively monitor agent performance and code changes, with the
  DGM archive providing a traceable lineage of modifications for review. At this stage, we have found
  8
  no evidence of harmful or malicious behavior in the generated agents, and the self-modifications have
  been primarily focused on improving coding capabilities.
  Conversely, a significant potential benefit of the self-improvement paradigm is that it could, in princi-
  ple, be directed toward enhancing safety and interpretability themselves. We conduct a preliminary
  investigation into how the DGM can be deployed in AI safety settings to develop countermeasures for
  FM hallucination (Appendix F). Just as the DGM learns to improve its coding capabilities, it could
  potentially discover and integrate better internal safeguards or modify itself for greater transparency
  (e.g., incorporating principles akin to Constitutional AI [8]), if such properties were included in
  its evaluation criteria [109]. This suggests a promising, albeit challenging, pathway in which self-
  improvement becomes a tool for building more trustworthy AI systems. Additional research could
  also explore weaving Constitutional AI in from the start, though the challenge would be incentivizing
  the system to retain these directives (an option worth exploring is to create an unmodifiable part of
  the system to be able to evaluate at halt the rest).
  The DGM demonstrates the potential of self-improving AI while still operating within safe research
  boundaries due to the current limitations of frontier FMs and effective mitigations like sandboxing.
  We include this safety discussion proactively to raise awareness about the emerging prospect of self-
  improving AI systems and their associated safety implications, particularly as these systems inevitably
  become more capable [10, 11, 20, 31, 142]. Accordingly, we advocate for continued investigation
  into the safe and beneficial evolution of AI-Generating Algorithms [20] and self-improving systems.
  6 Conclusion and Limitations
  We introduce the Darwin Gödel Machine (DGM), the first self-improving system powered by FMs
  with open-ended exploration, where progress on its evaluation benchmarks can directly translate
  into better self-improvement capabilities. We demonstrate the automatic discovery of better tools
  and FM systems, resulting in better performance on two benchmarks: SWE-bench and Polyglot.
  Through self-improvement and open-ended exploration, the DGM shows a continuous increase in
  performance, bringing us one step closer to self-accelerating, self-improving AI systems.
  We demonstrate that the DGM can autonomously achieve performance on par with openly available
  solutions. However, it still falls short of closed-source SoTA SWE-bench solutions. An open question
  is whether running the DGM for longer would continue to yield performance gains and eventually
  surpass closed-source solutions. These closed-source solutions often rely on elaborately handcrafted
  techniques developed by teams of highly skilled experts. Since FMs have yet to match the capabilities
  of such experts (e.g., in reasoning), the DGM currently requires extensive compute to discover
  improvements. A single run of the DGM on SWE-bench, as presented in Section 4, takes about 2
  weeks and incurs significant API costs (Appendix B.1). We hypothesize that further progress will
  require more efficient use of computational resources and the development of better reasoning skills.
  Since this version of the DGM is mainly powered by FMs, it is inherently limited by the capabilities
  of the underlying FM. Hence, an exciting future direction is to extend self-modification beyond just
  prompts or FM workflows, to include more computationally intensive methods, such as rewriting
  its own training script to update the FM itself. While this version of the DGM focuses on coding,
  AI systems are increasingly applied across a wide range of domains (e.g., computer vision, creative
  writing). Another promising extension is to develop self-improving AI systems capable of enhancing
  themselves beyond just the coding domain. A key assumption in this work is that coding benchmarks
  are a good reflection of the agent’s ability to self-improve, since the self-modification task requires
  the agent to modify its own codebase. However, one could envision an alternative approach that
  co-evolves the target task distribution, thereby removing the constraint of self-improvement being
  tied to a single objective, as in true open-ended processes. As discussed in Section 5, we must also
  continue to keep safety front and center as we explore this powerful technology.
  In conclusion, the DGM represents a significant step toward the automation of AI development
  through self-improving systems capable of editing their own codebase. While current limitations in
  compute and reasoning constrain its full potential, continued advances in FMs and infrastructure may
  unlock more powerful and general-purpose self-improvements. Provided that the safety concerns
  are carefully navigated (Section 5), the future of self-improving AI systems and AI-Generating
  Algorithms [20] holds immense promise to open-endedly evolve AI, continually rewriting or retraining
  itself in pursuit of greater capabilities aligned with human values.
  9
  Acknowledgments and Disclosure of Funding
  This research was supported by the Vector Institute, the Canada CIFAR AI Chairs program, a
  grant from Schmidt Futures, an NSERC Discovery Grant, and a generous donation from Rafael
  Cosman. Resources used in preparing this research were provided, in part, by the Province of
  Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute
  (https://vectorinstitute.ai/partnerships/current-partners/). Any opinions, findings, and conclusions or
  recommendations expressed in this material are those of the authors and do not necessarily reflect
  the views of the sponsors. We also thank Aaron Dharna, Ben Norman, Cédric Colas, and Shyam
  Sudhakaran for insightful discussions and feedback.
