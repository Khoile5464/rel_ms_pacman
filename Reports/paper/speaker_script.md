# Speaker script: Pixels to Power Pellets

Target length: about 12 minutes (15 slides, roughly 45 seconds each). Times in brackets are cumulative guides. Replace `[Author]` names. Numbers match `report.tex`; if you changed numbers on Overleaf, update them here too.

---

## 1. Title [0:00]
"Hi everyone. Our project is called *Pixels to Power Pellets*. We trained a reinforcement-learning agent to play Ms. Pac-Man from raw pixels, and we asked one question: when the agent gets stuck, **why**, and what actually gets it unstuck? We are Le Minh Khoi, Pham Quang Huy and Ngo Quang Huy, supervised by NamLG."

## 2. The problem [0:30]
"We started with a standard PPO agent with a CNN. It learns quickly: within about 25 thousand steps the score jumps. But then it stalls around 700 points. A random policy scores about 274, so 700 is better than random, but doubling training from one to two million steps changed nothing.
When we watched it play, it either stood still or repeated a pattern, and after eating a power pellet it did not chase the ghosts. That's a problem, because the ghosts are where the big points are.
Spoiler: our best agent reaches 1125. The rest of the talk is about what it took to get there, and what did *not* work."

## 3. What the agent sees and does [1:15]
"Quick setup. We use the Atari version of Ms. Pac-Man from Gymnasium: nine actions, and each agent step is four emulator frames. The agent sees the last four frames, resized to 84 by 84, either gray or RGB.
Points: dots give 10, a power pellet 50, and ghosts 200 up to 1600 in a chain. Notice that the big rewards need a *sequence*: eat the pellet, notice the ghosts turned blue, chase them before it wears off.
During training, losing a life ends an episode; for scoring we always play a full three-life game, and we always report the real game score, not the shaped reward."

## 4. Three hypotheses [2:00]
"We turned our suspicions into three hypotheses, each with a cheap experiment.
**H1:** idling is free. Dying is cheap and waiting costs nothing. So we add a death penalty and a per-step penalty.
**H2:** colour is lost. In gray, the blue frightened ghosts may look like the maze. So we try RGB input.
**H3:** no memory. Four frames can't remember 'I just ate a pellet'. So we try an LSTM and a Vision Transformer.
We implemented PPO, the LSTM and the ViT ourselves in PyTorch; only the emulator comes from Gymnasium."

## 5. Method: shaped reward and PPO [2:45]
"The learning reward is a log-compressed game reward, minus a death penalty when a life is lost, minus a constant step penalty. The log keeps big events bigger than dots without letting them dominate.
One important point: these penalties are *not* potential-based shaping, so they can change what the optimal policy is. We'll see that matter later.
PPO settings are standard: 32 parallel environments, 512 steps per rollout, gamma 0.99, clip 0.1. Everything ran on one laptop GPU."

## 6. Four policies [3:30]
"We compare four networks, all with the same actor and critic heads.
The baseline is the Nature-DQN CNN. The LSTM adds a 512-unit recurrent layer; we reset its state when a life is lost and build minibatches by splitting environments so sequences stay intact.
The ViT cuts the image into 12 by 12 patches, giving 49 tokens plus a class token, with four transformer layers.
The fourth one is ours: to test colour without throwing away what a gray agent already learned, we expand the first conv layer from 4 to 12 channels using the luminance weights, so the RGB network starts out behaving like the gray one."

## 7. Testing H2: does gray hide the ghosts? [4:15]
"Here is the evidence for H2 on a real frame. On the left, RGB after a power pellet: the ghosts are blue, 116 blue pixels. On the right, the same frame in gray. The frightened ghost's gray value is 109, exactly the same as one common maze gray. So it's plausible information is lost.
But, and this is the key twist, RGB did **not** improve the score. Both gray and RGB CNNs eat about 0.59 ghosts per game. This is a single-frame check, so it's suggestive, not proof."

## 8. Result for H1: penalties [5:00]
"Now H1, with two seeds and one million steps. A small constant penalty gets 651 on average. Starting with a big penalty, 20 and 0.1, gets only 499: about 150 points worse, and the entropy stays higher, so it learns slower. A curriculum that ramps penalties up gets 657, which is the same as the small penalty within noise.
So penalties alone did not cure idling, and bigger was not better."

## 9. H2 and H3 at 1M steps [5:45]
"At the same one-million-step budget, gray and RGB are indistinguishable: 653 and 665, with a standard error of about 25. The LSTM is a bit ahead and the ViT a bit behind. With one seed each, differences under about two standard errors are not decisive. At this budget, neither colour nor architecture breaks the plateau."

## 10. Long fine-tuning [6:30]
"What did break it was patience. We trained the LSTM and the ViT in stages, 13 million steps in total, with a small step penalty of 0.05 and a lower learning rate. Unlike the CNN, they kept improving.
In the last 300 games the ViT averages 1125 with a best game of 4720, the LSTM 900, and the CNNs about 650. Caveat: these runs are *not* under the same conditions as the one-million-step CNNs, so this is not a clean architecture comparison."

## 11. What changed in behaviour? [7:15]
"Is this just surviving longer? No. After about 8 million steps the ViT eats about 1.06 ghosts per game, up from 0.6, and 1.58 pellets per game, up from 1.0. The LSTM stays at 0.6. Since ghosts are worth 200 to 1600 points, nearly doubling ghost captures explains most of the gain. So the agent did start using power pellets, but only after more than ten million steps."

## 12. Failure mode: collapse [8:00]
"Now the failure. When we pushed the best ViT further with a bigger penalty, death penalty 5 and step penalty up to 1, the training score fell from about 1140 to 490 and never recovered. Our RGB run with step penalty 2 failed the same way.
Look at the value loss: about 1 to 2 when things work, 15 at step penalty 0.5, nearly 400 at 2.
Our explanation, which we have *not* tested with a counterfactual run: with a negative reward every step, dying early pays off whenever the death penalty is smaller than the step cost you expect to pay for the rest of the life. Practical lesson: a tenfold jump in value loss is a cheap early warning that your reward scale is wrong."

## 13. Evaluation is noisy [9:00]
"One important caveat about every number in this talk. The same gray model scored 828 in one 20-game evaluation and 608 in another. The same ViT checkpoint scored 748, 660 and 499 with the same seeds. The standard error of a 20-game mean is 50 to 100, so gaps under about 150 points are not interpretable.
That's why we lean on thousands of training games, call single-seed results trends, and treat 'ViT beats LSTM' as a hypothesis, because they differ in learning rate, stage timing and seed."

## 14. Conclusion [9:45]
"What we're confident about: long, low-learning-rate fine-tuning with a small step penalty lifted the ViT to 1125, mainly through about twice as many ghosts eaten; large penalties hurt and can collapse training; and neither RGB nor a penalty curriculum gave a measurable gain.
Limits: mostly one seed, confounded stages, and only 20-game evaluations. Next, we would run at least three seeds with 50 or more evaluation games, compare ViT and LSTM under identical budgets, rerun 13 million steps without the step penalty, and try an exploration bonus."

## 15. Ethics [10:45]
"Three risks. First, reward misspecification: our own fix for idling made quitting optimal; we'd log intended-behaviour metrics, run short pilots, and review reward changes. Second, compute and evidence: 23 runs and about 84 million steps; we'd report seeds and noise and run multi-seed pilots first. Third, a pixel-only agent could be turned into a game bot, so we'd share only emulator training code and state the intended scope.
Thank you. We're happy to take questions."

---

## Likely questions (short answers)
- **Why PPO and not DQN?** It parallelises well on one GPU; we implemented DQN too but have no Ms. Pac-Man results, so we don't claim anything about it.
- **Why didn't RGB help?** Either the information is recoverable from gray (e.g. ghost motion) or the agent can't yet exploit it at this budget; we didn't run the direct behavioural test around frightened ghosts.
- **Is ViT really better than LSTM?** Not established: different learning rate, stages and one seed.
- **Is the step penalty what helped?** Unknown: we have no 13M-step run without it, so we can't separate it from longer training.
- **How is the ghost count measured?** From exact reward values (200/400/800/1600); level-2 fruit worth 200 can be miscounted.
