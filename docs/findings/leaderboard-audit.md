# I ran Hyperliquid's 77 "top traders" through a deflated-Sharpe + Jensen-alpha audit. Their edge isn't skill.

*An honest, in-sample, reproducible look at whether the leaderboard ranks alpha — or just beta and survivorship.*

---

## The question

"Just copy the top traders" is one of the most repeated pieces of advice in perp trading. Hyperliquid even hands you a leaderboard of them. So I wanted to answer a narrow, testable question:

**When the leaderboard crowns someone a "top trader," is it measuring skill — or is it measuring leveraged exposure to a market that happened to go up?**

This matters because the two look *identical* on a PnL chart. A wallet that was long ETH with leverage during an ETH rally shows the same beautiful equity curve as a wallet with genuine edge. The only way to tell them apart is to decompose the returns.

So I did.

## What I tested

I took the 77 wallets a Hyperliquid leaderboard had already flagged as "top traders" and scored their copy-returns through a kill-test pipeline built on the standard academic tools:

- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado) — does the Sharpe survive after correcting for the number of trials and non-normality?
- **Jensen's alpha / beta decomposition** — once you regress out market beta (BTC and ETH), is there any residual skill?
- **Per-wallet survival** — does the cohort effect hold up wallet-by-wallet, or is it driven by a handful of outliers?

One important methodological note up front, because it's the whole ballgame: **I scored them in-sample — on the very window their "winning" was measured on.** That's the most *favorable* possible test for them. I'll explain below why that makes the finding stronger, not weaker.

## What I found

**The raw numbers look great.** The cohort's pooled returns are strongly positive (t = +38.97), and they even clear the deflated-Sharpe gate (DSR = +0.139). By every naive bar a leaderboard uses — positive returns, positive Sharpe, even deflated Sharpe — these look like winners.

**Then you regress out ETH beta, and the edge evaporates.**

- Jensen's alpha vs ETH: **t = -16.25**, with **beta = 0.79**.
- After subtracting a position-sized ETH buy-and-hold, the residual alpha is *negative*.

In plain terms: the "skill" is a leveraged long-ETH tilt, measured in a window where ETH went up. The leaderboard is, to a first approximation, ranking *who had the most ETH beta during an ETH rally*.

**Per wallet, it's no better.** Of the 77:
- 28 trade enough to evaluate (n >= 30).
- 24 have stable fits (n >= 100).
- **Only 4 of those 24 show real risk-adjusted alpha** (Jensen-t > 1.5 vs *both* BTC and ETH).

So ~17% of the *evaluable* "top traders" show something that survives beta-adjustment — even in-sample. The cohort as a whole does not.

## "But maybe it's a maker-rebate edge?"

This was the most interesting objection I had to rule out, because it's the kind of hidden, non-directional edge that *would* be real. Maybe these wallets aren't directional at all — maybe they're quietly farming maker rebates.

The data says no, decisively:

- The cohort's gross PnL is +$57.9M. Maker rebates earned: **$96k — that's 0.17% of gross PnL.**
- The cohort actually **pays ~$1.35M in net fees.** Only **1 of 77** wallets is a net-rebate-earner.

The per-fill signal is misleading here: the average maker fill carries a small rebate, but the cohort does the overwhelming majority of its *dollar volume* as takers, so in aggregate they're large net fee-payers. (Lesson: never read a fee structure off per-fill means.)

So these are **directional traders who pay fees**, not market-makers harvesting rebates. Which closes the escape hatch: their profit is genuinely directional — and that directional profit is ETH beta, not alpha. Two independent decompositions, same conclusion.

## Why I ran it in-sample on purpose

The 77 are post-hoc winner-selected — they're in the dataset *because* they won. Naively scoring them on the same window is in-sample, and in-sample inflates any apparent edge.

A clean out-of-sample test isn't possible on this data yet: the cohort was selected on 2026-05-30, so there's only ~2 days of post-selection data — not enough for a 7-day forward return to even resolve. I verified this rather than assume it.

So I deliberately ran the test **in-sample, which is a tailwind for the cohort.** The logic: *if their edge doesn't survive beta-adjustment even on their own most-favorable window, out-of-sample is hopeless.* The finding is conservative by construction.

## What this does and doesn't claim

I want to be precise, because the honest version is more useful than the dramatic one:

- **It claims:** the leaderboard cohort's apparent edge is market beta, not alpha, even scored in-sample; only 4 of 24 evaluable wallets show beta-adjusted alpha.
- **It does NOT claim:** "copy-trading these wallets loses money out-of-sample." That's untestable on <2 days of OOS data. The honest statement is *un-knowability* out-of-sample plus *failure* in-sample — which together still indict the leaderboard's selection.
- It does NOT claim those specific 4 wallets are fake. They'd need their own OOS test. The claim is "the leaderboard ranks beta," not "every wallet is a fraud."

## Why I'm posting this

I'm building tooling to answer exactly this kind of question — is a backtested or observed edge real, or is it beta / overfit / survivorship? The leaderboard audit is one thing the engine can do; the code and method are open so you can check the numbers yourself rather than take my word for it: **https://github.com/Michaellyu11/hl-validator**

But mostly I'm posting because I want to hear from people who think about this seriously. So, a genuine question for anyone who builds strategies:

**When your own backtest looks good, what makes you decide it's real — and not just beta or an overfit to one regime?** Gut, walk-forward, something formal, or do you just trust the equity curve?

I'd rather hear your honest method than have you take mine.

---

*Every number above is engine-computed and independently re-verified. Methodology, kill-test definitions, and an example verdict are in the repo. Corrections welcome — especially if you can poke a hole in the beta-decomposition.*
