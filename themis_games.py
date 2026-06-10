"""
Interactive Streamlit App illustrating the dynamics of three cooperation games,
based on Carl Rasmussen's Themis Mechanism talk.


Run with:
    streamlit run themis_games.py
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Themis: Cooperation Games",
    page_icon="🌍",
    layout="wide",
)

YOU_COLOR = "#e45756"
AGENT_COLOR = "#4c78a8"

st.title("Themis Mechanism: Cooperation Games")
st.markdown(
    "Three interactive games illustrating the dynamics of cooperation, by showing why some ways of organising a shared effort collapse into free-riding, while others make cooperation a self-interested choice. You play one player, all others are simulated agents with tunable parameters in the sidebar."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️  Simulation")

    CUR = st.selectbox(
        "Currency symbol",
        ["€", "$", "£"],
        index=0,
        help="Cosmetic only — sets the symbol used throughout the app.",
    )
    endowment = st.number_input(f"Endowment per player ({CUR})", 1.0, 1000.0, 1.0, 0.5)

    st.divider()
    st.subheader("Behaviour of agents")
    mean_coop = st.slider(
        "Mean cooperativeness",
        0.0,
        1.0,
        0.5,
        0.05,
        help="The fraction of their endowment a typical simulated agent wants to contribute.",
    )
    coop_spread = st.slider(
        "Spread between agents",
        0.0,
        0.5,
        0.15,
        0.01,
        help="Standard deviation of cooperativeness across the population of agents.",
    )

n_others = 99  # fixed: 100 players in total
n_total = n_others + 1
rng = np.random.default_rng()


def money(x: float, dec: int = 2) -> str:
    """Format a number with the chosen currency symbol."""
    return f"{CUR}{x:,.{dec}f}"


# ---------------------------------------------------------------------------
# Agent model
# ---------------------------------------------------------------------------
def sample_coops(n: int) -> np.ndarray:
    """Cooperativeness ~ N(mean_coop, coop_spread), clipped to [0, 1]."""
    return np.clip(rng.normal(mean_coop, coop_spread, size=n), 0.0, 1.0)


def agent_contributions(target_per_agent: float) -> np.ndarray:
    """A vector of contributions for the agents, given their per-agent target."""
    coops = sample_coops(n_others)
    raw = coops * target_per_agent
    return np.clip(raw, 0.0, endowment)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def contrib_chart(
    contribs: np.ndarray,
    payouts: np.ndarray,
    value_label: str,
):
    """Sorted bar chart of every player's contribution, highlighting you.

    contribs[0] / payouts[0] is *you*. Hovering a bar shows that player's
    contribution and their resulting final money.
    """
    sort_idx = np.argsort(contribs)
    user_pos = int(np.where(sort_idx == 0)[0][0])
    df = pd.DataFrame(
        {
            "rank": np.arange(len(contribs)),
            "value": contribs[sort_idx],
            "payout": np.asarray(payouts)[sort_idx],
            "who": ["You" if i == user_pos else "Other" for i in range(len(contribs))],
        }
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "rank:O",
                title="Players (sorted by contribution)",
                axis=alt.Axis(labels=False, ticks=False),
            ),
            y=alt.Y("value:Q", title=value_label),
            color=alt.Color(
                "who:N",
                scale=alt.Scale(
                    domain=["You", "Other"], range=[YOU_COLOR, AGENT_COLOR]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("who:N", title="Player"),
                alt.Tooltip("value:Q", title=value_label, format=".3f"),
                alt.Tooltip("payout:Q", title=f"Final money ({CUR})", format=".3f"),
            ],
        )
        .properties(height=240)
    )


def payoff_curve_chart(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    x_now: float,
    y_now: float,
    x_label: str,
):
    """Line chart of *your* final money as a function of *your own* contribution,
    holding everyone else fixed at this round's behaviour. The red dot marks
    the choice you actually made; the grey dashed line is your endowment."""
    df = pd.DataFrame({"x": grid_x, "y": grid_y})
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=3, color=AGENT_COLOR)
        .encode(
            x=alt.X("x:Q", title=x_label),
            y=alt.Y(
                "y:Q",
                title=f"Your final money ({CUR})",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("x:Q", title=x_label, format=".3f"),
                alt.Tooltip("y:Q", title=f"Your final money ({CUR})", format=".3f"),
            ],
        )
    )
    endow_rule = (
        alt.Chart(pd.DataFrame({"y": [endowment], "label": ["endowment"]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    you_df = pd.DataFrame({"x": [x_now], "y": [y_now]})
    you_rule = (
        alt.Chart(you_df).mark_rule(color=YOU_COLOR, strokeDash=[4, 3]).encode(x="x:Q")
    )
    you_pt = (
        alt.Chart(you_df)
        .mark_point(size=170, color=YOU_COLOR, filled=True, opacity=1)
        .encode(
            x="x:Q",
            y="y:Q",
            tooltip=[
                alt.Tooltip("x:Q", title="Your choice", format=".3f"),
                alt.Tooltip("y:Q", title=f"Your final money ({CUR})", format=".3f"),
            ],
        )
    )
    return (line + endow_rule + you_rule + you_pt).properties(height=280)


# ---------------------------------------------------------------------------
# Outcome overview
# ---------------------------------------------------------------------------
def show_payout_overview(
    payouts: np.ndarray,
    total_contribution: float,
    *,
    contribution_label: str | None = None,
    contribution_note: str | None = None,
    contribution_help: str = "Total money that actually went into the communal pot this round.",
):
    """Renders an overview of the outcome of one round.

    "Final money" = the part of the endowment a player did *not* invest, plus
    their share of the (doubled) pot — i.e. what they walk away with.
    """
    if contribution_label is None:
        contribution_label = f"Total contribution ({CUR})"

    user_payout = float(payouts[0])
    agents = payouts[1:]
    avg_payout = float(payouts.mean())
    best_agent = float(agents.max())
    worst_agent = float(agents.min())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        f"Your final money ({CUR})",
        f"{user_payout:.3f}",
        f"{user_payout - endowment:+.3f} vs start",
        help=f"Your final money = the part of your {money(endowment)} endowment you did **not** invest, plus your share of the (doubled) pot — what you walk away with. Green/up = you ended richer than your starting {money(endowment)}; red/down = you lost money overall.",
    )
    c2.metric(
        f"Average final money ({CUR})",
        f"{avg_payout:.3f}",
        f"{avg_payout - endowment:+.3f} vs start",
        help="Mean final money across everyone (you + all agents). Shows whether the group as a whole created or destroyed value.",
    )
    c3.metric(
        f"Max agent final money ({CUR})",
        f"{best_agent:.3f}",
        f"{best_agent - user_payout:+.3f} vs you",
        delta_color="off",
        help="Best-off simulated agent. If this is above your final money, someone did better than you — usually by contributing less.",
    )
    c4.metric(
        f"Min agent final money ({CUR})",
        f"{worst_agent:.3f}",
        f"{worst_agent - user_payout:+.3f} vs you",
        delta_color="off",
        help="Worst-off simulated agent. Together with the max, this is the spread of outcomes among the other players.",
    )
    c5.metric(
        contribution_label,
        f"{total_contribution:.2f}",
        contribution_note,
        delta_color="off",
        help=contribution_help,
    )

    st.caption(
        f"**Final money** = the part of your **{money(endowment)}** endowment you did *not* invest **+** your "
        "share of the doubled pot — i.e. what you actually walk away with. The first two cards compare it "
        f"against your **{money(endowment)}** starting endowment (green = gain, red = loss). "
        "“Max/min agent final money” show the best- and worst-off of the other players, measured **relative to you**."
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "1️⃣  Independent commitment",
        "2️⃣  Fixed budget",
        "3️⃣  Common commitment",
    ]
)

# ===========================================================================
# Game 1 – Independent commitment
# ===========================================================================
with tab1:
    st.subheader("Game 1 · Independent commitment")

    # --- 1. How it works ---------------------------------------------------
    st.markdown("#### How it works")
    st.markdown(f"""
Everyone starts with the same endowment of **{money(endowment)}**. Each round:

1. You choose how much of your endowment to put into a communal pot.
2. The pot is **doubled**.
3. The doubled pot is **split equally** among all **{n_total}** players — regardless of who paid in.

""")

    # --- 2. Your move ------------------------------------------------------
    st.markdown("#### Your move")
    user_contrib = st.slider(
        f"Your contribution ({CUR})",
        0.0,
        float(endowment),
        float(endowment) / 2,
        float(endowment) / 100,
        key="g1_user",
    )
    play1 = st.button("▶ Play", key="g1_play", type="primary")

    # --- 3. What happened --------------------------------------------------
    if play1:
        st.markdown("#### What happened")

        others = agent_contributions(endowment)
        contribs = np.concatenate([[user_contrib], others])
        pot = contribs.sum() * 2
        share = pot / n_total
        payouts = (endowment - contribs) + share

        show_payout_overview(
            payouts,
            contribs.sum(),
            contribution_note=f"of max {money(endowment * n_total, 0)}",
            contribution_help="Total money put into the pot by all players. It doubles, then is split equally.",
        )

        roi_per_euro = (2 / n_total) - 1
        st.info(
            f"📉 **Why defection wins here.** Every {money(1)} you put in is doubled and split "
            f"{n_total} ways, so it returns just **{money(2 / n_total, 2)}** to *you*. This is a return of "
            f"**{roi_per_euro * 100:.0f}%** on your own money. Collectively, cooperation is great: "
            f"if all {n_total} players gave everything, everyone would double up to "
            f"**{money(2 * endowment)}**. But individually your best move is to contribute nothing and "
            f"free-ride on the others."
        )

        st.markdown("##### Where everyone landed")
        st.altair_chart(
            contrib_chart(contribs, payouts, f"Contribution ({CUR})"),
            width="stretch",
        )

        st.markdown("##### Your final money vs. how much you put in")
        grid = np.linspace(0.0, float(endowment), 101)
        s_others = others.sum()
        curve = (endowment - grid) + 2.0 * (grid + s_others) / n_total
        st.altair_chart(
            payoff_curve_chart(
                grid,
                curve,
                user_contrib,
                float(payouts[0]),
                f"Your contribution ({CUR})",
            ),
            width="stretch",
        )
        st.caption(
            "This line is *your* final money as you vary your own contribution, holding everyone "
            "else fixed at this round's choices. It slopes **down**. Every extra unit you contribute lowers your own final money. "
            "Your best response is to be on the far left, by contributing nothing."
        )


# ===========================================================================
# Game 2 – Fixed budget
# ===========================================================================
with tab2:
    st.subheader("Game 2 · Fixed budget")

    # --- 1. How it works ---------------------------------------------------
    st.markdown("#### How it works")
    st.markdown(f"""
Everyone starts with the same endowment of **{money(endowment)}**. Similar to Game 1, the money can be invested into a communal pot that doubles and is shared equally, but only if the total pledges reach a certain **budget B**. Each round:

1. Everyone pledges an amount.
2. **If the pledges reach the budget**, exactly *B* is collected (pledges above *B* are
   refunded proportionally), the pot doubles to 2·*B*, and 2·*B* is split equally. So each
   player receives the **same fixed amount** from the pot, *no matter how much they personally paid*.
3. **If the pledges fall short**, the deal collapses and everyone simply keeps their money.

The crucial change from Game 1: once the budget is met, the benefit each person draws from
the pot is **fixed**. The size of the pie is settled with the only question: *who pays for it*?
""")

    # --- 2. Your move ------------------------------------------------------
    st.markdown("#### Your move")
    budget_frac = st.slider(
        "Budget (as a fraction of the total endowment in the room)",
        0.05,
        1.0,
        0.5,
        0.05,
        key="g2_budget",
    )
    budget = budget_frac * endowment * n_total
    fair_share = budget / n_total
    st.caption(
        f"Total budget **{money(budget)}** across {n_total} players → "
        f"fair share **{money(fair_share, 3)}** per player. "
        f"If the budget is met, every player receives a fixed **{money(2 * budget / n_total, 3)}** from the pot."
    )

    user_contrib2 = st.slider(
        f"Your pledge ({CUR})",
        0.0,
        float(endowment),
        float(fair_share),
        float(endowment) / 100,
        key="g2_user",
    )
    play2 = st.button("▶ Play", key="g2_play", type="primary")

    # --- 3. What happened --------------------------------------------------
    if play2:
        st.markdown("#### What happened")

        others = agent_contributions(endowment)
        pledges = np.concatenate([[user_contrib2], others])
        total = pledges.sum()

        if total >= budget:
            actual = pledges * (budget / total)  # take exactly B, proportionally
            share = 2 * budget / n_total
            payouts = (endowment - actual) + share
            success = True
        else:
            payouts = np.full(n_total, endowment)
            success = False

        if success:
            st.success(
                f"✅ **Budget met** — {money(total)} pledged ≥ {money(budget)} budget. "
                "The pot formed and pledges above the budget were refunded proportionally."
            )
        else:
            st.error(
                f"❌ **Budget missed** — {money(total)} pledged < {money(budget)} budget. "
                "Everyone tried to free-ride, the deal collapsed and all pledges were returned. "
                "Thus, everyone's final money falls back to the endowment."
            )

        show_payout_overview(
            payouts,
            total,
            contribution_note=f"vs budget {money(budget)}",
            contribution_help="Sum of everyone's pledges. The pot is capped at the budget; anything above is refunded. If the total falls short, the game fails.",
        )

        st.info(
            f"🎭 **Why this turns players into rivals.** Once the budget is reached, the pot pays a "
            f"**fixed {money(2 * budget / n_total, 3)}** to every player, regardless of who funded it. "
            f"The size of the pie is settled, so all that's left is a fight over **who pays**. Every unit "
            f"you pledge is a cost for a benefit that is already locked in and therefore a pure loss to you, "
            f"and a pure gain to whoever pledges less. That makes it a **zero-sum, adversarial** game: there is "
            f"a fixed budget to raise and players push the cost onto one another. Real cooperation can't emerge."
        )

        st.markdown("##### Where everyone landed")
        st.altair_chart(
            contrib_chart(pledges, payouts, f"Pledge ({CUR})"),
            width="stretch",
        )

        st.markdown("##### Your final money vs. how much you pledge")
        grid = np.linspace(0.0, float(endowment), 101)
        s_others = others.sum()
        totals = grid + s_others
        safe_totals = np.where(totals == 0, 1.0, totals)
        met = totals >= budget
        actual_u = np.where(met, grid * budget / safe_totals, 0.0)
        curve = np.where(met, (endowment - actual_u) + 2 * budget / n_total, endowment)
        st.altair_chart(
            payoff_curve_chart(
                grid, curve, user_contrib2, float(payouts[0]), f"Your pledge ({CUR})"
            ),
            width="stretch",
        )
        st.caption(
            "Your final money is **flat at your endowment** for scenarios where the group is below budget (the deal "
            "fails and you keep your money). If the budget is met, it **slopes down**. Now, pledging "
            "more only enlarges your share of the bill while your benefit stays fixed. Best response: "
            "pledge as little as you can while still hoping the others reach the budget."
        )


# ===========================================================================
# Game 3 – Common commitment
# ===========================================================================
with tab3:
    st.subheader("Game 3 · Common commitment")

    # --- 1. How it works ---------------------------------------------------
    st.markdown("#### How it works")
    st.markdown("""
You **pledge** any amount of your endowment. You can pick between
**two settlement rules** below. In both, the collected amount is **doubled** and split equally **only
among the players who actually took part**:

- **Option 1 — Common minimum.** Only the part of each pledge that is *common to everyone* (the
  **lowest pledge across all players**) is collected. Anything above that minimum is handed back.
  *Everyone* takes part, at the same common level.
- **Option 2 — Optimal threshold.** A threshold is chosen to **maximise the total money raised** for
  the pot. Every player who pledged **at least the threshold** takes part and contributes exactly the
  threshold. Players who pledged **less drop out** and simply keep their endowment (they take **no**
  share of the pot).
""")

    # --- 2. Settlement rule ------------------------------------------------
    st.markdown("#### Settlement rule")
    g3_mode = st.radio(
        "How should the common commitment be settled?",
        [
            "Option 1 — Common minimum (everyone shares the lowest pledge)",
            "Option 2 — Optimal threshold (maximise the total pot)",
        ],
        key="g3_mode",
        help=(
            "Option 1 collects the single lowest pledge from everyone, so one player can sets the "
            "common level for all. Option 2 instead picks the threshold that raises the most money, "
            "letting players who pledged below it drop out."
        ),
    )
    use_threshold = g3_mode.startswith("Option 2")

    # --- 3. Your move ------------------------------------------------------
    st.markdown("#### Your move")
    user_pledge = st.slider(
        f"Your pledge ({CUR})",
        0.0,
        float(endowment),
        float(endowment),
        float(endowment) / 100,
        key="g3_user",
    )
    play3 = st.button("▶ Play", key="g3_play", type="primary")

    # --- 4. What happened --------------------------------------------------
    if play3:
        st.markdown("#### What happened")

        others = agent_contributions(endowment)
        pledges = np.concatenate([[user_pledge], others])

        if not use_threshold:
            # ---------------- Option 1: common minimum --------------------
            common = pledges.min()
            payouts = np.full(
                n_total, endowment + common
            )  # = endowment - common + 2*common

            show_payout_overview(
                payouts,
                common * n_total,
                contribution_note=f"of max {money(endowment * n_total, 0)}",
                contribution_help=f"Only the common (minimum) pledge of {money(common, 3)} from each player is invested; the rest stays in everyone's pocket.",
            )
            st.caption(
                f"Notice **max = min = your final money ({money(float(payouts[0]), 3)})**: with common "
                f"commitment everyone receives exactly the same. The common commitment was "
                f"**{money(common, 3)}**, set by the least cooperative player. "
            )

            st.success(
                "✅ **Why honesty and generosity become safe:** You can only ever lose what you have "
                "**in common** with everyone else, so a free-rider who pledges little can't exploit you. "
                "They only lower the common amount for *everyone*, themselves included. Raising your own "
                "pledge never lowers your final money and might raise the common floor, so **pledging your full "
                "endowment is the dominant strategy.** "
            )

            st.markdown("##### Where everyone landed")
            bar = contrib_chart(pledges, payouts, f"Pledge ({CUR})")
            min_line = (
                alt.Chart(pd.DataFrame({"y": [common]}))
                .mark_rule(color="white", strokeDash=[6, 4], size=2)
                .encode(y="y:Q")
            )
            min_label = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "y": [common],
                            "text": [f"common commitment = {money(common, 3)}"],
                        }
                    )
                )
                .mark_text(align="left", dx=5, dy=-6, color="white", fontWeight="bold")
                .encode(x=alt.value(5), y="y:Q", text="text:N")
            )
            st.altair_chart(bar + min_line + min_label, width="stretch")

            st.markdown("##### Your final money vs. how much you pledge")
            grid = np.linspace(0.0, float(endowment), 101)
            others_min = others.min()
            curve = endowment + np.minimum(grid, others_min)
            st.altair_chart(
                payoff_curve_chart(
                    grid, curve, user_pledge, float(payouts[0]), f"Your pledge ({CUR})"
                ),
                width="stretch",
            )
            st.caption(
                f"Your final money **rises** as you pledge more, up to the lowest pledge among the others "
                f"({money(others_min, 3)}) and then goes **flat**. Crucially it never slopes down: pledging "
                f"more can only help or do nothing. Compare this with Games 1 and 2, where the curve always "
                f"fell. That sign flip is the "
                f"whole point of *common commitment*, and the core of the Themis idea."
            )

        else:
            # ---------------- Option 2: optimal threshold -----------------
            # Choose the threshold t that maximises t × (#players pledging ≥ t).
            # The optimum is always one of the pledged values, so we sort the
            # pledges descending and read off t × count for each. This is the
            # Themis elicitation: p̂ = argmax p × c(p).
            sorted_desc = np.sort(pledges)[::-1]
            totals = sorted_desc * np.arange(1, n_total + 1)
            best_k = int(np.argmax(totals))
            threshold = float(sorted_desc[best_k])

            participate = pledges >= threshold
            n_part = int(participate.sum())
            total_collected = threshold * n_part
            # The doubled pot is shared only among the participants.
            share = (2 * total_collected / n_part) if n_part > 0 else 0.0
            payouts = np.where(participate, (endowment - threshold) + share, endowment)

            you_in = bool(participate[0])
            you_payout = float(payouts[0])

            show_payout_overview(
                payouts,
                total_collected,
                contribution_label=f"Collected for pot ({CUR})",
                contribution_note=f"from {n_part} of {n_total} players",
                contribution_help=(
                    f"Option 2 picks the threshold {money(threshold, 3)} that maximises "
                    f"threshold × participants. Each of the {n_part} participants pays "
                    f"{money(threshold, 3)}; the {n_total - n_part} who pledged less keep their "
                    f"endowment and take no share of the pot."
                ),
            )

            if you_in:
                st.caption(
                    f"The optimal threshold is **{money(threshold, 3)}**. You pledged "
                    f"**{money(user_pledge, 3)}** ≥ threshold, so **you took part**: you paid "
                    f"{money(threshold, 3)} and your final money is **{money(you_payout, 3)}** "
                    f"(= {money(endowment - threshold, 3)} kept + {money(2 * threshold, 3)} from the pot). "
                    f"**{n_part} of {n_total}** players took part."
                )
            else:
                st.caption(
                    f"The optimal threshold is **{money(threshold, 3)}**. You pledged "
                    f"**{money(user_pledge, 3)}** < threshold, so **you dropped out** and simply kept your "
                    f"endowment of **{money(endowment, 3)}**. **{n_part} of {n_total}** players took part."
                )

            st.success(
                "🏛️ **This is the Themis elicitation process in miniature.** Instead of letting the single "
                "least-cooperative player set a low common level for everyone (Option 1), Option 2 picks the "
                f"threshold that **maximises the total raised**. In this scenario: threshold × participants = "
                f"{money(threshold, 3)} × {n_part} = **{money(total_collected, 2)}**. Players who won't meet "
                "the threshold simply drop out. Pledging your full endowment is your **dominant strategy**. It "
                "can only get you into the coalition or raise a threshold you yourself benefit from."
            )

            st.markdown("##### Where everyone landed")
            bar = contrib_chart(pledges, payouts, f"Pledge ({CUR})")
            thr_line = (
                alt.Chart(pd.DataFrame({"y": [threshold]}))
                .mark_rule(color="white", strokeDash=[6, 4], size=2)
                .encode(y="y:Q")
            )
            thr_label = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "y": [threshold],
                            "text": [
                                f"threshold = {money(threshold, 3)} ({n_part} take part)"
                            ],
                        }
                    )
                )
                .mark_text(align="left", dx=5, dy=-6, color="white", fontWeight="bold")
                .encode(x=alt.value(5), y="y:Q", text="text:N")
            )
            st.altair_chart(bar + thr_line + thr_label, width="stretch")
            st.caption(
                "Bars **at or above** the dashed threshold line are the participants (each contributes the "
                "threshold). Bars **below** it dropped out and kept their endowment. Hover any bar to see that "
                "player's final money. Participants end on the same level, dropouts stay at the endowment."
            )

            st.markdown("##### Your final money vs. how much you pledge")
            grid = np.linspace(0.0, float(endowment), 101)
            curve = np.empty_like(grid)
            for i, g in enumerate(grid):
                pl = np.concatenate([[g], others])
                sd = np.sort(pl)[::-1]
                tot = sd * np.arange(1, n_total + 1)
                thr = float(sd[int(np.argmax(tot))])
                curve[i] = (endowment + thr) if g >= thr else endowment
            st.altair_chart(
                payoff_curve_chart(
                    grid, curve, user_pledge, you_payout, f"Your pledge ({CUR})"
                ),
                width="stretch",
            )
            st.caption(
                "Your final money **never slopes down** as you pledge more: pledging at least the threshold "
                "gets you into the coalition (a jump up), and pledging more can only push the threshold, and "
                "with it your own final money upwards. As in Option 1, the curve never falls, so **pledging "
                "your full endowment stays the dominant strategy.** The difference is that a single stingy "
                "player can no longer shrink the pot for everyone. They just drop out."
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Based on slides by **Carl Edward Rasmussen** (Cambridge University) — "
    '"Climate Change Cooperation: the Themis Mechanism", Pioneer Centre for AI, '
    "Copenhagen, 22 April 2026."
)
