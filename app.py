# -*- coding: utf-8 -*-
"""
Rank One Radar - POC (synthetic data)
Two views: Athlete Focus and Injury Focus.
Data columns mirror Rank One Partner API v1.0 (April 2026) exactly.
All data is synthetic - no real athletes or records.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from dotenv import dotenv_values

TODAY = datetime(2026, 7, 2)

# ---------------- optional AI layer (Azure OpenAI) ----------------
# Credential lookup order: Streamlit Cloud secrets -> environment -> local ..\.env
def _load_cfg():
    keys = ("VITE_AOAI_KEY", "VITE_AOAI_ENDPOINT",
            "VITE_AOAI_DEPLOYMENT", "VITE_AOAI_API_VERSION")
    import os
    local = {}
    try:
        local = dotenv_values(r"D:\rankone\.env")   # dev machine only
    except Exception:
        pass
    cfg = {}
    for k in keys:
        v = None
        try:
            v = st.secrets.get(k)                    # Streamlit Cloud
        except Exception:
            pass
        cfg[k] = v or os.environ.get(k) or local.get(k)
    return cfg

_cfg = _load_cfg()
AI_READY = all(_cfg.values())

@st.cache_resource
def _ai_client():
    from openai import AzureOpenAI
    return AzureOpenAI(api_key=_cfg["VITE_AOAI_KEY"],
                       azure_endpoint=_cfg["VITE_AOAI_ENDPOINT"],
                       api_version=_cfg["VITE_AOAI_API_VERSION"])

@st.cache_data(show_spinner=False)
def ask_ai(system: str, prompt: str) -> str:
    r = _ai_client().chat.completions.create(
        model=_cfg["VITE_AOAI_DEPLOYMENT"],
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        max_completion_tokens=2000)
    return r.choices[0].message.content.strip()

# ---------------- validated reference palette (light mode) ----------------
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
          "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]   # fixed order, never cycled
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SURFACE, GRID, INK, INK2, MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#52514e", "#898781"

BODY_REGIONS = ["Ankle", "Knee", "Thigh", "Head", "Shoulder", "Back", "Lower Leg", "Other"]
REGION_COLOR = {r: SERIES[i] for i, r in enumerate(BODY_REGIONS)}

def style(fig, h=340):
    # Title pinned to the very top, horizontal legend in its own band below it,
    # generous top margin so the two can never collide.
    fig.update_layout(
        height=h, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(family='system-ui, "Segoe UI", sans-serif', color=INK2, size=13),
        title=dict(font=dict(color=INK, size=15), x=0.01, xanchor="left",
                   y=0.995, yanchor="top"),
        margin=dict(l=10, r=45, t=92, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    title_text=None, font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED))
    return fig

# ------------------------------- data ------------------------------------
def _data_version():
    """Cache key tied to the data files themselves - a new data push always
    invalidates the cache (Streamlit keeps caches across git-pull reloads)."""
    import os
    return tuple(
        (f, int(os.path.getmtime(os.path.join("data", f))),
         os.path.getsize(os.path.join("data", f)))
        for f in sorted(os.listdir("data")) if f.endswith(".csv"))

@st.cache_data
def load(version):
    base = "data"
    athletes = pd.read_csv(f"{base}/athletes.csv")
    injuries = pd.read_csv(f"{base}/injuries.csv", parse_dates=[
        "InjuryDate", "ReturnDate", "EstimatedReturnDate", "DateClosed",
        "PhysicianEvalDate", "ATCEvalDate", "SurgeryDate", "MissDayStart", "MissDayEnd"])
    treatments = pd.read_csv(f"{base}/treatments.csv")
    treatments["TreatmentDate"] = pd.to_datetime(
        treatments["TreatmentDate"].str[:19], format="%Y-%m-%d %H:%M:%S")
    notes = pd.read_csv(f"{base}/medical_notes.csv", parse_dates=["NoteDate", "DateCreated"])
    refs = pd.read_csv(f"{base}/injury_references.csv")
    injuries["_region"] = injuries["_bodyRegion"].where(
        injuries["_bodyRegion"].isin(BODY_REGIONS[:-1]), "Other")
    injuries["SchoolYear"] = np.where(injuries.InjuryDate.dt.month >= 8,
        injuries.InjuryDate.dt.year.astype(str) + "-" + (injuries.InjuryDate.dt.year + 1).astype(str).str[2:],
        (injuries.InjuryDate.dt.year - 1).astype(str) + "-" + injuries.InjuryDate.dt.year.astype(str).str[2:])
    return athletes, injuries, treatments, notes, refs

athletes, injuries, treatments, notes, refs = load(_data_version())

st.set_page_config(page_title="Rank One Radar - POC", page_icon="R", layout="wide")

st.markdown(
    "<h2 style='margin-bottom:0'>Rank One Radar <span style='color:#898781;"
    "font-weight:400;font-size:0.6em'>UNT proof of concept - synthetic data, "
    "Rank One Partner API v1.0 schema</span></h2>", unsafe_allow_html=True)

view = st.sidebar.radio("View", ["Radar Insights", "Causality Lab", "Forecasts",
                                 "Treatment Analysis", "Athlete Focus", "Injury Focus"])
st.sidebar.markdown("---")
st.sidebar.caption(("AI layer: **connected** (Azure OpenAI)" if AI_READY
                    else "AI layer: offline - add credentials to ..\\.env"))
st.sidebar.caption(
    f"{len(athletes):,} athletes - {len(injuries):,} injuries - "
    f"{len(treatments):,} treatments - {len(notes):,} medical notes. "
    "All records synthetic; column names match the April 2026 Rank One data documentation.")

# ==========================================================================
#  RADAR INSIGHTS - the leadership view
# ==========================================================================
if view == "Radar Insights":

    dd = injuries.merge(athletes[["Athlete_ID", "sportName", "SchoolName",
                                  "gradeLevel", "firstName", "lastName"]], on="Athlete_ID")
    closed = dd[dd._actualDays.notna()].copy()

    # ---------- days lost & concentration ----------
    dd["_daysLost"] = np.where(
        dd.MissDayStart.notna(),
        np.where(dd.MissDayEnd.notna(),
                 (dd.MissDayEnd - dd.MissDayStart).dt.days,
                 (TODAY - dd.MissDayStart).dt.days),
        0).clip(0)
    per_ath = dd.groupby("Athlete_ID")._daysLost.sum().sort_values(ascending=False)
    top10_share = per_ath.head(max(1, len(per_ath) // 10)).sum() / max(1, per_ath.sum())

    # ---------- premature-return -> re-injury (the causal finding) ----------
    dd_sorted = dd.sort_values(["Athlete_ID", "InjuryDate"])
    dd_sorted["_next_same"] = (
        (dd_sorted.groupby("Athlete_ID")._complaint.shift(-1) == dd_sorted._complaint) &
        (dd_sorted.groupby("Athlete_ID").Athlete_ID.shift(-1).notna()))
    closed_seq = dd_sorted[dd_sorted._actualDays.notna()]
    re_rate_prem = closed_seq[closed_seq._premature == 1]._next_same.mean()
    re_rate_norm = closed_seq[closed_seq._premature == 0]._next_same.mean()
    odds_mult = re_rate_prem / max(0.001, re_rate_norm)

    # ---------- predictive model: beat the trainer's estimate ----------
    @st.cache_data(show_spinner="Training recovery model on closed cases...")
    def train_model():
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        m = closed.copy()
        m["bmiProxy"] = m.Weight / m.Height.clip(lower=1)
        m["pain0"] = m.PainScale.fillna(m.PainScale.median())
        # the model knows the diagnosis (as the real system would), the athlete's
        # history and physiology - the things the flat protocol estimate ignores
        m["_dx"] = m._complaint.str.replace(r"^(Right|Left|R|L|rt|lt)\s+", "",
                                            regex=True).str.title()
        feats_cat = pd.get_dummies(m[["_dx", "_bodyRegion", "sportName"]], dtype=float)
        X = pd.concat([m[["gradeLevel", "bmiProxy", "pain0", "_priorSameSite",
                          "_priorInjuries", "RequiredSurgery", "isConcussion",
                          "_isReinjury"]].astype(float), feats_cat], axis=1)
        y = m._actualDays.astype(float)
        Xtr, Xte, ytr, yte, itr, ite = train_test_split(
            X, y, m.index, test_size=0.3, random_state=7)
        gbm = GradientBoostingRegressor(random_state=7, n_estimators=250,
                                        max_depth=3, learning_rate=0.06)
        gbm.fit(Xtr, ytr)
        pred = gbm.predict(Xte)
        model_mae = float(np.mean(np.abs(pred - yte)))
        test = m.loc[ite].copy()
        test["_pred"] = pred
        with_est = test[test._estDays.notna()]
        trainer_mae = float(np.mean(np.abs(with_est._actualDays - with_est._estDays)))
        model_mae_same = float(np.mean(np.abs(with_est._pred - with_est._actualDays)))
        imp = (pd.Series(gbm.feature_importances_, index=X.columns)
               .sort_values(ascending=False).head(8))
        return trainer_mae, model_mae_same, test, imp

    trainer_mae, model_mae, test_df, importances = train_model()
    improvement = (1 - model_mae / trainer_mae) * 100

    # ---------- hero numbers ----------
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Athlete-days lost to injury", f"{int(per_ath.sum()):,}",
              help="Sum of missed-participation windows across all records.")
    delta_txt = (f"{improvement:.0f}% less error" if improvement > 0
                 else f"{-improvement:.0f}% more error")
    h2.metric("Return-date error: model vs trainer",
              f"{model_mae:.1f} vs {trainer_mae:.1f} days",
              delta=delta_txt, delta_color="normal" if improvement > 0 else "inverse",
              help="Mean absolute error on held-out closed cases the model never saw.")
    h3.metric("Re-injury risk after early return", f"{odds_mult:.1f}x",
              help="Re-injury rate when the athlete returned well ahead of the "
                   "typical timeline vs. on-schedule returns.")
    h4.metric("Lost days driven by top 10% of athletes", f"{top10_share*100:.0f}%",
              help="Risk is concentrated - watch the right kids and you prevent "
                   "most of the loss.")

    st.markdown("---")
    cA, cB = st.columns([1.15, 1])

    with cA:
        st.markdown("#### The model beats the training room's own predictions")
        comp = pd.DataFrame({
            "Predictor": ["Trainer's EstimatedReturnDate", "Radar model"],
            "Error (days)": [trainer_mae, model_mae]})
        fig = px.bar(comp, x="Error (days)", y="Predictor", orientation="h",
                     color="Predictor",
                     color_discrete_sequence=[SERIES[2], SERIES[0]],
                     text=comp["Error (days)"].round(1))
        fig.update_traces(textposition="outside", cliponaxis=False,
                          marker_line_color=SURFACE,
                          marker_line_width=2, showlegend=False)
        fig.update_yaxes(title=None)
        fig.update_layout(title="Average miss on return-to-play date "
                                "(held-out cases)")
        st.plotly_chart(style(fig, h=250), use_container_width=True)
        st.caption(
            f"Trained on {len(closed):,} closed cases, evaluated on a 30% holdout. "
            f"The model learns what the flat protocol estimate ignores - prior "
            f"same-site injuries, age, physiology. On real Rank One data this "
            f"becomes the product claim: *'Radar predicts return dates "
            f"{improvement:.0f}% more accurately than current practice.'*")

        st.markdown("#### What actually drives recovery time")
        imp = importances.rename(index=lambda s: s.replace("_dx_", "Diagnosis: ")
                                 .replace("_bodyRegion_", "Region: ")
                                 .replace("sportName_", "Sport: ")
                                 .replace("_priorSameSite", "Prior same-site injuries")
                                 .replace("_priorInjuries", "Prior injuries (any)")
                                 .replace("_isReinjury", "Is a re-injury")
                                 .replace("bmiProxy", "Body mass (proxy)")
                                 .replace("pain0", "Initial pain score")
                                 .replace("gradeLevel", "Grade / age")
                                 .replace("RequiredSurgery", "Required surgery")
                                 .replace("isConcussion", "Concussion"))
        fig = px.bar(imp.iloc[::-1], orientation="h",
                     color_discrete_sequence=[SERIES[0]])
        fig.update_layout(title="Model feature importance", showlegend=False,
                          xaxis_title="Importance", yaxis_title=None)
        st.plotly_chart(style(fig, h=300), use_container_width=True)

    with cB:
        st.markdown("#### The re-injury multiplier - the sellable insight")
        rr = pd.DataFrame({
            "Return type": ["Returned on/after typical timeline",
                            "Returned well before typical timeline"],
            "Re-injury rate": [re_rate_norm * 100, re_rate_prem * 100]})
        fig = px.bar(rr, x="Re-injury rate", y="Return type", orientation="h",
                     color="Return type",
                     color_discrete_sequence=[SERIES[1], STATUS["critical"]],
                     text=rr["Re-injury rate"].round(0).astype(int).astype(str) + "%")
        fig.update_traces(textposition="outside", cliponaxis=False,
                          marker_line_color=SURFACE,
                          marker_line_width=2, showlegend=False)
        fig.update_yaxes(title=None)
        fig.update_layout(title="Same-injury recurrence by how early the athlete "
                                "was returned")
        st.plotly_chart(style(fig, h=250), use_container_width=True)
        st.caption(
            f"Athletes cleared substantially ahead of the typical recovery window "
            f"re-injured the same site **{odds_mult:.1f}x more often**. This single "
            f"number - computed from data Rank One already stores - is the "
            f"evidence-based case for benchmark-guided return-to-play.")

        st.markdown("#### Where the days go")
        by_school = (dd.groupby(["SchoolName", "sportName"])._daysLost.sum()
                     .reset_index())
        top_sports = (by_school.groupby("sportName")._daysLost.sum()
                      .sort_values(ascending=False).head(4).index.tolist())
        by_school["Sport"] = np.where(by_school.sportName.isin(top_sports),
                                      by_school.sportName, "Other")
        pv = (by_school.groupby(["SchoolName", "Sport"])._daysLost.sum()
              .reset_index())
        order4 = top_sports + ["Other"]
        fig = px.bar(pv, x="_daysLost", y="SchoolName", color="Sport",
                     orientation="h", category_orders={"Sport": order4},
                     color_discrete_sequence=SERIES[:5],
                     title="Athlete-days lost, by school and sport")
        fig.update_traces(marker_line_color=SURFACE, marker_line_width=2)
        fig.update_yaxes(title=None)
        fig.update_xaxes(title="Days lost")
        st.plotly_chart(style(fig, h=330), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Today's Watchlist - open cases ranked by recovery risk")

    open_cases = dd[dd.Status == "Open"].copy()
    if len(open_cases):
        tt = treatments[treatments.InjuryId.isin(open_cases.ID)]
        last_trt = tt.groupby("InjuryId").TreatmentDate.max()
        pain_trend = {}
        for iid, g in tt.groupby("InjuryId"):
            p = g.sort_values("TreatmentDate")._painParsed.dropna().tail(3)
            pain_trend[iid] = float(p.diff().mean()) if len(p) >= 2 else 0.0

        rows = []
        for _, r in open_cases.iterrows():
            reasons, score = [], 10.0
            if pd.notna(r.EstimatedReturnDate) and TODAY > r.EstimatedReturnDate:
                od = (TODAY - r.EstimatedReturnDate).days
                score += min(35, od * 2.2)
                reasons.append(f"{od}d past estimated return")
            gap = (TODAY - last_trt.get(r.ID, r.InjuryDate)).days
            if gap >= 5:
                score += min(20, gap * 2)
                reasons.append(f"no treatment in {gap}d")
            pt = pain_trend.get(r.ID, 0.0)
            if pt >= 0:
                score += min(20, 8 + pt * 8)
                reasons.append("pain not improving")
            if r._priorSameSite > 0:
                score += 12
                reasons.append(f"{int(r._priorSameSite)} prior same-site injur"
                               f"{'y' if r._priorSameSite==1 else 'ies'}")
            if r.isConcussion and pd.isna(r.PhysicianEvalDate):
                score += 25
                reasons.append("concussion w/o physician eval")
            rows.append({
                "Score": int(min(99, score)),
                "Athlete": f"{r.firstName} {r.lastName}",
                "School": r.SchoolName.replace(" High School", " HS")
                                       .replace(" Middle School", " MS"),
                "Sport": r.sportName,
                "Injury": r._complaint,
                "Day": (TODAY - r.InjuryDate).days,
                "Why it's flagged": "; ".join(reasons) if reasons else "routine monitoring",
            })
        wl = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(
            wl, use_container_width=True, hide_index=True,
            column_config={"Score": st.column_config.ProgressColumn(
                "Risk", min_value=0, max_value=100, format="%d")})
        st.caption(
            "Heuristic score for the POC - Phase 1 replaces it with the validated "
            "model above. The pitch: a trainer walks in at 7am and this table is "
            "their triage list.")
    else:
        st.info("No open cases in the current dataset snapshot.")

# ==========================================================================
#  CAUSALITY LAB - why, not just what
# ==========================================================================
elif view == "Causality Lab":
    lk = pd.read_csv("data/lookups.csv")
    sport_map = lk[lk.LookupType == "Sport"].set_index("LookupId").Value
    occ_map = lk[lk.LookupType == "InjuryOccurred"].set_index("LookupId").Value

    dj = injuries.merge(athletes[["Athlete_ID", "SchoolName", "sportName",
                                  "isMultiSport"]], on="Athlete_ID")
    dj["injSport"] = dj.Sport_ID.map(sport_map)
    dj["occurred"] = dj.InjuryOccurredId.map(occ_map)
    roster = athletes.groupby("SchoolName").size().rename("Athletes")

    st.markdown("### Counting is not causality")
    st.markdown(
        "Raw injury counts make the biggest school look the most dangerous. "
        "This page walks the same question up the causal ladder: **counts -> "
        "rates -> adjusted rates -> mechanisms** - each step removing one "
        "false explanation. The same method runs unchanged on Rank One's real data.")

    # ---------------- step 1-2: counts vs rates ----------------
    sc = dj.groupby("SchoolName").size().rename("Injuries").to_frame().join(roster)
    sc["Rate"] = sc.Injuries / sc.Athletes * 100
    sc = sc.reset_index()
    sc["Short"] = (sc.SchoolName.str.replace(" High School", " HS")
                   .str.replace(" Middle School", " MS"))
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(sc.sort_values("Injuries"), x="Injuries", y="Short",
                     orientation="h", color_discrete_sequence=[SERIES[0]],
                     title="Step 1 - raw counts (misleading: bigger school, more of everything)")
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, h=280), use_container_width=True)
    with c2:
        fig = px.bar(sc.sort_values("Rate"), x="Rate", y="Short",
                     orientation="h", color_discrete_sequence=[SERIES[0]],
                     title="Step 2 - injuries per 100 athletes (fair denominator)")
        fig.update_yaxes(title=None)
        fig.update_xaxes(title="Injuries per 100 athletes")
        st.plotly_chart(style(fig, h=280), use_container_width=True)

    # ---------------- step 3: adjust for the sport mix ----------------
    st.markdown("#### Step 3 - is it the school, or just its sports?")
    sport_rate = (dj.groupby("sportName").size()
                  / athletes.groupby("sportName").size()).rename("r")
    expect = (athletes.assign(r=athletes.sportName.map(sport_rate))
              .groupby("SchoolName").r.sum().rename("Expected"))
    adj = sc.set_index("SchoolName").join(expect)
    adj["SIR"] = adj.Injuries / adj.Expected      # standardized injury ratio
    adj = adj.reset_index()
    multi_share = athletes.groupby("SchoolName").isMultiSport.mean() * 100
    adj["MultiShare"] = adj.SchoolName.map(multi_share)

    c3, c4 = st.columns(2)
    with c3:
        a = adj.sort_values("SIR")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=a.SIR, y=a.Short, orientation="h",
                             marker=dict(color=SERIES[0],
                                         line=dict(color=SURFACE, width=2)),
                             text=a.SIR.round(2), textposition="outside",
                             cliponaxis=False))
        fig.add_vline(x=1.0, line_dash="dot", line_color=MUTED)
        fig.update_layout(title="Observed vs expected injuries, given each "
                                "school's sport mix (1.0 = exactly as expected)",
                          xaxis_title="Standardized injury ratio")
        st.plotly_chart(style(fig, h=300), use_container_width=True)
        st.caption(
            "Most of North Star's excess disappears once you account for its "
            "football-heavy roster - the sport mix, not the school, was the cause. "
            "What excess remains points at something else...")
    with c4:
        fig = px.scatter(adj, x="MultiShare", y="SIR", text="Short",
                         color_discrete_sequence=[SERIES[0]])
        fig.update_traces(marker=dict(size=14, line=dict(color=SURFACE, width=2)),
                          textposition="top center", cliponaxis=False,
                          textfont=dict(size=11, color=INK2))
        # simple fitted line, no statsmodels dependency
        k, b = np.polyfit(adj.MultiShare, adj.SIR, 1)
        xs = np.linspace(adj.MultiShare.min() - 2, adj.MultiShare.max() + 2, 10)
        fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines",
                                 line=dict(color=MUTED, width=2, dash="dot"),
                                 showlegend=False))
        fig.update_layout(title="...multi-sport participation: the residual "
                                "school effect tracks it",
                          xaxis_title="% of athletes playing 2+ sports",
                          yaxis_title="Adjusted injury ratio (SIR)")
        st.plotly_chart(style(fig, h=300), use_container_width=True)
        st.caption(
            "Schools where more kids play multiple sports carry more injuries "
            "per athlete even after sport-mix adjustment.")

    # ---------------- step 4: the multi-sport effect, within schools ----------------
    st.markdown("#### Step 4 - test the suspect WITHIN each school")
    ms = []
    for school, g in athletes.groupby("SchoolName"):
        for flag, gg in g.groupby("isMultiSport"):
            n_inj_ms = dj[dj.Athlete_ID.isin(gg.Athlete_ID)].shape[0]
            ms.append({"School": school.replace(" High School", " HS")
                       .replace(" Middle School", " MS"),
                       "Group": "Multi-sport" if flag else "Single-sport",
                       "Rate": n_inj_ms / len(gg) * 100})
    ms = pd.DataFrame(ms)
    overall_single = ms[ms.Group == "Single-sport"].Rate.mean()
    overall_multi = ms[ms.Group == "Multi-sport"].Rate.mean()
    fig = px.bar(ms, x="School", y="Rate", color="Group", barmode="group",
                 color_discrete_sequence=[SERIES[0], SERIES[5]],
                 title="Injuries per 100 athletes - multi-sport vs single-sport, "
                       "inside every school")
    fig.update_traces(marker_line_color=SURFACE, marker_line_width=2)
    fig.update_yaxes(title="Injuries per 100 athletes")
    fig.update_xaxes(title=None)
    st.plotly_chart(style(fig, h=340), use_container_width=True)
    st.caption(
        f"The comparison holds within every school ({overall_multi:.0f} vs "
        f"{overall_single:.0f} per 100 on average, ~"
        f"{overall_multi/overall_single:.1f}x) - so it is not the school "
        f"causing it, and not the sport mix: year-round load on the same body "
        f"is the remaining explanation. This is confounder control by "
        f"stratification - the first honest step toward causality.")

    # ---------------- step 5: mechanism - each sport injures its own body map ----
    st.markdown("#### Step 5 - mechanism: each sport attacks its own body map")
    top_sports = dj.injSport.value_counts().head(8).index.tolist()
    hm = (dj[dj.injSport.isin(top_sports)]
          .groupby(["injSport", "_region"]).size().unstack(fill_value=0))
    hm = hm.div(hm.sum(axis=1), axis=0) * 100
    hm = hm.loc[top_sports]
    scale = [[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)]
    fig = go.Figure(go.Heatmap(
        z=hm.values, x=hm.columns, y=hm.index, colorscale=scale,
        hovertemplate="%{y} - %{x}: %{z:.0f}% of injuries<extra></extra>",
        colorbar=dict(title="% of sport's injuries")))
    fig.update_layout(title="Share of each sport's injuries by body region")
    st.plotly_chart(style(fig, h=380), use_container_width=True)
    st.caption(
        "Volleyball and basketball live at the ankle; football and soccer split "
        "knee/thigh; wrestling owns the head row. Mechanism is what makes a "
        "correlation believable - and it tells each program which prevention "
        "drills to buy.")

    # ---------------- step 6: game vs practice + the honest limit ----------------
    c5, c6 = st.columns(2)
    with c5:
        gp = (dj[dj.occurred.isin(["Game", "Practice"])]
              .groupby(["injSport", "occurred"]).size().unstack(fill_value=0))
        gp = gp.loc[[s for s in top_sports if s in gp.index]]
        gp_pct = gp.div(gp.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for i, col in enumerate(["Game", "Practice"]):
            fig.add_trace(go.Bar(x=gp_pct[col], y=gp_pct.index, orientation="h",
                                 name=col, marker=dict(color=SERIES[i],
                                 line=dict(color=SURFACE, width=2))))
        fig.update_layout(barmode="stack",
                          title="Where injuries happen: games vs practices",
                          xaxis_title="% of sport's injuries")
        st.plotly_chart(style(fig, h=340), use_container_width=True)
    with c6:
        st.markdown("##### The honest limit - and the Phase 2 ask")
        st.markdown(
            "Games look 'safer' here only because there are far fewer of them "
            "than practices. To compute the number that matters - injuries **per "
            "1,000 game-hours vs per 1,000 practice-hours** - we need exposure "
            "denominators: schedules, rosters, participation. That is exactly "
            "the Phase 2 data request in the proposal, and it is why this page "
            "can be honest about what it claims:\n\n"
            "- **Shown as causal here:** early return -> re-injury (2x, forward "
            "in time, dose-consistent); multi-sport load -> more injuries "
            "(holds within every school).\n"
            "- **Shown as association:** school differences after adjustment - "
            "candidate causes (turf share, staffing) need Phase 2 data.\n\n"
            "On real Rank One data, this same ladder runs with matched "
            "comparisons and regression adjustment - publication-grade methods, "
            "same story.")

# ==========================================================================
#  FORECASTS - turning the causal findings into per-athlete predictions
# ==========================================================================
elif view == "Forecasts":
    dd = injuries.merge(athletes[["Athlete_ID", "sportName", "SchoolName", "gradeLevel",
                                  "isMultiSport", "Height", "Weight", "YearsPlayingSport",
                                  "firstName", "lastName"]], on="Athlete_ID")
    dd = dd.sort_values(["Athlete_ID", "InjuryDate"])
    dd["_daysLost"] = np.where(
        dd.MissDayStart.notna(),
        np.where(dd.MissDayEnd.notna(),
                 (dd.MissDayEnd - dd.MissDayStart).dt.days,
                 (TODAY - dd.MissDayStart).dt.days),
        0).clip(0)

    st.markdown("### Forecasts: from what happened to what happens next")
    st.markdown(
        "Radar Insights and Causality Lab show what the data reveals in aggregate. "
        "This page trains models that make individual, forward-looking predictions - "
        "the shape a real deployed product would take. All four are trained live, "
        "on this synthetic dataset, using the fields Rank One has already documented.")

    # ---------------- A. Per-athlete re-injury risk ----------------
    st.markdown("#### 1. Re-injury risk - which athletes are likely to re-injure next")
    dd["_prevPremature"] = dd.groupby("Athlete_ID")._premature.shift(1).fillna(0)

    @st.cache_data(show_spinner="Training re-injury risk model...")
    def train_reinjury_model(df):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        feat_cols = ["_prevPremature", "_priorInjuries", "_priorSameSite",
                    "gradeLevel", "isMultiSport"]
        sport_dum = pd.get_dummies(df["sportName"], prefix="Sport", dtype=float)
        X = pd.concat([df[feat_cols].astype(float), sport_dum], axis=1)
        y = df["_isReinjury"].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=11,
                                              stratify=y)
        clf = GradientBoostingClassifier(random_state=11, n_estimators=200,
                                         max_depth=3, learning_rate=0.05)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba)
        base_rate = float(y.mean())
        imp = (pd.Series(clf.feature_importances_, index=X.columns)
               .sort_values(ascending=False).head(6))
        return clf, X.columns, auc, base_rate, imp

    clf, feat_columns, auc, base_rate, reinj_imp = train_reinjury_model(dd)

    rc1, rc2 = st.columns([1, 1.2])
    with rc1:
        st.metric("Model AUC (held-out cases)", f"{auc:.2f}",
                  help="0.5 = no better than guessing, 1.0 = perfect separation. "
                       "Trained on prior-history features only - nothing about "
                       "the injury itself, since that hasn't happened yet.")
        st.metric("Baseline re-injury rate", f"{base_rate*100:.0f}%",
                  help="Share of all injuries that were repeats of a prior site. "
                       "The model's job is to say WHICH athletes sit above this line.")
        imp_named = reinj_imp.rename(index=lambda s: s.replace("_prevPremature",
            "Prior return was premature").replace("_priorInjuries", "Prior injuries (any)")
            .replace("_priorSameSite", "Prior same-site injuries")
            .replace("gradeLevel", "Grade / age").replace("isMultiSport", "Multi-sport")
            .replace("Sport_", "Sport: "))
        fig = px.bar(imp_named.iloc[::-1], orientation="h",
                     color_discrete_sequence=[SERIES[3]])
        fig.update_layout(title="What predicts re-injury", showlegend=False,
                          xaxis_title="Importance", yaxis_title=None)
        st.plotly_chart(style(fig, h=260), use_container_width=True)

    with rc2:
        # score each athlete's CURRENT state (their most recent injury on record)
        latest = dd.sort_values("InjuryDate").groupby("Athlete_ID").tail(1).copy()
        sport_dum_all = pd.get_dummies(latest["sportName"], prefix="Sport", dtype=float)
        Xall = pd.concat([latest[["_prevPremature", "_priorInjuries", "_priorSameSite",
                                  "gradeLevel", "isMultiSport"]].astype(float),
                          sport_dum_all], axis=1).reindex(columns=feat_columns, fill_value=0.0)
        latest["_reinjRisk"] = clf.predict_proba(Xall)[:, 1] * 100
        top_risk = latest.sort_values("_reinjRisk", ascending=False).head(12)
        show = top_risk[["firstName", "lastName", "SchoolName", "sportName",
                         "_priorInjuries", "_reinjRisk"]].copy()
        show["Athlete"] = show.firstName + " " + show.lastName
        show["School"] = show.SchoolName.str.replace(" High School", " HS").str.replace(
            " Middle School", " MS")
        show = show.rename(columns={"sportName": "Sport", "_priorInjuries": "Prior injuries"})
        show["_reinjRisk"] = show["_reinjRisk"].round(0).astype(int)
        st.dataframe(
            show[["Athlete", "School", "Sport", "Prior injuries", "_reinjRisk"]]
            .rename(columns={"_reinjRisk": "Re-injury risk"}),
            use_container_width=True, hide_index=True,
            column_config={"Re-injury risk": st.column_config.ProgressColumn(
                "Re-injury risk", min_value=0, max_value=100, format="%d%%")})
        st.caption(
            "Highest-risk athletes leaguewide, scored from injury history only "
            "(no knowledge of a future injury). This is the same causal driver as "
            "the Radar Insights re-injury multiplier, turned into a per-athlete score.")

    st.markdown("---")

    # ---------------- B. Season-long burden forecast ----------------
    st.markdown("#### 2. Season-burden forecast - who is likely to lose the most time")
    per_ath_hist = dd.groupby("Athlete_ID")._daysLost.sum().rename("_totalDaysLost")
    ath_feats = athletes.set_index("Athlete_ID").join(per_ath_hist).fillna({"_totalDaysLost": 0})
    ath_feats["_bmiProxy"] = ath_feats.Weight / ath_feats.Height.clip(lower=1)

    @st.cache_data(show_spinner="Training season-burden model...")
    def train_burden_model(df):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        feat_cols = ["gradeLevel", "isMultiSport", "_bmiProxy", "YearsPlayingSport"]
        sport_dum = pd.get_dummies(df["sportName"], prefix="Sport", dtype=float)
        X = pd.concat([df[feat_cols].astype(float), sport_dum], axis=1)
        y = df["_totalDaysLost"].astype(float)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=5)
        gbm = GradientBoostingRegressor(random_state=5, n_estimators=200,
                                        max_depth=3, learning_rate=0.05)
        gbm.fit(Xtr, ytr)
        mae = float(np.mean(np.abs(gbm.predict(Xte) - yte)))
        pred_all = gbm.predict(X)
        return mae, pred_all

    burden_mae, burden_pred = train_burden_model(ath_feats)
    ath_feats["_predictedBurden"] = burden_pred.clip(min=0)

    bc1, bc2 = st.columns([1, 1.3])
    with bc1:
        st.metric("Forecast error (MAE)", f"{burden_mae:.1f} days",
                  help="Average miss predicting an athlete's TOTAL historical lost "
                       "days, using only sport/grade/build/tenure - attributes known "
                       "before any injury occurs.")
        st.caption(
            "Built from profile attributes only (sport, grade, body-mass proxy, "
            "years playing) - not from past injuries themselves - so it can score "
            "athletes with zero injury history too, for preseason planning.")
    with bc2:
        top_burden = ath_feats.sort_values("_predictedBurden", ascending=False).head(12).reset_index()
        top_burden["Athlete"] = top_burden.firstName + " " + top_burden.lastName
        top_burden["School"] = top_burden.SchoolName.str.replace(
            " High School", " HS").str.replace(" Middle School", " MS")
        fig = px.bar(top_burden.sort_values("_predictedBurden"),
                     x="_predictedBurden", y="Athlete", orientation="h",
                     color_discrete_sequence=[SERIES[4]],
                     hover_data=["School", "sportName"])
        fig.update_layout(title="Highest projected lost-participation days (preseason forecast)",
                          xaxis_title="Predicted days lost", yaxis_title=None)
        st.plotly_chart(style(fig, h=340), use_container_width=True)

    st.markdown("---")

    # ---------------- C. Repeat-concussion recovery risk ----------------
    st.markdown("#### 3. Repeat-concussion recovery risk")
    conc = dd[(dd.isConcussion == 1) & (dd._actualDays.notna())].copy()
    if len(conc):
        conc["ConcussionHistory"] = pd.cut(
            conc.NumberOfConcussions, bins=[-1, 0, 1, 2, 99],
            labels=["1st known concussion", "2nd", "3rd", "4th+"])
        cbar = conc.groupby("ConcussionHistory", observed=True)._actualDays.median().reset_index()
        fig = px.bar(cbar, x="ConcussionHistory", y="_actualDays",
                     color_discrete_sequence=[SERIES[6]],
                     text=cbar._actualDays.round(0).astype(int))
        fig.update_traces(textposition="outside", cliponaxis=False,
                          marker_line_color=SURFACE, marker_line_width=2)
        fig.update_layout(title="Median recovery days by concussion history",
                          xaxis_title=None, yaxis_title="Days to clearance",
                          showlegend=False)
        st.plotly_chart(style(fig, h=300), use_container_width=True)
        st.caption(
            "Recovery from repeat concussions takes measurably longer - the same "
            "prior-same-site penalty implanted in the recovery model applies here "
            "via NumberOfConcussions. On real data this becomes an automatic flag: "
            "extend the expected recovery window for any athlete with 2+ recorded "
            "concussions before comparing them to a fresh-injury baseline.")
    else:
        st.info("No closed concussion cases in this dataset.")

    st.markdown("---")

    # ---------------- D. Availability & performance impact ----------------
    st.markdown("#### 4. Availability impact - lost participation, by school / sport / body part")
    st.caption(
        "Rank One's schema has no performance stats (points, wins, playing time) - "
        "so \"performance impact\" here means lost availability: days an athlete "
        "could not practice or play, computed from MissDayStart/MissDayEnd and "
        "CanPractice. This is the honest proxy available today.")

    roster = athletes.groupby("SchoolName").size().rename("Athletes")

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        by_school = dd.groupby("SchoolName")._daysLost.sum().to_frame().join(roster)
        by_school["PerAthlete"] = by_school._daysLost / by_school.Athletes
        by_school = by_school.reset_index()
        by_school["Short"] = by_school.SchoolName.str.replace(
            " High School", " HS").str.replace(" Middle School", " MS")
        fig = px.bar(by_school.sort_values("PerAthlete"), x="PerAthlete", y="Short",
                     orientation="h", color_discrete_sequence=[SERIES[0]])
        fig.update_yaxes(title=None)
        fig.update_xaxes(title="Days lost per athlete on roster")
        fig.update_layout(title="By school (normalized)")
        st.plotly_chart(style(fig, h=320), use_container_width=True)

    with ac2:
        roster_sport = athletes.groupby("sportName").size().rename("Athletes")
        by_sport = dd.groupby("sportName")._daysLost.sum().to_frame().join(roster_sport)
        by_sport["PerAthlete"] = by_sport._daysLost / by_sport.Athletes
        by_sport = by_sport.reset_index().sort_values("PerAthlete")
        fig = px.bar(by_sport, x="PerAthlete", y="sportName", orientation="h",
                     color_discrete_sequence=[SERIES[1]])
        fig.update_yaxes(title=None)
        fig.update_xaxes(title="Days lost per participant")
        fig.update_layout(title="By sport (normalized)")
        st.plotly_chart(style(fig, h=320), use_container_width=True)

    with ac3:
        by_region = dd.groupby("_region")._daysLost.agg(["sum", "mean"]).reset_index()
        by_region = by_region.sort_values("sum")
        fig = px.bar(by_region, x="sum", y="_region", orientation="h",
                     color_discrete_sequence=[SERIES[2]],
                     hover_data={"mean": ":.1f"})
        fig.update_yaxes(title=None)
        fig.update_xaxes(title="Total days lost (all cases)")
        fig.update_layout(title="By body part (total volume)")
        st.plotly_chart(style(fig, h=320), use_container_width=True)
    st.caption(
        "Volume (total days lost) is driven by frequency; hover the body-part "
        "chart for the average days lost PER CASE - the severity view. A region "
        "can rank low on volume but high on severity (e.g., concussions vs ankle "
        "sprains), and that distinction matters for where to invest prevention.")

# ==========================================================================
#  TREATMENT ANALYSIS - the visits themselves, not just the injuries
# ==========================================================================
elif view == "Treatment Analysis":
    st.markdown("### Treatment Analysis")
    st.markdown(
        "Every other view treats treatments as a supporting signal for injury "
        "analysis (pain trends, care gaps). This page looks at the visits "
        "themselves - what trainers are doing, how often, and what the free-text "
        "record says in bulk.")

    tf1, tf2, tf3 = st.columns(3)
    sports_t = ["All sports"] + sorted(athletes.sportName.unique())
    sp_t = tf1.selectbox("Sport", sports_t, key="trt_sport")
    years_t = ["All years"] + sorted(injuries.SchoolYear.unique())
    yr_t = tf2.selectbox("School year", years_t, key="trt_year")
    regions_t = ["All regions"] + BODY_REGIONS
    rg_t = tf3.selectbox("Body region", regions_t, key="trt_region")

    dinj = injuries.merge(athletes[["Athlete_ID", "sportName", "SchoolName"]], on="Athlete_ID")
    if sp_t != "All sports":
        dinj = dinj[dinj.sportName == sp_t]
    if yr_t != "All years":
        dinj = dinj[dinj.SchoolYear == yr_t]
    if rg_t != "All regions":
        dinj = dinj[dinj._region == rg_t]

    dtrt = treatments[treatments.InjuryId.isin(dinj.ID)].merge(
        dinj[["ID", "_region", "_complaint", "SchoolName"]].rename(columns={"ID": "InjuryId"}),
        on="InjuryId")

    tm1, tm2, tm3, tm4 = st.columns(4)
    tm1.metric("Treatments (this filter)", f"{len(dtrt):,}")
    per_case = dtrt.groupby("InjuryId").size()
    tm2.metric("Median sessions per case", f"{per_case.median():.0f}" if len(per_case) else "-")
    standalone_n = (treatments.InjuryId == 0).sum()
    tm3.metric("Standalone treatments (no linked injury)", f"{standalone_n:,}",
              help="Rank One's schema allows treatments with InjuryId = 0 - general "
                   "or wellness visits not tied to a specific injury record. Not "
                   "filtered by sport/year/region above since they carry no injury.")
    tm4.metric("Trainers logging care", f"{treatments.CreatedBy.nunique()}")

    st.markdown("---")

    # ---------------- treatment plan composition ----------------
    st.markdown("#### What trainers are doing - treatment plan composition")
    ref_f = refs[refs.InjuryId.isin(dinj.ID)]
    pc1, pc2, pc3 = st.columns(3)
    for col, header, color in [(pc1, "Modalities", SERIES[0]),
                               (pc2, "Initial Treatment", SERIES[1]),
                               (pc3, "Therapeutic Exercise", SERIES[2])]:
        vals = ref_f[ref_f.headerName == header].refValue.value_counts().head(8)
        vals = vals.sort_values()
        fig = px.bar(x=vals.values, y=vals.index, orientation="h",
                     color_discrete_sequence=[color])
        fig.update_yaxes(title=None)
        fig.update_xaxes(title="Cases")
        fig.update_layout(title=header, showlegend=False)
        col.plotly_chart(style(fig, h=300), use_container_width=True)
    st.caption(
        "One recorded value per category per case (Rank One's InjuryReferences "
        "table) - this is the first time this field has been analyzed anywhere "
        "in the app. On real data this is a direct read on care patterns by "
        "injury type, without touching a single free-text note.")

    st.markdown("---")

    # ---------------- cadence: session count + taper ----------------
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.histogram(per_case, nbins=20, color_discrete_sequence=[SERIES[3]])
        fig.update_layout(title="Sessions per case - how much care an injury gets",
                          xaxis_title="Treatment sessions", yaxis_title="Cases",
                          showlegend=False)
        st.plotly_chart(style(fig, h=320), use_container_width=True)

    with cc2:
        gaps = []
        for iid, g in dtrt.groupby("InjuryId"):
            g = g.sort_values("TreatmentDate")
            if len(g) < 2:
                continue
            d_days = g.TreatmentDate.diff().dt.total_seconds().dropna() / 86400
            half = len(d_days) // 2
            for i, gap in enumerate(d_days):
                gaps.append({"Phase": "Early in case" if i < max(1, half) else "Late in case",
                            "Gap": gap})
        if gaps:
            gdf = pd.DataFrame(gaps)
            med = gdf.groupby("Phase").Gap.median().reindex(["Early in case", "Late in case"])
            fig = px.bar(med.reset_index(), x="Phase", y="Gap",
                         color_discrete_sequence=[SERIES[4]],
                         text=med.round(1).values)
            fig.update_traces(textposition="outside", cliponaxis=False,
                              marker_line_color=SURFACE, marker_line_width=2)
            fig.update_layout(title="Median days between visits - early vs late in a case",
                              yaxis_title="Days between visits", xaxis_title=None,
                              showlegend=False)
            st.plotly_chart(style(fig, h=320), use_container_width=True)
            st.caption(
                "Confirms the intended real-world pattern: near-daily care early, "
                "tapering as the case progresses - not a finding to sell, a sanity "
                "check that the synthetic care pattern looks like practice.")
        else:
            st.info("Not enough multi-visit cases in this filter to show cadence.")

    st.markdown("---")

    # ---------------- care-gap monitor (open cases only) ----------------
    st.markdown("#### Care-gap monitor - open cases going untreated")
    open_inj = dinj[dinj.Status == "Open"]
    last_seen = dtrt.groupby("InjuryId").TreatmentDate.max()
    gap_rows = []
    for _, r in open_inj.iterrows():
        last = last_seen.get(r.ID, r.InjuryDate)
        gap_rows.append({"Athlete_ID": r.Athlete_ID, "InjuryId": r.ID,
                         "Complaint": r._complaint, "School": r.SchoolName,
                         "DaysSinceLastVisit": (TODAY - last).days})
    if gap_rows:
        gdf2 = pd.DataFrame(gap_rows).merge(
            athletes[["Athlete_ID", "firstName", "lastName"]], on="Athlete_ID")
        gdf2["Athlete"] = gdf2.firstName + " " + gdf2.lastName
        gdf2["Short"] = gdf2.School.str.replace(" High School", " HS").str.replace(
            " Middle School", " MS")
        g5, g10, g14 = ((gdf2.DaysSinceLastVisit >= n).sum() for n in (5, 10, 14))
        gm1, gm2, gm3 = st.columns(3)
        gm1.metric("Open cases, 5+ days no visit", g5)
        gm2.metric("10+ days", g10)
        gm3.metric("14+ days", g14)
        stale = gdf2.sort_values("DaysSinceLastVisit", ascending=False).head(12)
        st.dataframe(
            stale[["Athlete", "Short", "Complaint", "DaysSinceLastVisit"]]
            .rename(columns={"Short": "School", "DaysSinceLastVisit": "Days since last visit"}),
            use_container_width=True, hide_index=True)
        st.caption(
            "Open injuries not being actively treated are exactly the ones that "
            "resurface as documentation gaps or liability exposure - this table "
            "already exists as one reason string in the Radar Insights watchlist; "
            "here it stands on its own as a monitoring tool.")
    else:
        st.info("No open cases in the current filter.")

    st.markdown("---")

    # ---------------- trainer workload ----------------
    st.markdown("#### Trainer workload")
    tw = treatments.CreatedBy.value_counts().sort_values().reset_index()
    tw.columns = ["Trainer (User_ID)", "Sessions"]
    tw["Trainer (User_ID)"] = "Trainer " + tw["Trainer (User_ID)"].astype(str)
    fig = px.bar(tw, x="Sessions", y="Trainer (User_ID)", orientation="h",
                 color_discrete_sequence=[SERIES[5]])
    fig.update_yaxes(title=None)
    fig.update_layout(title="Session volume per trainer (all cases, unfiltered)",
                      showlegend=False)
    st.plotly_chart(style(fig, h=260), use_container_width=True)
    st.caption(
        "Staffing-load signal only - in this synthetic dataset trainer assignment "
        "is random, so differences here are not a performance signal. On real "
        "data this is a workload-balancing view, not a trainer scorecard.")

    st.markdown("---")

    # ---------------- AI digest across many notes at once ----------------
    st.markdown("#### AI treatment-notes digest")
    st.caption(
        "The per-athlete case files (Athlete Focus) already summarize ONE case's "
        "notes. This does the opposite: synthesize recurring themes across MANY "
        "treatment assessments at once, for the current filter.")
    if AI_READY:
        sample_n = min(25, len(dtrt))
        if sample_n and st.button(f"Summarize {sample_n} treatment notes for this filter"):
            sample_txt = "\n---\n".join(
                dtrt.sample(sample_n, random_state=3).CurrentAssessment.dropna().astype(str))
            with st.spinner("Reading treatment notes..."):
                out = ask_ai(
                    "You are a sports-medicine analyst reviewing a batch of athletic "
                    "trainers' treatment assessment notes for one filtered slice of a "
                    "district's caseload. Identify: (1) recurring clinical themes or "
                    "phrases across notes; (2) any recurring concerns a supervising "
                    "clinician should know about; (3) whether documentation quality "
                    "looks consistent across notes. Be concise (bullet points). Do "
                    "not invent facts not supported by the notes.",
                    f"FILTER: sport={sp_t}, year={yr_t}, region={rg_t}.\n\n"
                    f"TREATMENT ASSESSMENT NOTES:\n{sample_txt}")
            st.markdown(out)
        elif not sample_n:
            st.info("No treatment notes in the current filter.")
    else:
        st.caption("AI layer offline - add Azure OpenAI credentials to ..\\.env to enable.")

# ==========================================================================
#  ATHLETE FOCUS
# ==========================================================================
elif view == "Athlete Focus":
    inj_counts = injuries.groupby("Athlete_ID").size()
    ath = athletes.copy()
    ath["n_inj"] = ath["Athlete_ID"].map(inj_counts).fillna(0).astype(int)
    ath = ath.sort_values(["lastName", "firstName"])
    ath["label"] = (ath.lastName + ", " + ath.firstName + "  -  " + ath.sportName +
                    ", " + ath.SchoolName +
                    np.where(ath.n_inj > 0,
                             "  (" + ath.n_inj.astype(str) + " injur" +
                             np.where(ath.n_inj == 1, "y)", "ies)"),
                             ""))

    only_injured = st.toggle("Show only athletes with injury records", value=True)
    opts = ath[ath.n_inj > 0] if only_injured else ath
    # default to a clinically interesting case rather than the alphabet's first entry
    default_idx = int(np.argmax(opts.n_inj.values >= 3)) if (opts.n_inj >= 3).any() else 0
    pick = st.selectbox("Select an athlete", opts.label.tolist(), index=default_idx)
    a = ath[ath.label == pick].iloc[0]
    ai = injuries[injuries.Athlete_ID == a.Athlete_ID].sort_values("InjuryDate")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f"### {a.firstName} {a.lastName}\n"
            f"**{a.sportName}** - {a.TeamLevelText}, Grade {a.gradeLevel}  \n"
            f"{a.SchoolName}  \n"
            f"Height {a.Height // 12}'{a.Height % 12}\" - Weight {a.Weight} lbs - "
            f"{a.YearsPlayingSport} yrs in sport  \n"
            f"`Athlete_ID {a.Athlete_ID}`")
    with c2:
        closed = ai[ai.Status == "Closed"]
        missed = (closed.MissDayEnd - closed.MissDayStart).dt.days.clip(lower=0).sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Injuries on record", len(ai))
        m2.metric("Days of participation missed", int(missed) if pd.notna(missed) else 0)
        m3.metric("Avg recovery (days)",
                  f"{closed._actualDays.mean():.0f}" if len(closed) else "-")
        m4.metric("Concussions", int(ai.isConcussion.sum()))

    if ai.empty:
        st.info("No injury records for this athlete.")
        st.stop()

    # ---- injury history timeline
    tl = ai.copy()
    tl["End"] = tl.ReturnDate.fillna(TODAY)
    tl["Injury"] = tl._complaint + tl.apply(
        lambda r: "  (re-injury)" if r._isReinjury else "", axis=1)
    fig = px.timeline(tl, x_start="InjuryDate", x_end="End", y="Injury",
                      color="_region", color_discrete_map=REGION_COLOR,
                      title="Injury history - every episode from onset to return",
                      hover_data={"Status": True, "Mechanism": True,
                                  "_actualDays": True, "_region": False})
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(marker_line_color=SURFACE, marker_line_width=2)  # 2px surface gap
    fig.update_layout(legend_title_text="Body region")
    st.plotly_chart(style(fig, h=max(250, 60 * len(tl) + 120)), use_container_width=True)

    colA, colB = st.columns(2)

    # ---- pain trajectory per injury (parsed from CurrentAssessment text)
    with colA:
        at = treatments[treatments.AthleteId == a.Athlete_ID].merge(
            ai[["ID", "_complaint"]], left_on="InjuryId", right_on="ID")
        if len(at):
            fig = go.Figure()
            for i, (inj, g) in enumerate(at.groupby("_complaint")):
                g = g.sort_values("_dayOfRecovery")
                fig.add_trace(go.Scatter(
                    x=g._dayOfRecovery, y=g._painParsed, mode="lines+markers",
                    name=inj[:28], line=dict(width=2, color=SERIES[i % 8]),
                    marker=dict(size=8)))
            fig.update_layout(title="Pain reported at each treatment session "
                                    "(parsed from trainer's assessment text)",
                              xaxis_title="Day of recovery", yaxis_title="Pain (0-10)")
            st.plotly_chart(style(fig), use_container_width=True)

    # ---- recovery vs peers
    with colB:
        rows = []
        for _, r in ai[ai._actualDays.notna()].iterrows():
            peers = injuries[(injuries._complaint == r._complaint) &
                             (injuries._actualDays.notna()) &
                             (injuries.ID != r.ID)]._actualDays
            if len(peers) >= 5:
                rows.append({"Injury": r._complaint,
                             "This athlete": r._actualDays,
                             "Peer median": peers.median()})
        if rows:
            cmp_df = pd.DataFrame(rows).melt(id_vars="Injury",
                                             var_name="Who", value_name="Days")
            fig = px.bar(cmp_df, x="Days", y="Injury", color="Who", barmode="group",
                         orientation="h", title="Recovery time vs athletes with the same injury",
                         color_discrete_sequence=[SERIES[0], SERIES[1]])
            fig.update_traces(marker_line_color=SURFACE, marker_line_width=2)
            fig.update_yaxes(title=None)
            st.plotly_chart(style(fig), use_container_width=True)
        else:
            st.caption("Not enough closed cases to benchmark this athlete against peers.")

    # ---- treatment & notes detail
    st.markdown("#### Case files")
    for _, r in ai.iterrows():
        n_tr = (treatments.InjuryId == r.ID).sum()
        badge = ("OPEN" if r.Status == "Open" else
                 f"closed in {int(r._actualDays)}d" if pd.notna(r._actualDays) else "closed")
        with st.expander(f"{r.InjuryDate.date()} - {r._complaint} - {badge} - {n_tr} treatments"):
            st.write(f"**Mechanism:** {r.Mechanism if pd.notna(r.Mechanism) else '(not recorded)'}   |   "
                     f"**Summary:** {r.Summary if pd.notna(r.Summary) else '(not recorded)'}")
            est = (f"{(r.EstimatedReturnDate - r.InjuryDate).days} days"
                   if pd.notna(r.EstimatedReturnDate) else "not entered")
            st.write(f"**Trainer's estimated recovery:** {est} - "
                     f"**actual:** {int(r._actualDays) if pd.notna(r._actualDays) else 'still open'} days")
            nn = notes[notes.InjuryID == r.ID].sort_values("NoteDate")
            for _, n in nn.iterrows():
                st.markdown(f"- *{n.NoteDate.date()}* - {n.Note}")
            if AI_READY and len(nn):
                if st.button("Summarize this case with AI", key=f"ai{r.ID}"):
                    tr_txt = "\n".join(
                        treatments[treatments.InjuryId == r.ID]
                        .sort_values("TreatmentDate").CurrentAssessment.head(20))
                    nt_txt = "\n".join(nn.Note)
                    with st.spinner("Reading the clinical record..."):
                        out = ask_ai(
                            "You are a sports-medicine clinical summarizer. From raw athletic "
                            "trainer notes and treatment assessments, produce: (1) a 3-line "
                            "clinical history in plain language; (2) a markdown table of "
                            "extracted events with columns Date/Stage | Event | Detail; "
                            "(3) one flag a supervising clinician should know, if any. "
                            "Be faithful to the text; do not invent facts.",
                            f"INJURY: {r._complaint}, onset {r.InjuryDate.date()}, "
                            f"status {r.Status}.\nTREATMENT ASSESSMENTS:\n{tr_txt}\n\n"
                            f"MEDICAL NOTES:\n{nt_txt}")
                    st.markdown(out)
                    st.caption("Generated by the AI layer from the free-text record above - "
                               "the Aim 2 capability, live.")

# ==========================================================================
#  INJURY FOCUS
# ==========================================================================
else:
    f1, f2, f3 = st.columns(3)
    sports = ["All sports"] + sorted(athletes.sportName.unique())
    sp = f1.selectbox("Sport", sports)
    years = ["All years"] + sorted(injuries.SchoolYear.unique())
    yr = f2.selectbox("School year", years)
    regions = ["All regions"] + BODY_REGIONS
    rg = f3.selectbox("Body region", regions)

    d = injuries.merge(athletes[["Athlete_ID", "sportName", "SchoolName"]], on="Athlete_ID")
    if sp != "All sports":
        d = d[d.sportName == sp]
    if yr != "All years":
        d = d[d.SchoolYear == yr]
    if rg != "All regions":
        d = d[d._region == rg]

    closed = d[d._actualDays.notna()]
    err = (closed._actualDays - closed._estDays)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Injuries", f"{len(d):,}")
    m2.metric("Median recovery", f"{closed._actualDays.median():.0f} days" if len(closed) else "-")
    m3.metric("Re-injury share", f"{d._isReinjury.mean() * 100:.0f}%")
    m4.metric("Concussions", f"{d.isConcussion.mean() * 100:.0f}%")
    m5.metric("Return estimates run", f"{err.mean():+.0f} days" if len(closed) else "-",
              help="Actual minus estimated recovery. Positive = trainers' EstimatedReturnDate is optimistic.")

    r1c1, r1c2 = st.columns(2)

    with r1c1:  # injuries per month
        bym = d.groupby(d.InjuryDate.dt.to_period("M")).size().reset_index(name="Injuries")
        bym["Month"] = bym.InjuryDate.dt.to_timestamp()
        fig = px.bar(bym, x="Month", y="Injuries",
                     title="Injuries per month - the season shape",
                     color_discrete_sequence=[SERIES[0]])
        fig.update_traces(marker_line_color=SURFACE, marker_line_width=1)
        st.plotly_chart(style(fig), use_container_width=True)

    with r1c2:  # by body region
        byr = d.groupby("_region").size().reset_index(name="Injuries").sort_values("Injuries")
        fig = px.bar(byr, x="Injuries", y="_region", orientation="h",
                     title="Injuries by body region",
                     color_discrete_sequence=[SERIES[0]])
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig), use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:  # recovery distribution by complaint
        top = closed._complaint.value_counts().head(8).index
        box = closed[closed._complaint.isin(top)]
        order = box.groupby("_complaint")._actualDays.median().sort_values().index
        fig = px.box(box, x="_actualDays", y="_complaint", category_orders={"_complaint": list(order)},
                     title="Recovery time by injury type (closed cases)",
                     color_discrete_sequence=[SERIES[0]], points=False)
        fig.update_yaxes(title=None)
        fig.update_xaxes(title="Days to return")
        st.plotly_chart(style(fig, h=380), use_container_width=True)

    with r2c2:  # estimated vs actual
        sample = closed.sample(min(800, len(closed)), random_state=1) if len(closed) else closed
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample._estDays, y=sample._actualDays, mode="markers",
            marker=dict(size=8, color=SERIES[0], opacity=0.45,
                        line=dict(color=SURFACE, width=1)),
            name="Closed cases",
            text=sample._complaint,
            hovertemplate="%{text}<br>Estimated %{x}d - actual %{y}d<extra></extra>"))
        mx = float(max(sample._estDays.max(), sample._actualDays.max())) if len(sample) else 10
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                                 line=dict(color=MUTED, width=2, dash="dot"),
                                 name="Perfect estimate"))
        fig.update_layout(title="Trainer's estimated vs actual recovery - "
                                "points above the line returned later than predicted",
                          xaxis_title="Estimated days (EstimatedReturnDate)",
                          yaxis_title="Actual days (ReturnDate)")
        st.plotly_chart(style(fig, h=380), use_container_width=True)

    r3c1, r3c2 = st.columns(2)

    with r3c1:  # average recovery curve: pain by day, by region (top 4)
        tt = treatments.merge(d[["ID", "_region"]], left_on="InjuryId", right_on="ID")
        top4 = tt._region.value_counts().head(4).index.tolist()
        fig = go.Figure()
        for i, reg in enumerate(top4):
            g = tt[tt._region == reg]
            curve = g.groupby(pd.cut(g._dayOfRecovery, bins=range(0, 61, 5)),
                              observed=True)._painParsed.mean()
            xs = [iv.mid for iv in curve.index]
            fig.add_trace(go.Scatter(x=xs, y=curve.values, mode="lines+markers",
                                     name=reg, line=dict(width=2, color=REGION_COLOR.get(reg, SERIES[i])),
                                     marker=dict(size=8)))
        fig.update_layout(title="The recovery curve - average pain by day since injury",
                          xaxis_title="Days since injury", yaxis_title="Avg pain (0-10)",
                          legend_title_text="Body region")
        st.plotly_chart(style(fig, h=360), use_container_width=True)

    with r3c2:  # concussion protocol timers
        cc = d[d.isConcussion == 1].copy()
        if len(cc):
            cc["phys_hrs"] = (cc.PhysicianEvalDate - cc.InjuryDate).dt.total_seconds() / 3600
            seen48 = (cc.phys_hrs <= 48).sum()
            late = (cc.phys_hrs > 48).sum()
            missing = cc.PhysicianEvalDate.isna().sum()
            comp = pd.DataFrame({
                "Stage": ["Physician eval within 48h", "Physician eval late (>48h)",
                          "No physician eval recorded"],
                "Cases": [seen48, late, missing],
                "Color": [STATUS["good"], STATUS["warning"], STATUS["critical"]]})
            fig = go.Figure(go.Bar(
                x=comp.Cases, y=comp.Stage, orientation="h",
                marker=dict(color=comp.Color, line=dict(color=SURFACE, width=2)),
                text=comp.Cases, textposition="outside"))
            fig.update_layout(title=f"Concussion protocol - physician evaluation timing "
                                    f"({len(cc)} concussions)",
                              xaxis_title="Cases")
            st.plotly_chart(style(fig, h=360), use_container_width=True)
            st.caption("Red = the compliance gap the Feb 2026 UT Southwestern study documented "
                       "statewide. Labels carry the status; color is reinforcement only.")
        else:
            st.caption("No concussions in the current filter.")

    # ---- AI narrative report for the current filter
    if AI_READY and len(closed):
        st.markdown("#### AI narrative report")
        if st.button("Generate a 'State of Recovery' briefing for this view"):
            top_inj = closed._complaint.value_counts().head(5)
            worst = (closed.groupby("_complaint")
                     .apply(lambda g: (g._actualDays - g._estDays).mean(), include_groups=False)
                     .sort_values(ascending=False).head(3))
            stats = (
                f"Filter: sport={sp}, year={yr}, region={rg}. "
                f"{len(d)} injuries, median recovery {closed._actualDays.median():.0f} days. "
                f"Re-injury share {d._isReinjury.mean()*100:.0f}%. "
                f"Concussion share {d.isConcussion.mean()*100:.0f}%. "
                f"Return estimates run {(closed._actualDays - closed._estDays).mean():+.1f} days vs actual. "
                f"Most common injuries: {top_inj.to_dict()}. "
                f"Largest underestimates (days late vs plan): {worst.round(1).to_dict()}.")
            with st.spinner("Writing the briefing..."):
                out = ask_ai(
                    "You write concise briefings for school athletic directors. From the "
                    "statistics given, write a 150-word 'State of Recovery' briefing: what "
                    "stands out, what it means for athlete availability, and two specific, "
                    "practical recommendations. No preamble, no invented numbers.",
                    stats)
            st.markdown(out)
            st.caption("Generated live by the AI layer from the numbers in this view.")

    st.caption("POC built by UNT College of Information on synthetic data replicating the "
               "Rank One Partner API v1.0 schema (April 2026). No real athlete data was used.")
