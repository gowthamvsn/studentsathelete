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
@st.cache_data
def load():
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

athletes, injuries, treatments, notes, refs = load()

st.set_page_config(page_title="Rank One Radar - POC", page_icon="R", layout="wide")

st.markdown(
    "<h2 style='margin-bottom:0'>Rank One Radar <span style='color:#898781;"
    "font-weight:400;font-size:0.6em'>UNT proof of concept - synthetic data, "
    "Rank One Partner API v1.0 schema</span></h2>", unsafe_allow_html=True)

view = st.sidebar.radio("View", ["Radar Insights", "Athlete Focus", "Injury Focus"])
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
