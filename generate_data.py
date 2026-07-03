# -*- coding: utf-8 -*-
"""
Synthetic data generator matching Rank One's Partner API v1.0 schema
(Injury-Treatment-Documentation, April 2026).

Column names reproduce the documented tables EXACTLY (Injury 3.1, Treatment 3.2,
Medical Notes 3.3, InjuryReferences 2.2). Athlete demographics use the field
names of the documented API payloads.

Realism features (deliberate, mirroring real EMR data):
  - trainer shorthand and inconsistent clinical voice in notes/assessments
  - missing values (PainScale, EstimatedReturnDate, MissDay fields)
  - imported Historical records, standalone treatments (InjuryId = 0)
  - some injuries with no notes at all; admin/phone-call notes mixed in
  - concussion notes in SCAT/RTP-stage language
All data is 100% synthetic. No real athletes, schools, or records.
"""
import numpy as np
import pandas as pd
import uuid
import random
import os
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
random.seed(42)

OUT = r"D:\rankone\poc_app\data"
os.makedirs(OUT, exist_ok=True)

N_ATHLETES = 2000
TODAY = datetime(2026, 7, 2)

# ---------------------------------------------------------------- lookups
SPORTS = {
    25: ("Soccer",        0.145),
    3:  ("Football",      0.200),
    7:  ("Basketball",    0.130),
    12: ("Volleyball",    0.095),
    5:  ("Baseball",      0.085),
    9:  ("Softball",      0.075),
    14: ("Track & Field", 0.085),
    18: ("Wrestling",     0.055),
    21: ("Cross Country", 0.055),
    28: ("Cheer",         0.035),
    30: ("Tennis",        0.025),
    32: ("Swimming",      0.015),
}
SPORT_IDS = list(SPORTS.keys())
SPORT_P = np.array([SPORTS[k][1] for k in SPORT_IDS]); SPORT_P /= SPORT_P.sum()

SCHOOLS = {
    2737: "North Star High School",
    2738: "Cedar Creek High School",
    2739: "Prairie View High School",
    2740: "Lakeside High School",
    2741: "Red Oak Middle School",
    2742: "Bluebonnet Middle School",
}
SCHOOL_IDS = list(SCHOOLS.keys())

# Each school has a CHARACTER - these are the implanted causes the app's
# Causality Lab exists to recover:
#   mix   - sport-emphasis multipliers (football school vs soccer school)
#   multi - share of athletes playing 2+ sports (small schools need everyone)
#   turf  - share of field-sport time on artificial turf
SCHOOL_PROFILES = {
    2737: {"mix": {3: 2.0, 18: 1.3},          "multi": 0.16, "turf": 0.85},  # North Star: football/turf school
    2738: {"mix": {},                          "multi": 0.24, "turf": 0.50},  # Cedar Creek: balanced
    2739: {"mix": {25: 1.9, 14: 1.5, 21: 1.4}, "multi": 0.30, "turf": 0.20},  # Prairie View: soccer/track, grass
    2740: {"mix": {7: 1.8, 12: 1.6, 32: 1.5},  "multi": 0.12, "turf": 0.50},  # Lakeside: court sports, specialized
    2741: {"mix": {},                          "multi": 0.45, "turf": 0.40},  # Red Oak MS: kids play everything
    2742: {"mix": {},                          "multi": 0.45, "turf": 0.40},  # Bluebonnet MS
}
MULTISPORT_INJURY_MULT = 1.5   # the implanted causal effect of year-round load

TEAM_LEVELS = {1: "Varsity", 2: "Junior Varsity", 3: "Freshman", 4: "7th/8th Grade"}
SURFACES = {378859: "Artificial Turf", 378860: "Natural Grass",
            378861: "Hardwood Court", 378862: "Track Surface", 378863: "Mat",
            378864: "Pool Deck"}
OCCURRED = {378690: "Practice", 378691: "Game", 378692: "Conditioning",
            378693: "Off-Season Training"}
NATURE = {378674: "Acute", 378675: "Overuse", 378676: "Chronic"}

FIRST_M = ["Jordan","Alex","Casey","Hayden","Cameron","Dakota","Marcus","Diego","Chen",
           "Malik","Ethan","Noah","Liam","Jose","Tyler","Cole","Omar","Raj","Aiden",
           "Caleb","Xavier","Dylan","Brady","Trevon","Luis","Miguel","Andre","Kai",
           "Jaxon","Colt","Hunter","Weston","Zion","Trey","Dominic","Isaiah","Grant",
           "Bryce","Angel","Roman","Silas","Knox","Dante","Emmett","Rhett","Judah"]
FIRST_F = ["Taylor","Riley","Morgan","Avery","Quinn","Peyton","Skyler","Reese","Emerson",
           "Rowan","Sage","Elena","Aisha","Priya","Sofia","Maya","Zara","Ana","Grace",
           "Jada","Nina","Lily","Camila","Harper","Brooklyn","Kenzie","Addison","Reagan",
           "Marisol","Ximena","Layla","Nora","Ivy","Selena","Amara","Journee","Paisley",
           "Trinity","Alondra","Kya","Mabel","Sierra","Tatum","Wren","Estrella"]
LAST = ["Martinez","Johnson","Williams","Nguyen","Patel","Brown","Garcia","Kim","Davis",
        "Lopez","Wilson","Anderson","Thomas","Hernandez","Moore","Jackson","White",
        "Harris","Chen","Lewis","Walker","Young","Allen","King","Wright","Scott",
        "Torres","Hill","Flores","Green","Adams","Baker","Nelson","Rivera","Campbell",
        "Mitchell","Carter","Roberts","Gomez","Phillips","Evans","Turner","Diaz",
        "Parker","Cruz","Edwards","Collins","Reyes","Stewart","Morris","Morales",
        "Murphy","Cook","Rogers","Gutierrez","Ortiz","Morgan","Cooper","Peterson",
        "Bailey","Reed","Kelly","Howard","Ramos","Cox","Ward","Richardson","Watson",
        "Brooks","Chavez","Wood","James","Bennett","Gray","Mendoza","Ruiz","Hughes",
        "Price","Alvarez","Castillo","Sanders","Vargas","McCoy","Okafor","Haddad"]

TRAINER_IDS = [10618, 41006, 86868, 52277, 60930]
PHYSICIANS = ["Dr. Okafor", "Dr. Reyes", "Dr. Lindqvist", "Dr. Banerjee", "Dr. Whitfield"]

# (complaint template, body region, sports it fits, median recovery days, sd, share)
INJURY_PROFILES = [
    ("{side} Ankle Sprain",        "Ankle",     None,                     14,  6, 0.170),
    ("{side} Knee Sprain",         "Knee",      None,                     21, 10, 0.095),
    ("{side} ACL Tear",            "Knee",      [3,25,7,12,28],          240, 45, 0.018),
    ("{side} Hamstring Strain",    "Thigh",     [3,25,14],                18,  8, 0.085),
    ("{side} Quadriceps Contusion","Thigh",     [3,25,7],                  7,  3, 0.065),
    ("{side} Shoulder Instability","Shoulder",  [3,18,5,9,12,32],         28, 14, 0.058),
    ("Concussion",                 "Head",      None,                     16,  7, 0.085),
    ("Low Back Strain",            "Back",      None,                     12,  6, 0.058),
    ("{side} Wrist Fracture",      "Wrist",     [3,7,18,28],              42, 12, 0.028),
    ("{side} Finger Sprain",       "Hand",      [7,12,5,9],                6,  3, 0.065),
    ("Shin Splints",               "Lower Leg", [14,21,7,28],             20, 10, 0.065),
    ("{side} Groin Strain",        "Hip/Groin", [3,25,18],                15,  7, 0.048),
    ("{side} Elbow Tendinitis",    "Elbow",     [5,9,12,30,32],           17,  8, 0.038),
    ("{side} Meniscus Tear",       "Knee",      [3,25,7,18],              90, 30, 0.018),
    ("Heat Illness",               "Systemic",  [3,14,21,25],              4,  2, 0.028),
    ("{side} Achilles Tendinitis", "Ankle",     [7,14,21],                24, 10, 0.028),
    ("{side} Patellar Tendinitis", "Knee",      [7,12,14,28],             21,  9, 0.030),
    ("{side} Rotator Cuff Strain", "Shoulder",  [5,9,32,30],              25, 11, 0.020),
]

MECHANISMS = {
    "Ankle":    ["Inversion landing from jump", "Stepped in hole on field",
                 "Rolled ankle during agility drill", "Contact tackle, ankle everted",
                 "Landed on opponent's foot after header", "Missed base, rolled ankle",
                 "Awkward dismount landing"],
    "Knee":     ["Non-contact pivot/cut", "Contact to lateral knee",
                 "Awkward landing from rebound", "Planted foot, twisted knee",
                 "Knee-to-knee collision", "Hyperextension during slide tackle",
                 "Repetitive jumping load"],
    "Thigh":    ["Sprint acceleration", "Helmet contact to thigh",
                 "Overstretch during kick", "Deceleration on wet field",
                 "Late-game sprint, felt a pull"],
    "Shoulder": ["Fall on outstretched arm", "Tackle with arm abducted",
                 "Repetitive overhead throwing", "Dove for ball, landed on shoulder",
                 "High pitch count doubleheader", "Butterfly set volume"],
    "Head":     ["Helmet-to-helmet contact", "Head to ground contact",
                 "Elbow to head going for rebound", "Collision with another player",
                 "Header collision, ball struck temple", "Takedown, head hit mat",
                 "Basket toss fall"],
    "Back":     ["Repetitive rotation and extension", "Lifting during strength session",
                 "Back handspring practice volume", "Awkward twist during takedown"],
    "Wrist":    ["Fall on outstretched hand", "Jammed during blocking drill",
                 "Landed hard on mat", "Checked into base"],
    "Hand":     ["Ball contact to fingertip", "Jammed finger on rebound",
                 "Caught in jersey during tackle"],
    "Lower Leg":["Gradual onset with mileage increase", "Repetitive impact on hard surface",
                 "New shoes, increased volume", "Transition from grass to track"],
    "Hip/Groin":["Overstretch during lateral movement", "Sprint acceleration",
                 "Split-second change of direction", "Sparring takedown defense"],
    "Elbow":    ["Repetitive throwing load", "Overhead serve volume",
                 "Curveball volume increase", "Weight room overload"],
    "Systemic": ["High heat index conditioning session", "Two-a-day practice in August",
                 "Inadequate hydration before afternoon session"],
}

INITIAL_TREATMENTS = ["Cold Pack", "Compression Wrap", "Elevation", "Immobilization",
                      "RICE Protocol", "Removed from Activity", "Referred to Physician",
                      "Crutches Issued", "Sling Applied", "Taping/Bracing"]
MODALITIES = ["Ice Bath", "Ultrasound", "Electrical Stimulation", "Cold Whirlpool",
              "Heat Pack", "Massage", "Compression Boots", "Laser Therapy",
              "Cupping", "Game Ready", "Kinesio Taping"]
THER_EX = ["Ankle Alphabet", "Theraband Series", "Single-Leg Balance", "Wall Sits",
           "Hip Bridge Progression", "Rotator Cuff Series", "Core Stabilization",
           "Eccentric Calf Raises", "Quad Sets", "Vestibular Exercises",
           "BOSU Balance Work", "Nordic Hamstring Curls", "Scap Stabilization",
           "Monster Walks", "Step-Down Progression"]

SEASON_WEIGHT = {8: 1.7, 9: 1.6, 10: 1.4, 11: 1.1, 1: 1.0, 2: 1.1, 3: 1.2,
                 4: 1.1, 5: 0.9, 12: 0.8, 6: 0.35, 7: 0.25}

def guid():
    return str(uuid.uuid4()).upper()

def side_variant(complaint):
    """Real trainers write the same injury five different ways."""
    v = complaint
    r = rng.random()
    if r < 0.30:
        v = v.replace("Right ", "R ").replace("Left ", "L ")
    elif r < 0.45:
        v = v.lower()
    elif r < 0.55:
        v = v.replace("Right ", "rt ").replace("Left ", "lt ")
    return v

# ------------------------- clinical voice: assessments -------------------------
def assessment_text(complaint, pain, frac, slow, region):
    mod = random.choice(MODALITIES); ex = random.choice(THER_EX)
    disp = side_variant(complaint)
    pain_txt = random.choice([
        f"Pain {pain}/10.", f"Pain today {pain}/10.", f"c/o pain {pain}/10.",
        f"Rates pain {pain}/10.", f"pain {pain}/10 w/ activity."])
    if pain == 0:
        pain_txt = random.choice(["Denies pain.", "Pain free today.", "0/10."])
    if rng.random() < 0.12:
        pain_txt = ""  # trainer didn't record pain this session
    if frac < 0.25:
        body = random.choice([
            f"Acute phase. {mod}, compression. Edema present.",
            f"Swelling noted, TTP over injury site. {mod} 15 min.",
            f"Initial rehab session. Gentle ROM only. {mod} post.",
            f"NWB per protocol. {mod} + elevation."])
    elif frac < 0.55:
        body = random.choice([
            f"Started {ex}, tolerance fair.",
            f"ROM improving. Added {ex}.",
            f"Subacute. {ex} x3 sets, mild soreness after.",
            f"Progressed HEP. {mod} post-session.",
            f"Swelling down. {ex} today, no increase in sx."])
    elif frac < 0.85:
        body = random.choice([
            f"{ex} advanced. Functional testing next week.",
            f"Good session. Sport-specific movement introduced.",
            f"Strength 4+/5. Continue {ex} progression.",
            f"Jogged 10 min no sx. Progressing per plan."])
    else:
        body = random.choice([
            "Late stage. Sport-specific drills, cleared pending final eval.",
            "Full practice trial next. Reviewed prevention program.",
            "Functional hop tests symmetric. RTP eval scheduled.",
            "Non-contact practice completed without sx."])
    if slow and frac < 0.6 and rng.random() < 0.5:
        body += random.choice([
            " Progress slower than expected.", " Sx persist w/ activity.",
            " Plateau noted - modified plan.", " Still guarded with movement."])
    return " ".join(x for x in [f"{disp}.", pain_txt, body] if x)

# ------------------------- clinical voice: notes -------------------------
def initial_note(complaint, mech, pain, region, occurred):
    disp = side_variant(complaint)
    return random.choice([
        f"Initial eval. {disp}. Mechanism: {mech.lower()}. Pain {pain}/10, "
        f"TTP over injury site, ROM limited. Plan: {random.choice(INITIAL_TREATMENTS).lower()}, "
        f"re-eval 48 hrs.",
        f"Athlete c/o {disp.lower()} after {occurred.lower()}. {mech}. "
        f"Swelling +1, strength {random.choice(['3/5','4-/5','4/5'])}. "
        f"Started {random.choice(INITIAL_TREATMENTS).lower()}. Parent notified.",
        f"Evaluated {disp.lower()}. {mech}. Neuro intact, pulses WNL. "
        f"Will treat conservatively and monitor.",
        f"S: c/o {disp.lower()}, pain {pain}/10. O: edema, guarding. "
        f"A: {disp.lower()}, acute. P: {random.choice(INITIAL_TREATMENTS)}, "
        f"HEP issued, f/u tomorrow."])

def concussion_notes(inj_date, phys_eval, n_prior):
    out = [(0, random.choice([
        "SCAT administered on sideline. Symptom score 34, memory recall 3/5. "
        "Removed from play immediately. Parents given home instructions.",
        "Suspected concussion - removed from activity per protocol. HA 6/10, "
        "dizziness, photophobia. No LOC. Baseline comparison pending.",
        "Head injury eval. Balance testing abnormal (BESS 18 errors). "
        "Symptomatic - initiated concussion protocol, notified parents."]))]
    if phys_eval is not None:
        out.append(((phys_eval - inj_date).days,
            random.choice([
                f"Seen by {random.choice(PHYSICIANS)}. Dx confirmed concussion. "
                "Cognitive rest 48h, then graduated RTP protocol.",
                f"{random.choice(PHYSICIANS)} eval complete - cleared to begin "
                "stepwise RTP once asymptomatic 24h."])))
    out.append((int(rng.integers(4, 9)), random.choice([
        "Stage 2 light aerobic - stationary bike 15 min, asymptomatic.",
        "Symptom score down to 6. HA resolved, mild fogginess only.",
        "Stage 3 sport-specific drills, no sx during or after.",
        "ImPACT retest within baseline range. Progressing protocol."])))
    if n_prior > 0 and rng.random() < 0.6:
        out.append((2, f"Hx of {n_prior} prior concussion(s) - progressing "
                       "conservatively, extended stages per physician."))
    return out

def progress_note(frac, complaint, slow):
    if slow and rng.random() < 0.5:
        return random.choice([
            "Reports increased swelling after first full-intensity session. "
            "Regressed one stage, will reassess Thursday.",
            "Plateau in progress. Discussed w/ team physician - considering "
            "imaging if no change in 1 wk.",
            "Athlete admits doing extra running at home against instructions. "
            "Re-educated on load management.",
            "Sx returned during practice trial. Held out, back to modified plan."])
    return random.choice([
        "Follow-up. Gradual improvement, continue current plan.",
        "felt gd today, cont current plan",
        "ROM WNL. Strength improving. Progressed HEP.",
        "No sx with jogging. Add cutting next session.",
        "Compliance good. Swelling resolved.",
        "Missed 2 sessions (family travel). Resuming progression today."])

def admin_note():
    return random.choice([
        f"Parent called re: MRI scheduling - referred to {random.choice(PHYSICIANS)}'s office.",
        "Insurance pre-auth submitted for imaging.",
        f"{random.choice(PHYSICIANS)} office faxed clearance form - filed to record.",
        "Coach notified of participation status via app.",
        "Athlete fitted for functional brace, wear at all practices.",
        "Discussed nutrition and sleep with athlete - handout provided."])

def clearance_note(complaint):
    return random.choice([
        "Final eval. Functional testing passed, symmetric strength. Cleared for "
        "full RTP. Prevention program reviewed.",
        f"Cleared by {random.choice(PHYSICIANS)} for full participation. "
        "Maintenance HEP 2x/wk for 4 wks.",
        "RTP complete. Athlete and parent counseled on recurrence signs.",
        "Released to full activity. Will spot-check taping first week back."])

# ---------------------------------------------------------------- athletes
BODY = {  # sport: (male height mu, male weight mu, female height mu, female weight mu)
    3: (70, 190, 65, 140), 7: (72, 175, 68, 145), 12: (70, 165, 68, 140),
    18: (68, 160, 63, 125), 21: (68, 135, 64, 110), 14: (69, 155, 65, 125),
    25: (69, 155, 64, 125), 5: (70, 170, 0, 0), 9: (0, 0, 65, 140),
    28: (68, 150, 63, 115), 30: (69, 150, 65, 125), 32: (70, 155, 66, 130),
}

athletes = []
for i in range(N_ATHLETES):
    aid = 108000000 + i
    school_id = int(rng.choice(SCHOOL_IDS, p=[0.24, 0.22, 0.20, 0.18, 0.08, 0.08]))
    prof = SCHOOL_PROFILES[school_id]
    # sport choice reflects the school's character (its "mix")
    w = SPORT_P.copy()
    for sid_, mult_ in prof["mix"].items():
        w[SPORT_IDS.index(sid_)] *= mult_
    w = w / w.sum()
    sport_id = int(rng.choice(SPORT_IDS, p=w))
    if sport_id == 5:
        gender = "M"
    elif sport_id in (9, 28, 12):
        gender = "F" if sport_id != 12 else rng.choice(["F", "M"], p=[0.8, 0.2])
    else:
        gender = rng.choice(["M", "F"])
    # multi-sport participation: the school's culture decides how common it is
    is_multi = rng.random() < prof["multi"]
    second_sport = None
    if is_multi:
        pool = [s for s in SPORT_IDS if s != sport_id
                and not (gender == "M" and s in (9, 28))
                and not (gender == "F" and s == 5)]
        second_sport = int(rng.choice(pool))
    is_ms = school_id in (2741, 2742)
    team_level = 4 if is_ms else int(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20]))
    grade = int(rng.integers(7, 9)) if is_ms else int(rng.integers(9, 13))
    hm, wm, hf, wf = BODY.get(sport_id, (68, 160, 64, 130))
    h_mu, w_mu = (hm, wm) if gender == "M" else (hf, wf)
    grade_adj = (grade - 10) * 0.7
    height = int(rng.normal(h_mu + grade_adj, 2.6))
    weight = int(rng.normal(w_mu + grade_adj * 6, w_mu * 0.13))
    athletes.append({
        "Athlete_ID": aid,
        "athleteGuid": guid(),
        "firstName": random.choice(FIRST_M if gender == "M" else FIRST_F),
        "middleName": random.choice(["", "", "", "J.", "M.", "L.", "A.", "R."]),
        "lastName": random.choice(LAST),
        "gender": gender,
        "gradeLevel": grade,
        "Sport_ID": sport_id,
        "sportName": SPORTS[sport_id][0],
        "SchoolId": school_id,
        "SchoolName": SCHOOLS[school_id],
        "TeamLevel": team_level,
        "TeamLevelText": TEAM_LEVELS[team_level],
        "YearsPlayingSport": max(1, int(rng.normal(grade - 5, 1.5))),
        "Height": max(56, min(80, height)),
        "Weight": max(85, min(295, weight)),
        "isMultiSport": int(is_multi),
        "secondSport_ID": second_sport,
        "sportsPlayed": (SPORTS[sport_id][0] if not is_multi
                         else f"{SPORTS[sport_id][0]}; {SPORTS[second_sport][0]}"),
    })
athletes_df = pd.DataFrame(athletes)

# ---------------------------------------------------------------- injuries
def pick_profile(sport_id):
    weights = []
    for prof in INJURY_PROFILES:
        w = prof[5]
        if prof[2] is not None and sport_id not in prof[2]:
            w *= 0.10
        if prof[0] == "Concussion":
            w *= {3: 2.2, 25: 1.4, 18: 1.3, 28: 1.5, 7: 0.9}.get(sport_id, 0.55)
        weights.append(w)
    weights = np.array(weights) / sum(weights)
    return INJURY_PROFILES[int(rng.choice(len(INJURY_PROFILES), p=weights))]

def sample_injury_date(year_start):
    month = int(rng.choice(list(SEASON_WEIGHT.keys()),
                p=np.array(list(SEASON_WEIGHT.values())) / sum(SEASON_WEIGHT.values())))
    year = year_start if month >= 8 else year_start + 1
    day = int(rng.integers(1, 29))
    hour = int(rng.choice([7, 15, 16, 17, 18, 19], p=[0.1, 0.25, 0.25, 0.2, 0.15, 0.05]))
    return datetime(year, month, day, hour, int(rng.integers(0, 60)))

injuries, references, treatments, notes = [], [], [], []
inj_id = 1200000
trt_id = 8500000
note_id = 420000
ref_id = 1

# injury-proneness varies by sport (collision sports higher) and by athlete
SPORT_RISK = {3: 1.45, 18: 1.25, 25: 1.15, 28: 1.10, 7: 1.00, 12: 0.95,
              14: 0.90, 5: 0.85, 9: 0.85, 21: 0.80, 30: 0.60, 32: 0.55}

for _, ath in athletes_df.iterrows():
    # expected injuries over ~3 years; individual frailty multiplier adds spread;
    # multi-sport athletes carry year-round load - the implanted causal effect
    lam = 0.95 * SPORT_RISK.get(ath.Sport_ID, 1.0) * float(rng.gamma(2.2, 1 / 2.2))
    if ath.isMultiSport:
        lam *= MULTISPORT_INJURY_MULT
    n_inj = int(rng.poisson(lam))
    if n_inj == 0:
        continue                      # plenty of athletes stay healthy
    history = []                      # (profile, side, premature, complaint)
    # draw all dates first and SORT, so injury k really precedes injury k+1 -
    # the causal premature-return -> re-injury chain must run forward in time
    inj_dates = []
    for k in range(n_inj):
        year_start = int(rng.choice([2023, 2024, 2025], p=[0.29, 0.33, 0.38]))
        d = sample_injury_date(year_start)
        if d > TODAY - timedelta(days=2):
            d = TODAY - timedelta(days=int(rng.integers(2, 320)))
        inj_dates.append(d)
    inj_dates.sort()
    for k in range(n_inj):
        inj_date = inj_dates[k]

        # multi-sport athletes get hurt in their second sport too
        inj_sport = ath.Sport_ID
        if ath.isMultiSport and ath.secondSport_ID and rng.random() < 0.35:
            inj_sport = int(ath.secondSport_ID)

        # ---- re-injury is CAUSAL, not random: a premature prior return is the
        #      dominant driver, prior injury count adds to it. The model in the
        #      app exists to rediscover exactly this structure.
        if history:
            last = history[-1]
            p_re = 0.10 + (0.32 if last["premature"] else 0.0) + 0.04 * len(history)
            if rng.random() < min(0.75, p_re):
                profile, side = last["profile"], last["side"]
                is_reinjury = True
            else:
                profile = pick_profile(inj_sport)
                side = rng.choice(["Right", "Left"])
                is_reinjury = False
        else:
            profile = pick_profile(inj_sport)
            side = rng.choice(["Right", "Left"])
            is_reinjury = False

        complaint = profile[0].format(side=side)
        region = profile[1]
        median_days, sd_days = profile[3], profile[4]

        # ---- recovery duration has learnable drivers:
        #      prior same-site injuries, age, body mass, overuse nature, re-injury
        n_same_site = sum(1 for h in history if h["complaint"] == complaint)
        base = max(2, rng.normal(median_days, sd_days * 0.65))
        base *= (1 + 0.13 * n_same_site)                       # scar tissue tax
        base *= (1 + 0.025 * (ath.gradeLevel - 7))             # older heal slower
        bmi_proxy = ath.Weight / max(1, ath.Height)
        base *= (1 + max(0.0, (bmi_proxy - 2.55)) * 0.35)      # mass slows lower-body healing
        if is_reinjury:
            base *= rng.uniform(1.20, 1.65)
        # slow recoveries are not random either: high initial pain, prior damage
        # to the same site, and overuse conditions all predispose to stalling
        pain0_pre = int(np.clip(rng.normal(6 if median_days > 15 else 4.5, 1.5), 1, 10))
        p_slow = (0.03 + 0.06 * (pain0_pre >= 7) + 0.06 * (n_same_site > 0)
                  + 0.05 * (nature_id_pre := ("Tendinitis" in profile[0]
                                              or profile[0] == "Shin Splints")))
        slow_case = rng.random() < p_slow
        premature = (not slow_case) and rng.random() < 0.12     # cleared too early
        if slow_case:
            base *= rng.uniform(1.5, 2.4)
        if premature:
            base *= rng.uniform(0.55, 0.75)                     # back before ready
        actual_days = int(round(base))

        est_missing = rng.random() < 0.12          # trainer never entered an estimate
        # The trainer estimates from protocol knowledge (textbook timeline for the
        # diagnosis) with an optimism bias - NOT from the athlete's individual risk
        # factors (prior same-site history, age, body mass) and NOT from the future.
        # The ML model's edge comes from learning exactly those ignored factors.
        est_days = max(2, int(round(median_days * rng.normal(0.88, 0.15))))
        est_return = None if est_missing else inj_date + timedelta(days=est_days)

        is_conc = complaint == "Concussion"
        return_date = inj_date + timedelta(days=actual_days)
        still_open = return_date > TODAY
        status = "Open" if still_open else "Closed"

        surgery = complaint.endswith(("ACL Tear", "Meniscus Tear", "Wrist Fracture")) and rng.random() < 0.85
        surgery_date = inj_date + timedelta(days=int(rng.integers(4, 21))) if surgery else None

        if inj_sport in (7, 12):
            surface_id = 378861
        elif inj_sport == 18:
            surface_id = 378863
        elif inj_sport == 32:
            surface_id = 378864
        elif inj_sport in (14, 21):
            surface_id = int(rng.choice([378862, 378860], p=[0.7, 0.3]))
        else:
            # field sports play on whatever their school has
            p_turf = SCHOOL_PROFILES[ath.SchoolId]["turf"]
            surface_id = int(rng.choice([378859, 378860], p=[p_turf, 1 - p_turf]))

        occurred_id = int(rng.choice(list(OCCURRED.keys()), p=[0.52, 0.33, 0.10, 0.05]))
        nature_id = 378675 if "Tendinitis" in profile[0] or profile[0] == "Shin Splints" \
            else (378676 if rng.random() < 0.05 else 378674)
        pain0 = pain0_pre                            # same pain that drives outcome
        pain_missing = rng.random() < 0.12          # PainScale never recorded
        mech = random.choice(MECHANISMS[region])
        historical = rng.random() < 0.04            # imported from previous system

        at_eval = inj_date + timedelta(hours=float(rng.uniform(0.3, 4))) if is_conc else None
        phys_eval = None
        if is_conc:
            r = rng.random()
            if r < 0.65:
                phys_eval = inj_date + timedelta(hours=float(rng.uniform(6, 48)))
            elif r < 0.85:
                phys_eval = inj_date + timedelta(hours=float(rng.uniform(72, 240)))
        n_prior_conc = int(rng.choice([0, 1, 2, 3], p=[0.62, 0.24, 0.10, 0.04])) if is_conc else 0

        rom = region in ("Ankle", "Knee", "Shoulder", "Elbow", "Wrist")
        miss_tracked = actual_days >= 3 and rng.random() > 0.15   # 15% never filled in
        miss_start = inj_date.date() if miss_tracked else None

        injuries.append({
            "ID": inj_id, "User_ID": int(rng.choice(TRAINER_IDS)),
            "Athlete_ID": ath.Athlete_ID, "Sport_ID": inj_sport,
            "InjuryDate": inj_date,
            "InjuryDateOffset": inj_date.strftime("%Y-%m-%d %H:%M:%S -05:00"),
            "InjuryDateUnknown": int(historical and rng.random() < 0.5),
            "InjuryTimeUnknown": int(rng.random() < 0.10),
            "CanPractice": int(actual_days < 3),
            "ReturnDate": None if still_open else return_date,
            "SendMessage": 1, "Message": None, "Notes": None,
            "Status": status,
            "DateClosed": None if still_open else return_date,
            "MissDayStart": miss_start,
            "MissDayEnd": None if (still_open or not miss_tracked) else return_date.date(),
            "CreateDate": inj_date + timedelta(minutes=int(rng.integers(8, 2880 if historical else 240))),
            "EditDate": inj_date + timedelta(minutes=int(rng.integers(8, 240))),
            "Archived": 0, "InjuryGUID": guid(), "Historical": int(historical),
            "Mechanism": mech if rng.random() > 0.08 else None,   # sometimes blank
            "InjuryNatureId": nature_id, "InjuryOccurredId": occurred_id,
            "SurfaceTypeId": surface_id,
            "LocationID": None, "LocationType": None, "Location": None,
            "Summary": (f"Athlete sustained {side_variant(complaint).lower()} during "
                        f"{OCCURRED[occurred_id].lower()}. {mech}."
                        if rng.random() > 0.10 else None),
            "FollowUpPlanID": 1,
            "EstimatedReturnDate": est_return,
            "PROMDegree": f"0-{int(rng.integers(90, 140))}" if rom and rng.random() > 0.3 else None,
            "AROMDegree": f"0-{int(rng.integers(60, 120))}" if rom and rng.random() > 0.3 else None,
            "Flexion": str(int(rng.integers(1, 5))) if rom and rng.random() > 0.5 else None,
            "Extension": str(int(rng.integers(1, 5))) if rom and rng.random() > 0.5 else None,
            "RequiredSurgery": int(surgery),
            "ModifiedBy": int(rng.choice(TRAINER_IDS)),
            "CreatedDate": inj_date.strftime("%Y-%m-%d %H:%M:%S -05:00"),
            "ModifiedDate": inj_date.strftime("%Y-%m-%d %H:%M:%S -05:00"),
            "CreationSource": int(rng.choice([1, 2], p=[0.6, 0.4])),
            "isConcussion": int(is_conc),
            "PhysicianEvalDate": phys_eval,
            "ConcussionLocation": ("Home Venue" if rng.random() < 0.6 else "Away Venue") if is_conc else None,
            "LossOfConsciousness": int(rng.random() < 0.10) if is_conc else None,
            "UnconsciousLength": None,
            "HeadContactItem": (random.choice(["Another Player", "Ground", "Ball", "Equipment"])
                                if is_conc else None),
            "WearingHelmet": (int(inj_sport == 3) if is_conc else None),
            "YearsPlayingSport": ath.YearsPlayingSport,
            "TeamLevel": ath.TeamLevel,
            "ATCEvalDate": at_eval,
            "NumberOfConcussions": n_prior_conc,
            "LossOfConInLastConcussion": int(n_prior_conc > 0 and rng.random() < 0.15),
            "MostRecentConcussionMonth": int(rng.integers(1, 13)) if n_prior_conc else 0,
            "MostRecentConcussionYear": int(inj_date.year - rng.integers(1, 3)) if n_prior_conc else 0,
            "Height": ath.Height, "Weight": ath.Weight,
            "SurgeryDate": surgery_date,
            "NoteToGuardians": 1, "AttachTheInjuryPdfOverview": 0,
            "EnableTwoWayCommunication": 1, "NoteToGuardiansMessage": None,
            "EMR": 2,
            "PainScale": None if pain_missing else pain0,
            "_complaint": complaint, "_bodyRegion": region,
            "_isReinjury": int(is_reinjury),
            "_actualDays": actual_days if not still_open else None,
            "_estDays": None if est_missing else est_days,
            "_slowCase": int(slow_case),
            "_premature": int(premature),
            "_priorSameSite": n_same_site,
            "_priorInjuries": len(history),
        })

        for header, val in [("Complaints", side_variant(complaint)),
                            ("Initial Treatment", random.choice(INITIAL_TREATMENTS)),
                            ("Modalities", random.choice(MODALITIES)),
                            ("Therapeutic Exercise", random.choice(THER_EX))]:
            references.append({"RefId": ref_id, "InjuryId": inj_id, "headerName": header,
                               "primaryRecord": True, "refValue": val})
            ref_id += 1

        # ---------------- treatments (skip most for historical imports)
        # Real cadence: near-daily early, tapering to weekly/biweekly late in
        # long rehabs - so long cases keep receiving care to the end.
        horizon = min(actual_days, (TODAY - inj_date).days)
        n_trt = 1 if historical else max(1, min(40, int(horizon / rng.uniform(1.8, 4.2))))
        pain = float(pain0)
        t_date = inj_date + timedelta(hours=float(rng.uniform(0.5, 6)))
        for t in range(n_trt):
            frac = t / max(1, n_trt - 1)
            if slow_case and frac < 0.6:
                pain += rng.normal(0.15, 0.55)
            else:
                pain -= (pain0 / max(2, n_trt)) * rng.uniform(0.6, 1.5)
            pain = float(np.clip(pain + rng.normal(0, 0.3), 0, 10))
            treatments.append({
                "InjuryTreatmentId": trt_id, "InjuryTreatmentGUID": guid(),
                "InjuryId": inj_id, "AthleteId": ath.Athlete_ID,
                "SportId": inj_sport,
                "TreatmentDate": t_date.strftime("%Y-%m-%d %H:%M:%S -05:00"),
                "CurrentAssessment": assessment_text(complaint, int(round(pain)),
                                                     frac, slow_case, region),
                "EmailSent": int(rng.random() < 0.15), "EmailSentBy": None, "EmailSentDate": None,
                "Active": 1,
                "CreatedDate": t_date.strftime("%Y-%m-%d %H:%M:%S -05:00"),
                "CreatedBy": int(rng.choice(TRAINER_IDS)),
                "ModifiedDate": t_date.strftime("%Y-%m-%d %H:%M:%S -05:00"),
                "ModifiedBy": int(rng.choice(TRAINER_IDS)),
                "SchoolId": ath.SchoolId, "EMR": 3,
                "_painParsed": int(round(pain)),
                "_dayOfRecovery": (t_date - inj_date).days,
            })
            trt_id += 1
            # spread sessions over the WHOLE recovery: daily early, sparser late
            gap = max(1.0, (horizon / max(1, n_trt)) * rng.uniform(0.6, 1.5))
            if rng.random() < 0.10:
                gap += rng.uniform(3, 9)            # weekend/holiday/no-show gaps
            t_date += timedelta(days=float(gap))
            if t_date > TODAY:
                break

        # ---------------- notes (10% of injuries have none; historical usually bare)
        if not historical and rng.random() > 0.10:
            note_list = [(0, initial_note(complaint, mech, pain0, region,
                                          OCCURRED[occurred_id]))]
            if is_conc:
                note_list += concussion_notes(inj_date, phys_eval, n_prior_conc)
            if surgery and surgery_date:
                note_list.append(((surgery_date - inj_date).days, random.choice([
                    "Athlete underwent surgical repair. Procedure successful. "
                    "Post-op plan initiated. PT to begin in 2 weeks.",
                    f"Surgery completed by {random.choice(PHYSICIANS)}. "
                    "Post-op protocol received and filed. NWB 2 wks."])))
            n_mid = int(rng.integers(0, 4))
            for _ in range(n_mid):
                day = int(rng.uniform(0.15, 0.9) * max(3, horizon))
                note_list.append((day, progress_note(day / max(1, actual_days),
                                                     complaint, slow_case)))
            if rng.random() < 0.30:
                note_list.append((int(rng.uniform(0.2, 0.8) * max(3, horizon)), admin_note()))
            if not still_open:
                note_list.append((actual_days, clearance_note(complaint)))
            for offset, text in note_list:
                nd = inj_date + timedelta(days=int(offset))
                if nd > TODAY:
                    continue
                notes.append({
                    "ID": note_id, "UserID": int(rng.choice(TRAINER_IDS)),
                    "InjuryID": inj_id, "Note": text,
                    "NoteDate": nd.strftime("%Y-%m-%d 00:00:00"),
                    "DateCreated": (nd + timedelta(hours=float(rng.uniform(1, 12)))
                                    ).strftime("%Y-%m-%d %H:%M:%S"),
                })
                note_id += 1

        history.append({"profile": profile, "side": side,
                        "premature": premature and not still_open,
                        "complaint": complaint})
        inj_id += 1

# ---------------- standalone treatments (InjuryId = 0, per doc 2.1) ----------------
maintenance = ["Pre-practice ankle taping.", "Recovery flush - compression boots 20 min.",
               "Post-game ice bath.", "Preventive shoulder band work.",
               "General soreness - massage 15 min, no injury opened.",
               "Cramping during practice - fluids + stretching, monitored.",
               "Wrist taping before meet.", "Hip flexor stretch session."]
for _ in range(650):
    ath = athletes_df.sample(1, random_state=None).iloc[0]
    t_date = datetime(2025, 8, 1) + timedelta(days=float(rng.uniform(0, 330)))
    if t_date > TODAY:
        continue
    treatments.append({
        "InjuryTreatmentId": trt_id, "InjuryTreatmentGUID": guid(),
        "InjuryId": 0,                       # standalone, exactly as the doc allows
        "AthleteId": ath.Athlete_ID, "SportId": ath.Sport_ID,
        "TreatmentDate": t_date.strftime("%Y-%m-%d %H:%M:%S -05:00"),
        "CurrentAssessment": random.choice(maintenance),
        "EmailSent": 0, "EmailSentBy": None, "EmailSentDate": None,
        "Active": 1,
        "CreatedDate": t_date.strftime("%Y-%m-%d %H:%M:%S -05:00"),
        "CreatedBy": int(rng.choice(TRAINER_IDS)),
        "ModifiedDate": t_date.strftime("%Y-%m-%d %H:%M:%S -05:00"),
        "ModifiedBy": int(rng.choice(TRAINER_IDS)),
        "SchoolId": ath.SchoolId, "EMR": 3,
        "_painParsed": None, "_dayOfRecovery": None,
    })
    trt_id += 1

injuries_df = pd.DataFrame(injuries)
treatments_df = pd.DataFrame(treatments)
notes_df = pd.DataFrame(notes)
refs_df = pd.DataFrame(references)

lookups = pd.DataFrame(
    [{"LookupType": "Sport", "LookupId": k, "Value": v[0]} for k, v in SPORTS.items()] +
    [{"LookupType": "SurfaceType", "LookupId": k, "Value": v} for k, v in SURFACES.items()] +
    [{"LookupType": "InjuryOccurred", "LookupId": k, "Value": v} for k, v in OCCURRED.items()] +
    [{"LookupType": "InjuryNature", "LookupId": k, "Value": v} for k, v in NATURE.items()] +
    [{"LookupType": "TeamLevel", "LookupId": k, "Value": v} for k, v in TEAM_LEVELS.items()] +
    [{"LookupType": "School", "LookupId": k, "Value": v} for k, v in SCHOOLS.items()]
)

athletes_df.to_csv(f"{OUT}/athletes.csv", index=False)
injuries_df.to_csv(f"{OUT}/injuries.csv", index=False)
treatments_df.to_csv(f"{OUT}/treatments.csv", index=False)
notes_df.to_csv(f"{OUT}/medical_notes.csv", index=False)
refs_df.to_csv(f"{OUT}/injury_references.csv", index=False)
lookups.to_csv(f"{OUT}/lookups.csv", index=False)

dist = injuries_df.groupby("Athlete_ID").size().value_counts().sort_index()
print(f"athletes:           {len(athletes_df):>7,}  "
      f"(injured: {injuries_df.Athlete_ID.nunique():,}; injuries-per-athlete dist: {dist.to_dict()})")
print(f"injuries:           {len(injuries_df):>7,}  (open: {(injuries_df.Status=='Open').sum():,}, "
      f"concussions: {injuries_df.isConcussion.sum():,}, re-injuries: {injuries_df._isReinjury.sum():,}, "
      f"historical: {injuries_df.Historical.sum():,})")
print(f"treatments:         {len(treatments_df):>7,}  (standalone InjuryId=0: "
      f"{(treatments_df.InjuryId==0).sum():,})")
print(f"medical notes:      {len(notes_df):>7,}")
print(f"injury references:  {len(refs_df):>7,}")
print(f"missing PainScale:  {injuries_df.PainScale.isna().mean()*100:.0f}%  |  "
      f"missing estimate: {injuries_df.EstimatedReturnDate.isna().mean()*100:.0f}%  |  "
      f"blank mechanism: {injuries_df.Mechanism.isna().mean()*100:.0f}%")
