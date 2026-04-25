"""All numeric constants for the FreshPrice RL environment.

Source of truth: FreshPrice_SDD.md
Do not use magic numbers inline — reference these constants.
"""

# ---------------------------------------------------------------------------
# Episode structure
# ---------------------------------------------------------------------------
DAYS_PER_EPISODE: int = 7
TICKS_PER_DAY: int = 96              # 24 hours x 4 ticks per hour (15-min resolution)
TOTAL_TICKS: int = TICKS_PER_DAY * DAYS_PER_EPISODE  # 672
TICK_DURATION_MINUTES: int = 15
TICKS_PER_BRIEF: int = 8             # Brief fires every 8 ticks = every 2 simulated hours
BRIEFS_PER_DAY: int = TICKS_PER_DAY // TICKS_PER_BRIEF  # 12
BRIEFS_PER_EPISODE: int = DAYS_PER_EPISODE * BRIEFS_PER_DAY  # 84

# ---------------------------------------------------------------------------
# Expiry urgency thresholds (hours)
# ---------------------------------------------------------------------------
URGENCY_WATCH_HRS: float = 72.0       # > 72h = FRESH, 24-72h = WATCH
URGENCY_URGENT_HRS: float = 24.0      # 6-24h = URGENT
URGENCY_CRITICAL_HRS: float = 6.0     # <= 6h = CRITICAL

# ---------------------------------------------------------------------------
# Pricing engine (Engine 1) constants
# ---------------------------------------------------------------------------
PRICE_MULTIPLIER_MIN: float = 0.25
PRICE_MULTIPLIER_MAX: float = 1.0
FLOOR_PRICE_MARGIN: float = 0.05      # 5% above unit cost
MAX_FLASH_SALES_PER_CATEGORY_PER_DAY: int = 1

# ---------------------------------------------------------------------------
# Reward component r1 (pricing)
# Stored positive — reward.py applies as negative where needed.
# ---------------------------------------------------------------------------
R1_URGENCY_CLEARANCE_BONUS: float = 0.15     # per unit sold within 4h of expiry
R1_NEAR_EXPIRY_HOURS: float = 4.0
R1_EXPIRED_UNIT_PENALTY: float = 0.80        # per unit expired unsold
R1_ANTIHACK_EARLY_DISCOUNT: float = 0.40     # price_mult < 0.35 with hours > 48
R1_ANTIHACK_BELOW_FLOOR: float = 0.40        # proposed price < floor_price

# Anti-hack thresholds for pricing
ANTIHACK_EARLY_DISCOUNT_PRICE_THRESHOLD: float = 0.35   # below this = suspicious
ANTIHACK_EARLY_DISCOUNT_HOURS_THRESHOLD: float = 48.0   # above this = suspicious

# ---------------------------------------------------------------------------
# Parse-failure handling — closes a reward-leak surface where a brief that
# fails to parse can still earn positive WRR delta from natural sales.
# ---------------------------------------------------------------------------
PARSE_FAIL_REWARD_PENALTY: float = 0.30   # subtracted from per-brief reward
PARSE_FAIL_FLAG_THRESHOLD: float = 0.0    # any positive reward + parse_fail = anti-hack
VALIDATION_FAIL_REWARD_PENALTY: float = 0.20  # smaller than parse-fail (parsed but invalid)

# ---------------------------------------------------------------------------
# Farmer engine (Engine 2) constants
# ---------------------------------------------------------------------------
MAX_ACTIVE_FARMER_OFFERS: int = 3
FARMER_OPS_COST_PER_KG: float = 8.0          # operational cost per kg (handling, cold chain)
VIABILITY_SHELF_LIFE_SAFETY_FACTOR: float = 1.5  # shelf life must cover sell-through × this

# ---------------------------------------------------------------------------
# Reward component r2 (farmer)
# Stored positive — reward.py applies as negative where needed.
# ---------------------------------------------------------------------------
R2_CLEARED_BATCH_BONUS: float = 0.20         # per batch fully cleared before expiry
R2_MISSED_OPPORTUNITY_PENALTY: float = 0.50   # declined offer with viability > 0.70
R2_MISSED_OPPORTUNITY_VIABILITY_THRESHOLD: float = 0.70
R2_RECKLESS_ACCEPT_PENALTY: float = 0.60     # accepted with viability < 0.30
ANTIHACK_RECKLESS_ACCEPT_VIABILITY_MAX: float = 0.30

# ---------------------------------------------------------------------------
# Trend engine (Engine 3) constants
# ---------------------------------------------------------------------------
TREND_SCORE_THRESHOLD: float = 65.0
ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER: float = 2.0  # hard cap at 2x weekly velocity
TREND_COOLDOWN_HRS: float = 72.0          # same category cannot trigger again for 72h
TREND_SIGNAL_EXPIRY_HRS: float = 48.0

# Composite score weights (from SDD Section 03 TrendSignal entity)
TREND_WEIGHT_RECIPE_SIMPLICITY: float = 0.25
TREND_WEIGHT_INGREDIENT_RARITY: float = 0.30
TREND_WEIGHT_VIEW_VELOCITY: float = 0.25
TREND_WEIGHT_LOCAL_RELEVANCE: float = 0.10
TREND_WEIGHT_HISTORICAL_CONVERSION: float = 0.10

# ---------------------------------------------------------------------------
# Reward component r3 (trend)
# ---------------------------------------------------------------------------
R3_PERFECT_TIMING_BONUS: float = 0.25         # trend stock sold at full price
R3_OVERTRADE_PENALTY: float = 0.30            # trend stock requires > 40% discount
R3_OVERTRADE_DISCOUNT_THRESHOLD: float = 0.40

# ---------------------------------------------------------------------------
# WRR (unified metric)
# ---------------------------------------------------------------------------
# WRR = revenue_recovered_from_atrisk / cost_of_atrisk_inventory
# at_risk = all batches that were URGENT or CRITICAL at any point this episode

# ---------------------------------------------------------------------------
# Reward weights for WRR components (used in WRRRewardEngine)
# ---------------------------------------------------------------------------
WRR_WEIGHT_R1: float = 0.40
WRR_WEIGHT_R2: float = 0.30
WRR_WEIGHT_R3: float = 0.30

# ---------------------------------------------------------------------------
# Risk buffer
# ---------------------------------------------------------------------------
RISK_BUFFER_INITIAL_SEED_RS: float = 5000.0
RISK_BUFFER_PROFIT_CONTRIBUTION_PCT: float = 0.05  # 5% of each profitable farmer batch

# ---------------------------------------------------------------------------
# Notification credits
# ---------------------------------------------------------------------------
NOTIFICATION_CREDITS_PER_CATEGORY_PER_DAY: int = 3

# ---------------------------------------------------------------------------
# Curriculum promotion
# ---------------------------------------------------------------------------
CURRICULUM_PROMOTION_WRR_THRESHOLD: float = 0.70
CURRICULUM_PROMOTION_WINDOW: int = 5          # consecutive eval episodes

# ---------------------------------------------------------------------------
# Brief quality scoring weights
# ---------------------------------------------------------------------------
BRIEF_QUALITY_WEIGHT_SITUATION: float = 1.0 / 3.0
BRIEF_QUALITY_WEIGHT_VIABILITY: float = 1.0 / 3.0
BRIEF_QUALITY_WEIGHT_DIRECTIVE: float = 1.0 / 3.0

# ---------------------------------------------------------------------------
# Model save & eval
# ---------------------------------------------------------------------------
EVAL_EPISODES_AFTER_SAVE: int = 5
SAVE_WRR_DEGRADATION_TOLERANCE: float = 0.03  # 3%

# ---------------------------------------------------------------------------
# LLM generation parameters
# ---------------------------------------------------------------------------
LLM_MAX_NEW_TOKENS: int = 800
LLM_TEMPERATURE: float = 0.3
LLM_TIMEOUT_SECONDS: int = 40

# ---------------------------------------------------------------------------
# Simulation defaults (for market state initialization)
# ---------------------------------------------------------------------------
DEFAULT_CATEGORIES: list[str] = [
    "fruits",
    "vegetables",
    "dairy",
    "mushrooms",
    "leafy_greens",
    "herbs",
]

# Shelf life ranges per category (hours)
CATEGORY_SHELF_LIFE: dict[str, tuple[float, float]] = {
    "fruits": (48.0, 120.0),
    "vegetables": (36.0, 96.0),
    "dairy": (72.0, 168.0),
    "mushrooms": (24.0, 72.0),
    "leafy_greens": (18.0, 48.0),
    "herbs": (24.0, 72.0),
}

# Base demand velocity (units/hour) per category
CATEGORY_BASE_VELOCITY: dict[str, float] = {
    "fruits": 3.0,
    "vegetables": 2.5,
    "dairy": 4.0,
    "mushrooms": 1.5,
    "leafy_greens": 2.0,
    "herbs": 1.0,
}

# Weekend demand multiplier
WEEKEND_DEMAND_MULTIPLIER: float = 1.5

# Festival day demand spike multiplier
FESTIVAL_DEMAND_MULTIPLIER: float = 2.5

# Supplier delay: reduces incoming stock by this fraction
SUPPLIER_DELAY_STOCK_FRACTION: float = 0.3

# ---------------------------------------------------------------------------
# Carbon footprint (Suggestion 3)
# ---------------------------------------------------------------------------
CO2_PER_KG_FOOD_WASTE: float = 2.5          # kg CO2 per kg food wasted (WRAP UK estimate)
WATER_LITRES_PER_KG_FOOD_WASTE: float = 1000.0  # litres wasted per kg food wasted

# Average weight per unit per category (kg)
CATEGORY_AVG_WEIGHT_KG: dict[str, float] = {
    "fruits":      0.30,
    "vegetables":  0.20,
    "dairy":       0.50,
    "mushrooms":   0.15,
    "leafy_greens": 0.10,
    "herbs":       0.05,
    "bakery":      0.25,
    "packaged":    0.40,
}

# Average retail price per unit per category (Rs) — used for units-sold estimation
CATEGORY_AVG_PRICE_RS: dict[str, float] = {
    "fruits":      65.0,
    "vegetables":  37.5,
    "dairy":       70.0,
    "mushrooms":   37.5,
    "leafy_greens": 25.0,
    "herbs":       25.0,
    "bakery":      50.0,
    "packaged":   105.0,
}

# ---------------------------------------------------------------------------
# Weather shocks (Suggestion 7)
# ---------------------------------------------------------------------------
WEATHER_RAIN_DEMAND_MULTIPLIER: float = 0.75        # rain reduces footfall
WEATHER_HOT_FRUITS_MULTIPLIER: float = 1.25         # heat spikes cold-fruit demand
WEATHER_HOT_DAIRY_MULTIPLIER: float = 0.85          # heat suppresses heavy dairy
WEATHER_COLD_BAKERY_DAIRY_MULTIPLIER: float = 1.15  # cold drives warm/comfort foods

# Festival demand multipliers by category
FESTIVAL_DEMAND_MULTIPLIERS: dict[str, float] = {
    "fruits":      2.5,
    "dairy":       2.0,
    "herbs":       1.8,
    "vegetables":  1.5,
    "bakery":      2.2,
    "packaged":    1.3,
    "mushrooms":   1.4,
    "leafy_greens": 1.3,
}

# Sports-event demand spike (packaged snacks, dairy)
SPORTS_EVENT_PACKAGED_MULTIPLIER: float = 1.4
SPORTS_EVENT_DAIRY_MULTIPLIER: float = 1.2

# ---------------------------------------------------------------------------
# Multi-store coordination (Suggestion 5)
# ---------------------------------------------------------------------------
INTER_STORE_TRANSFER_COST_RS_PER_KG: float = 5.0   # cold-chain transfer cost
INTER_STORE_MAX_TRANSFER_PCT: float = 0.30          # max 30% of batch can be transferred
MIN_UNITS_FOR_TRANSFER: int = 5                     # minimum transfer threshold

# ---------------------------------------------------------------------------
# Long-horizon mode (Theme #2)
# ---------------------------------------------------------------------------
LONG_HORIZON_DAYS: int = 30                        # 30-day month-long horizon
LONG_HORIZON_TICKS: int = LONG_HORIZON_DAYS * TICKS_PER_DAY   # 2880
SPARSE_REWARD_INTERVAL_DAYS: int = 7               # reward paid weekly, not per-brief
LONG_HORIZON_BRIEFS_PER_EPISODE: int = (
    LONG_HORIZON_DAYS * BRIEFS_PER_DAY              # 30 days * 12 briefs = 360
)

# Long-horizon prompt truncation. The tick-context prompt is forcibly
# trimmed to this many characters AFTER the notebook block has been
# prepended. Forces the agent to externalize state into the notebook —
# without it, briefs from day 3 are gone by day 27.
LONG_HORIZON_PROMPT_TAIL_CHARS: int = 3500
# Brief-history window inside the prompt (most recent K briefs only)
LONG_HORIZON_RECENT_BRIEFS_WINDOW: int = 4

# Plan-adherence reward (r4): bonus per honored commitment, penalty per
# broken commitment. Stored positive — applied as +/- in reward.py.
R4_HONORED_COMMITMENT_BONUS: float = 0.08
R4_BROKEN_COMMITMENT_PENALTY: float = 0.10

# Token-scaled reasoning reward (r5, Mercor sub-prize).
# r5 = clip(token_count / R5_TOKEN_TARGET, 0, R5_CAP) iff quality >= R5_QUALITY_FLOOR
R5_TOKEN_TARGET: int = 600          # full reward at ~600 tokens of brief
R5_CAP: float = 0.20                # caps at 20% of WRR delta — never dominant
R5_QUALITY_FLOOR: float = 0.55      # below this, longer briefs get nothing
R5_TOKEN_HARD_CAP: int = 1200       # beyond this, no extra credit (anti-bloat)

# Cooperation index (multi-store, multi-agent). Computed in MarketCommonsEnv.
COOPERATION_INDEX_TRANSFER_WEIGHT: float = 0.35
COOPERATION_INDEX_MESSAGE_WEIGHT: float = 0.25
COOPERATION_INDEX_PARETO_WEIGHT: float = 0.40

# ---------------------------------------------------------------------------
# FRESHPRICE 7-ENGINE REWARD WEIGHTS (Store Efficiency Score = SES)
# Source: FreshPrice_Strategy.docx Section 7. SES = sum(w_i * r_i).
# Used as primary curriculum-promotion metric (SES >= 0.70 over 5 evals).
# ---------------------------------------------------------------------------

SES_WEIGHT_R1_PRICING:    float = 0.28   # Engine 1 - daily pricing decisions
SES_WEIGHT_R2_FARMER:     float = 0.18   # Engine 2 - farmer offer accept/decline
SES_WEIGHT_R3_TREND:      float = 0.15   # Engine 3 - social-trend purchase orders
SES_WEIGHT_R4_INTRAFLEET: float = 0.12   # Engine 4 - inter-store transfers
SES_WEIGHT_R5_MICROMFG:   float = 0.10   # Engine 5 - micro-manufacturer routing
SES_WEIGHT_R6_EVENT:      float = 0.10   # Engine 6 - event pre-positioning
SES_WEIGHT_R7_SURPLUSBOX: float = 0.07   # Engine 7 - weekly surplus box

# Sanity: weights sum to 1.0 - verified at module load by tests.

# ---------------------------------------------------------------------------
# Engine 4 - Intra-Fleet Rebalancing (TRANSFER directive). Multi-store only.
# r4_intrafleet = revenue_recovered / waste_prevented - transfer_cost. The
# multi-store engine already exists in multi_store_env.py; these are the
# reward shaping constants used in r4 computation.
# ---------------------------------------------------------------------------
R4_TRANSFER_COST_PENALTY_PER_KG: float = 0.05   # opportunity cost per kg ferried
R4_TRANSFER_REVENUE_BONUS:       float = 0.20   # per unit sold post-transfer
R4_RECKLESS_TRANSFER_PENALTY:    float = 0.30   # transfer that wastes the source

# ---------------------------------------------------------------------------
# Engine 5 - Micro-Manufacturer Pipeline. Routes near-expiry batches to
# registered processors (juice bars, pickle makers, bakeries) for partial
# revenue recovery instead of zero-revenue spoilage.
# r5 = recovered_rs / (batch_cost * unsold_fraction). Floor: 30% recovery
# is positive reward vs. zero for waste.
# ---------------------------------------------------------------------------
MICROMFG_RECOVERY_RATIO_DEFAULT: float = 0.35   # processor pays ~35% of cost
MICROMFG_HOURS_TO_EXPIRY_FLOOR:  float = 12.0   # eligible window: <=12 hrs
R5_LATE_ROUTING_PENALTY:         float = 0.25   # routed too late to use
R5_EARLY_ROUTING_PENALTY:        float = 0.30   # routed when discount could clear

# ---------------------------------------------------------------------------
# Engine 6 - Event Pre-Positioning. Pre-stocks for detected events.
# r6 = event_day_revenue_uplift / baseline; bonus on zero stockouts;
# penalty if pre-stock spoils unsold.
# ---------------------------------------------------------------------------
EVENT_PRESTOCK_LEAD_HOURS_MIN:    float = 4.0    # min hrs ahead to count as "pre"
EVENT_PRESTOCK_LEAD_HOURS_MAX:    float = 48.0   # supplier lead time cap
R6_EVENT_NO_STOCKOUT_BONUS:       float = 0.30   # full coverage bonus
R6_EVENT_OVERSTOCK_PENALTY:       float = 0.20   # unsold pre-stock post-event
R6_EVENT_DEMAND_UPLIFT_DENOMINATOR: float = 100.0  # normaliser for r6 scaling

# ---------------------------------------------------------------------------
# Engine 7 - Surplus Box Subscription. Friday assembly of near-expiry
# items into a 1.5-2 kg box; weekly dispatch to subscribers.
# r7 = subscriber_retention * (box_revenue / box_cost).
# ---------------------------------------------------------------------------
SURPLUS_BOX_DEFAULT_SUBSCRIBERS:    int   = 25
SURPLUS_BOX_TARGET_WEIGHT_KG_MIN:   float = 1.5
SURPLUS_BOX_TARGET_WEIGHT_KG_MAX:   float = 2.0
SURPLUS_BOX_PRICE_PER_KG:           float = 80.0
R7_FIVE_STAR_BONUS_PER_SUBSCRIBER:  float = 0.10
R7_CANCEL_PENALTY_PER_SUBSCRIBER:   float = 0.30

# ---------------------------------------------------------------------------
# (Optional, non-SES) Blinkit/Zepto-style realism extras. Kept for the
# /commons/* dashboard panels but NOT in the SES path. Numbering uses
# semantic names rather than r6/r7 to avoid colliding with the user's
# Engines 6 and 7 above.
# Public sources: Blinkit dark-store ops post-mortems, Zepto Plus
# launch press, FY24 ICRA QCom report.
# ---------------------------------------------------------------------------

# Rider/courier pool. A single dark store typically runs 6-10 riders.
BLINKIT_DEFAULT_RIDER_COUNT: int = 6
# Average ferry time per order (store -> pin -> store) at a 1.5 km median.
BLINKIT_RIDER_FERRY_MINUTES: float = 14.0
# Promised ETA tiers. The brief can opt batches into Express by setting
# eta_tier="EXPRESS" in a directive; otherwise standard applies.
BLINKIT_EXPRESS_PROMISE_MINUTES: float = 10.0
BLINKIT_STANDARD_PROMISE_MINUTES: float = 30.0
# r6_delivery_quality: penalty per order whose freshness clock expires
# while it sat in the rider queue (this is the *real* spoilage in QCom,
# not shelf-spoilage).
R6_TRANSIT_SPOILAGE_PENALTY: float = 0.45
# r6 bonus per order delivered within promised ETA (caps at +0.20 per brief).
R6_ON_TIME_BONUS: float = 0.04
R6_BONUS_CAP_PER_BRIEF: float = 0.20

# Consumer cohort weights (must sum to 1.0). Premium cohort is small
# in volume but disproportionately drives loyalty-program revenue.
COHORT_WEIGHT_PREMIUM: float = 0.18    # Blinkit Plus / Zepto Pass holders
COHORT_WEIGHT_MASS: float = 0.62
COHORT_WEIGHT_BARGAIN: float = 0.20
# Per-cohort price elasticities (demand = ratio**elasticity).
COHORT_ELASTICITY_PREMIUM: float = 0.40
COHORT_ELASTICITY_MASS: float = 1.20
COHORT_ELASTICITY_BARGAIN: float = 2.50
# Per-cohort ETA tolerance (minutes). Premium cohort walks if late.
COHORT_ETA_TOLERANCE_PREMIUM_MIN: float = 12.0
COHORT_ETA_TOLERANCE_MASS_MIN: float = 25.0
COHORT_ETA_TOLERANCE_BARGAIN_MIN: float = 60.0
# Per-cohort freshness tolerance: max URGENT/CRITICAL ratio they accept.
COHORT_FRESHNESS_TOL_PREMIUM: float = 0.05
COHORT_FRESHNESS_TOL_MASS: float = 0.50
COHORT_FRESHNESS_TOL_BARGAIN: float = 1.00

# Liquidation channel — B2B firesale for dead stock.
# Recovery ratio: fraction of original price recovered when LIQUIDATEd.
LIQUIDATION_RECOVERY_RATIO: float = 0.18
# Anti-hack: liquidating non-CRITICAL stock is a violation. The agent
# cannot dump fresh inventory to game R1.
LIQUIDATION_MIN_URGENCY_HOURS: float = 6.0
LIQUIDATION_RECKLESS_PENALTY: float = 0.50
# r7_liquidation: positive reward when expected discount-driven sell-through
# is *less* than what we recovered via B2B; negative otherwise.
R7_LIQUIDATION_BONUS: float = 0.10
R7_LIQUIDATION_RECKLESS: float = 0.50
