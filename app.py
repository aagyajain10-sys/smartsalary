import sqlite3
import calendar
import hashlib
import hmac
import secrets
from pathlib import Path
from datetime import date

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Salary & Expense Manager",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "model_output"
DB_PATH = BASE_DIR / "expenses.db"
MODEL_PATH = OUTPUT_DIR / "ridge_regression.joblib"

FEATURES = [
    "days_elapsed",
    "days_remaining",
    "cumulative_spending",
    "average_daily_spending",
    "days_in_month"
]

CATEGORIES = [
    "Food", "Transport", "Shopping", "Bills", "Rent",
    "Education", "Health", "Entertainment", "Groceries",
    "Personal care", "Travel", "Other"
]

# ============================================================
# DATABASE AND AUTHENTICATION
# ============================================================

def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200_000
    )
    return salt.hex(), digest.hex()


def verify_password(password, salt_hex, password_hash):
    try:
        _, candidate = hash_password(password, bytes.fromhex(salt_hex))
        return hmac.compare_digest(candidate, password_hash)
    except (ValueError, TypeError):
        return False


def initialize_database():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS monthly_plans (
                user_id INTEGER NOT NULL,
                month TEXT NOT NULL,
                salary REAL NOT NULL DEFAULT 0,
                savings_goal REAL NOT NULL DEFAULT 0,
                fixed_expenses REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, month),
                FOREIGN KEY (user_id)
                    REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS personal_expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                expense_date TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                category TEXT NOT NULL,
                description TEXT DEFAULT '',
                payment_method TEXT DEFAULT 'Not specified',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id)
                    REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expenses_user_date
            ON personal_expenses(user_id, expense_date)
        """)


def register_user(display_name, username, password):
    salt, password_hash = hash_password(password)

    try:
        with db_connect() as conn:
            cursor = conn.execute("""
                INSERT INTO users
                    (display_name, username, salt, password_hash)
                VALUES (?, ?, ?, ?)
            """, (display_name.strip(), username.strip(), salt, password_hash))
            return int(cursor.lastrowid), None
    except sqlite3.IntegrityError:
        return None, "That username is already registered."


def authenticate_user(username, password):
    with db_connect() as conn:
        row = conn.execute("""
            SELECT id, display_name, salt, password_hash
            FROM users
            WHERE username = ? COLLATE NOCASE
        """, (username.strip(),)).fetchone()

    if row and verify_password(password, row["salt"], row["password_hash"]):
        return {"id": int(row["id"]), "name": row["display_name"]}
    return None


def get_plan(user_id, month):
    with db_connect() as conn:
        row = conn.execute("""
            SELECT salary, savings_goal, fixed_expenses
            FROM monthly_plans
            WHERE user_id = ? AND month = ?
        """, (user_id, month)).fetchone()

    return tuple(float(value) for value in row) if row else (0.0, 0.0, 0.0)


def save_plan(user_id, month, salary, savings_goal, fixed_expenses):
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO monthly_plans
                (user_id, month, salary, savings_goal, fixed_expenses)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, month)
            DO UPDATE SET
                salary = excluded.salary,
                savings_goal = excluded.savings_goal,
                fixed_expenses = excluded.fixed_expenses
        """, (user_id, month, salary, savings_goal, fixed_expenses))


def get_expenses(user_id, month):
    with db_connect() as conn:
        df = pd.read_sql_query("""
            SELECT id, expense_date, amount, category,
                   description, payment_method
            FROM personal_expenses
            WHERE user_id = ?
              AND substr(expense_date, 1, 7) = ?
            ORDER BY expense_date DESC, id DESC
        """, conn, params=(user_id, month))

    if not df.empty:
        df["expense_date"] = pd.to_datetime(df["expense_date"]).dt.date
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def get_previous_month_totals(user_id, before_month, limit=6):
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT substr(expense_date, 1, 7) AS month,
                   SUM(amount) AS total
            FROM personal_expenses
            WHERE user_id = ?
              AND substr(expense_date, 1, 7) < ?
            GROUP BY substr(expense_date, 1, 7)
            ORDER BY month DESC
            LIMIT ?
        """, (user_id, before_month, limit)).fetchall()
    return [float(row["total"]) for row in rows]


def add_expense(user_id, expense_date, amount, category, description, payment_method):
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO personal_expenses
                (user_id, expense_date, amount, category,
                 description, payment_method)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id, expense_date.isoformat(), float(amount),
            category, description.strip(), payment_method
        ))


def delete_expense(user_id, expense_id):
    with db_connect() as conn:
        conn.execute("""
            DELETE FROM personal_expenses
            WHERE id = ? AND user_id = ?
        """, (int(expense_id), user_id))


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def money(value):
    return f"₹{float(value):,.2f}"


# ============================================================
# INITIALIZE APP + PREMIUM DARK DESIGN
# ============================================================

initialize_database()

st.markdown("""
<style>
:root {
    --bg: #0B0D12;
    --panel: #131722;
    --panel-2: #191F2C;
    --panel-3: #202838;
    --blue: #4C8DFF;
    --blue-2: #2F6FE4;
    --blue-soft: #172C4D;
    --text: #F2F5FB;
    --muted: #A2AEC2;
    --line: #293244;
    --green: #35D39A;
    --red: #FF747F;
    --amber: #F4BD50;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}
.block-container {
    max-width: 1480px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
#MainMenu, footer {visibility: hidden;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: var(--text) !important;
    letter-spacing: -0.035em;
    font-weight: 700 !important;
}
.stApp p, .stApp label, .stApp [data-testid="stCaptionContainer"] {
    color: var(--muted);
}
.stApp hr {border-color: var(--line) !important;}

/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background: #0F131C !important;
}
[data-testid="stSidebar"] {
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: var(--text) !important;
}

/* Inputs */
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="base-input"],
.stApp [data-baseweb="textarea"],
.stApp [data-testid="stNumberInput"] [data-baseweb="input"],
.stApp [data-testid="stTextInput"] [data-baseweb="input"] {
    background: var(--panel-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}
.stApp input, .stApp textarea,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--blue) !important;
    background: transparent !important;
}
.stApp input::placeholder, .stApp textarea::placeholder {
    color: #77849B !important;
    -webkit-text-fill-color: #77849B !important;
}
.stApp [data-baseweb="input"]:focus-within,
.stApp [data-baseweb="base-input"]:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 1px var(--blue) !important;
}
.stApp input:-webkit-autofill {
    -webkit-box-shadow: 0 0 0 1000px var(--panel-2) inset !important;
    -webkit-text-fill-color: var(--text) !important;
}

/* Buttons */
.stButton > button, .stFormSubmitButton > button,
.stDownloadButton > button {
    background: var(--panel-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    min-height: 2.7rem;
    font-weight: 650 !important;
    transition: all .18s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--blue) !important;
    background: var(--panel-3) !important;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"],
button[kind="primary"] {
    background: var(--blue) !important;
    color: white !important;
    border-color: var(--blue) !important;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover,
button[kind="primary"]:hover {background: var(--blue-2) !important;}

/* Forms and containers */
[data-testid="stForm"] {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line) !important;
    border-radius: 16px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #171D29, #111620) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.15rem !important;
    min-height: 125px;
    box-shadow: 0 8px 24px rgba(0,0,0,.12);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: .88rem !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
    color: var(--text) !important;
    font-weight: 750 !important;
    letter-spacing: -.035em;
}
[data-testid="stMetricDelta"] {font-weight: 650;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 7px;
    background: var(--panel) !important;
    border: 1px solid var(--line);
    border-radius: 13px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 9px 15px;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--blue-soft) !important;
    color: var(--text) !important;
}
.stTabs [aria-selected="true"] p {color: var(--text) !important;}

/* Dataframe, alerts, links */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stAlert"] {border-radius: 12px !important;}
.stApp a {color: #83B2FF !important;}
.stCheckbox label, .stRadio label {color: var(--text) !important;}

/* Hero and page header */
.hero-shell {
    background:
        radial-gradient(circle at 88% 18%, rgba(76,141,255,.22), transparent 32%),
        linear-gradient(135deg, #171E2B 0%, #10141E 72%);
    border: 1px solid #2A3549;
    border-radius: 24px;
    padding: 1.7rem 1.8rem;
    margin: .1rem 0 1.35rem 0;
}
.eyebrow {
    color: #8EB8FF;
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: .65rem;
}
.hero-title {
    color: #F5F7FC;
    font-size: clamp(1.7rem, 3vw, 2.55rem);
    font-weight: 800;
    letter-spacing: -.055em;
    line-height: 1.12;
    margin: 0 0 .75rem 0;
}
.hero-copy {
    color: #AAB6CA;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 720px;
}
.pill {
    display: inline-block;
    border: 1px solid #31466A;
    color: #BBD4FF;
    background: rgba(23,44,77,.65);
    padding: .35rem .7rem;
    border-radius: 999px;
    font-size: .78rem;
    margin: .7rem .35rem 0 0;
}
.section-label {
    color: #8EB8FF;
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.auth-panel {
    background: linear-gradient(155deg, #171D29, #11151E);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1.4rem;
}
div[data-testid="stVerticalBlock"] {gap: .85rem;}
/* Tablets and smaller screens */
@media (max-width: 800px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    .hero-shell {
        padding: 1.25rem;
    }
}

/* Mobile phones */
@media (max-width: 480px) {
    .block-container {
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    .hero-shell {
        padding: 1rem;
    }

    h1 {
        font-size: 1.6rem !important;
    }

    h2 {
        font-size: 1.3rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
    }

    .stButton button {
        width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOGIN / REGISTRATION SCREEN
# ============================================================

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.user_name = None

if st.session_state.user_id is None:
    left, right = st.columns([1.12, 0.88], gap="large")

    with left:
        st.markdown("""
        <div class="hero-shell">
            <div class="eyebrow">PERSONAL FINANCE • MADE CLEAR</div>
            <div class="hero-title">Make every rupee<br>work smarter.</div>
            <div class="hero-copy">
                Plan your salary, protect your savings, track everyday spending,
                and see how much you can comfortably spend next.
            </div>
            <span class="pill">Monthly budget planning</span>
            <span class="pill">Daily expense tracking</span>
            <span class="pill">Personal spending insights</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Your money, in one place")
        st.markdown("""
        - **Plan first:** set salary, savings, and fixed commitments.
        - **Track simply:** record an expense in a few seconds.
        - **Stay informed:** see your remaining balance and spending trends.
        """)

        st.caption("A personal finance research prototype. Estimates are guidance, not guarantees.")

    with right:
        st.markdown('<div class="section-label">WELCOME</div>', unsafe_allow_html=True)
        st.markdown("## Your finance workspace")
        st.markdown("Log in to continue, or create an account to get started.")

        login_tab, register_tab = st.tabs(["Log in", "Create account"])

        with login_tab:
            with st.form("login_form"):
                login_username = st.text_input("Username", placeholder="Enter your username")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                login_clicked = st.form_submit_button(
                    "Log in to dashboard",
                    type="primary",
                    use_container_width=True
                )

            if login_clicked:
                user = authenticate_user(login_username, login_password)
                if user:
                    st.session_state.user_id = user["id"]
                    st.session_state.user_name = user["name"]
                    st.rerun()
                else:
                    st.error("Username or password is incorrect.")

        with register_tab:
            with st.form("register_form"):
                new_name = st.text_input("Your name", placeholder="e.g. Aatya")
                new_username = st.text_input("Create a username", placeholder="At least 3 characters")
                new_password = st.text_input("Create a password", type="password", placeholder="At least 8 characters")
                confirm_password = st.text_input("Confirm password", type="password")
                register_clicked = st.form_submit_button(
                    "Create my account",
                    type="primary",
                    use_container_width=True
                )

            if register_clicked:
                if not new_name.strip() or not new_username.strip():
                    st.error("Enter your name and a username.")
                elif len(new_username.strip()) < 3:
                    st.error("Username must contain at least 3 characters.")
                elif len(new_password) < 8:
                    st.error("Use a password with at least 8 characters.")
                elif new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    new_id, error = register_user(new_name, new_username, new_password)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.user_id = new_id
                        st.session_state.user_name = new_name.strip()
                        st.success("Account created successfully.")
                        st.rerun()

        st.caption("Your account keeps your salary plans and expense history separate from other users.")

    st.stop()


# ============================================================
# LOGGED-IN USER + SIDEBAR
# ============================================================

user_id = int(st.session_state.user_id)
current_month = date.today().strftime("%Y-%m")
start_year = 2015
end_year = date.today().year

month_options = [
    f"{year:04d}-{month:02d}"
    for year in range(start_year, end_year + 1)
    for month in range(1, 13)
    if f"{year:04d}-{month:02d}" <= current_month
]

with st.sidebar:
    st.markdown("## 💠 MoneySpace")
    st.caption("PERSONAL FINANCE")
    st.markdown("---")
    st.markdown(f"### Welcome, {st.session_state.user_name} 👋")
    st.caption("Your personal finance dashboard")

    if st.button("Log out", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.user_name = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Workspace")
    selected_month = st.selectbox(
        "Choose month",
        month_options,
        index=month_options.index(current_month)
    )

    st.markdown("---")
    st.markdown("### Monthly salary plan")
    saved_salary, saved_savings, saved_fixed = get_plan(user_id, selected_month)

    with st.form("plan_form"):
        salary = st.number_input("Monthly salary (₹)", min_value=0.0, value=saved_salary, step=500.0)
        savings_goal = st.number_input("Savings to reserve first (₹)", min_value=0.0, value=saved_savings, step=500.0)
        fixed_expenses = st.number_input("Fixed bills / commitments (₹)", min_value=0.0, value=saved_fixed, step=500.0)
        save_plan_clicked = st.form_submit_button("Save monthly plan", type="primary", use_container_width=True)

    if save_plan_clicked:
        if savings_goal + fixed_expenses > salary:
            st.error("Savings and fixed expenses cannot be more than your salary.")
        else:
            save_plan(user_id, selected_month, salary, savings_goal, fixed_expenses)
            st.success("Monthly plan saved.")
            st.rerun()

    st.markdown("---")
    st.caption("Tip: record expenses regularly for more useful personal insights.")


# ============================================================
# SPENDING CALCULATIONS (PRESERVED)
# ============================================================

month_df = get_expenses(user_id, selected_month)

days_in_month = calendar.monthrange(
    int(selected_month[:4]), int(selected_month[5:7])
)[1]

month_total = float(month_df["amount"].sum()) if not month_df.empty else 0.0
spendable_amount = max(0.0, salary - savings_goal - fixed_expenses)
remaining_money = spendable_amount - month_total
is_current_month = selected_month == current_month

if is_current_month:
    elapsed_day = date.today().day
    days_left_including_today = days_in_month - date.today().day + 1
    today_total = (
        float(month_df.loc[month_df["expense_date"] == date.today(), "amount"].sum())
        if not month_df.empty else 0.0
    )
else:
    elapsed_day = days_in_month
    days_left_including_today = 0
    today_total = 0.0

daily_target = spendable_amount / days_in_month if days_in_month else 0.0

if is_current_month and days_left_including_today:
    remaining_daily_limit = max(0.0, remaining_money / days_left_including_today)
else:
    remaining_daily_limit = 0.0

# Existing personal estimate logic is intentionally preserved.
previous_month_totals = get_previous_month_totals(user_id, selected_month, limit=6)
predicted_month_end = None
prediction_label = "Month-end spending estimate"
prediction_note = ""

if is_current_month:
    if month_total > 0 and elapsed_day > 0:
        predicted_month_end = (month_total / elapsed_day) * days_in_month
        prediction_label = "Personal spending-rate estimate"
        prediction_note = "Based on expenses you recorded this month. The estimate may change as you add expenses."
    elif len(previous_month_totals) >= 3:
        recent_months = previous_month_totals[:3]
        predicted_month_end = sum(recent_months) / len(recent_months)
        prediction_label = "Average of your previous months"
        prediction_note = "No expenses recorded this month yet. This estimate uses your three most recent months with recorded expenses."
    else:
        prediction_note = "Not enough personal expense history to make a reliable estimate yet. Record expenses and check again."
else:
    predicted_month_end = month_total
    prediction_label = "Recorded spending for selected month"
    prediction_note = "This is the total recorded for this month, not a prediction."


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(f"""
<div class="hero-shell">
    <div class="eyebrow">YOUR PERSONAL FINANCE OVERVIEW</div>
    <div class="hero-title">Good money habits start here.</div>
    <div class="hero-copy">
        Welcome back, {st.session_state.user_name}. Review your monthly plan,
        track your spending, and keep your next financial decision clear.
    </div>
    <span class="pill">Selected month: {selected_month}</span>
    <span class="pill">Currency: INR ₹</span>
</div>
""", unsafe_allow_html=True)

tab_dashboard, tab_add, tab_history = st.tabs([
    "◈  Dashboard",
    "＋  Add an expense",
    "▤  Expense history"
])


# ============================================================
# DASHBOARD
# ============================================================

with tab_dashboard:
    st.markdown("### Financial snapshot")
    st.caption(f"Your salary plan and recorded spending for {selected_month}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly salary", money(salary))
    c2.metric("Savings reserved", money(savings_goal))
    c3.metric("Spent this month", money(month_total))
    c4.metric(
        "Money remaining",
        money(remaining_money),
        delta="Within plan" if remaining_money >= 0 else "Over planned amount",
        delta_color="normal" if remaining_money >= 0 else "inverse"
    )

    st.markdown("")
    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        st.markdown("### Your money plan")
        plan_df = pd.DataFrame({
            "Type": ["Savings reserved", "Fixed expenses", "Spent so far", "Still available"],
            "Amount (₹)": [savings_goal, fixed_expenses, month_total, max(0.0, remaining_money)]
        })
        st.bar_chart(plan_df.set_index("Type"), use_container_width=True)
        st.caption(f"Flexible spending available after savings and fixed bills: **{money(spendable_amount)}**")

    with right:
        st.markdown("### Daily spending guidance")
        st.metric("Suggested daily amount at start of month", money(daily_target))

        if is_current_month:
            st.metric("Suggested amount per remaining day", money(remaining_daily_limit))
            st.write(f"Today's recorded expenses: **{money(today_total)}**")

            if remaining_money < 0:
                st.error(f"You are {money(abs(remaining_money))} over your available spending amount.")
            elif remaining_daily_limit < daily_target:
                st.warning("Your remaining daily allowance is lower because of spending already recorded.")
            else:
                st.success("You are within your available spending amount.")
        else:
            st.info("Live remaining-day guidance is shown for the current month.")

    st.markdown("---")
    st.markdown("### Month-end spending outlook")
    p1, p2 = st.columns(2)

    if predicted_month_end is not None:
        p1.metric(prediction_label, money(predicted_month_end))
        if is_current_month and spendable_amount > 0:
            difference = predicted_month_end - spendable_amount
            p2.metric(
                "Estimate vs spendable amount",
                money(difference),
                delta="Above plan" if difference > 0 else "Within plan",
                delta_color="inverse" if difference > 0 else "normal"
            )
        else:
            p2.metric("Spendable amount", money(spendable_amount))
    else:
        p1.metric("Month-end spending estimate", "Not available")
        p2.metric("Forecast status", "Waiting for data")

    st.caption(prediction_note)

    if not month_df.empty:
        st.markdown("---")
        chart_left, chart_right = st.columns(2, gap="large")

        with chart_left:
            st.markdown("### Spending by day")
            chart_df = month_df.copy()
            chart_df["Day"] = pd.to_datetime(chart_df["expense_date"]).dt.day
            daily = chart_df.groupby("Day")["amount"].sum().reindex(
                range(1, days_in_month + 1), fill_value=0
            )
            daily_chart = pd.DataFrame({
                "Daily spending": daily,
                "Cumulative spending": daily.cumsum()
            })
            st.line_chart(daily_chart, use_container_width=True)

        with chart_right:
            st.markdown("### Spending by category")
            category_totals = month_df.groupby("category")["amount"].sum().sort_values(ascending=False)
            if not category_totals.empty:
                st.bar_chart(category_totals, use_container_width=True)
    else:
        st.info("No personal expenses have been recorded for this month. Open **Add an expense** to start building your history.")


# ============================================================
# ADD EXPENSE
# ============================================================

with tab_add:
    st.markdown("### Record an expense")
    st.caption(f"New expenses will be saved under {selected_month}.")

    form_left, form_right = st.columns([1.1, 0.9], gap="large")

    with form_left:
        with st.form("expense_form", clear_on_submit=True):
            default_date = date.today()
            if selected_month != current_month:
                default_date = date(int(selected_month[:4]), int(selected_month[5:7]), 1)

            expense_date = st.date_input("Expense date", value=default_date)
            amount = st.number_input("Amount (₹)", min_value=0.0, value=0.0, step=10.0, format="%.2f")
            category = st.selectbox("Category", CATEGORIES)
            description = st.text_input("Description (optional)", placeholder="e.g. lunch, bus fare, groceries")
            payment_method = st.selectbox(
                "Payment method",
                ["UPI", "Cash", "Debit card", "Credit card", "Bank transfer", "Other", "Not specified"]
            )
            add_clicked = st.form_submit_button("Save expense", type="primary", use_container_width=True)

        if add_clicked:
            if expense_date.strftime("%Y-%m") != selected_month:
                st.error("The expense date must be in the selected month.")
            elif amount <= 0:
                st.error("Enter an amount greater than ₹0.")
            else:
                add_expense(user_id, expense_date, amount, category, description, payment_method)
                st.success(f"Saved {money(amount)} for {expense_date.strftime('%d %b %Y')}.")
                st.rerun()

    with form_right:
        st.markdown("""
        <div class="hero-shell">
            <div class="eyebrow">QUICK REMINDER</div>
            <div style="color:#F5F7FC;font-size:1.35rem;font-weight:750;margin-bottom:.6rem;">
                Small entries. Clearer decisions.
            </div>
            <div class="hero-copy">
                Add expenses as you make them. Your dashboard updates with your
                monthly total, remaining amount, daily guidance, and category breakdown.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Spent so far this month", money(month_total))
        st.metric("Remaining from your spendable amount", money(remaining_money))


# ============================================================
# EXPENSE HISTORY
# ============================================================

with tab_history:
    st.markdown("### Expense history")
    st.caption(f"Saved transactions for {selected_month}.")

    if month_df.empty:
        st.info("There are no saved expenses for this month yet.")
    else:
        display_df = month_df.rename(columns={
            "expense_date": "Date",
            "amount": "Amount (₹)",
            "category": "Category",
            "description": "Description",
            "payment_method": "Payment method"
        })

        st.dataframe(
            display_df[["id", "Date", "Amount (₹)", "Category", "Description", "Payment method"]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download this month's expenses (CSV)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name=f"personal_expenses_{selected_month}.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("### Delete an expense")
        expense_options = {
            (
                f"#{int(row.id)} · {row.expense_date} · {money(row.amount)} · "
                f"{row.category} · {row.description}"
            ): int(row.id)
            for row in month_df.itertuples(index=False)
        }

        chosen_label = st.selectbox("Select expense to delete", list(expense_options.keys()))
        confirm_delete = st.checkbox("I confirm I want to permanently delete this expense")

        if st.button("Delete selected expense", disabled=not confirm_delete):
            delete_expense(user_id, expense_options[chosen_label])
            st.success("Expense deleted.")
            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Smart Salary & Expense Manager · Personal finance research prototype. "
    "Each user's records are filtered by account ID. Before public deployment, "
    "add HTTPS, secure session management, backups, and production-grade database security. "
    "Spending estimates are guidance, not guarantees."
)
