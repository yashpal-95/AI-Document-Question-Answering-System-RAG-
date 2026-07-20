# ============================================================
#  app.py  –  Streamlit frontend with User Management
# ============================================================

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0f14;
    color: #e8e8e8;
}
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1f2330;
}
h1 { font-size: 2rem !important; font-weight: 800 !important; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; color: #a78bfa; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #94a3b8; }

/* Auth card */
.auth-card {
    background: #1a1d27;
    border: 1px solid #252836;
    border-radius: 16px;
    padding: 36px 40px;
    max-width: 460px;
    margin: 60px auto;
    text-align: center;
}
.auth-title { font-size: 1.8rem; font-weight: 800; color: #e8e8e8; margin-bottom: 6px; }
.auth-sub   { font-size: 0.9rem; color: #64748b; margin-bottom: 28px; }

/* User card in select list */
.user-card {
    background: #1a1d27;
    border: 1px solid #252836;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.user-name  { font-weight: 700; font-size: 1rem; color: #e8e8e8; }
.user-id    { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #a78bfa; }
.user-login { font-size: 0.78rem; color: #475569; }

/* Session badge */
.session-badge {
    background: #1e2130;
    border: 1px solid #2d3248;
    border-radius: 999px;
    padding: 4px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #c4b5fd;
    display: inline-block;
    margin-bottom: 6px;
}

.metric-card {
    background: #1a1d27; border: 1px solid #252836;
    border-radius: 12px; padding: 18px 22px;
    text-align: center; margin-bottom: 8px;
}
.metric-label { font-size: 0.75rem; color: #64748b; letter-spacing: 1px; text-transform: uppercase; }
.metric-value { font-size: 2rem; font-weight: 800; font-family: 'DM Mono', monospace; margin: 6px 0 2px; }
.metric-badge { font-size: 0.85rem; }
.score-good     { color: #4ade80; }
.score-warn     { color: #facc15; }
.score-moderate { color: #fb923c; }
.score-bad      { color: #f87171; }

.answer-box {
    background: #1a1d27; border-left: 3px solid #a78bfa;
    border-radius: 0 12px 12px 0; padding: 18px 22px;
    font-family: 'DM Mono', monospace; font-size: 0.92rem;
    line-height: 1.7; color: #e2e8f0; margin: 12px 0;
}
.summary-box {
    background: #13161e; border: 1px solid #252836;
    border-radius: 10px; padding: 14px 18px;
    font-size: 0.88rem; color: #94a3b8; margin-top: 8px;
}
.warn-box {
    background: #1f1510; border: 1px solid #78350f;
    border-radius: 10px; padding: 14px 18px;
    font-size: 0.88rem; color: #fbbf24; margin: 10px 0;
}
.info-box {
    background: #0f1a2e; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 14px 18px;
    font-size: 0.88rem; color: #93c5fd; margin: 10px 0;
}
.file-pill {
    display: inline-block; background: #1e2130;
    border: 1px solid #2d3248; border-radius: 999px;
    padding: 4px 14px; font-size: 0.82rem; color: #c4b5fd;
    font-family: 'DM Mono', monospace; margin: 3px 4px 3px 0;
}
.chat-item {
    background: #1a1d27; border: 1px solid #252836;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
}
.chat-q  { color: #a78bfa; font-weight: 600; font-size: 0.88rem; margin-bottom: 6px; }
.chat-a  { color: #e2e8f0; font-family: 'DM Mono', monospace; font-size: 0.85rem; line-height: 1.6; }
.chat-ts { color: #475569; font-size: 0.75rem; margin-top: 6px; }

.stButton > button {
    background: #a78bfa !important; color: #0d0f14 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-family: 'Syne', sans-serif !important;
}
.stButton > button:hover { background: #c4b5fd !important; }
.stTextInput > div > div > input, .stTextArea textarea {
    background: #1a1d27 !important; border: 1px solid #252836 !important;
    color: #e8e8e8 !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stFileUploader"] {
    background: #1a1d27; border: 1px dashed #2d3248; border-radius: 12px; padding: 8px;
}
hr { border-color: #1f2330 !important; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ──────────────────────────────────────────────

def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API. Is the server running?"
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

def api_post(path, **kwargs):
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=600, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API. Is the server running?"
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

def api_delete(path, **kwargs):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API. Is the server running?"
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

def score_class(value, invert=False):
    v = (100 - value) if invert else value
    if v >= 80: return "score-good"
    elif v >= 60: return "score-warn"
    elif v >= 35: return "score-moderate"
    return "score-bad"

def metric_card(label, value, badge, css_class):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value:.0f}</div>
        <div class="metric-badge">{badge}</div>
    </div>""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────

if "logged_in"    not in st.session_state: st.session_state.logged_in    = False
if "user_id"      not in st.session_state: st.session_state.user_id      = ""
if "display_name" not in st.session_state: st.session_state.display_name = ""
if "auth_tab"     not in st.session_state: st.session_state.auth_tab     = "login"


# ════════════════════════════════════════════════════════════
#  AUTH GATE  –  shown when not logged in
# ════════════════════════════════════════════════════════════

if not st.session_state.logged_in:

    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-title'>🧠 RAG Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-sub'>Your personal document intelligence system</div>", unsafe_allow_html=True)

    tab_login, tab_register, tab_select = st.tabs(["🔑 Login", "✨ New User", "👥 Select User"])

    # ── Tab 1: Login with existing user_id ───────────────────
    with tab_login:
        st.markdown("")
        login_id = st.text_input("Enter your User ID", placeholder="e.g. user_01", key="login_id_input")

        if st.button("🔑 Login", use_container_width=True, key="login_btn"):
            if not login_id.strip():
                st.warning("Please enter your User ID.")
            else:
                data, err = api_post("/users/login", json={"user_id": login_id.strip()})
                if err:
                    # Show friendly message if user not found
                    if "does not exist" in str(err):
                        st.error(f"❌ User **'{login_id.strip()}'** does not exist.")
                        st.markdown(
                            '<div class="info-box">💡 Switch to the <strong>New User</strong> tab to register, or <strong>Select User</strong> to pick from existing users.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.error(err)
                else:
                    user = data.get("user", {})
                    st.session_state.logged_in    = True
                    st.session_state.user_id      = user.get("user_id", login_id.strip())
                    st.session_state.display_name = user.get("display_name", login_id.strip())
                    st.rerun()

    # ── Tab 2: Register new user ──────────────────────────────
    with tab_register:
        st.markdown("")
        new_name = st.text_input("Your name", placeholder="e.g. Alice", key="new_name_input")

        if st.button("✨ Create Account", use_container_width=True, key="register_btn"):
            if not new_name.strip():
                st.warning("Please enter your name.")
            else:
                data, err = api_post("/users/create", json={"display_name": new_name.strip()})
                if err:
                    st.error(err)
                else:
                    user = data.get("user", {})
                    st.success(f"✅ Account created! Your User ID is **`{user['user_id']}`** — save this to login next time.")
                    st.markdown(
                        f'<div class="info-box">🪪 Your unique ID: <strong>{user["user_id"]}</strong><br>Use this to login in the future.</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.logged_in    = True
                    st.session_state.user_id      = user["user_id"]
                    st.session_state.display_name = user["display_name"]
                    st.rerun()

    # ── Tab 3: Select from existing users ─────────────────────
    with tab_select:
        st.markdown("")
        users_data, err = api_get("/users")

        if err:
            st.error(err)
        elif not users_data or not users_data.get("users"):
            st.info("No users registered yet. Create a new account first.")
        else:
            users = users_data["users"]
            st.markdown(f"<span style='color:#64748b;font-size:0.85rem;'>{len(users)} registered user(s)</span>", unsafe_allow_html=True)
            st.markdown("")

            for u in users:
                uid   = u.get("user_id", "")
                name  = u.get("display_name", uid)
                login = str(u.get("last_login", ""))[:10]

                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(f"""
                    <div class="user-card">
                        <div>
                            <div class="user-name">👤 {name}</div>
                            <div class="user-id">{uid}</div>
                            <div class="user-login">Last login: {login}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col_btn:
                    if st.button("Login", key=f"sel_{uid}"):
                        data, err = api_post("/users/login", json={"user_id": uid})
                        if err:
                            st.error(err)
                        else:
                            st.session_state.logged_in    = True
                            st.session_state.user_id      = uid
                            st.session_state.display_name = name
                            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()   # ← nothing below renders until logged in


# ════════════════════════════════════════════════════════════
#  MAIN APP  –  shown after login
# ════════════════════════════════════════════════════════════

user_id      = st.session_state.user_id
display_name = st.session_state.display_name

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG Assistant")
    st.markdown("---")

    st.markdown(f'<div class="session-badge">👤 {display_name}</div>', unsafe_allow_html=True)
    st.markdown(f"<span style='color:#475569;font-size:0.78rem;'>ID: <code>{user_id}</code></span>", unsafe_allow_html=True)

    st.markdown("")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in    = False
        st.session_state.user_id      = ""
        st.session_state.display_name = ""
        st.rerun()

    st.markdown("---")
    page = st.radio("Navigation", [
        "💬 Ask",
        "📂 My Documents",
        "🕘 Chat History",
        "⚙️ Settings",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#475569;'>RAG System v2.0<br>Ollama + LangGraph + MongoDB</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: Ask
# ════════════════════════════════════════════════════════════

if page == "💬 Ask":
    st.markdown(f"# 💬 Ask Your Documents")
    st.markdown(f"<span style='color:#64748b;'>Logged in as <strong>{display_name}</strong> · <code>{user_id}</code></span>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Check if user has any PDFs uploaded ──────────────────
    pdf_data, pdf_err = api_get(f"/pdfs/{user_id}")
    user_pdfs = pdf_data.get("files", []) if pdf_data else []

    if pdf_err:
        st.error(pdf_err)
        st.stop()

    if not user_pdfs:
        # Block the page entirely — no PDFs uploaded yet
        st.markdown("""
        <div style='text-align:center; padding: 60px 20px;'>
            <div style='font-size:4rem;'>📂</div>
            <div style='font-size:1.4rem; font-weight:800; color:#e8e8e8; margin:16px 0 8px;'>No Documents Uploaded</div>
            <div style='font-size:0.95rem; color:#64748b; max-width:380px; margin:0 auto 24px;'>
                You haven't uploaded any PDFs yet.<br>
                Go to <strong>My Documents</strong> to upload a PDF first,
                then come back here to ask questions.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_c, col_btn_c, col_c2 = st.columns([2, 1, 2])
        with col_btn_c:
            if st.button("📂 Go to My Documents", use_container_width=True):
                st.session_state["_nav"] = "📂 My Documents"
                st.rerun()
        st.stop()

    # ── PDF selector ─────────────────────────────────────────
    st.markdown("### 📄 Select PDF to Query")

    pdf_options = ["🗂️ All Documents"] + user_pdfs
    selected_pdf = st.selectbox(
        "Choose which PDF to query",
        options=pdf_options,
        label_visibility="collapsed",
        key="pdf_selector",
    )

    if selected_pdf == "🗂️ All Documents":
        st.markdown(
            '<div class="summary-box">🗂️ Querying across <strong>all your uploaded documents</strong>.</div>',
            unsafe_allow_html=True,
        )
        selected_filename = None
    else:
        st.markdown(
            f'<div class="summary-box">📄 Querying only: <strong>{selected_pdf}</strong></div>',
            unsafe_allow_html=True,
        )
        selected_filename = selected_pdf

    st.markdown("")

    # ── Question input ────────────────────────────────────────
    question = st.text_area(
        "Your question",
        placeholder="What skills are mentioned in the document?",
        height=110,
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        ask_clicked = st.button("🔍 Ask", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking... (may take up to a minute)"):
                payload = {
                    "user_id":  user_id,
                    "question": question,
                }
                # Pass selected filename if user picked a specific PDF
                if selected_filename:
                    payload["selected_pdf"] = selected_filename

                data, err = api_post("/ask", json=payload)

            if err:
                st.error(err)
            else:
                scores = data.get("scores", {})
                acc    = scores.get("total_accuracy_score", 0)
                hall   = scores.get("total_hallucination_score", 100)

                # Show which PDF was queried
                queried = data.get("queried_pdf", "All Documents")
                st.markdown(f"<span style='color:#475569;font-size:0.82rem;'>📄 Queried: <code>{queried}</code></span>", unsafe_allow_html=True)

                st.markdown("### 💡 Answer")
                st.markdown(f'<div class="answer-box">{data["answer"]}</div>', unsafe_allow_html=True)

                st.markdown("### 📊 Evaluation Scores")
                c1, c2 = st.columns(2)
                with c1:
                    metric_card("ACCURACY", acc, scores.get("accuracy_label", ""), score_class(acc))
                with c2:
                    metric_card("HALLUCINATION", hall, scores.get("hallucination_label", ""), score_class(hall, invert=True))

                summary = scores.get("evaluation_summary", "")
                if summary:
                    st.markdown(f'<div class="summary-box">🔍 <strong>Eval summary:</strong> {summary}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: My Documents
# ════════════════════════════════════════════════════════════

elif page == "📂 My Documents":
    st.markdown("# 📂 My Documents")
    st.markdown(f"<span style='color:#64748b;'><code>{user_id}</code></span>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⬆️ Upload a PDF")
    st.markdown(
        '<div class="summary-box" style="margin-bottom:12px;">📎 Only <strong>.pdf</strong> files are accepted. JPG, PNG and other formats are not supported.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],                    # strictly PDF only
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded:
        # Double-check mime type even if browser bypassed the type filter
        if uploaded.type != "application/pdf" or not uploaded.name.lower().endswith(".pdf"):
            st.error("❌ Invalid file type. Please upload a PDF file only (.pdf). Images (JPG, PNG) and other formats are not accepted.")
        else:
            st.markdown(f'<span class="file-pill">📄 {uploaded.name} · {round(uploaded.size/1024, 1)} KB</span>', unsafe_allow_html=True)
            if st.button("📤 Index this PDF"):
                with st.spinner(f"Indexing '{uploaded.name}'..."):
                    data, err = api_post(
                        f"/upload?user_id={user_id}",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    )
                if err: st.error(err)
                else:
                    st.success(data.get("message", "Uploaded!"))
                    st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Uploaded Files")
    data, err = api_get(f"/pdfs/{user_id}")

    if err:
        st.error(err)
    elif not data or not data.get("files"):
        st.info("No PDFs uploaded yet.")
    else:
        files = data["files"]
        st.markdown(f"<span style='color:#64748b;font-size:0.85rem;'>{len(files)} file(s)</span>", unsafe_allow_html=True)

        for fname in files:
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(f'<span class="file-pill">📄 {fname}</span>', unsafe_allow_html=True)
            with col_del:
                if st.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                    st.session_state[f"confirm_{fname}"] = True

            if st.session_state.get(f"confirm_{fname}"):
                st.markdown(f'<div class="warn-box">⚠️ Delete <strong>{fname}</strong> from disk, ChromaDB and MongoDB?</div>', unsafe_allow_html=True)
                c1, c2, _ = st.columns([1, 1, 4])
                with c1:
                    if st.button("✅ Yes", key=f"yes_{fname}"):
                        result, err = api_delete("/pdfs/delete_one", json={"user_id": user_id, "filename": fname})
                        st.session_state.pop(f"confirm_{fname}", None)
                        if err: st.error(err)
                        else:
                            st.success(result.get("status", "Deleted."))
                            st.rerun()
                with c2:
                    if st.button("❌ No", key=f"cancel_{fname}"):
                        st.session_state.pop(f"confirm_{fname}", None)
                        st.rerun()


# ════════════════════════════════════════════════════════════
#  PAGE: Chat History
# ════════════════════════════════════════════════════════════

elif page == "🕘 Chat History":
    st.markdown("# 🕘 Chat History")
    st.markdown(f"<span style='color:#64748b;'><code>{user_id}</code></span>", unsafe_allow_html=True)
    st.markdown("---")

    data, err = api_get(f"/chat_history/{user_id}?limit=50")

    if err:
        st.error(err)
    elif not data or not data.get("history"):
        st.info("No chat history yet. Ask a question first!")
    else:
        history = data["history"]
        col_count, col_btn = st.columns([3, 2])
        with col_count:
            st.markdown(f"<span style='color:#64748b;font-size:0.85rem;'>{len(history)} conversation(s)</span>", unsafe_allow_html=True)
        with col_btn:
            if st.button("🗑️ Delete All Chat History"):
                st.session_state["confirm_chat_delete"] = True

        if st.session_state.get("confirm_chat_delete"):
            st.markdown('<div class="warn-box">⚠️ Delete ALL chat history? PDFs are NOT affected.</div>', unsafe_allow_html=True)
            c1, c2, _ = st.columns([1, 1, 4])
            with c1:
                if st.button("✅ Yes, delete"):
                    result, err = api_delete("/chat_history/delete", json={"user_id": user_id})
                    st.session_state.pop("confirm_chat_delete", None)
                    if err: st.error(err)
                    else:
                        st.success(result.get("status", "Done"))
                        st.rerun()
            with c2:
                if st.button("❌ Cancel", key="cancel_chat"):
                    st.session_state.pop("confirm_chat_delete", None)
                    st.rerun()

        st.markdown("---")
        for item in history:
            ts = str(item.get("timestamp", ""))[:19].replace("T", " ")
            st.markdown(f"""
            <div class="chat-item">
                <div class="chat-q">❓ {item.get('question', '')}</div>
                <div class="chat-a">{item.get('answer', '')}</div>
                <div class="chat-ts">🕐 {ts}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: Settings
# ════════════════════════════════════════════════════════════

elif page == "⚙️ Settings":
    st.markdown("# ⚙️ Settings")
    st.markdown(f"<span style='color:#64748b;'>Account: <strong>{display_name}</strong> · <code>{user_id}</code></span>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Delete chat history only ─────────────────────────────
    st.markdown("### 🗑️ Delete Chat History Only")
    st.markdown('<div class="summary-box">Removes all Q&A logs from MongoDB. PDFs and indexed data remain intact.</div>', unsafe_allow_html=True)
    st.markdown("")

    if st.button("🗑️ Delete My Chat History"):
        st.session_state["settings_confirm_chat"] = True

    if st.session_state.get("settings_confirm_chat"):
        st.markdown('<div class="warn-box">⚠️ Are you sure? This cannot be undone.</div>', unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("✅ Yes", key="yes_chat_set"):
                result, err = api_delete("/chat_history/delete", json={"user_id": user_id})
                st.session_state.pop("settings_confirm_chat", None)
                if err: st.error(err)
                else: st.success(result.get("status", "Done"))
        with c2:
            if st.button("❌ Cancel", key="cancel_chat_set"):
                st.session_state.pop("settings_confirm_chat", None)
                st.rerun()

    st.markdown("---")

    # ── Delete all user data ─────────────────────────────────
    st.markdown("### 🚨 Delete My Account & All Data")
    st.markdown('<div class="warn-box">⚠️ <strong>This cannot be undone.</strong> Deletes your account, all PDFs (disk + ChromaDB), all chat history and all metadata from MongoDB.</div>', unsafe_allow_html=True)
    st.markdown("")

    confirm = st.checkbox("I understand this permanently deletes my account and all data")
    if confirm:
        if st.button("🚨 Delete My Account"):
            st.session_state["settings_confirm_all"] = True

    if st.session_state.get("settings_confirm_all"):
        st.markdown(f'<div class="warn-box">⚠️ Final confirmation: permanently delete everything for <strong>{display_name}</strong> ({user_id})?</div>', unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("✅ Yes, delete all", key="yes_all_set"):
                result, err = api_delete("/delete_user_data", json={"user_id": user_id})
                st.session_state.pop("settings_confirm_all", None)
                if err:
                    st.error(err)
                else:
                    st.success("Account and all data deleted.")
                    # Auto logout after account deletion
                    st.session_state.logged_in    = False
                    st.session_state.user_id      = ""
                    st.session_state.display_name = ""
                    st.rerun()
        with c2:
            if st.button("❌ Cancel", key="cancel_all_set"):
                st.session_state.pop("settings_confirm_all", None)
                st.rerun()

    st.markdown("---")

    # ── ChromaDB Debug View ──────────────────────────────────
    st.markdown("### 🔬 ChromaDB Debug")
    st.markdown('<div class="summary-box">Shows what embeddings and chunks are currently stored in ChromaDB for your account.</div>', unsafe_allow_html=True)
    st.markdown("")

    col_db, col_nuke = st.columns([1, 1])
    with col_db:
        if st.button("🔍 Inspect My ChromaDB Chunks"):
            data, err = api_get(f"/debug/chromadb/{user_id}")
            if err:
                st.error(err)
            else:
                st.markdown(f"**Total chunks stored:** `{data['total_chunks']}`")
                if data["files"]:
                    for fname, count in data["files"].items():
                        st.markdown(f'<span class="file-pill">📄 {fname} · {count} chunks</span>', unsafe_allow_html=True)
                else:
                    st.info("No chunks found in ChromaDB for your account.")

    with col_nuke:
        if st.button("🧹 Nuke Orphan Chunks"):
            result, err = api_delete("/debug/chromadb/nuke_orphans")
            if err: st.error(err)
            else: st.success(result.get("status", "Done"))

    st.markdown("---")

    # ── API Status ───────────────────────────────────────────
    st.markdown("### 🔌 API Connection")
    _, err = api_get(f"/pdfs/{user_id}")
    if err:
        st.error(f"API unreachable: {err}")
    else:
        st.success(f"✅ Connected to `{API_BASE}`") 
