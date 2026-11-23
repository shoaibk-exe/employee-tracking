
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* --------------------------------------------------
   GLOBAL
-------------------------------------------------- */

html, body, .stApp, p, label, h1, h2, h3, h4, li, td, th, button, input, textarea {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* light-dark() follows the color-scheme Streamlit sets on .stApp. */
.stApp {
    --emp-input: light-dark(#F0F2F6, #2A255C);
    --emp-surface: light-dark(#ffffff, #1C1844);
    --emp-border: light-dark(rgba(15, 23, 42, .08), rgba(199, 201, 240, .16));
    --emp-ring: light-dark(rgba(15, 23, 42, .06), rgba(199, 201, 240, .10));
    --emp-lift: light-dark(rgba(15, 23, 42, .22), rgba(0, 0, 0, .6));
    --emp-button-shadow: 0 4px 12px light-dark(rgba(79, 70, 229, .25), rgba(79, 70, 229, .35));
}

.block-container {
    padding-top: 2.2rem;
    max-width: 1280px;
}

/* --------------------------------------------------
   HEADINGS
-------------------------------------------------- */

h1 {
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

/* --------------------------------------------------
   BORDERED CONTAINERS ONLY
   Do not style every nested vertical block.
-------------------------------------------------- */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    border: 1px solid var(--emp-border) !important;
}

/* --------------------------------------------------
   FORMS
-------------------------------------------------- */

[data-testid="stForm"] {
    border: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    min-width: 0 !important;
}

[data-testid="stForm"] [data-testid="stVerticalBlock"] {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    gap: 0.75rem !important;
}

/* --------------------------------------------------
   INPUT FIELDS
-------------------------------------------------- */

[data-testid="InputInstructions"] {
    display: none !important;
}
.stTextInput,
.stTextInput > div,
[data-testid="stTextInputRootElement"] {
    width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}

[data-testid="stTextInputRootElement"],
[data-testid="stNumberInputContainer"],
[data-testid="stTextAreaRootElement"],
[data-testid="stSelectbox"] > div:last-child > div {
    background-color: var(--emp-input) !important;
    border-radius: 10px !important;
    box-sizing: border-box !important;
}

[data-testid="stTextInputRootElement"] input {
    width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

.stTextInput label {
    margin-bottom: 0.4rem !important;
    font-weight: 600 !important;
}

[data-testid="stForm"] [data-testid="stWidgetLabel"] {
    margin-bottom: 0.35rem !important;
}

/* --------------------------------------------------
   BUTTONS
   Streamlit secondary buttons were white-on-white in
   light theme. Force solid indigo + white labels.
-------------------------------------------------- */

[data-testid="stButton"] button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondaryFormSubmit"],
[data-testid="stBaseButton-primaryFormSubmit"],
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all .2s ease;
    background: #312E81 !important;
    background-color: #312E81 !important;
    background-image: none !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #4338CA !important;
    opacity: 1 !important;
}

[data-testid="stButton"] button *,
[data-testid="stBaseButton-secondary"] *,
[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-secondaryFormSubmit"] *,
[data-testid="stBaseButton-primaryFormSubmit"] *,
.stButton > button *,
.stFormSubmitButton > button *,
[data-testid="stButton"] button p,
.stButton > button p,
.stFormSubmitButton > button p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}

[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"],
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    background-color: #4F46E5 !important;
    border: 0 !important;
    box-shadow: var(--emp-button-shadow);
}

[data-testid="stButton"] button:hover,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.1);
}

[data-testid="stButton"] button:disabled,
[data-testid="stBaseButton-secondary"]:disabled,
.stButton > button:disabled,
.stFormSubmitButton > button:disabled {
    opacity: 0.45 !important;
}

/* --------------------------------------------------
   METRICS
-------------------------------------------------- */

[data-testid="stMetric"] {
    background: var(--emp-surface);
    border-radius: 20px;
    padding: 18px 22px;
    box-shadow: 0 0 0 1px var(--emp-ring), 0 10px 30px -14px var(--emp-lift);
}

[data-testid="stMetricValue"] {
    font-weight: 800;
}

/* --------------------------------------------------
   IMAGES
-------------------------------------------------- */

[data-testid="stImage"] img {
    border-radius: 14px;
}

/* --------------------------------------------------
   CAMERA FORM SPACING
-------------------------------------------------- */

[data-testid="stForm"] .stTextInput {
    margin-bottom: 0.35rem !important;
}

[data-testid="stForm"] .stCheckbox {
    margin-top: 0.35rem !important;
}

[data-testid="stForm"] .stFormSubmitButton {
    margin-top: 0.5rem !important;
}

/* --------------------------------------------------
   RESPONSIVE LAYOUT
-------------------------------------------------- */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    [data-testid="stTextInputRootElement"] {
        width: 100% !important;
    }

    .st-key-emp_theme {
        right: 3.4rem;
        width: 132px;
    }

}

/* Header switch. Fixed so it stays beside the menu on every page. */
.st-key-emp_theme {
    position: fixed;
    top: 0.72rem;
    right: 8.4rem;
    z-index: 1000002;
    width: 168px;
    height: 36px;
    margin: 0;
}

</style>
"""


def apply():

    st.markdown(
        CSS,
        unsafe_allow_html=True
    )

    logo = Image.new(
        "RGBA",
        (260, 56),
        (0, 0, 0, 0)
    )

    d = ImageDraw.Draw(logo)

    d.rounded_rectangle(
        [4, 8, 44, 48],
        12,
        fill="#8B5CF6"
    )

    d.ellipse(
        [15, 19, 33, 37],
        outline="#0E0B2B",
        width=3
    )

    d.text(
        (58, 14),
        "EMPTracker",
        fill="white",
        font=ImageFont.load_default(size=26)
    )

    st.logo(
        logo,
        size="large"
    )


_THEME_HTML = """
<div class="switch" role="group" aria-label="Appearance">
    <button type="button" id="light">Light</button>
    <button type="button" id="dark">Dark</button>
</div>
"""

_THEME_CSS = """
.switch {
    display: flex;
    height: 36px;
    padding: 3px;
    box-sizing: border-box;
    border-radius: 10px;
    background: rgba(14, 11, 43, 0.94);
    border: 1px solid rgba(199, 201, 240, 0.32);
}
button {
    flex: 1;
    border: 0;
    border-radius: 8px;
    background: transparent;
    color: #C7C9F0;
    font: 600 13px 'Plus Jakarta Sans', sans-serif;
    cursor: pointer;
}
button.on {
    background: #4F46E5;
    color: #fff;
}
"""

_THEME_JS = """
export default function (component) {
    const root = component.parentElement;
    const host = root && (root.host || root);
    if (!root || !host || host.dataset.bound) {
        return;
    }
    host.dataset.bound = "1";

    const light = root.querySelector("#light");
    const dark = root.querySelector("#dark");

    function keyFor(path) {
        return "stActiveTheme-" + path + "-v2";
    }

    function storedMode() {
        const raw = localStorage.getItem(keyFor(location.pathname));
        if (!raw) {
            return null;
        }
        try {
            const mode = JSON.parse(raw);
            return mode === "Light" || mode === "Dark" ? mode : null;
        } catch (error) {
            return null;
        }
    }

    function activeMode() {
        const saved = storedMode();
        if (saved) {
            return saved;
        }
        const app = document.querySelector(".stApp");
        const scheme = app ? getComputedStyle(app).colorScheme : "light";
        return scheme === "dark" ? "Dark" : "Light";
    }

    function paint() {
        const mode = activeMode();
        light.classList.toggle("on", mode === "Light");
        dark.classList.toggle("on", mode === "Dark");
    }

    function paths() {
        const found = new Set([location.pathname]);
        document.querySelectorAll("a[href]").forEach((anchor) => {
            const url = new URL(anchor.href, location.origin);
            if (url.origin === location.origin && !url.pathname.startsWith("/_stcore")) {
                found.add(url.pathname);
            }
        });
        return found;
    }

    function apply(mode) {
        if (activeMode() === mode && storedMode() === mode) {
            return;
        }
        paths().forEach((path) => {
            localStorage.setItem(keyFor(path), JSON.stringify(mode));
        });
        location.reload();
    }

    light.onclick = () => apply("Light");
    dark.onclick = () => apply("Dark");
    paint();
}
"""

_theme_switch = st.components.v2.component(
    "theme_switch",
    html=_THEME_HTML,
    css=_THEME_CSS,
    js=_THEME_JS,
)


def theme_switch():
    """Light / Dark control pinned to the header on every page."""
    _theme_switch(key="emp_theme", width=168, height=36)
