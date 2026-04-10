"""Subscribe endpoint — sign up for weekly Tidus AI pricing reports.

Routes:
  GET  /subscribe          → branded HTML sign-up form
  POST /api/v1/subscribe   → JSON API (email, name?) → adds to config/subscribers.yaml
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator

from tidus.reporting.subscribers import add_subscriber, load_subscribers

router = APIRouter(tags=["Reports"])

# ── HTML subscribe page ────────────────────────────────────────────────────────

_SUBSCRIBE_PAGE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Subscribe — Tidus AI Pricing Reports</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#f0f2f5;color:#1a1a2e;min-height:100vh;
     display:flex;align-items:center;justify-content:center;padding:20px;}
.card{background:white;border-radius:16px;padding:48px;max-width:480px;width:100%;
      box-shadow:0 4px 24px rgba(0,0,0,0.08);}
.logo{font-size:28px;font-weight:800;color:#0f3460;letter-spacing:-0.5px;margin-bottom:4px;}
.logo span{color:#a78bfa;}
.badge{display:inline-block;background:#f0f2f5;color:#533483;font-size:11px;
       font-weight:700;text-transform:uppercase;letter-spacing:1.5px;
       padding:3px 10px;border-radius:20px;margin-bottom:24px;}
h1{font-size:24px;font-weight:700;color:#1a1a2e;line-height:1.3;margin-bottom:10px;}
p{font-size:14px;color:#666;line-height:1.6;margin-bottom:24px;}
.features{list-style:none;margin-bottom:28px;}
.features li{font-size:13px;color:#555;padding:6px 0;
             display:flex;align-items:center;gap:8px;}
.features li::before{content:"✓";color:#533483;font-weight:700;}
label{display:block;font-size:13px;font-weight:600;color:#444;margin-bottom:6px;}
input{width:100%;padding:10px 14px;border:1.5px solid #ddd;border-radius:8px;
      font-size:14px;color:#1a1a2e;outline:none;transition:border-color 0.2s;}
input:focus{border-color:#533483;}
.field{margin-bottom:16px;}
button{width:100%;padding:13px;background:linear-gradient(135deg,#0f3460,#533483);
       color:white;border:none;border-radius:8px;font-size:15px;font-weight:700;
       cursor:pointer;margin-top:8px;transition:opacity 0.2s;}
button:hover{opacity:0.9;}
.note{font-size:12px;color:#aaa;text-align:center;margin-top:16px;line-height:1.5;}
.note a{color:#533483;text-decoration:none;}
.success{background:#f0fff4;border:1.5px solid #86efac;border-radius:8px;
         padding:16px;text-align:center;display:none;}
.success p{color:#166534;font-weight:600;margin:0;}
@media(max-width:520px){.card{padding:28px 20px;}}
</style>
</head>
<body>
<div class="card">
  <div class="logo">tidus<span>.</span>ai</div>
  <div class="badge">Weekly Report</div>
  <h1>Stay ahead of AI pricing shifts</h1>
  <p>Get a weekly digest of AI model price changes, new model launches,
     and market intelligence — delivered straight to your inbox.</p>
  <ul class="features">
    <li>Price changes across 40+ models from 10+ vendors</li>
    <li>New model discoveries with capability breakdowns</li>
    <li>Market narratives &amp; competitive analysis</li>
    <li>Free forever · Open source · No spam</li>
  </ul>
  <div id="success" class="success"><p>&#x2713; You're subscribed! Expect your first report next Sunday.</p></div>
  <form id="subForm">
    <div class="field">
      <label for="email">Email address</label>
      <input type="email" id="email" name="email" placeholder="you@company.com" required>
    </div>
    <div class="field">
      <label for="name">Name <span style="font-weight:400;color:#aaa">(optional)</span></label>
      <input type="text" id="name" name="name" placeholder="Your name">
    </div>
    <button type="submit">Subscribe &rarr;</button>
  </form>
  <p class="note">
    Self-hosted on your own Tidus instance. Powered by
    <a href="https://github.com/kensterinvest/tidus">Tidus v1.1.0</a>.<br>
    To unsubscribe, set <code>active: false</code> in <code>config/subscribers.yaml</code>.
  </p>
</div>
<script>
document.getElementById('subForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const btn = this.querySelector('button');
  btn.textContent = 'Subscribing…';
  btn.disabled = true;
  try {
    const res = await fetch('/api/v1/subscribe', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        email: document.getElementById('email').value,
        name:  document.getElementById('name').value || ''
      })
    });
    if (res.ok) {
      this.style.display = 'none';
      document.getElementById('success').style.display = 'block';
    } else {
      const d = await res.json();
      alert(d.detail || 'Subscription failed. Please try again.');
      btn.textContent = 'Subscribe →';
      btn.disabled = false;
    }
  } catch {
    alert('Network error. Please try again.');
    btn.textContent = 'Subscribe →';
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/subscribe", response_class=HTMLResponse, include_in_schema=False)
async def subscribe_page() -> HTMLResponse:
    """Branded HTML subscribe form."""
    return HTMLResponse(content=_SUBSCRIBE_PAGE)


class SubscribeRequest(BaseModel):
    email: str
    name: str = ""

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email address")
        return v


@router.post("/subscribe", summary="Subscribe to weekly pricing reports")
async def subscribe(req: SubscribeRequest) -> JSONResponse:
    """Add an email to the weekly pricing report subscriber list.

    Idempotent: re-subscribing a previously unsubscribed address reactivates it.
    """
    try:
        add_subscriber(req.email, req.name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        status_code=200,
        content={
            "status": "subscribed",
            "email": req.email,
            "message": "You'll receive the next weekly AI pricing report.",
        },
    )


@router.get("/subscribers", summary="List active subscribers (admin)")
async def list_subscribers() -> dict:
    """Return count and email list of active subscribers."""
    subs = load_subscribers()
    return {
        "count": len(subs),
        "subscribers": [{"email": s.email, "name": s.name} for s in subs],
    }
