"""Dependency-free web UI for the KernelAgent task service."""

from __future__ import annotations

import json


def render_task_ui() -> str:
    """Return the self-contained task dashboard HTML."""
    return _TASK_UI


def render_admin_console() -> str:
    """Return the frontend-only machine administration console."""
    return _ADMIN_CONSOLE


def render_auth_page(next_path: str = "/v1/ui") -> str:
    """Return the login and signup page."""
    safe_next = json.dumps(
        next_path if next_path.startswith("/") and not next_path.startswith("//") else "/v1/ui"
    )
    return _AUTH_PAGE.replace("__NEXT_PATH__", safe_next)


def render_console_access_denied() -> str:
    """Return a friendly page for authenticated non-admin users."""
    return _CONSOLE_ACCESS_DENIED


_CONSOLE_ACCESS_DENIED = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="light"><title>Console Access · Anvil</title>
  <style>
    :root{--bg:#edf2f7;--panel:#fff;--line:#d6e0e9;--text:#17212b;--muted:#667585;--accent:#f06431;--shadow:0 22px 65px rgba(45,64,82,.14)}
    *{box-sizing:border-box}body{margin:0;min-width:320px;min-height:100vh;display:grid;place-items:center;padding:24px;color:var(--text);background:radial-gradient(circle at 10% 0,rgba(240,100,49,.16),transparent 32rem),radial-gradient(circle at 100% 100%,rgba(36,118,200,.11),transparent 30rem),var(--bg);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
    main{width:min(520px,100%);padding:48px;text-align:center;border:1px solid var(--line);border-radius:20px;background:var(--panel);box-shadow:var(--shadow)}.mark{width:48px;height:48px;margin:0 auto 24px;display:grid;place-items:center;border:1px solid rgba(240,100,49,.3);border-radius:14px;color:var(--accent);background:rgba(240,100,49,.08);font:700 18px ui-monospace,monospace}h1{margin:0;font-size:27px;letter-spacing:-.04em}p{margin:12px auto 28px;color:var(--muted)}a{display:inline-block;padding:10px 16px;border-radius:9px;color:#fff;background:linear-gradient(135deg,#ef7b45,var(--accent));text-decoration:none;font-weight:750}@media(max-width:560px){main{padding:38px 26px}}
  </style>
</head>
<body><main><div class="mark">A</div><h1>Console Access Is Restricted</h1><p>Only admin users can access the console.</p><a href="/v1/ui">Return to the task UI</a></main></body>
</html>"""


_AUTH_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="light"><title>Sign In · Anvil</title>
  <style>
    :root{--bg:#edf2f7;--panel:#fff;--line:#d6e0e9;--text:#17212b;--muted:#667585;--accent:#f06431;--red:#c93649;--shadow:0 22px 65px rgba(45,64,82,.16)}
    *{box-sizing:border-box}body{margin:0;min-width:320px;min-height:100vh;display:grid;place-items:center;padding:24px;color:var(--text);background:radial-gradient(circle at 10% 0,rgba(240,100,49,.17),transparent 32rem),radial-gradient(circle at 100% 100%,rgba(36,118,200,.12),transparent 30rem),var(--bg);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
    .shell{width:min(920px,100%);display:grid;grid-template-columns:1fr 1fr;overflow:hidden;border:1px solid var(--line);border-radius:20px;background:var(--panel);box-shadow:var(--shadow)}
    .intro{padding:56px 48px;color:#fff;background:linear-gradient(145deg,#17212b,#263a4d);display:flex;flex-direction:column;justify-content:space-between}.mark{width:44px;height:44px;display:grid;place-items:center;border:1px solid rgba(255,255,255,.25);border-radius:13px;color:#ff956d;background:rgba(255,255,255,.07);font:700 17px ui-monospace,monospace}.intro h1{margin:44px 0 12px;font-size:38px;line-height:1.08;letter-spacing:-.05em}.intro p{margin:0;color:#b8c7d4}.note{margin-top:70px;padding-top:18px;border-top:1px solid rgba(255,255,255,.13);font-size:12px}
    .auth{padding:46px 44px}.tabs{display:flex;gap:5px;margin-bottom:30px;padding:4px;border-radius:10px;background:#eef3f7}.tab{flex:1;padding:9px;border:0;border-radius:7px;color:var(--muted);background:transparent;cursor:pointer;font:700 13px inherit}.tab.active{color:var(--text);background:#fff;box-shadow:0 2px 8px rgba(45,64,82,.1)}h2{margin:0;font-size:25px;letter-spacing:-.035em}.sub{margin:7px 0 25px;color:var(--muted)}label{display:block;margin:14px 0 7px;font-size:12px;font-weight:750}input{width:100%;height:44px;padding:0 12px;border:1px solid #c5d1dc;border-radius:9px;outline:none;font:inherit}input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(240,100,49,.1)}button[type=submit]{width:100%;height:44px;margin-top:22px;border:0;border-radius:9px;color:#fff;background:linear-gradient(135deg,#ef7b45,var(--accent));cursor:pointer;font-weight:750}.error{min-height:22px;margin-top:12px;color:var(--red);font-size:12px}.signup-copy{display:none}.auth.signup .login-copy{display:none}.auth.signup .signup-copy{display:block}@media(max-width:700px){.shell{grid-template-columns:1fr}.intro{padding:30px}.intro h1{margin-top:25px;font-size:30px}.note{display:none}.auth{padding:32px 28px}}
  </style>
</head>
<body><main class="shell"><section class="intro"><div><div class="mark">A∿</div><h1>Build Faster Kernels.</h1><p>Sign in to your anvil workspace to create and monitor GPU optimization tasks.</p></div><p class="note">Self-service signup creates a general account. Administrators use the credentials supplied by the service owner.</p></section>
  <section id="auth" class="auth"><div class="tabs"><button class="tab active" data-mode="login" type="button">Sign in</button><button class="tab" data-mode="signup" type="button">Create account</button></div><div class="login-copy"><h2>Welcome Back</h2><p class="sub">Enter your account details to continue.</p></div><div class="signup-copy"><h2>Create Your Account</h2><p class="sub">New accounts can access the task UI.</p></div>
  <form id="form"><label for="username">Username</label><input id="username" autocomplete="username" minlength="3" maxlength="32" required><label for="password">Password</label><input id="password" type="password" autocomplete="current-password" minlength="8" maxlength="256" required><button id="submit" type="submit">Sign in</button><div id="error" class="error" role="alert"></div></form></section></main>
  <script>'use strict';let mode='login';const auth=document.getElementById('auth'),password=document.getElementById('password'),error=document.getElementById('error'),submit=document.getElementById('submit');document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{mode=tab.dataset.mode;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===tab));auth.classList.toggle('signup',mode==='signup');password.autocomplete=mode==='signup'?'new-password':'current-password';submit.textContent=mode==='signup'?'Create account':'Sign in';error.textContent='';}));document.getElementById('form').addEventListener('submit',async event=>{event.preventDefault();error.textContent='';submit.disabled=true;try{const response=await fetch('/v1/auth/'+mode,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:document.getElementById('username').value,password:password.value})});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Unable to continue.');window.location.assign(__NEXT_PATH__);}catch(e){error.textContent=e.message;}finally{submit.disabled=false;}});</script>
</body></html>"""


_ADMIN_CONSOLE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Machine Admin · Anvil</title>
  <style>
    :root { --bg:#edf2f7; --panel:#fff; --soft:#f7f9fb; --line:#dbe3eb; --text:#17212b; --muted:#667585; --subtle:#8d99a6; --accent:#f06431; --green:#168b5b; --red:#c93649; --shadow:0 18px 48px rgba(45,64,82,.1); }
    * { box-sizing:border-box; }
    body { margin:0; min-width:320px; min-height:100vh; color:var(--text); background:radial-gradient(circle at 8% -10%,rgba(240,100,49,.14),transparent 30rem),radial-gradient(circle at 100% 10%,rgba(36,118,200,.1),transparent 28rem),var(--bg); font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    button,input { font:inherit; }
    .topbar { height:68px; padding:0 30px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); background:rgba(255,255,255,.86); backdrop-filter:blur(18px); }
    .brand,.nav { display:flex; align-items:center; gap:12px; }
    .mark { width:34px; height:34px; display:grid; place-items:center; border:1px solid rgba(240,100,49,.42); border-radius:10px; color:var(--accent); background:rgba(240,100,49,.09); font:700 15px ui-monospace,monospace; }
    .brand strong { display:block; font-size:16px; letter-spacing:-.02em; }
    .brand small { display:block; color:var(--muted); }
    .nav a,.nav button { padding:7px 11px; border:0; border-radius:8px; color:var(--muted); text-decoration:none; font:650 12px inherit; background:transparent; cursor:pointer; }
    .nav a:hover { background:#eef2f6; color:var(--text); }
    .nav .active { color:var(--accent); background:rgba(240,100,49,.08); }
    main { width:min(1400px,100%); margin:0 auto; padding:38px 28px 60px; }
    .console-shell { display:grid; grid-template-columns:220px minmax(0,1fr); gap:28px; align-items:start; }
    .side-panel { position:sticky; top:28px; padding:10px; border:1px solid var(--line); border-radius:14px; background:rgba(255,255,255,.92); box-shadow:var(--shadow); }
    .side-title { padding:9px 10px 12px; color:var(--subtle); font-size:10px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .side-item { width:100%; padding:11px 12px; display:flex; align-items:center; gap:11px; border:0; border-radius:9px; color:var(--muted); background:transparent; cursor:pointer; text-align:left; font-weight:700; }
    .side-item:hover { color:var(--text); background:#f1f5f8; }
    .side-item.active { color:var(--accent); background:rgba(240,100,49,.09); }
    .side-icon { width:25px; height:25px; display:grid; place-items:center; border:1px solid currentColor; border-radius:7px; font:700 11px ui-monospace,monospace; opacity:.75; }
    .console-view[hidden] { display:none; }
    .hero { display:flex; justify-content:space-between; align-items:end; gap:24px; margin-bottom:28px; }
    .eyebrow { margin:0 0 7px; color:var(--accent); font-size:10px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    h1 { margin:0; font-size:32px; letter-spacing:-.045em; }
    .hero p { max-width:600px; margin:9px 0 0; color:var(--muted); }
    .dummy { flex:none; padding:7px 11px; border:1px solid #e8c395; border-radius:999px; color:#8c5c17; background:#fff8e9; font-size:11px; font-weight:700; }
    .layout { display:grid; grid-template-columns:340px minmax(0,1fr); gap:22px; align-items:start; }
    .panel { overflow:hidden; border:1px solid var(--line); border-radius:14px; background:linear-gradient(180deg,#fff,#fafbfd); box-shadow:var(--shadow); }
    .panel-head { padding:18px 20px; border-bottom:1px solid var(--line); }
    .panel-head h2 { margin:0; font-size:16px; }
    .panel-head p { margin:5px 0 0; color:var(--muted); font-size:12px; }
    form { padding:20px; }
    label { display:block; margin-bottom:8px; color:#455464; font-size:12px; font-weight:700; }
    input { width:100%; height:42px; padding:0 12px; border:1px solid #c8d3dd; border-radius:9px; color:var(--text); outline:none; background:#fff; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    input:focus { border-color:rgba(240,100,49,.7); box-shadow:0 0 0 3px rgba(240,100,49,.09); }
    .error { min-height:21px; margin:6px 0 10px; color:var(--red); font-size:11px; }
    .btn { min-height:39px; padding:8px 13px; border:1px solid #cbd5df; border-radius:9px; color:var(--text); background:#eef3f7; cursor:pointer; font-weight:700; }
    .btn:hover { background:#e5ebf1; }
    .primary { width:100%; border-color:#ef784c; color:#fff; background:linear-gradient(135deg,#ef7b45,var(--accent)); box-shadow:0 8px 20px rgba(240,100,49,.18); }
    .primary:hover { background:linear-gradient(135deg,#f28552,#e95926); }
    .stats { display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-top:14px; }
    .stat { padding:12px; border:1px solid var(--line); border-radius:10px; background:var(--soft); }
    .stat span { display:block; color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.06em; }
    .stat strong { display:block; margin-top:4px; font-size:20px; }
    .list-head { padding:17px 20px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); }
    .list-head h2 { margin:0; font-size:16px; }
    .count { color:var(--muted); font-size:12px; }
    .machine { min-height:73px; padding:14px 20px; display:grid; grid-template-columns:minmax(180px,1fr) 120px 110px 40px; gap:16px; align-items:center; border-bottom:1px solid var(--line); }
    .machine:last-child { border-bottom:0; }
    .ip { font:700 13px ui-monospace,SFMono-Regular,Menlo,monospace; }
    .meta { margin-top:4px; color:var(--subtle); font-size:11px; }
    .status { width:max-content; padding:4px 8px; border:1px solid rgba(22,139,91,.22); border-radius:999px; color:var(--green); background:rgba(22,139,91,.07); font-size:10px; font-weight:800; text-transform:uppercase; }
    .role { color:var(--muted); font-size:12px; }
    .delete { width:34px; height:34px; border:1px solid transparent; border-radius:8px; color:var(--subtle); background:transparent; cursor:pointer; font-size:19px; line-height:1; }
    .delete:hover { border-color:rgba(201,54,73,.25); color:var(--red); background:rgba(201,54,73,.06); }
    .empty { padding:70px 20px; color:var(--muted); text-align:center; }
    .empty strong { display:block; margin-bottom:5px; color:var(--text); }
    .toast { position:fixed; right:24px; bottom:24px; padding:11px 15px; border:1px solid var(--line); border-radius:10px; background:#fff; box-shadow:var(--shadow); opacity:0; transform:translateY(8px); pointer-events:none; transition:.2s; }
    .toast.show { opacity:1; transform:none; }
    .modal-backdrop { position:fixed; inset:0; z-index:50; display:grid; place-items:center; padding:20px; background:rgba(23,33,43,.48); backdrop-filter:blur(3px); opacity:0; visibility:hidden; transition:opacity .18s,visibility .18s; }
    .modal-backdrop.open { opacity:1; visibility:visible; }
    .modal { width:min(430px,100%); padding:24px; border:1px solid var(--line); border-radius:15px; background:#fff; box-shadow:0 24px 70px rgba(23,33,43,.25); transform:translateY(8px) scale(.98); transition:transform .18s; }
    .modal-backdrop.open .modal { transform:none; }
    .modal-icon { width:42px; height:42px; margin-bottom:16px; display:grid; place-items:center; border-radius:12px; color:var(--red); background:rgba(201,54,73,.08); font-size:22px; font-weight:800; }
    .modal h2 { margin:0 0 8px; font-size:19px; letter-spacing:-.02em; }
    .modal p { margin:0; color:var(--muted); }
    .modal-ip { display:block; margin:13px 0 0; padding:10px 12px; border:1px solid var(--line); border-radius:8px; color:var(--text); background:var(--soft); font:700 12px ui-monospace,SFMono-Regular,Menlo,monospace; }
    .modal-actions { margin-top:22px; display:flex; justify-content:flex-end; gap:9px; }
    .confirm-delete { border-color:var(--red); color:#fff; background:var(--red); }
    .confirm-delete:hover { background:#ad293b; }
    .dashboard-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }
    .dashboard-card { min-height:132px; padding:20px; border:1px solid var(--line); border-radius:14px; background:linear-gradient(180deg,#fff,#fafbfd); box-shadow:var(--shadow); }
    .dashboard-label { color:var(--muted); font-size:11px; font-weight:750; letter-spacing:.06em; text-transform:uppercase; }
    .dashboard-value { margin-top:12px; font-size:34px; font-weight:760; letter-spacing:-.05em; }
    .dashboard-meta { margin-top:5px; color:var(--subtle); font-size:11px; }
    .dashboard-panels { margin-top:16px; display:grid; grid-template-columns:1fr 1.4fr; gap:16px; }
    .dashboard-panel { padding:20px; }
    .dashboard-panel h2 { margin:0; font-size:16px; }
    .dashboard-panel-copy { margin:5px 0 18px; color:var(--muted); font-size:12px; }
    .breakdown-row { margin-top:14px; }
    .breakdown-head { display:flex; align-items:center; justify-content:space-between; gap:14px; color:var(--muted); font-size:12px; }
    .breakdown-head strong { color:var(--text); }
    .track { height:7px; margin-top:7px; overflow:hidden; border-radius:999px; background:#e8eef3; }
    .fill { width:0; height:100%; border-radius:inherit; background:linear-gradient(90deg,var(--accent),#f58a55); transition:width .3s; }
    .fill.blue { background:linear-gradient(90deg,#2476c8,#60a4e8); }
    .job-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:9px; }
    .job-cell { padding:12px; border:1px solid var(--line); border-radius:10px; background:var(--soft); }
    .job-cell span { display:block; color:var(--muted); font-size:10px; text-transform:uppercase; }
    .job-cell strong { display:block; margin-top:4px; font-size:20px; }
    .dashboard-refresh { border-color:rgba(240,100,49,.3); color:var(--accent); background:rgba(240,100,49,.06); }
    @media(max-width:1100px){ .dashboard-grid{grid-template-columns:repeat(2,1fr)} }
    @media(max-width:900px){ .console-shell{grid-template-columns:1fr}.side-panel{position:static;display:flex;align-items:center}.side-title{padding:9px 12px}.side-item{width:auto}.dashboard-panels{grid-template-columns:1fr} }
    @media(max-width:760px){ .topbar{padding:0 16px}.brand small{display:none}.nav a:first-child{display:none} main{padding:24px 16px}.side-title{display:none}.side-item{flex:1;justify-content:center}.hero{align-items:start;flex-direction:column}.layout{grid-template-columns:1fr}.machine{grid-template-columns:1fr 90px 34px}.machine .role{display:none}.dashboard-grid{grid-template-columns:1fr 1fr}.job-grid{grid-template-columns:repeat(2,1fr)} }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><div class="mark">A</div><div><strong>Anvil</strong><small>Administration</small></div></div>
    <nav class="nav"><a href="/v1/ui">UI</a><a class="active" href="/v1/console">Console</a><button type="button" onclick="fetch('/v1/auth/logout',{method:'POST'}).then(()=>location.assign('/v1/auth'))">Sign out</button></nav>
  </header>
  <main>
    <div class="console-shell">
      <aside class="side-panel" aria-label="Console sections">
        <div class="side-title">Information</div>
        <button class="side-item active" type="button" data-view="machines" aria-selected="true"><span class="side-icon">M</span>Machine Information</button>
        <button class="side-item" type="button" data-view="dashboard" aria-selected="false"><span class="side-icon">D</span>Dashboard</button>
      </aside>
      <div>
        <section id="machines-view" class="console-view">
          <section class="hero"><div><p class="eyebrow">Infrastructure</p><h1>Machine Registry</h1><p>Add and remove worker machines available to anvil. This preview stores changes only in your browser.</p></div><span class="dummy">Frontend preview</span></section>
          <div class="layout">
            <section class="panel">
              <div class="panel-head"><h2>Add a Machine</h2><p>Enter an IPv4 or IPv6 address.</p></div>
              <form id="machine-form" novalidate>
                <label for="machine-ip">Machine IP address</label>
                <input id="machine-ip" name="ip" placeholder="10.0.0.42" autocomplete="off" spellcheck="false">
                <div id="ip-error" class="error" aria-live="polite"></div>
                <button class="btn primary" type="submit">Add machine</button>
                <div class="stats"><div class="stat"><span>Registered</span><strong id="registered">0</strong></div><div class="stat"><span>Available</span><strong id="available">0</strong></div></div>
              </form>
            </section>
            <section class="panel">
              <div class="list-head"><h2>Registered Machines</h2><span class="count" id="count"></span></div>
              <div id="machine-list"></div>
            </section>
          </div>
        </section>
        <section id="dashboard-view" class="console-view" hidden>
          <section class="hero"><div><p class="eyebrow">Monitor</p><h1>Service Dashboard</h1><p>Monitor registered users, submitted jobs, queue activity, and execution outcomes.</p></div><button id="dashboard-refresh" class="btn dashboard-refresh" type="button">Refresh data</button></section>
          <div class="dashboard-grid">
            <article class="dashboard-card"><div class="dashboard-label">Registered Users</div><div id="dash-users" class="dashboard-value">—</div><div id="dash-user-meta" class="dashboard-meta">Loading account data</div></article>
            <article class="dashboard-card"><div class="dashboard-label">Submitted Jobs</div><div id="dash-jobs" class="dashboard-value">—</div><div class="dashboard-meta">All recorded submissions</div></article>
            <article class="dashboard-card"><div class="dashboard-label">Active Jobs</div><div id="dash-active" class="dashboard-value">—</div><div id="dash-active-meta" class="dashboard-meta">Queued and running</div></article>
            <article class="dashboard-card"><div class="dashboard-label">Success Rate</div><div id="dash-success-rate" class="dashboard-value">—</div><div id="dash-updated" class="dashboard-meta">Waiting for refresh</div></article>
          </div>
          <div class="dashboard-panels">
            <section class="panel dashboard-panel"><h2>User Roles</h2><p class="dashboard-panel-copy">Registered accounts by access level.</p><div class="breakdown-row"><div class="breakdown-head"><span>General Users</span><strong id="dash-general">—</strong></div><div class="track"><div id="dash-general-bar" class="fill"></div></div></div><div class="breakdown-row"><div class="breakdown-head"><span>Administrators</span><strong id="dash-admin">—</strong></div><div class="track"><div id="dash-admin-bar" class="fill blue"></div></div></div></section>
            <section class="panel dashboard-panel"><h2>Job Lifecycle</h2><p class="dashboard-panel-copy">Current totals across persisted and in-memory jobs.</p><div class="job-grid"><div class="job-cell"><span>Queued</span><strong id="dash-queued">—</strong></div><div class="job-cell"><span>Running</span><strong id="dash-running">—</strong></div><div class="job-cell"><span>Succeeded</span><strong id="dash-succeeded">—</strong></div><div class="job-cell"><span>Failed</span><strong id="dash-failed">—</strong></div><div class="job-cell"><span>Canceled</span><strong id="dash-canceled">—</strong></div><div class="job-cell"><span>GPU Workers</span><strong id="dash-gpus">—</strong></div></div></section>
          </div>
        </section>
      </div>
    </div>
  </main>
  <div id="delete-modal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="delete-title" aria-describedby="delete-description">
    <div class="modal">
      <div class="modal-icon" aria-hidden="true">!</div>
      <h2 id="delete-title">Delete This Machine?</h2>
      <p id="delete-description">This will remove the machine from the local registry.</p>
      <span id="delete-ip" class="modal-ip"></span>
      <div class="modal-actions"><button id="cancel-delete" class="btn" type="button">No, keep it</button><button id="confirm-delete" class="btn confirm-delete" type="button">Yes, delete</button></div>
    </div>
  </div>
  <div id="toast" class="toast" role="status"></div>
  <script>
    (() => {
      const seed = [{ip:'10.24.0.18',added:'Preview machine'},{ip:'10.24.0.27',added:'Preview machine'},{ip:'fd12:3456:789a::8',added:'Preview machine'}];
      let machines;
      try { machines = JSON.parse(localStorage.getItem('kernelagent.console.machines')) || seed; } catch (_) { machines = seed; }
      const list = document.getElementById('machine-list'); const input = document.getElementById('machine-ip'); const error = document.getElementById('ip-error');
      const modal = document.getElementById('delete-modal'); const confirmDelete = document.getElementById('confirm-delete'); const cancelDelete = document.getElementById('cancel-delete'); let pendingDelete = null; let deleteTrigger = null;
      document.querySelectorAll('.side-item').forEach(button => button.addEventListener('click', () => { const view=button.dataset.view; document.querySelectorAll('.side-item').forEach(item => { const active=item===button; item.classList.toggle('active',active); item.setAttribute('aria-selected',String(active)); }); document.getElementById('machines-view').hidden=view!=='machines'; document.getElementById('dashboard-view').hidden=view!=='dashboard'; if(view==='dashboard')loadDashboard(); }));
      const escapeHtml = value => value.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      const validIp = value => { if (value.includes(':')) return /^[0-9a-f:]+$/i.test(value) && value.includes(':') && value.length <= 45; const p=value.split('.'); return p.length===4 && p.every(x => /^\d{1,3}$/.test(x) && Number(x)<=255 && String(Number(x))===x); };
      const save = () => { try { localStorage.setItem('kernelagent.console.machines', JSON.stringify(machines)); } catch (_) {} };
      const toast = message => { const el=document.getElementById('toast'); el.textContent=message; el.classList.add('show'); clearTimeout(toast.timer); toast.timer=setTimeout(()=>el.classList.remove('show'),2200); };
      const setText = (id,value) => { document.getElementById(id).textContent=value; };
      const loadDashboard = async () => { try { const response=await fetch('/v1/console/stats'); if(!response.ok)throw new Error(`HTTP ${response.status}`); const data=await response.json(); const users=data.users,jobs=data.jobs,infra=data.infrastructure; setText('dash-users',users.total); setText('dash-user-meta',`${users.general} general · ${users.admin} admin`); setText('dash-jobs',jobs.total); setText('dash-active',jobs.active); setText('dash-active-meta',`${jobs.queued} queued · ${jobs.running} running`); const decided=jobs.succeeded+jobs.unsuccessful; setText('dash-success-rate',decided?`${Math.round(jobs.succeeded/decided*100)}%`:'—'); setText('dash-updated',`Updated ${new Date().toLocaleTimeString()}`); setText('dash-general',users.general); setText('dash-admin',users.admin); document.getElementById('dash-general-bar').style.width=`${users.total?users.general/users.total*100:0}%`; document.getElementById('dash-admin-bar').style.width=`${users.total?users.admin/users.total*100:0}%`; setText('dash-queued',jobs.queued); setText('dash-running',jobs.running); setText('dash-succeeded',jobs.succeeded); setText('dash-failed',jobs.unsuccessful); setText('dash-canceled',jobs.canceled); setText('dash-gpus',infra.gpu_workers); } catch(error) { toast(`Dashboard refresh failed: ${error.message}`); } };
      const render = () => {
        document.getElementById('registered').textContent=machines.length; document.getElementById('available').textContent=machines.length; document.getElementById('count').textContent=`${machines.length} machine${machines.length===1?'':'s'}`;
        if (!machines.length) { list.innerHTML='<div class="empty"><strong>No machines registered</strong>Add a machine by IP to get started.</div>'; return; }
        list.innerHTML=machines.map((m,i)=>`<div class="machine"><div><div class="ip">${escapeHtml(m.ip)}</div><div class="meta">${escapeHtml(m.added || 'Added locally')}</div></div><span class="status">Available</span><span class="role">GPU worker</span><button class="delete" data-index="${i}" aria-label="Delete ${escapeHtml(m.ip)}" title="Delete machine">×</button></div>`).join('');
      };
      document.getElementById('machine-form').addEventListener('submit', event => { event.preventDefault(); const ip=input.value.trim().toLowerCase(); error.textContent=''; if(!validIp(ip)){ error.textContent='Enter a valid IPv4 or IPv6 address.'; input.focus(); return; } if(machines.some(m=>m.ip.toLowerCase()===ip)){ error.textContent='This machine is already registered.'; input.focus(); return; } machines.unshift({ip,added:'Added just now · browser only'}); save(); render(); input.value=''; toast(`Added ${ip}`); });
      const closeDeleteModal = () => { modal.classList.remove('open'); pendingDelete=null; if(deleteTrigger){ deleteTrigger.focus(); deleteTrigger=null; } };
      list.addEventListener('click', event => { const button=event.target.closest('.delete'); if(!button)return; pendingDelete=Number(button.dataset.index); deleteTrigger=button; document.getElementById('delete-ip').textContent=machines[pendingDelete].ip; modal.classList.add('open'); cancelDelete.focus(); });
      cancelDelete.addEventListener('click', closeDeleteModal);
      modal.addEventListener('click', event => { if(event.target===modal) closeDeleteModal(); });
      document.addEventListener('keydown', event => { if(event.key==='Escape' && modal.classList.contains('open')) closeDeleteModal(); });
      confirmDelete.addEventListener('click', () => { if(pendingDelete===null)return; const removed=machines[pendingDelete]; machines.splice(pendingDelete,1); save(); render(); modal.classList.remove('open'); pendingDelete=null; deleteTrigger=null; toast(`Removed ${removed.ip}`); });
      document.getElementById('dashboard-refresh').addEventListener('click',loadDashboard);
      render();
      loadDashboard();
      window.setInterval(loadDashboard,10000);
    })();
  </script>
</body>
</html>"""


_TASK_UI = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Anvil</title>
  <style>
    :root {
      --bg: #edf2f7;
      --panel: #ffffff;
      --panel-2: #f8fafc;
      --panel-3: #eef3f7;
      --line: #cbd5df;
      --line-soft: #dde5ed;
      --text: #17212b;
      --muted: #637181;
      --subtle: #8995a2;
      --accent: #f06431;
      --accent-2: #e9793f;
      --green: #168b5b;
      --blue: #2476c8;
      --yellow: #a8760d;
      --red: #d64557;
      --radius: 14px;
      --shadow: 0 16px 45px rgba(48, 67, 86, .10);
    }

    * { box-sizing: border-box; }
    html { min-width: 320px; background: var(--bg); }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 15% -15%, rgba(255, 122, 69, .13), transparent 32rem),
        radial-gradient(circle at 95% 15%, rgba(36, 118, 200, .10), transparent 30rem),
        var(--bg);
      font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select, textarea { font: inherit; }
    button, a { -webkit-tap-highlight-color: transparent; }

    .app { min-height: 100vh; }
    .topbar {
      height: 68px;
      padding: 0 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line-soft);
      background: rgba(255, 255, 255, .86);
      backdrop-filter: blur(18px);
      position: sticky;
      top: 0;
      z-index: 20;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
      width: 34px;
      height: 34px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(255, 122, 69, .45);
      border-radius: 10px;
      color: var(--accent-2);
      background: linear-gradient(145deg, rgba(255, 122, 69, .18), rgba(255, 122, 69, .04));
      font: 700 15px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
      box-shadow: inset 0 0 20px rgba(255, 122, 69, .08);
    }
    .brand-name { font-size: 16px; font-weight: 700; letter-spacing: -.02em; }
    .brand-sub { color: var(--muted); font-size: 12px; }
    .health-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: rgba(255, 255, 255, .88);
      font-size: 12px;
    }
    .health-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--subtle); }
    .health-chip.ok .health-dot { background: var(--green); box-shadow: 0 0 10px rgba(66, 211, 146, .75); }
    .health-chip.bad .health-dot { background: var(--red); }
    .top-actions { display:flex; align-items:center; gap:9px; }
    .console-link { padding:7px 11px; border:1px solid var(--line); border-radius:8px; color:var(--text); background:#fff; text-decoration:none; font-size:12px; font-weight:700; }
    .console-link:hover { border-color:rgba(240,100,49,.42); color:var(--accent-2); background:rgba(240,100,49,.05); }
    .signout { padding:7px 10px; border:0; border-radius:8px; color:var(--muted); background:transparent; cursor:pointer; font-size:12px; font-weight:700; }
    .signout:hover { color:var(--text); background:#eef3f7; }

    .shell {
      width: min(1560px, 100%);
      margin: 0 auto;
      padding: 24px 28px 40px;
      display: grid;
      grid-template-columns: 370px minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--line-soft);
      border-radius: var(--radius);
      background: linear-gradient(180deg, rgba(255, 255, 255, .98), rgba(248, 250, 252, .98));
      box-shadow: var(--shadow);
    }
    .composer { position: sticky; top: 92px; overflow: hidden; }
    .panel-head {
      padding: 18px 20px;
      border-bottom: 1px solid var(--line-soft);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .eyebrow {
      margin: 0 0 3px;
      color: var(--accent-2);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .13em;
      text-transform: uppercase;
    }
    h1, h2, h3, p { margin-top: 0; }
    h1 { margin-bottom: 0; font-size: 19px; letter-spacing: -.025em; }
    h2 { margin-bottom: 0; font-size: 16px; }
    h3 { margin-bottom: 10px; font-size: 13px; }

    .form { padding: 18px 20px 20px; }
    .field { margin-bottom: 15px; }
    .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    label { display: block; margin: 0 0 7px; color: #465565; font-size: 12px; font-weight: 600; }
    .hint { color: var(--subtle); font-weight: 400; }
    input, select, textarea {
      width: 100%;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 9px;
      outline: none;
      background: #ffffff;
      transition: border-color .18s, box-shadow .18s;
    }
    input, select { height: 39px; padding: 0 11px; }
    textarea {
      padding: 11px 12px;
      resize: vertical;
      min-height: 84px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
      tab-size: 4;
    }
    #pytorch-code { min-height: 272px; }
    input:focus, select:focus, textarea:focus {
      border-color: rgba(255, 122, 69, .7);
      box-shadow: 0 0 0 3px rgba(255, 122, 69, .09);
    }
    .btn {
      min-height: 38px;
      padding: 8px 13px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 9px;
      color: var(--text);
      background: var(--panel-3);
      cursor: pointer;
      font-weight: 650;
      transition: transform .12s, border-color .18s, background .18s;
    }
    .btn:hover { border-color: #aab7c4; background: #e5ebf1; }
    .btn:active { transform: translateY(1px); }
    .btn:disabled { opacity: .45; cursor: not-allowed; }
    .btn-primary {
      width: 100%;
      border-color: #ff8a55;
      color: #160a04;
      background: linear-gradient(135deg, var(--accent-2), var(--accent));
      box-shadow: 0 8px 22px rgba(255, 122, 69, .18);
    }
    .btn-primary:hover { border-color: #ffc088; background: linear-gradient(135deg, #ffc274, #ff8754); }
    .btn-danger { border-color: rgba(255, 102, 120, .35); color: #ff9aa6; background: rgba(255, 102, 120, .08); }
    .btn-small { min-height: 32px; padding: 5px 10px; font-size: 12px; }

    .workspace { min-width: 0; }
    .upload-box { padding: 12px; border: 1px dashed #b9c7d3; border-radius: 10px; background: #f8fafc; }
    .upload-box input { height: auto; padding: 0; border: 0; background: transparent; }
    .upload-help { margin: 8px 0 0; color: var(--subtle); font-size: 10px; }
    .upload-status { min-height: 18px; margin-top: 6px; color: var(--muted); font-size: 11px; }
    .upload-status.error { color: var(--red); }
    .upload-support { margin: 9px 0 0; padding-top: 9px; border-top: 1px solid var(--line-soft); color: var(--muted); font-size: 10px; line-height: 1.55; }
    .upload-support strong { color: #465565; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
    .metric {
      min-height: 86px;
      padding: 15px 17px;
      border: 1px solid var(--line-soft);
      border-radius: 12px;
      background: rgba(255, 255, 255, .82);
    }
    .metric-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
    .metric-value { margin-top: 7px; font-size: 23px; font-weight: 720; letter-spacing: -.04em; }
    .metric-value.text { font-size: 15px; line-height: 1.8; color: var(--green); }

    .task-panel { overflow: hidden; }
    .toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .tabs { display: flex; gap: 4px; padding: 3px; border-radius: 9px; background: #e8eef4; }
    .tab {
      min-height: 30px;
      padding: 5px 10px;
      border: 0;
      border-radius: 7px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
      font-size: 12px;
    }
    .tab.active { color: var(--text); background: var(--panel-3); }
    .task-list { min-height: 330px; }
    .task-row {
      width: 100%;
      padding: 15px 20px;
      display: grid;
      grid-template-columns: minmax(190px, 1.5fr) 110px 90px 120px 90px;
      gap: 16px;
      align-items: center;
      border: 0;
      border-bottom: 1px solid var(--line-soft);
      color: inherit;
      text-align: left;
      background: transparent;
      cursor: pointer;
      transition: background .16s;
    }
    .task-row:hover, .task-row.selected { background: rgba(36, 118, 200, .045); }
    .task-row.selected { box-shadow: inset 3px 0 var(--accent); }
    .task-id { overflow: hidden; text-overflow: ellipsis; font: 600 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .task-stage { margin-top: 3px; overflow: hidden; color: var(--muted); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
    .cell-label { display: none; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .muted { color: var(--muted); }
    .small { font-size: 12px; }
    .badge {
      width: fit-content;
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: rgba(99, 113, 129, .05);
      font-size: 10px;
      font-weight: 750;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .badge.running { color: #8bc1ff; border-color: rgba(98, 168, 255, .3); background: rgba(98, 168, 255, .08); }
    .badge.queued { color: #f6d67b; border-color: rgba(239, 199, 94, .3); background: rgba(239, 199, 94, .08); }
    .badge.succeeded { color: #7be3ad; border-color: rgba(66, 211, 146, .3); background: rgba(66, 211, 146, .08); }
    .badge.failed, .badge.timed_out, .badge.lost { color: #ff94a0; border-color: rgba(255, 102, 120, .3); background: rgba(255, 102, 120, .08); }
    .badge.canceled { color: #adb6c0; }
    .empty { min-height: 300px; padding: 48px 20px; display: grid; place-items: center; color: var(--muted); text-align: center; }
    .empty-icon { margin-bottom: 10px; color: var(--subtle); font: 28px/1 ui-monospace, monospace; }

    .detail { margin-top: 18px; overflow: hidden; }
    .detail.hidden { display: none; }
    .detail-actions { display: flex; gap: 8px; }
    .detail-body { padding: 20px; }
    .summary-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 18px; }
    .summary-item { padding: 11px 12px; border: 1px solid var(--line-soft); border-radius: 10px; background: #f7f9fb; }
    .summary-item span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }
    .summary-item strong { display: block; margin-top: 5px; overflow: hidden; text-overflow: ellipsis; font-size: 12px; white-space: nowrap; }
    .detail-grid { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(300px, .9fr); gap: 16px; }
    .section-card { min-width: 0; padding: 15px; border: 1px solid var(--line-soft); border-radius: 11px; background: #f8fafc; }
    .section-title { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .code-block {
      max-height: 340px;
      margin: 0;
      padding: 12px;
      overflow: auto;
      border: 1px solid #1d2731;
      border-radius: 8px;
      color: #dbe6ef;
      background: #18222d;
      font: 11px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .events { max-height: 390px; overflow: auto; }
    .event { padding: 10px 0; border-bottom: 1px solid var(--line-soft); }
    .event:last-child { border-bottom: 0; }
    .event-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .event-type { color: var(--blue); font: 600 11px/1.4 ui-monospace, monospace; }
    .event-time { color: var(--subtle); font-size: 10px; }
    .event-payload { margin-top: 6px; color: var(--muted); font: 10px/1.5 ui-monospace, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
    .artifact { padding: 10px 0; display: flex; align-items: center; justify-content: space-between; gap: 10px; border-bottom: 1px solid var(--line-soft); }
    .artifact:last-child { border-bottom: 0; }
    .artifact-name { overflow: hidden; font: 600 11px/1.45 ui-monospace, monospace; text-overflow: ellipsis; white-space: nowrap; }
    .artifact-meta { color: var(--subtle); font-size: 10px; }
    .download { color: var(--accent-2); text-decoration: none; font-size: 12px; font-weight: 650; }
    .download:hover { text-decoration: underline; }
    .error-box { margin-bottom: 16px; padding: 12px; border: 1px solid rgba(255, 102, 120, .25); border-radius: 9px; color: #ffabb5; background: rgba(255, 102, 120, .07); white-space: pre-wrap; }

    .toast-stack { position: fixed; right: 22px; bottom: 22px; z-index: 50; display: grid; gap: 9px; }
    .toast { max-width: 380px; padding: 11px 14px; border: 1px solid var(--line); border-radius: 10px; color: var(--text); background: #ffffff; box-shadow: var(--shadow); animation: enter .2s ease-out; }
    .toast.error { border-color: rgba(255, 102, 120, .4); }
    @keyframes enter { from { opacity: 0; transform: translateY(8px); } }
    .spin { animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    @media (max-width: 1100px) {
      .shell { grid-template-columns: 330px minmax(0, 1fr); padding-inline: 18px; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .task-row { grid-template-columns: minmax(170px, 1.5fr) 100px 80px 95px; }
      .task-row > :nth-child(5) { display: none; }
      .summary-grid { grid-template-columns: repeat(3, 1fr); }
      .detail-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      .topbar { height: 60px; padding: 0 15px; }
      .brand-sub { display: none; }
      .shell { display: block; padding: 14px 12px 30px; }
      .composer { position: static; margin-bottom: 14px; }
      .metrics { gap: 8px; }
      .metric { min-height: 72px; padding: 12px; }
      .metric-value { font-size: 19px; }
      .panel-head { padding: 15px; align-items: flex-start; }
      .form, .detail-body { padding: 15px; }
      .toolbar { justify-content: flex-end; }
      .tabs { width: 100%; overflow-x: auto; justify-content: flex-start; }
      .task-row { padding: 14px 15px; grid-template-columns: 1fr 90px; gap: 10px; }
      .task-row > :nth-child(3), .task-row > :nth-child(4), .task-row > :nth-child(5) { display: none; }
      .summary-grid { grid-template-columns: repeat(2, 1fr); }
      .detail-actions { flex-direction: column; }
      .toast-stack { left: 12px; right: 12px; bottom: 12px; }
      .toast { max-width: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">A∿</div>
        <div>
          <div class="brand-name">Anvil</div>
          <div class="brand-sub">GPU Kernel Generation & Optimization</div>
        </div>
      </div>
      <div class="top-actions"><a class="console-link" href="/v1/console">Console</a><div id="health-chip" class="health-chip">
        <span class="health-dot"></span>
        <span id="health-text">正在连接服务</span>
      </div><button class="signout" type="button" onclick="fetch('/v1/auth/logout',{method:'POST'}).then(()=>location.assign('/v1/auth'))">退出</button></div>
    </header>

    <main class="shell">
      <aside class="panel composer">
        <div class="panel-head">
          <div><p class="eyebrow">New Workload</p><h1>创建 Kernel 任务</h1></div>
          <span class="badge">CUDA</span>
        </div>
        <form id="task-form" class="form">
          <div class="field">
            <label for="submission-file">Candidate submission <span class="hint">optional</span></label>
            <div class="upload-box">
              <input id="submission-file" type="file" accept=".py,.cu,.cpp,.cc,.cxx,.c,.json,.zip,.tar.gz,.tgz">
              <p class="upload-help">Accepts Python run() sources, C/C++ or CUDA sources, JSON solutions, and multi-file .zip/.tar.gz/.tgz archives. Archives must contain submission.py or a PyBind host entry point.</p>
              <div id="upload-status" class="upload-status" aria-live="polite"></div>
              <p class="upload-support"><strong>Currently supported:</strong> <code>.py</code> with top-level <code>run()</code>; <code>.cpp</code>/<code>.cc</code>/<code>.cxx</code>/<code>.c</code> with <code>run()</code> and <code>PYBIND11_MODULE</code>; standalone <code>.cu</code>; SOL-style <code>.json</code> with <code>spec</code> and <code>sources</code>; and <code>.zip</code>/<code>.tar.gz</code>/<code>.tgz</code> containing <code>submission.py</code> or a PyBind host entry point. These formats are normalized and supplied to the generation agent as candidate source; native SOL compilation and evaluation are not yet available.</p>
            </div>
          </div>
          <div class="field">
            <label for="pytorch-code">PyTorch 参考实现 <span class="hint">correctness + workloads · 必填</span></label>
            <textarea id="pytorch-code" spellcheck="false" required>import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x, y):
        return x + y

def get_inputs():
    return [
        torch.randn(1048576, device="cuda"),
        torch.randn(1048576, device="cuda"),
    ]

def get_init_inputs():
    return []</textarea>
          </div>
          <div class="field-row">
            <div class="field">
              <label for="runner">Harness Runner</label>
              <select id="runner"><option value="claude">Claude Code</option><option value="pi">pi</option><option value="codex">Codex</option></select>
            </div>
            <div class="field">
              <label for="target-hardware">Target Hardware</label>
              <select id="target-hardware"><option value="H200">NVIDIA H200</option><option value="B200">NVIDIA B200</option><option value="H20">NVIDIA H20</option><option value="H100">NVIDIA H100</option><option value="A100">NVIDIA A100</option><option value="昇腾">昇腾</option><option value="寒武纪">寒武纪</option></select>
            </div>
          </div>
          <div class="field-row">
            <div class="field">
              <label for="language">Kernel 语言</label>
              <select id="language"><option value="triton">Triton</option><option value="cutedsl">CuTe DSL</option></select>
            </div>
            <div class="field">
              <label for="rounds">最大优化轮数</label>
              <input id="rounds" type="number" min="1" max="100" value="5">
            </div>
          </div>
          <div class="field-row">
            <div class="field">
              <label for="timeout">超时时间 <span class="hint">秒</span></label>
              <input id="timeout" type="number" min="30" max="86400" value="7200">
            </div>
          </div>
          <div class="field">
            <label for="instructions">额外优化要求 <span class="hint">可选</span></label>
            <textarea id="instructions" spellcheck="false" placeholder="例如：优先优化大尺寸连续张量的吞吐性能"></textarea>
          </div>
          <div class="field">
            <label for="test-code">附加测试代码 <span class="hint">可选</span></label>
            <textarea id="test-code" spellcheck="false" placeholder="# 可选的 Python 正确性测试"></textarea>
          </div>
          <button id="submit-btn" class="btn btn-primary" type="submit">提交到 GPU 队列 <span>→</span></button>
        </form>
      </aside>

      <section class="workspace">
        <div class="metrics">
          <div class="metric"><div class="metric-label">服务状态</div><div id="metric-status" class="metric-value text">连接中</div></div>
          <div class="metric"><div class="metric-label">GPU Workers</div><div id="metric-gpus" class="metric-value">—</div></div>
          <div class="metric"><div class="metric-label">运行中</div><div id="metric-running" class="metric-value">—</div></div>
          <div class="metric"><div class="metric-label">队列</div><div id="metric-queue" class="metric-value">—</div></div>
        </div>

        <section class="panel task-panel">
          <div class="panel-head">
            <div><p class="eyebrow">Work Queue</p><h2>任务中心</h2></div>
            <div class="toolbar">
              <div id="tabs" class="tabs">
                <button class="tab active" data-filter="all" type="button">全部</button>
                <button class="tab" data-filter="active" type="button">进行中</button>
                <button class="tab" data-filter="succeeded" type="button">成功</button>
                <button class="tab" data-filter="failed" type="button">异常</button>
              </div>
              <button id="refresh-btn" class="btn btn-small" type="button" title="立即刷新">↻ 刷新</button>
            </div>
          </div>
          <div id="task-list" class="task-list"><div class="empty"><div><div class="empty-icon">⌁</div>正在读取任务…</div></div></div>
        </section>

        <section id="detail" class="panel detail hidden">
          <div class="panel-head">
            <div><p class="eyebrow">Task Detail</p><h2 id="detail-title" class="mono">—</h2></div>
            <div class="detail-actions">
              <button id="cancel-btn" class="btn btn-small btn-danger" type="button">取消任务</button>
              <button id="close-detail" class="btn btn-small" type="button">关闭</button>
            </div>
          </div>
          <div class="detail-body">
            <div id="task-error"></div>
            <div id="summary-grid" class="summary-grid"></div>
            <div class="detail-grid">
              <div class="section-card">
                <div class="section-title"><h3>运行事件</h3><span id="event-count" class="muted small">0 条</span></div>
                <div id="events" class="events"><div class="muted small">暂无事件</div></div>
              </div>
              <div>
                <div class="section-card" style="margin-bottom:16px">
                  <div class="section-title"><h3>生成产物</h3><span id="artifact-count" class="muted small">0 个</span></div>
                  <div id="artifacts"><div class="muted small">暂无产物</div></div>
                </div>
                <div class="section-card">
                  <h3>结构化结果</h3>
                  <pre id="result" class="code-block">任务尚未返回结果</pre>
                </div>
              </div>
            </div>
          </div>
        </section>
      </section>
    </main>
  </div>
  <div id="toasts" class="toast-stack" aria-live="polite"></div>

  <script>
    'use strict';

    const state = { tasks: [], filter: 'all', selectedId: null, health: null, uploadName: null, uploadContent: null, uploadError: null, hardwareTouched: false };
    const terminal = new Set(['succeeded', 'failed', 'canceled', 'timed_out', 'lost']);
    const abnormal = new Set(['failed', 'timed_out', 'lost', 'canceled']);
    const $ = (id) => document.getElementById(id);

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"]/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: {'Content-Type': 'application/json', ...(options.headers || {})},
      });
      let payload = null;
      const text = await response.text();
      if (text) {
        try { payload = JSON.parse(text); } catch (_) { payload = text; }
      }
      if (!response.ok) {
        const detail = payload && payload.detail ? payload.detail : payload;
        throw new Error(typeof detail === 'string' ? detail : `HTTP ${response.status}`);
      }
      return payload;
    }

    function toast(message, kind = '') {
      const node = document.createElement('div');
      node.className = `toast ${kind}`;
      node.textContent = message;
      $('toasts').appendChild(node);
      window.setTimeout(() => node.remove(), 4200);
    }

    function formatTime(value) {
      if (!value) return '—';
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', {hour12: false});
    }

    function elapsed(task) {
      if (!task.started_at) return '—';
      const end = task.finished_at ? new Date(task.finished_at) : new Date();
      const seconds = Math.max(0, Math.floor((end - new Date(task.started_at)) / 1000));
      if (seconds < 60) return `${seconds}s`;
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
      return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    }

    function formatBytes(bytes) {
      if (!Number.isFinite(bytes)) return '—';
      const units = ['B', 'KB', 'MB', 'GB'];
      let value = bytes, index = 0;
      while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
      return `${value.toFixed(index ? 1 : 0)} ${units[index]}`;
    }

    function statusLabel(status) {
      return ({queued:'排队中', running:'运行中', succeeded:'成功', failed:'失败', canceled:'已取消', timed_out:'已超时', lost:'已丢失'})[status] || status;
    }

    async function loadHealth() {
      try {
        const health = await api('/healthz');
        state.health = health;
        const ok = health.status === 'ok';
        $('health-chip').className = `health-chip ${ok ? 'ok' : 'bad'}`;
        const runners = (health.runner_backends || []).join(' / ');
        $('health-text').textContent = `${ok ? '服务正常' : '服务降级'}${runners ? ` · ${runners}` : ''}`;
        $('metric-status').textContent = ok ? 'ONLINE' : 'DEGRADED';
        $('metric-status').style.color = ok ? 'var(--green)' : 'var(--yellow)';
        $('metric-gpus').textContent = health.gpu_workers.length;
        $('metric-running').textContent = health.running_tasks;
        $('metric-queue').textContent = `${health.queue_size} / ${health.queue_capacity}`;
        if (!state.hardwareTouched && health.target_hardware) {
          const target = $('target-hardware');
          if (![...target.options].some((option) => option.value === health.target_hardware)) {
            target.add(new Option(health.target_hardware, health.target_hardware));
          }
          target.value = health.target_hardware;
        }
      } catch (error) {
        $('health-chip').className = 'health-chip bad';
        $('health-text').textContent = '连接失败';
        $('metric-status').textContent = 'OFFLINE';
        $('metric-status').style.color = 'var(--red)';
      }
    }

    function filteredTasks() {
      if (state.filter === 'all') return state.tasks;
      if (state.filter === 'active') return state.tasks.filter((task) => ['queued', 'running'].includes(task.status));
      if (state.filter === 'failed') return state.tasks.filter((task) => abnormal.has(task.status));
      return state.tasks.filter((task) => task.status === state.filter);
    }

    function renderTasks() {
      const tasks = filteredTasks();
      if (!tasks.length) {
        $('task-list').innerHTML = '<div class="empty"><div><div class="empty-icon">⌁</div>当前筛选条件下没有任务</div></div>';
        return;
      }
      $('task-list').innerHTML = tasks.map((task) => `
        <button class="task-row ${task.id === state.selectedId ? 'selected' : ''}" data-task-id="${escapeHtml(task.id)}" type="button">
          <div><div class="task-id">${escapeHtml(task.id)}</div><div class="task-stage">${escapeHtml(task.stage || task.operation || '等待调度')}</div></div>
          <div><span class="badge ${escapeHtml(task.status)}">${escapeHtml(statusLabel(task.status))}</span></div>
          <div class="small"><span class="cell-label">GPU </span>${escapeHtml(task.gpu_id ?? '—')}</div>
          <div class="small muted">${escapeHtml(formatTime(task.created_at))}</div>
          <div class="small mono">${escapeHtml(elapsed(task))}</div>
        </button>`).join('');
      document.querySelectorAll('[data-task-id]').forEach((node) => node.addEventListener('click', () => selectTask(node.dataset.taskId)));
    }

    async function loadTasks(silent = false) {
      try {
        state.tasks = await api('/v1/tasks?offset=0&limit=100');
        renderTasks();
        if (state.selectedId) await loadDetail(state.selectedId, true);
      } catch (error) {
        if (!silent) toast(`读取任务失败：${error.message}`, 'error');
      }
    }

    function summaryItem(label, value) {
      return `<div class="summary-item"><span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong></div>`;
    }

    function isVisibleArtifact(artifact) {
      const path = artifact.relative_path || artifact.name || '';
      const parts = path.split('/');
      if (parts.some((part) => part.startsWith('.run_') || part === '__pycache__')) return false;
      return !['stdout.txt', 'stderr.txt'].includes((artifact.name || '').toLowerCase())
        && !path.endsWith('.pyc');
    }

    async function selectTask(taskId) {
      state.selectedId = taskId;
      renderTasks();
      $('detail').classList.remove('hidden');
      await loadDetail(taskId);
      $('detail').scrollIntoView({behavior: 'smooth', block: 'start'});
    }

    async function loadDetail(taskId, silent = false) {
      try {
        const [task, events] = await Promise.all([
          api(`/v1/tasks/${encodeURIComponent(taskId)}`),
          api(`/v1/tasks/${encodeURIComponent(taskId)}/events?after=0&limit=1000`),
        ]);
        if (state.selectedId !== taskId) return;
        renderDetail(task, events);
      } catch (error) {
        if (!silent) toast(`读取任务详情失败：${error.message}`, 'error');
      }
    }

    function renderDetail(task, events) {
      $('detail-title').textContent = task.id;
      $('summary-grid').innerHTML = [
        summaryItem('状态', statusLabel(task.status)),
        summaryItem('阶段', task.stage || '—'),
        summaryItem('GPU', task.gpu_id ?? '—'),
        summaryItem('Runner', task.runner_backend || '—'),
        summaryItem('耗时', elapsed(task)),
      ].join('');
      $('cancel-btn').disabled = terminal.has(task.status);
      $('task-error').innerHTML = task.error ? `<div class="error-box">${escapeHtml(task.error)}</div>` : '';
      $('result').textContent = task.result ? JSON.stringify(task.result, null, 2) : '任务尚未返回结果';

      const artifacts = (task.artifacts || []).filter(isVisibleArtifact);
      $('artifact-count').textContent = `${artifacts.length} 个`;
      $('artifacts').innerHTML = artifacts.length ? artifacts.map((artifact) => `
        <div class="artifact">
          <div style="min-width:0"><div class="artifact-name" title="${escapeHtml(artifact.relative_path)}">${escapeHtml(artifact.name)}</div><div class="artifact-meta">${escapeHtml(formatBytes(artifact.size_bytes))}</div></div>
          <a class="download" href="/v1/tasks/${encodeURIComponent(task.id)}/artifacts/${encodeURIComponent(artifact.id)}" download>下载 ↓</a>
        </div>`).join('') : '<div class="muted small">暂无产物</div>';

      $('event-count').textContent = `${events.length} 条`;
      $('events').innerHTML = events.length ? events.slice().reverse().map((event) => `
        <div class="event">
          <div class="event-top"><span class="event-type">#${event.sequence} ${escapeHtml(event.type)}</span><span class="event-time">${escapeHtml(formatTime(event.created_at))}</span></div>
          <div class="event-payload">${escapeHtml(JSON.stringify(event.payload, null, 2))}</div>
        </div>`).join('') : '<div class="muted small">暂无事件</div>';
    }

    async function submitTask(event) {
      event.preventDefault();
      if (state.uploadError) {
        toast(`提交失败：${state.uploadError}`, 'error');
        return;
      }
      const button = $('submit-btn');
      button.disabled = true;
      button.textContent = '正在提交…';
      const payload = {
        pytorch_code: $('pytorch-code').value,
        runner_backend: $('runner').value,
        kernel_language: $('language').value,
        max_rounds: Number($('rounds').value),
        timeout_seconds: Number($('timeout').value),
        target_hardware: $('target-hardware').value,
      };
      if (state.uploadName) {
        payload.submission_filename = state.uploadName;
        payload.submission_content = state.uploadContent;
      }
      const instructions = $('instructions').value.trim();
      const testCode = $('test-code').value.trim();
      if (instructions) payload.extra_instructions = instructions;
      if (testCode) payload.test_code = testCode;
      try {
        const created = await api('/v1/tasks', {method: 'POST', body: JSON.stringify(payload)});
        toast(`任务已提交：${created.task_id}`);
        state.filter = 'all';
        syncTabs();
        await loadTasks();
        await selectTask(created.task_id);
      } catch (error) {
        toast(`提交失败：${error.message}`, 'error');
      } finally {
        button.disabled = false;
        button.innerHTML = '提交到 GPU 队列 <span>→</span>';
      }
    }

    function fileExtension(name) {
      const lower = name.toLowerCase();
      if (lower.endsWith('.tar.gz')) return '.tar.gz';
      const index = lower.lastIndexOf('.');
      return index >= 0 ? lower.slice(index) : 'unknown';
    }

    function encodeBase64(buffer) {
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let offset = 0; offset < bytes.length; offset += 32768) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
      }
      return btoa(binary);
    }

    async function loadSubmissionFile(event) {
      const file = event.target.files[0];
      const status = $('upload-status');
      state.uploadName = null;
      state.uploadContent = null;
      state.uploadError = null;
      status.className = 'upload-status';
      status.textContent = '';
      if (!file) return;
      const extension = fileExtension(file.name);
      const supported = new Set(['.py', '.cu', '.cpp', '.cc', '.cxx', '.c', '.json', '.zip', '.tar.gz', '.tgz']);
      if (!supported.has(extension)) {
        state.uploadError = `${extension} format is currently not supported`;
        status.classList.add('error');
        status.textContent = state.uploadError;
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        state.uploadError = 'Uploaded file is too large; the current limit is 10 MB';
        status.classList.add('error');
        status.textContent = state.uploadError;
        return;
      }
      try {
        state.uploadName = file.name;
        state.uploadContent = encodeBase64(await file.arrayBuffer());
        status.textContent = `${file.name} loaded as the candidate solution`;
      } catch (_) {
        state.uploadError = 'The selected file could not be read';
        status.classList.add('error');
        status.textContent = state.uploadError;
      }
    }

    async function cancelSelected() {
      if (!state.selectedId || !window.confirm('确定取消这个任务？运行中的 runner 及其子进程会被终止。')) return;
      try {
        await api(`/v1/tasks/${encodeURIComponent(state.selectedId)}/cancel`, {method: 'POST'});
        toast('取消请求已提交');
        await loadTasks();
      } catch (error) {
        toast(`取消失败：${error.message}`, 'error');
      }
    }

    function syncTabs() {
      document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.filter === state.filter));
    }

    $('task-form').addEventListener('submit', submitTask);
    $('submission-file').addEventListener('change', loadSubmissionFile);
    $('target-hardware').addEventListener('change', () => { state.hardwareTouched = true; });
    $('refresh-btn').addEventListener('click', async () => { await Promise.all([loadHealth(), loadTasks()]); toast('已刷新'); });
    $('cancel-btn').addEventListener('click', cancelSelected);
    $('close-detail').addEventListener('click', () => { state.selectedId = null; $('detail').classList.add('hidden'); renderTasks(); });
    document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => { state.filter = tab.dataset.filter; syncTabs(); renderTasks(); }));

    Promise.all([loadHealth(), loadTasks()]);
    window.setInterval(loadHealth, 5000);
    window.setInterval(() => loadTasks(true), 4000);
  </script>
</body>
</html>
"""
