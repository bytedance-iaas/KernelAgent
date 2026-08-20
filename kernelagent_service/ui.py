"""Dependency-free web UI for the KernelAgent task service."""

from __future__ import annotations


def render_task_ui() -> str:
    """Return the self-contained task dashboard HTML."""
    return _TASK_UI


_TASK_UI = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>KernelAgent Console</title>
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
        <div class="brand-mark">K∿</div>
        <div>
          <div class="brand-name">KernelAgent Console</div>
          <div class="brand-sub">GPU kernel generation & optimization</div>
        </div>
      </div>
      <div id="health-chip" class="health-chip">
        <span class="health-dot"></span>
        <span id="health-text">正在连接服务</span>
      </div>
    </header>

    <main class="shell">
      <aside class="panel composer">
        <div class="panel-head">
          <div><p class="eyebrow">New workload</p><h1>创建 Kernel 任务</h1></div>
          <span class="badge">CUDA</span>
        </div>
        <form id="task-form" class="form">
          <div class="field">
            <label for="pytorch-code">PyTorch 参考实现 <span class="hint">必填</span></label>
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
              <label for="language">Kernel 语言</label>
              <select id="language"><option value="triton">Triton</option><option value="cutedsl">CuTe DSL</option></select>
            </div>
          </div>
          <div class="field-row">
            <div class="field">
              <label for="rounds">最大优化轮数</label>
              <input id="rounds" type="number" min="1" max="100" value="5">
            </div>
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
            <div><p class="eyebrow">Work queue</p><h2>任务中心</h2></div>
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
            <div><p class="eyebrow">Task detail</p><h2 id="detail-title" class="mono">—</h2></div>
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

    const state = { tasks: [], filter: 'all', selectedId: null, health: null };
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

      const artifacts = task.artifacts || [];
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
      const button = $('submit-btn');
      button.disabled = true;
      button.textContent = '正在提交…';
      const payload = {
        pytorch_code: $('pytorch-code').value,
        runner_backend: $('runner').value,
        kernel_language: $('language').value,
        max_rounds: Number($('rounds').value),
        timeout_seconds: Number($('timeout').value),
      };
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
