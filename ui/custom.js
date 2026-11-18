/* OpenAmika — the Custom brain panel, shared by Setup → Brain → Custom and the first-run wizard: her four roles, each
   on a model of an OpenAI-compatible server with its own thinking, and the servers themselves.
   OA.customPanel(host, options) fills host and returns { load }. options:
     compact   the wizard's order (servers before roles) and a new server named "My server"
     use       (body, done) → Promise: post /api/build_f/custom/use and call done(answer); the Setup page passes its
               switch with Unlock, the wizard a plain post
     onChange  called after anything changed what her roles can run (the Setup page reloads its model lists) */
(function () {
  "use strict";
  const { toast, getJSON, postAction, el } = window.OA;

  const ROLES = ["surface", "deep", "reflection", "memory_write"];
  const ROLE_TEXT = { surface: ["Surface", "instant reply"], deep: ["Deep", "her real voice"],
                      reflection: ["Reflection", "background coach"], memory_write: ["Memory writer", "remembers"] };
  const JSON_MODES = [["json_object", "json_object · most servers"], ["json_schema", "json_schema · LM Studio"],
                      ["none", "none · the server refuses both"]];
  const SWITCHES = [["none", "Don't send · the server decides"],
                    ["reasoning_effort", "reasoning_effort · OpenAI-style APIs, routers, Groq"],
                    ["chat_template_kwargs", "chat_template_kwargs · vLLM, llama.cpp (on or off)"]];
  // the efforts most roles use are buttons; the rare ones and "send nothing" (null, "" in a select) sit behind ⋯
  const EFFORT_BUTTONS = [["none", "Off"], ["low", "Low"], ["medium", "Med"], ["high", "High"]];
  const EFFORT_MORE = [["", "Server decides"], ["minimal", "Minimal"], ["xhigh", "Extra high"], ["max", "Max"]];
  const SMALL_CONTEXT = 16384;      // her prompts take about 8k tokens plus the answer
  const MORE = "more";              // the ⋯ placeholder of the rare-effort select

  const chatModels = (endpoint) => (endpoint.models || []).filter((m) => m.kind === "chat");
  const contextText = (n) => n >= 1000000 ? Math.round(n / 1048576 * 10) / 10 + "M" : Math.round(n / 1024) + "k";
  const pickValue = (endpoint, model) => JSON.stringify([endpoint, model]);
  const roleList = (roles) => roles.map((r) => ROLE_TEXT[r][0].toLowerCase()).join(", ");

  async function plainUse(body, done) {
    try { done(await postAction("/api/build_f/custom/use", "custom-use", body)); }
    catch (error) { toast(String(error.message || error), true); }
  }

  function customPanel(host, options) {
    const opts = Object.assign({ compact: false, use: plainUse, onChange: () => {} }, options || {});
    host.classList.add("cu-panel");
    let info = null;              // GET /api/build_f/custom: the servers, their models, which roles use them
    let models = null;            // GET /api/build_f/models: what a role that is not on a custom server runs
    let gen = 0;                  // a status read still in flight is older than any answer shown after it started
    const rows = {};              // role → { endpoint, model, effort } as shown; endpoint null: stays where it is
    const openModels = new Set(); // server ids whose Models fold is open
    const editing = new Map();    // server id ("" for a new one) → its form as typed
    const checking = new Set();   // server ids being asked for their models
    let using = false;

    const endpoints = () => (info && info.endpoints) || [];
    const find = (id) => endpoints().find((e) => e.id === id) || null;
    const storedEffort = (endpoint, role) => {
      const value = (endpoint.reasoning_effort || {})[role];
      return value === undefined ? null : value;
    };
    function baseline(role) {
      const used = ((info && info.in_use) || {})[role];
      const endpoint = used && find(used.endpoint);
      return endpoint ? { endpoint: used.endpoint, model: used.model, effort: storedEffort(endpoint, role) }
                      : { endpoint: null, model: null, effort: null };
    }
    function changed(role) {
      const row = rows[role], base = baseline(role);
      return !!row && (row.endpoint !== base.endpoint || row.model !== base.model || row.effort !== base.effort);
    }
    const valid = (row) => !row.endpoint || (find(row.endpoint) && chatModels(find(row.endpoint)).some((m) => m.name === row.model));

    // A new answer keeps the last known reachability of a server whose address and key did not change (only GET
    // /api/build_f/custom asks the servers), and the role picks not used yet while their models still exist.
    function setInfo(next) {
      gen++;
      const keep = {};
      if (info) for (const role of ROLES) if (changed(role)) keep[role] = rows[role];
      const before = new Map(endpoints().map((e) => [e.id, e]));
      for (const endpoint of next.endpoints || []) {
        const old = before.get(endpoint.id), key = endpoint.key || {}, oldKey = (old && old.key) || {};
        if (endpoint.reachable === null && old && old.base_url === endpoint.base_url && oldKey.present === key.present && oldKey.hint === key.hint) {
          endpoint.reachable = old.reachable; endpoint.error = old.error;
        }
      }
      info = next;
      for (const role of ROLES) rows[role] = keep[role] && valid(keep[role]) ? keep[role] : baseline(role);
      render();
    }

    async function load(givenModels) {
      const mine = ++gen;
      try {
        const [status, roleModels] = await Promise.all([getJSON("/api/build_f/custom"),
          givenModels ? Promise.resolve(givenModels) : getJSON("/api/build_f/models").catch(() => null)]);
        if (mine !== gen) return;
        if (roleModels) models = roleModels;
        setInfo(status);
      } catch (error) {
        if (mine === gen) host.replaceChildren(el("p", { class: "card-hint", text: "status unavailable: " + (error.message || error) }));
      }
    }

    // ── her four roles ──
    function staysText(role) {
      const roleOptions = (models && models.options && models.options[role]) || [];
      const item = roleOptions.find((o) => o.key === ((models && models.selections) || {})[role]);
      return (item && item.name ? item.name : "another provider") + " · stays";
    }
    function modelOptions(select) {
      for (const endpoint of endpoints()) {
        const chat = chatModels(endpoint);
        if (!chat.length) continue;
        select.append(el("optgroup", { label: endpoint.name }, ...chat.map((m) => el("option", { value: pickValue(endpoint.id, m.name),
          text: m.name + " · " + endpoint.name + (m.context_length ? " · " + contextText(m.context_length) : "") }))));
      }
    }
    function setModel(role, value) {
      if (!value) { rows[role] = baseline(role); return; }
      const [id, model] = JSON.parse(value);
      const endpoint = find(id), row = rows[role];
      // a role that already ran on a custom server keeps the thinking shown; one that comes over takes the server's
      rows[role] = { endpoint: id, model, effort: row.endpoint ? row.effort : storedEffort(endpoint, role) };
    }
    function warnings(role) {
      const row = rows[role], endpoint = row.endpoint && find(row.endpoint);
      if (!endpoint) return [];
      const out = [];
      const m = chatModels(endpoint).find((x) => x.name === row.model);
      if (m && m.context_length && m.context_length < SMALL_CONTEXT) {
        out.push(m.name + "'s context window is " + m.context_length.toLocaleString("en-US") +
                 " tokens; her prompts take about 8k plus the answer: use a model or server setting of 16k or more.");
      }
      // the memory writer's answer cap (6,000 tokens) holds its thinking too: a long think cuts its JSON off
      if (role === "memory_write" && endpoint.reasoning_switch !== "none" && row.effort !== null &&
          (endpoint.reasoning_switch === "chat_template_kwargs" ? row.effort !== "none" : !["none", "minimal", "low"].includes(row.effort))) {
        out.push((endpoint.reasoning_switch === "chat_template_kwargs" ? "With thinking on" : "Above Low") +
                 ", the memory writer's thinking can use up its 6,000-token answer and cut its JSON off.");
      }
      return out;
    }
    function thinking(role) {
      const row = rows[role], endpoint = row.endpoint && find(row.endpoint);
      if (!endpoint) return el("span", { class: "cu-fixed", text: "—" });
      if (endpoint.reasoning_switch === "none") {
        return el("span", { class: "cu-fixed", text: "server decides",
          title: endpoint.name + " is set to send no thinking setting: Edit → Advanced → How to send thinking." });
      }
      const onOff = endpoint.reasoning_switch === "chat_template_kwargs";
      const set = (effort) => { rows[role] = Object.assign({}, rows[role], { effort }); render(); };
      const thinks = row.effort !== null && row.effort !== "none";
      const seg = el("div", { class: "seg", role: "group", "aria-label": ROLE_TEXT[role][0] + " thinking" });
      for (const [value, text] of onOff ? [["none", "Off"], ["on", "On"]] : EFFORT_BUTTONS) {
        const on = onOff ? (value === "on" ? thinks : row.effort === "none") : row.effort === value;
        // on/off servers read any effort but none as on: keep the one stored, else low
        const effort = value === "on" ? (thinks ? row.effort : "low") : value;
        seg.append(el("button", { type: "button", class: on ? "on" : "", "aria-pressed": String(on), text,
                                  onclick: () => set(effort) }));
      }
      const more = onOff ? EFFORT_MORE.slice(0, 1) : EFFORT_MORE;
      const rare = more.find(([value]) => (value || null) === row.effort);
      // a native select is as wide as its longest choice: an invisible one over a label as wide as the one shown
      const select = el("select", { "aria-label": ROLE_TEXT[role][0] + ": more thinking choices" },
        el("option", { value: MORE, text: "⋯" }), ...more.map(([value, text]) => el("option", { value, text })));
      select.value = rare ? rare[0] : MORE;
      select.addEventListener("change", () => { if (select.value !== MORE) set(select.value || null); });
      seg.append(el("span", { class: "more" + (rare ? " on" : "") }, el("span", { text: rare ? rare[1] : "⋯" }), select));
      return seg;
    }
    function rolesBlock() {
      const box = el("div", { class: "cu-roles" });
      const all = el("select", { "aria-label": "Set all four roles to" }, el("option", { value: "", text: "— pick a model —" }));
      modelOptions(all);
      const first = rows[ROLES[0]];
      if (first.endpoint && ROLES.every((r) => rows[r].endpoint === first.endpoint && rows[r].model === first.model)) {
        all.value = pickValue(first.endpoint, first.model);
      }
      all.addEventListener("change", () => {
        if (!all.value) return;
        for (const role of ROLES) setModel(role, all.value);
        render();
      });
      box.append(el("div", { class: "cu-all" }, el("span", { text: "Set all four to" }), all),
                 el("div", { class: "cu-head" }, el("span", { text: "Role" }), el("span", { text: "Model" }), el("span", { text: "Thinking" })));
      for (const role of ROLES) {
        const row = rows[role];
        const select = el("select", { "aria-label": ROLE_TEXT[role][0] + " model" });
        if (!baseline(role).endpoint) select.append(el("option", { value: "", text: staysText(role) }));
        modelOptions(select);
        select.value = row.endpoint ? pickValue(row.endpoint, row.model) : "";
        select.addEventListener("change", () => { setModel(role, select.value); render(); });
        const warn = warnings(role);
        box.append(el("div", { class: "cu-row" + (changed(role) ? " changed" : ""), "data-role": role },
          el("div", { class: "who" }, el("b", { text: ROLE_TEXT[role][0] }), el("span", { text: ROLE_TEXT[role][1] })),
          el("div", { class: "model" }, select),
          el("div", { class: "think" }, el("small", { text: "Thinking" }), thinking(role)),
          warn.length ? el("p", { class: "warn", text: warn.join(" ") }) : null));
      }
      const pending = ROLES.filter(changed);
      const onCustom = ROLES.some((r) => baseline(r).endpoint);
      const use = el("button", { type: "button", class: "btn small primary", text: "Use these", onclick: useThese });
      use.disabled = using || !pending.length;
      box.append(el("div", { class: "cu-foot" }, use,
        pending.length ? el("button", { type: "button", class: "btn small", text: "Undo",
          onclick: () => { for (const role of ROLES) rows[role] = baseline(role); render(); } }) : null,
        el("span", { class: "card-hint", text: pending.length ? pending.length + " role" + (pending.length > 1 ? "s" : "") + " changed · not used yet"
          : onCustom ? "This is what she runs now." : "Pick a model for a role, then Use these." })));
      return box;
    }
    async function useThese() {
      const pending = ROLES.filter(changed);
      if (!pending.length || using) return;
      // every role shown on a custom server, so the answer is exactly what the table says; the others stay
      const roles = {};
      for (const role of ROLES) {
        const row = rows[role], endpoint = row.endpoint && find(row.endpoint);
        if (!endpoint) continue;
        roles[role] = { endpoint: row.endpoint, model: row.model };
        if (endpoint.reasoning_switch !== "none") roles[role].effort = row.effort;
      }
      const warning = pending.map(warnings).flat()[0] || "";
      const summary = pending.map((r) => ROLE_TEXT[r][0] + " → " + rows[r].model).join(", ");
      using = true;
      render();
      try {
        await opts.use({ roles }, (answer) => {
          for (const role of ROLES) rows[role] = null;     // the answer is the new baseline
          setInfo(Object.assign({}, info, { endpoints: answer.endpoints || info.endpoints, in_use: answer.in_use || {} }));
          toast("Now running: " + summary + "." + (warning ? " " + warning : ""), !!warning);
          opts.onChange();
        });
      } finally {
        using = false;
        render();
      }
    }

    // ── the servers ──
    async function detect(id) {
      const endpoint = find(id);
      if (!endpoint || checking.has(id)) return;
      checking.add(id);
      render();
      try {
        const answer = await postAction("/api/build_f/custom/detect", "custom-detect", { id });
        checking.delete(id);
        setInfo(answer);
        const found = find(id), n = found ? chatModels(found).length : 0;
        if (!n) openModels.add(id);
        toast(n ? n + " chat model" + (n > 1 ? "s" : "") + " at " + found.name + "."
          : endpoint.name + " answered, but lists no chat model: load one on the server, or add its id by hand under Models.", !n);
      } catch (error) {
        checking.delete(id);
        const current = find(id);
        if (current) Object.assign(current, { reachable: false, error: String(error.message || error) });
        openModels.add(id);
        render();
        toast(String(error.message || error), true);
      }
      opts.onChange();
    }
    async function handModel(endpoint, name, remove) {
      try {
        setInfo(await postAction("/api/build_f/custom/model", "custom-model", remove ? { id: endpoint.id, name, remove: true } : { id: endpoint.id, name }));
        toast(remove ? name + " removed." : name + " added: pick it for a role above.");
        opts.onChange();
      } catch (error) { toast(String(error.message || error), true); }
    }

    function serverCard(endpoint) {
      if (editing.has(endpoint.id)) return editor(endpoint);
      const chat = chatModels(endpoint), key = endpoint.key || {};
      const runs = ROLES.filter((r) => (((info.in_use || {})[r]) || {}).endpoint === endpoint.id);
      const used = new Set(runs.map((r) => info.in_use[r].model));
      const busy = checking.has(endpoint.id);
      const state = busy ? "asking for its models…" : endpoint.reachable === false ? (endpoint.error || "not answering at " + endpoint.base_url)
        : endpoint.reachable ? "answering" : "";
      const refresh = el("button", { type: "button", class: "btn small", text: "Refresh models", onclick: () => detect(endpoint.id) });
      refresh.disabled = busy;
      const fold = el("details", { class: "oa-details cu-models" }, el("summary", { text: "Models (" + chat.length + ")" }));
      fold.open = openModels.has(endpoint.id) || !chat.length;
      fold.addEventListener("toggle", () => { if (fold.open) openModels.add(endpoint.id); else openModels.delete(endpoint.id); });
      const typed = el("input", { type: "text", maxlength: "200", autocomplete: "off", placeholder: "its id as the server knows it",
                                  "data-k": "hand:" + endpoint.id, "aria-label": "Add a model to " + endpoint.name + " by hand" });
      const add = () => {
        const name = typed.value.trim();
        if (!name) { toast("Type the model's id as the server knows it.", true); typed.focus(); return; }
        handModel(endpoint, name, false);
      };
      typed.addEventListener("keydown", (e) => { if (e.key === "Enter") add(); });
      fold.append(el("div", { class: "inner" },
        ...(chat.length ? chat.map((m) => el("div", { class: "cu-mrow" },
          el("span", { class: "mono", text: m.name, title: m.name }),
          el("span", { class: "card-hint", text: [m.context_length ? contextText(m.context_length) + " context" : "",
            m.source === "manual" ? "added by hand" : "", used.has(m.name) ? "in use" : ""].filter(Boolean).join(" · ") }),
          m.source === "manual" && !used.has(m.name)
            ? el("button", { type: "button", class: "editorial-link", text: "Remove →", onclick: () => handModel(endpoint, m.name, true) }) : null))
          : [el("p", { class: "card-hint", text: "No chat model listed: load one on the server and press Refresh models, or add its id by hand." })]),
        el("div", { class: "form-row cu-hand" },
          el("div", { class: "field", style: "flex:2" }, el("label", { text: "Add a model by hand" }), typed),
          el("button", { type: "button", class: "btn small", text: "Add", onclick: add }))));
      return el("div", { class: "cu-srv", "data-server": endpoint.id },
        el("div", { class: "cu-srv-head" },
          el("span", { class: "dot " + (busy ? "warn" : endpoint.reachable === true ? "ok" : endpoint.reachable === false ? "err" : "") }),
          el("span", { class: "name", text: endpoint.name }),
          el("span", { class: "addr mono", text: endpoint.base_url, title: endpoint.base_url }),
          el("span", { class: "acts" }, refresh,
            el("button", { type: "button", class: "btn small", text: "Edit", onclick: () => { editing.set(endpoint.id, draftOf(endpoint)); render(); } }))),
        el("div", { class: "cu-meta" },
          el("span", { class: "tag keep", text: key.present ? "key " + (key.hint || "saved") : "no key" }),
          el("span", { class: "tag", text: chat.length + " model" + (chat.length === 1 ? "" : "s") }),
          el("span", { class: endpoint.reachable === false && !busy ? "bad" : "",
                       text: [state, runs.length ? "runs " + roleList(runs) : ""].filter(Boolean).join(" · ") })),
        fold);
    }

    const draftOf = (endpoint) => endpoint
      ? { name: endpoint.name, url: endpoint.base_url, keyMode: (endpoint.key || {}).present ? "keep" : "none", key: "",
          json: endpoint.json_mode, think: endpoint.reasoning_switch, advanced: false }
      : { name: opts.compact ? "My server" : "", url: "", keyMode: "none", key: "", json: "json_object", think: "none", advanced: false };

    function editor(endpoint) {
      const id = endpoint ? endpoint.id : "";
      const d = editing.get(id);
      const hint = endpoint && endpoint.key && endpoint.key.hint;
      const input = (field, attrs) => {
        const node = el("input", Object.assign({ type: "text", autocomplete: "off", "data-k": field + ":" + id }, attrs));
        node.value = d[field];
        node.addEventListener("input", () => { d[field] = node.value; });
        return node;
      };
      const mode = (keyMode) => () => { d.keyMode = keyMode; d.key = ""; focusNext = "key:" + id; render(); };
      const keyInput = () => {
        const node = input("key", { type: "password", placeholder: d.keyMode === "replace" ? "paste the new key" : "paste the key",
                                    "aria-label": "Key" });
        node.addEventListener("keydown", (e) => { if (e.key === "Enter") save(); });
        return node;
      };
      const button = (text, onclick, cls) => el("button", { type: "button", class: "btn small" + (cls ? " " + cls : ""), text, onclick });
      const keyRow = el("div", { class: "cu-key" }, el("span", { class: "lbl", text: "Key" }),
        ...({
          keep: () => [el("span", { class: "val" }, "Saved · ", el("span", { class: "mono", text: hint || "…" })),
                       button("Replace", mode("replace")), button("Remove", mode("remove"), "danger")],
          replace: () => [keyInput(), button("Cancel", mode("keep"))],
          remove: () => [el("span", { class: "val warn", text: (hint || "The key") + " is removed when you press Save" }), button("Undo", mode("keep"))],
          none: () => [el("span", { class: "val muted", text: "None · only if the server asks for one" }), button("Add a key", mode("add"))],
          add: () => [keyInput(), button("Cancel", mode("none"))],
        })[d.keyMode]());
      const choose = (field, choices) => {
        const node = el("select", { "data-k": field + ":" + id }, ...choices.map(([value, text]) => el("option", { value, text })));
        node.value = d[field];
        node.addEventListener("change", () => { d[field] = node.value; });
        return node;
      };
      const advanced = el("details", { class: "oa-details cu-adv" }, el("summary", { text: "Advanced · JSON and thinking" }),
        el("div", { class: "inner" },
          el("div", { class: "field" }, el("label", { text: "JSON for her memories" }), choose("json", JSON_MODES),
            el("p", { class: "card-hint", text: "How the memory writer asks for JSON. LM Studio wants json_schema; pick none only if the server refuses both." })),
          el("div", { class: "field" }, el("label", { text: "How to send thinking" }), choose("think", SWITCHES),
            el("p", { class: "card-hint", text: "How each role's Thinking reaches the server. chat_template_kwargs knows only on and off. " +
                                                "\"Don't send\" leaves it to the server, and the roles' Thinking says \"server decides\"." }))));
      advanced.open = d.advanced;
      advanced.addEventListener("toggle", () => { d.advanced = advanced.open; });
      const saveButton = button(endpoint ? "Save" : "Connect", save, "primary");
      async function save() {
        const body = { name: d.name.trim(), base_url: d.url.trim(), json_mode: d.json, reasoning_switch: d.think };
        if (!body.name) { toast("Give the server a name.", true); return; }
        if (!body.base_url) { toast("Enter the server's address, e.g. http://127.0.0.1:1234/v1", true); return; }
        if (endpoint) body.id = endpoint.id;
        if ((d.keyMode === "replace" || d.keyMode === "add") && d.key.trim()) body.key = d.key.trim();
        if (d.keyMode === "remove") body.clear_key = true;
        const before = new Set(endpoints().map((e) => e.id));
        saveButton.disabled = true;
        try {
          const answer = await postAction("/api/build_f/custom/endpoint", "custom-endpoint", body);
          editing.delete(id);
          setInfo(answer);
          // a new server is added last; a new address or key is asked for its models again
          const saved = endpoint ? find(endpoint.id) : (answer.endpoints || []).filter((e) => !before.has(e.id)).pop();
          if (saved && (!endpoint || saved.base_url !== endpoint.base_url || body.key || body.clear_key)) detect(saved.id);
          else { toast("Saved."); opts.onChange(); }
        } catch (error) {
          saveButton.disabled = false;
          toast(String(error.message || error), true);
        }
      }
      async function remove() {
        if (!window.confirm("Remove " + endpoint.name + ((endpoint.key || {}).present ? " and its saved key?" : "?"))) return;
        try {
          editing.delete(id);
          setInfo(await postAction("/api/build_f/custom/endpoint", "custom-endpoint", { id, remove: true }));
          toast(endpoint.name + " removed.");
          opts.onChange();
        } catch (error) { editing.set(id, d); toast(String(error.message || error), true); }
      }
      // the first server has nothing to go back to
      const cancel = endpoint || endpoints().length ? button("Cancel", () => { editing.delete(id); render(); }) : null;
      return el("div", { class: "cu-srv editing", "data-server": id },
        el("div", { class: "cu-srv-head" },
          endpoint ? el("span", { class: "dot " + (endpoint.reachable === true ? "ok" : endpoint.reachable === false ? "err" : "") }) : null,
          el("span", { class: "name", text: endpoint ? "Editing " + endpoint.name : endpoints().length ? "New server" : "Your server" })),
        el("div", { class: "cu-edit" },
          el("div", { class: "form-row" },
            el("div", { class: "field" }, el("label", { text: "Name" }), input("name", { maxlength: "60", placeholder: "e.g. LM Studio" })),
            el("div", { class: "field", style: "flex:2" }, el("label", { text: "Address" }), input("url", { placeholder: "http://127.0.0.1:1234/v1" }))),
          keyRow, advanced,
          el("div", { class: "cu-edit-foot" }, saveButton, cancel,
            endpoint ? null : el("span", { class: "card-hint", text: "saves it and asks for its models" }),
            el("span", { class: "spacer" }),
            endpoint ? button("Remove server", remove, "danger") : null)));
    }

    // a redraw keeps the field being typed in, and focuses the key field a Replace or Add a key just opened
    let focusNext = null;
    function render() {
      if (!info) return;
      const active = document.activeElement && host.contains(document.activeElement) && document.activeElement.getAttribute("data-k");
      const caret = active && typeof document.activeElement.selectionStart === "number"
        ? [document.activeElement.selectionStart, document.activeElement.selectionEnd] : null;
      const list = endpoints();
      if (!list.length && !editing.has("")) editing.set("", draftOf(null));
      const anyModel = list.some((e) => chatModels(e).length);
      const roles = [el("div", { class: "cu-h" }, el("h4", { text: "Her four roles" }),
                       el("span", { class: "card-hint", text: "a model and how hard it thinks, per role" })),
                     anyModel ? rolesBlock() : el("p", { class: "card-hint", text: list.length
                       ? "No chat model yet: press Refresh models on a server, or add a model by hand under its Models."
                       : "Add your server below: its address, and a key if it asks for one." })];
      const servers = [el("div", { class: "cu-h" }, el("h4", { text: "Servers" })), ...list.map(serverCard),
        editing.has("") ? editor(null)
          : el("button", { type: "button", class: "btn small cu-add", text: "+ Add a server", onclick: () => { editing.set("", draftOf(null)); focusNext = "name:"; render(); } })];
      const hint = el("p", { class: "card-hint cu-hint" },
        "Her prompts take about 8k tokens plus the answer: give each model a context window of 16k or more on its server (LM Studio: ",
        el("b", { text: "Context Length" }), " when you load it; llama.cpp: ", el("code", { text: "-c 16384" }), "; vLLM: ",
        el("code", { text: "--max-model-len 16384" }), "; cloud APIs need nothing). Her cost meter counts these calls at $0.");
      host.replaceChildren(...(opts.compact ? (anyModel ? [...servers, ...roles] : servers) : [...roles, ...servers]), hint);
      const target = focusNext || active;
      focusNext = null;
      const node = target && host.querySelector('[data-k="' + target + '"]');
      if (node) {
        node.focus();
        if (caret && target === active && typeof node.setSelectionRange === "function") node.setSelectionRange(caret[0], caret[1]);
      }
    }

    return { load };
  }

  window.OA.customPanel = customPanel;
})();
