"""
Purpose
-------
Watches user_request.txt for plain-English requests, forwards them to a
local Ollama instance in a background daemon thread, and applies the
returned bounded controller parameters while the CARLA simulation keeps
running uninterrupted.

Safety contract
---------------
  - The LLM may only tune the bounded high-level parameters listed below.
  - Steering is never set by this module — the deterministic controller
    always computes steer_cmd from state and route.
  - If Ollama is slow, fails, or returns garbage, the previous safe
    parameters remain active and driving continues.

Supported controllers
---------------------
  pure_pursuit  stanley  pid  mpc

Usage
-----
    sup = LiveLLMSupervisor(controller_type="pure_pursuit")
    # Once per simulated second inside the CARLA control loop:
    changed, params, status, reason, request = sup.poll()
    if changed:
        controller.lookahead_base = params["lookahead_base"]
        run_target_kmh = params["target_speed_kmh"]
"""

import json
import os
import threading
import urllib.request
import urllib.error


_MODULE_DIR     = os.path.dirname(os.path.abspath(__file__))
_REQUEST_FILE   = os.path.join(_MODULE_DIR, "user_request.txt")
_PARAMS_FILE    = os.path.join(_MODULE_DIR, "live_llm_params.json")

_OLLAMA_URL       = "http://localhost:11434/api/generate"
_OLLAMA_MODEL_GPU = "llama3.2:1b"        # primary — uses GPU if available
_OLLAMA_MODEL_CPU = "llama3.2-3b-cpu"   # fallback — forced to CPU
_OLLAMA_TIMEOUT   = 300   # seconds — long enough for cold CPU inference

# Per-model generation options.
_GPU_OPTIONS = {
    "temperature": 0,
    "num_predict": 100,
}
_CPU_OPTIONS = {
    "num_gpu":     0,
    "temperature": 0,
    "num_predict": 100,
}


# Per-controller defaults (target_speed_kmh = 10 for interactive demo)


_DEFAULTS = {
    "pure_pursuit": {
        "target_speed_kmh": 15.0,
        "lookahead_base":    6.0,
        "lookahead_gain":    0.5,
    },
    "stanley": {
        "target_speed_kmh": 15.0,
        "k":                 1.0,
        "softening_speed":   1.0,
    },
    "pid": {
        "target_speed_kmh": 15.0,
        "Kp_cte":            0.25,
        "Kd_cte":            0.10,
        "K_heading":         0.80,
    },
    "mpc": {
        "target_speed_kmh": 15.0,
        "q_lateral":         1.0,
        "q_heading":         0.5,
        "r_steer":           0.1,
        "r_steer_change":    0.2,
    },
}


# Safety bounds — enforced after every Ollama response


_BOUNDS = {
    "pure_pursuit": {
        "target_speed_kmh": (8.0,  20.0),
        "lookahead_base":   (4.0,  10.0),
        "lookahead_gain":   (0.2,   1.2),
    },
    "stanley": {
        "target_speed_kmh": (8.0,  20.0),
        "k":                (0.2,   2.0),
        "softening_speed":  (0.5,   3.0),
    },
    "pid": {
        "target_speed_kmh": (8.0,  20.0),
        "Kp_cte":           (0.05,  0.6),
        "Kd_cte":           (0.0,   0.4),
        "K_heading":        (0.2,   1.5),
    },
    "mpc": {
        "target_speed_kmh": (8.0,  20.0),
        "q_lateral":        (0.5,   4.0),
        "q_heading":        (0.2,   3.0),
        "r_steer":          (0.05,  1.0),
        "r_steer_change":   (0.05,  1.5),
    },
}

SUPPORTED_CONTROLLERS = tuple(_DEFAULTS.keys())

# ---------------------------------------------------------------------------
# Per-controller JSON templates — shown in the prompt so the model knows
# the exact key names and value format expected.
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "pure_pursuit": (
        '{"target_speed_kmh":12,"lookahead_base":6.5,'
        '"lookahead_gain":0.3,"reason":"ok"}'
    ),
    "stanley": (
        '{"target_speed_kmh":10,"k":1.5,'
        '"softening_speed":2.0,"reason":"ok"}'
    ),
    "pid": (
        '{"target_speed_kmh":12,"Kp_cte":0.25,'
        '"Kd_cte":0.10,"K_heading":0.80,"reason":"ok"}'
    ),
    "mpc": (
        '{"target_speed_kmh":12,"q_lateral":2.0,"q_heading":0.8,'
        '"r_steer":0.15,"r_steer_change":0.8,"reason":"ok"}'
    ),
}


# ---------------------------------------------------------------------------
# Supervisor class
# ---------------------------------------------------------------------------

class LiveLLMSupervisor(object):
    """
    Asynchronous live LLM parameter supervisor for CARLA LKA controllers.

    Watches user_request.txt for changes.  When the text changes and is
    non-empty, an Ollama request is started in a daemon thread so the CARLA
    loop is never blocked.  When Ollama responds with valid JSON, the active
    parameters are updated and live_llm_params.json is written.

    Args:
        controller_type: One of "pure_pursuit", "stanley", "pid", "mpc".
    """

    def __init__(self, controller_type="pure_pursuit"):
        if controller_type not in _DEFAULTS:
            raise ValueError(
                "LiveLLMSupervisor: unsupported controller '{}'. "
                "Choose from: {}".format(controller_type, list(SUPPORTED_CONTROLLERS))
            )
        self.controller_type = controller_type
        self._defaults       = dict(_DEFAULTS[controller_type])
        self._bounds         = _BOUNDS[controller_type]

        # --- Shared state (guarded by _lock) ---
        self._active_params   = dict(self._defaults)
        self._llm_status      = "idle"   # "idle" | "thinking" | "ok" | "error"
        self._llm_reason      = ""
        self._params_updated  = False    # set True by thread; cleared by poll()

        self._lock                      = threading.Lock()
        self._request_in_progress       = False   # True while a background thread is running
        self._last_submitted_request    = ""       # stored the moment a request is sent
        self._last_completed_request    = ""       # stored when the thread exits (ok or error)

        self._ensure_files()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def poll(self):
        """
        Read user_request.txt and react to any change.

        Call once per simulated second from the CARLA control loop.

        Returns:
            Tuple (changed, params, llm_status, llm_reason, request_text):
                changed      – True if Ollama just wrote new params since
                               the previous poll() call.
                params       – Dict of current active parameters.
                llm_status   – "idle" | "thinking" | "ok" | "error"
                llm_reason   – Reason string from the last LLM response.
                request_text – Current (stripped) content of user_request.txt.
        """
        # Harvest results from a finished thread.
        changed = False
        with self._lock:
            if self._params_updated:
                self._params_updated = False
                changed = True
            params = dict(self._active_params)
            status = self._llm_status
            reason = self._llm_reason

        request_text = self._read_request()

        # Submit a new Ollama request only when ALL four conditions hold:
        #   1. File contains a non-empty string.
        #   2. Text differs from the last request we submitted.
        #   3. Text differs from the last request that completed (ok or error).
        #   4. No thread is currently running.
        # Conditions 2 and 3 together ensure a request is never re-submitted
        # after completion even if _request_in_progress has already been cleared.
        if (request_text
                and request_text != self._last_submitted_request
                and request_text != self._last_completed_request
                and not self._request_in_progress):
            self._last_submitted_request = request_text   # store BEFORE thread starts
            self._request_in_progress    = True
            with self._lock:
                self._llm_status = "thinking"
            t = threading.Thread(
                target=self._run_ollama,
                args=(request_text, params),
                daemon=True,
            )
            t.start()
            print("[LLM] New user request detected")
            print("[LLM] Querying Ollama...")

        return changed, params, status, reason, request_text

    @property
    def active_params(self):
        """Thread-safe copy of the current active parameters."""
        with self._lock:
            return dict(self._active_params)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run_ollama(self, request_text, current_params):
        """
        Daemon thread: POST to Ollama, parse the response, and update
        shared state.  Clears _request_in_progress when done regardless of outcome.
        """
        try:
            prompt     = self._build_prompt(request_text, current_params)
            new_params, reason = self._call_ollama(prompt, current_params)

            if new_params is not None:
                clamped = self._clamp(new_params)
                with self._lock:
                    self._active_params  = clamped
                    self._llm_reason     = reason
                    self._llm_status     = "ok"
                    self._params_updated = True
                self._write_params(clamped, reason)
                print("[LLM] Parameters updated: "
                      "speed={:.1f}km/h".format(clamped["target_speed_kmh"]))
                if reason:
                    print("[LLM] Reason: {}".format(reason))
            else:
                with self._lock:
                    self._llm_status = "error"
                print("[LLM] Both GPU and CPU fallback failed; keeping previous parameters.")

        except Exception as exc:
            with self._lock:
                self._llm_status = "error"
            print("[LLM] Unexpected error in background thread: {} — "
                  "keeping previous parameters.".format(exc))
        finally:
            self._last_completed_request = request_text
            self._request_in_progress    = False

    # ------------------------------------------------------------------
    # Ollama HTTP call
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt, current_params):
        """
        Try the GPU model first.  If it returns truncated/malformed JSON,
        retry once with a minimal repair prompt before trying the CPU fallback.

        Returns:
            (params_dict, reason_str) on success from any attempt.
            (None, "")               if all attempts fail.
        """
        params, reason, parse_error = self._try_model(
            _OLLAMA_MODEL_GPU, _GPU_OPTIONS, prompt, current_params)
        if params is not None:
            print("[LLM] Parameters received from GPU model {}".format(
                _OLLAMA_MODEL_GPU))
            return params, reason

        # JSON parse failure only: retry the same GPU model once with a
        # shorter repair prompt before giving up on GPU.
        if parse_error:
            print("[LLM] GPU parse error — retrying with repair prompt ...")
            repair = self._build_repair_prompt(current_params)
            params, reason, _ = self._try_model(
                _OLLAMA_MODEL_GPU, _GPU_OPTIONS, repair, current_params)
            if params is not None:
                print("[LLM] Parameters received from GPU model {} (repair)".format(
                    _OLLAMA_MODEL_GPU))
                return params, reason

        print("[LLM] GPU model failed — trying CPU fallback ({}) ...".format(
            _OLLAMA_MODEL_CPU))
        params, reason, _ = self._try_model(
            _OLLAMA_MODEL_CPU, _CPU_OPTIONS, prompt, current_params)
        if params is not None:
            print("[LLM] Parameters received from CPU fallback model {}".format(
                _OLLAMA_MODEL_CPU))
            return params, reason

        return None, ""

    def _try_model(self, model, options, prompt, current_params):
        """
        POST to the Ollama /api/generate endpoint for a single model attempt.

        Args:
            model:          Ollama model name string.
            options:        Dict of generation options sent in the payload.
            prompt:         Prompt string.
            current_params: Current parameter dict used as fallback for missing keys.

        Returns:
            (params_dict, reason_str, False) on success.
            (None, "", False)                on HTTP or connection error.
            (None, "", True)                 on JSON parse error (signals retry).
        """
        payload = {
            "model":      model,
            "prompt":     prompt,
            "stream":     False,
            "format":     "json",
            "keep_alive": "30m",
            "options":    options,
        }
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            _OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            print("[LLM] HTTP error ({}): {}".format(model, exc))
            return None, "", False
        except Exception as exc:
            print("[LLM] Connection error ({}): {}".format(model, exc))
            return None, "", False

        # Ollama wraps the model output in a top-level "response" key.
        try:
            outer      = json.loads(raw)
            inner_text = outer.get("response", "")
            inner      = json.loads(inner_text)
        except (ValueError, KeyError) as exc:
            print("[LLM] JSON parse error ({}): {}".format(model, exc))
            return None, "", True   # True = parse error, eligible for retry

        # Extract tunable params; fall back to current values for missing keys.
        params = {}
        for key in self._bounds:
            if key in inner:
                try:
                    params[key] = float(inner[key])
                except (TypeError, ValueError):
                    params[key] = current_params.get(key, self._defaults[key])
            else:
                params[key] = current_params.get(key, self._defaults[key])

        reason = str(inner.get("reason", ""))
        return params, reason, False

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, request_text, current_params):
        """Build a compact bounded prompt with JSON template. JSON output only."""
        params_inline = ", ".join(
            "{}={}".format(k, current_params.get(k, self._defaults[k]))
            for k in self._bounds
        )
        bounds_inline = ", ".join(
            "{}={}-{}".format(k, lo, hi)
            for k, (lo, hi) in self._bounds.items()
        )
        template = _TEMPLATES.get(self.controller_type, "")
        return (
            "Tune {ctrl}. JSON only. reason: max 3 words.\n"
            "Current: {params}\n"
            "Request: \"{request}\"\n"
            "Bounds: {bounds}\n"
            "Template: {template}"
        ).format(
            ctrl     = self.controller_type,
            params   = params_inline,
            request  = request_text,
            bounds   = bounds_inline,
            template = template,
        )

    def _build_repair_prompt(self, current_params):
        """Minimal prompt for a single JSON retry after a parse failure."""
        template = _TEMPLATES.get(self.controller_type, "")
        params_inline = ", ".join(
            "{}={}".format(k, current_params.get(k, self._defaults[k]))
            for k in self._bounds
        )
        return (
            "Return valid JSON for {ctrl}. "
            "Use these values: {params}. "
            "Exact structure: {template}"
        ).format(
            ctrl     = self.controller_type,
            params   = params_inline,
            template = template,
        )

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    def _ensure_files(self):
        """Create user_request.txt and live_llm_params.json if absent."""
        if not os.path.isfile(_REQUEST_FILE):
            try:
                with open(_REQUEST_FILE, "w") as fh:
                    fh.write("")
                print("[LLM] Created {}".format(_REQUEST_FILE))
            except IOError as exc:
                print("[LLM] Warning: could not create request file: {}".format(exc))

        if not os.path.isfile(_PARAMS_FILE):
            self._write_params(self._defaults, reason="defaults")

    def _read_request(self):
        """Return stripped content of user_request.txt, or '' on error."""
        try:
            with open(_REQUEST_FILE, "r") as fh:
                return fh.read().strip()
        except IOError:
            return ""

    def _write_params(self, params, reason=""):
        """Write params + reason to live_llm_params.json."""
        doc = dict(params)
        doc["reason"] = reason
        try:
            with open(_PARAMS_FILE, "w") as fh:
                json.dump(doc, fh, indent=2)
        except IOError as exc:
            print("[LLM] Warning: could not write params file: {}".format(exc))

    # ------------------------------------------------------------------
    # Bounds enforcement
    # ------------------------------------------------------------------

    def _clamp(self, params):
        """Return a new dict with all values clipped to their safety bounds."""
        result = {}
        for key, (lo, hi) in self._bounds.items():
            val        = params.get(key, self._defaults[key])
            result[key] = max(lo, min(hi, val))
        return result
