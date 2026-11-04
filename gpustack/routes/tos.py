"""TOS ingest API and the in-product TOS deploy form."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from gpustack.api.exceptions import InvalidException
from gpustack.schemas.models import ModelSource, SourceEnum
from gpustack.worker import tos_storage

router = APIRouter()
ui_router = APIRouter()

TOS_SOURCE_FIELDS = [
    {
        "name": "tos_endpoint",
        "label": "TOS Endpoint",
        "required": True,
        "placeholder": "https://tos-cn-shanghai.volces.com",
    },
    {
        "name": "tos_region",
        "label": "Region",
        "required": True,
        "placeholder": "cn-shanghai",
    },
    {
        "name": "tos_bucket",
        "label": "Bucket",
        "required": True,
        "placeholder": "your-bucket",
    },
    {
        "name": "tos_model",
        "label": "Model id",
        "required": True,
        "placeholder": "qwen3",
    },
    {
        "name": "tos_version",
        "label": "Version",
        "required": True,
        "placeholder": "v1",
    },
    {
        "name": "tos_release_path",
        "label": "Release path",
        "required": False,
        "placeholder": "lingji-models/qwen3/v1",
    },
]


class TosIngestRequest(BaseModel):
    local_path: str = Field(..., min_length=1)
    tos_endpoint: str
    tos_region: str
    tos_bucket: str
    tos_model: str
    tos_version: str
    tos_release_path: Optional[str] = None


class TosIngestResponse(BaseModel):
    source: SourceEnum = SourceEnum.TOS
    tos_endpoint: str
    tos_region: str
    tos_bucket: str
    tos_model: str
    tos_version: str
    tos_release_path: str
    uri: str
    manifest_key: str
    size: int
    files: List[str]
    model_create: Dict[str, Any]


def model_sources() -> List[Dict[str, Any]]:
    return [
        {
            "source": SourceEnum.HUGGING_FACE.value,
            "label": "Hugging Face",
            "ingest": False,
        },
        {
            "source": SourceEnum.MODEL_SCOPE.value,
            "label": "ModelScope",
            "ingest": False,
        },
        {
            "source": SourceEnum.LOCAL_PATH.value,
            "label": "Local Path",
            "ingest": False,
        },
        {
            "source": SourceEnum.TOS.value,
            "label": "TOS",
            "ingest": True,
            "fields": TOS_SOURCE_FIELDS,
            "deploy_path": "/tos-deploy",
        },
    ]


def _location(req: TosIngestRequest) -> tos_storage.TosLocation:
    source = ModelSource(
        source=SourceEnum.TOS,
        tos_endpoint=req.tos_endpoint,
        tos_region=req.tos_region,
        tos_bucket=req.tos_bucket,
        tos_model=req.tos_model,
        tos_version=req.tos_version,
        tos_release_path=req.tos_release_path,
    )
    return tos_storage.location_from_source(source)


def _source_options_html() -> str:
    options = []
    for item in model_sources():
        selected = " selected" if item["source"] == SourceEnum.TOS.value else ""
        options.append(
            f'<option value="{item["source"]}"{selected}>{item["label"]}</option>'
        )
    return "\n      ".join(options)


def _tos_fields_html() -> str:
    parts = []
    for field in TOS_SOURCE_FIELDS:
        required = " *" if field["required"] else ""
        placeholder = field.get("placeholder") or ""
        parts.append(
            f'<label>{field["label"]}{required}</label>'
            f'<input data-tos="{field["name"]}" placeholder="{placeholder}"/>'
        )
    return "\n  ".join(parts)


def ingest_release(
    req: TosIngestRequest,
    client: Optional[tos_storage.TosObjectClient] = None,
) -> TosIngestResponse:
    location = _location(req)
    local = Path(req.local_path)
    if not local.is_dir():
        raise InvalidException(message="local_path must be an existing directory")
    if client is None:
        client = tos_storage.SdkTosClient(location.endpoint, location.region)
    try:
        manifest = tos_storage.upload_release(location, local, client)
    except tos_storage.TosStorageError as exc:
        raise InvalidException(message=str(exc)) from exc
    model_create = {
        "source": SourceEnum.TOS.value,
        "tos_endpoint": location.endpoint,
        "tos_region": location.region,
        "tos_bucket": location.bucket,
        "tos_model": location.model,
        "tos_version": location.version,
        "tos_release_path": location.release_path,
    }
    return TosIngestResponse(
        tos_endpoint=location.endpoint,
        tos_region=location.region,
        tos_bucket=location.bucket,
        tos_model=location.model,
        tos_version=location.version,
        tos_release_path=location.release_path,
        uri=location.uri,
        manifest_key=location.manifest_key,
        size=tos_storage.manifest_total_size(manifest),
        files=[entry["path"] for entry in manifest["files"]],
        model_create=model_create,
    )


@router.get("/sources")
async def get_model_sources():
    """Source dropdown metadata for the deploy form, including TOS fields."""
    return {"items": model_sources()}


@router.post("/releases", response_model=TosIngestResponse)
async def create_tos_release(req: TosIngestRequest):
    """Publish a local directory (or on-disk hub snapshot) as an immutable TOS release."""
    return await asyncio.to_thread(ingest_release, req)


@ui_router.get("/tos-deploy", include_in_schema=False)
async def tos_deploy_page():
    return HTMLResponse(_tos_deploy_html())


def _tos_deploy_html() -> str:
    return _TOS_DEPLOY_HTML.replace(
        "__SOURCE_OPTIONS__",
        _source_options_html(),
    ).replace(
        "__TOS_FIELDS__",
        _tos_fields_html(),
    )


_TOS_DEPLOY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>GPUStack TOS deploy</title>
  <style>
    :root { font-family: ui-sans-serif, system-ui, sans-serif; color: #111; }
    body { max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.4rem; }
    label { display: block; margin: 0.75rem 0 0.25rem; font-weight: 600; }
    input, select, textarea { width: 100%; box-sizing: border-box; padding: 0.45rem; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
    button { margin-top: 1rem; margin-right: 0.5rem; padding: 0.5rem 0.9rem; }
    pre { background: #f4f4f5; padding: 0.75rem; overflow: auto; }
    .note { color: #444; font-size: 0.95rem; }
    .err { color: #b00020; }
  </style>
</head>
<body>
  <h1>TOS model source</h1>
  <p class="note">
    Preferred path: ingest weights into TOS from a directory on this server
    (local tree or an on-disk HF/ModelScope snapshot), then create a deployment
    that selects worker and GPUs. The assigned worker pulls and verifies the
    release. Hugging Face, ModelScope, and local_path stay on the main Deploy
    dropdown. Credentials are the server/worker <code>TOS_*</code> env vars.
  </p>
  <label>Source</label>
  <select id="source">
      __SOURCE_OPTIONS__
  </select>
  <div id="tos-fields">
  __TOS_FIELDS__
  </div>
  <label>Local directory on this server</label>
  <input id="local_path" placeholder="/data/snapshots/qwen3"/>
  <label>Deployment name (optional, for create)</label>
  <input id="name" placeholder="qwen3-tos"/>
  <label>GPU IDs (one per line, e.g. worker-1:cuda:0)</label>
  <textarea id="gpu_ids" rows="3"></textarea>
  <label>API token</label>
  <input id="token" type="password" autocomplete="off"/>
  <button type="button" id="ingest">Ingest to TOS</button>
  <button type="button" id="create">Create TOS deployment</button>
  <pre id="out"></pre>
  <script>
    const API = "/v2";
    const out = document.getElementById("out");
    function token() { return document.getElementById("token").value.trim(); }
    function headers() {
      const h = {"Content-Type": "application/json"};
      if (token()) h["Authorization"] = "Bearer " + token();
      return h;
    }
    function tosBody() {
      const body = {local_path: document.getElementById("local_path").value.trim()};
      document.querySelectorAll("[data-tos]").forEach((el) => {
        if (el.value.trim()) body[el.dataset.tos] = el.value.trim();
      });
      return body;
    }
    function fillSources(items) {
      const sel = document.getElementById("source");
      const current = sel.value || "tos";
      sel.innerHTML = "";
      (items || []).forEach((item) => {
        const opt = document.createElement("option");
        opt.value = item.source;
        opt.textContent = item.label;
        if (item.source === current) opt.selected = true;
        sel.appendChild(opt);
      });
      const tos = (items || []).find((i) => i.source === "tos") || {fields: []};
      const box = document.getElementById("tos-fields");
      const previous = {};
      document.querySelectorAll("[data-tos]").forEach((el) => {
        previous[el.dataset.tos] = el.value;
      });
      box.innerHTML = "";
      (tos.fields || []).forEach((f) => {
        const lab = document.createElement("label");
        lab.textContent = f.label + (f.required ? " *" : "");
        const input = document.createElement("input");
        input.dataset.tos = f.name;
        input.placeholder = f.placeholder || "";
        if (previous[f.name]) input.value = previous[f.name];
        box.appendChild(lab);
        box.appendChild(input);
      });
    }
    async function loadSources() {
      const res = await fetch(API + "/tos/sources", {headers: headers()});
      if (!res.ok) return;
      const data = await res.json();
      fillSources(data.items || []);
    }
    async function ingest() {
      out.textContent = "ingesting…";
      const res = await fetch(API + "/tos/releases", {
        method: "POST", headers: headers(), body: JSON.stringify(tosBody())
      });
      const data = await res.json();
      out.textContent = JSON.stringify(data, null, 2);
      if (!res.ok) throw new Error(data.message || res.statusText);
      return data;
    }
    function modelCreateFromForm() {
      const body = tosBody();
      return {
        source: "tos",
        tos_endpoint: body.tos_endpoint,
        tos_region: body.tos_region,
        tos_bucket: body.tos_bucket,
        tos_model: body.tos_model,
        tos_version: body.tos_version,
        tos_release_path: body.tos_release_path || undefined,
      };
    }
    document.getElementById("ingest").onclick = () => ingest().catch((e) => {
      out.className = "err"; out.textContent = String(e);
    });
    document.getElementById("create").onclick = async () => {
      try {
        out.className = "";
        const source = document.getElementById("source").value;
        if (source && source !== "tos") {
          throw new Error(
            "Hugging Face, ModelScope, and local_path stay on the main Deploy form"
          );
        }
        const local = document.getElementById("local_path").value.trim();
        const published = local
          ? await ingest()
          : {model_create: modelCreateFromForm(), tos_model: tosBody().tos_model};
        const gpuIds = document.getElementById("gpu_ids").value
          .split(/[\\s,]+/).filter(Boolean);
        const payload = Object.assign({
          name: document.getElementById("name").value.trim() || published.tos_model,
          replicas: 1,
          backend: "vLLM",
        }, published.model_create);
        if (gpuIds.length) payload.gpu_selector = {gpu_ids: gpuIds};
        const res = await fetch(API + "/models", {
          method: "POST", headers: headers(), body: JSON.stringify(payload)
        });
        const data = await res.json();
        out.textContent = JSON.stringify(data, null, 2);
        if (!res.ok) throw new Error(data.message || res.statusText);
      } catch (e) {
        out.className = "err"; out.textContent = String(e);
      }
    };
    document.getElementById("token").addEventListener("change", () => {
      loadSources().catch(() => {});
    });
    loadSources().catch(() => {});
  </script>
</body>
</html>
"""
