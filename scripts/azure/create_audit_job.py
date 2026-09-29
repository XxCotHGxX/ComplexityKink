"""Create (or update) an Azure Container Apps Job that runs the independent audit.

Secrets (auditor API key, ACR password) are fetched from the Azure CLI at run
time, written to a temporary YAML only for the duration of the `az` call, and
stored as Container Apps secrets. Nothing secret is committed.

Usage:
    python scripts/azure/create_audit_job.py --job ckr-audit-primary \
        --auditor-account DataPipeline0 --deployment Phi-4-reasoning \
        --input audit_input.jsonl --output audit_primary_phi4_reasoning.jsonl --workers 200
Start / monitor:
    az containerapp job start -g ComplexityKinkResearch -n ckr-audit-primary
    az containerapp job execution list -g ComplexityKinkResearch -n ckr-audit-primary -o table
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

RG = "ComplexityKinkResearch"
ENV = "managedEnvironment-GPU"
ACR = "qwen359b"
STORAGE_NAME = "ckrcameraready"


def az(*args: str) -> str:
    return subprocess.run(["az", *args], check=True, capture_output=True, text=True,
                          shell=os.name == "nt").stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--auditor-account", required=True)
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--image-tag", default="1")
    ap.add_argument("--replica-timeout", type=int, default=86400)
    args = ap.parse_args()

    env_id = az("containerapp", "env", "show", "-g", RG, "-n", ENV, "--query", "id", "-o", "tsv")
    location = az("containerapp", "env", "show", "-g", RG, "-n", ENV, "--query", "location", "-o", "tsv")
    endpoint = az("cognitiveservices", "account", "show", "-g", RG, "-n", args.auditor_account,
                  "--query", "properties.endpoint", "-o", "tsv").rstrip("/")
    auditor_key = az("cognitiveservices", "account", "keys", "list", "-g", RG, "-n", args.auditor_account,
                     "--query", "key1", "-o", "tsv")
    acr_password = az("acr", "credential", "show", "-n", ACR, "--query", "passwords[0].value", "-o", "tsv")

    spec = {
        "location": location,
        "properties": {
            "environmentId": env_id,
            "workloadProfileName": "Consumption",
            "configuration": {
                "triggerType": "Manual",
                "replicaTimeout": args.replica_timeout,
                "replicaRetryLimit": 1,
                "manualTriggerConfig": {"parallelism": 1, "replicaCompletionCount": 1},
                "registries": [{"server": f"{ACR}.azurecr.io", "username": ACR,
                                "passwordSecretRef": "acr-password"}],
                "secrets": [{"name": "acr-password", "value": acr_password},
                            {"name": "auditor-key", "value": auditor_key}],
            },
            "template": {
                "containers": [{
                    "name": "audit",
                    "image": f"{ACR}.azurecr.io/ckr-audit:{args.image_tag}",
                    "resources": {"cpu": 2.0, "memory": "4Gi"},
                    "env": [
                        {"name": "AUDIT_DEPLOYMENT", "value": args.deployment},
                        {"name": "AUDIT_INPUT", "value": args.input},
                        {"name": "AUDIT_OUTPUT", "value": args.output},
                        {"name": "AUDIT_WORKERS", "value": str(args.workers)},
                        {"name": "AUDIT_ENDPOINT", "value": endpoint},
                        {"name": "AUDIT_API_KEY", "secretRef": "auditor-key"},
                    ],
                    "volumeMounts": [{"volumeName": "ckr", "mountPath": "/mnt/ckr"}],
                }],
                "volumes": [{"name": "ckr", "storageType": "AzureFile", "storageName": STORAGE_NAME}],
            },
        },
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(spec, f)
        exists = subprocess.run(["az", "containerapp", "job", "show", "-g", RG, "-n", args.job, "-o", "none"],
                                capture_output=True, shell=os.name == "nt").returncode == 0
        verb = "update" if exists else "create"
        az("containerapp", "job", verb, "-g", RG, "-n", args.job, "--yaml", path, "-o", "none")
        print(f"{verb}d job {args.job}: {args.deployment} on {args.input} -> {args.output}")
    finally:
        os.remove(path)


if __name__ == "__main__":
    main()
