#!/usr/bin/env python3
"""Create GitHub repositories and push both projects.

Usage:
    python3 push_to_github.py <GITHUB_PAT>

Requires a GitHub Personal Access Token (classic) with 'repo' scope.
Visit https://github.com/settings/tokens/new to create one.

Creates:
  - github.com/SaranshDhage/harness-engg-phase1   (Harness_Engg-1)
  - github.com/SaranshDhage/adf-study             (this repo)
"""
import subprocess, sys, os
from pathlib import Path

def run(cmd, cwd=None, check=True):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"ERROR: {cmd}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

def create_and_push(token: str, owner: str, repo_name: str,
                     local_path: str, description: str) -> None:
    import urllib.request, json

    print(f"\n{'='*60}")
    print(f"  Creating {owner}/{repo_name}")

    # Create repo via GitHub API
    payload = json.dumps({
        "name": repo_name,
        "description": description,
        "private": False,
        "auto_init": False,
    }).encode()
    req = urllib.request.Request(
        "https://api.github.com/user/repos",
        data=payload,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            remote_url = data["clone_url"]
            print(f"  Created: {data['html_url']}")
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        if "already exists" in body.get("message", ""):
            remote_url = f"https://github.com/{owner}/{repo_name}.git"
            print(f"  Repo already exists: {remote_url}")
        else:
            print(f"  ERROR: {body}")
            return

    # Set remote and push
    auth_url = remote_url.replace("https://", f"https://{token}@")
    run(f"git remote remove origin 2>/dev/null; git remote add origin {auth_url}",
        cwd=local_path, check=False)
    run(f"git push -u origin master --force", cwd=local_path)
    print(f"  Pushed to {remote_url}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    token = sys.argv[1]
    owner = "SaranshDhage"
    root  = Path(__file__).resolve().parent

    create_and_push(
        token, owner, "harness-engg-phase1",
        str(root.parent / "Harness_Engg-1"),
        "Harness Engineering Phase 1: FSM + Structured Planning + Validation Gate "
        "(arXiv:2608.26197). Baseline for the ADF study.",
    )
    create_and_push(
        token, owner, "adf-study",
        str(root),
        "Agent Degrees of Freedom (ADF): execution determinism as a predictable "
        "function of harness constraint. Multi-session research project.",
    )
    print("\nDone. Both repositories are now on GitHub.")
