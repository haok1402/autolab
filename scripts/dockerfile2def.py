#!/usr/bin/env python3
"""Convert an AutoLab task Dockerfile into an Apptainer definition file.

Only the directives AutoLab tasks actually use are supported:
FROM / ENV / RUN / COPY / WORKDIR / ARG / CMD / SHELL.

Build context is staged at /.build-ctx and the instructions are replayed
in order inside %post, so COPY/RUN/WORKDIR ordering matches Docker.
"""
import re
import shlex
import sys
from pathlib import Path


def join_continuations(text: str) -> list[str]:
    lines, buf = [], ""
    for raw in text.splitlines():
        if raw.lstrip().startswith("#") and not buf:
            continue
        if raw.rstrip().endswith("\\"):
            buf += raw.rstrip() + "\n"
            continue
        buf += raw
        if buf.strip():
            lines.append(buf)
        buf = ""
    if buf.strip():
        lines.append(buf)
    return lines


def q(v: str) -> str:
    """Quote a value, but let $VAR references expand like Docker's ENV does."""
    return f'"{v}"' if "$" in v else shlex.quote(v)


def unwrap(s: str) -> str:
    """Collapse backslash-continuations into a single line."""
    return s.replace("\\\n", " ")


def parse_env(rest: str) -> list[tuple[str, str]]:
    rest = unwrap(rest).strip()
    if "=" not in rest.split()[0]:                       # ENV KEY value
        k, _, v = rest.partition(" ")
        return [(k, v.strip())]
    pairs, out = shlex.split(unwrap(rest)), []  # ENV K=V K2=V2
    for p in pairs:
        k, _, v = p.partition("=")
        out.append((k, v))
    return out


def convert(dockerfile: Path) -> str:
    instrs = join_continuations(dockerfile.read_text())
    base, post, envs, args = None, [], [], {}

    for line in instrs:
        m = re.match(r"^\s*([A-Za-z]+)\s+(.*)$", line, re.S)
        if not m:
            continue
        op, rest = m.group(1).upper(), m.group(2)

        if op == "FROM":
            base = rest.split()[0]
        elif op == "ARG":
            k, _, v = rest.strip().partition("=")
            args[k] = v
            post.append(f"export {k}={q(v)}")
        elif op == "ENV":
            for k, v in parse_env(rest):
                envs.append((k, v))
                post.append(f"export {k}={q(v)}")
        elif op == "WORKDIR":
            d = rest.strip()
            post.append(f"mkdir -p {d} && cd {d}")
        elif op == "COPY":
            parts = shlex.split(unwrap(rest))
            parts = [p for p in parts if not p.startswith("--")]
            *srcs, dst = parts
            if dst.endswith("/") or len(srcs) > 1:
                post.append(f"mkdir -p {dst}")
                for s in srcs:
                    post.append(f"cp -a /.build-ctx/{s} {dst}")
            else:
                post.append(f"mkdir -p $(dirname {dst})")
                post.append(f"cp -a /.build-ctx/{srcs[0]} {dst}")
        elif op == "RUN":
            post.append(rest.strip())
        elif op in ("CMD", "ENTRYPOINT", "SHELL", "LABEL", "EXPOSE", "USER"):
            post.append(f"# (ignored) {op} {rest.strip()[:60]}")

    if base is None:
        raise SystemExit(f"no FROM found in {dockerfile}")

    env_block = "\n".join(f"    export {k}={q(v)}" for k, v in envs)
    post_block = "\n".join("    " + l for l in post)
    return f"""Bootstrap: docker
From: {base}

%files
    . /.build-ctx

%post
    set -e
    export DEBIAN_FRONTEND=noninteractive
{post_block}
    rm -rf /.build-ctx
    mkdir -p /logs/verifier /tests

%environment
{env_block}

%runscript
    exec /bin/bash "$@"
"""


if __name__ == "__main__":
    print(convert(Path(sys.argv[1])))
