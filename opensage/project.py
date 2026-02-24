"""
OpenSage Project Context Detector.

Detects the language, frameworks, testing setup, CI/CD configuration,
and overall structure of the current project directory. This context
is injected into every agent task so the AI understands the codebase
before making any changes.

Designed to work on both new projects (minimal structure) and large
legacy codebases.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Map language name -> indicators (file extensions and config file names)
_LANG_INDICATORS: Dict[str, List[str]] = {
    "python":     [".py", "requirements.txt", "Pipfile", "pyproject.toml", "setup.py", "setup.cfg", "tox.ini"],
    "javascript": [".js", ".jsx", "package.json", ".npmrc"],
    "typescript": [".ts", ".tsx", "tsconfig.json"],
    "java":       [".java", "pom.xml", "build.gradle", "build.gradle.kts", ".gradle"],
    "go":         [".go", "go.mod", "go.sum"],
    "rust":       [".rs", "Cargo.toml", "Cargo.lock"],
    "ruby":       [".rb", "Gemfile", "Gemfile.lock", ".ruby-version"],
    "cpp":        [".cpp", ".cc", ".cxx", ".hxx", ".hpp"],
    "c":          [".c"],
    "csharp":     [".cs", ".csproj", ".sln"],
    "php":        [".php", "composer.json", "composer.lock"],
    "swift":      [".swift", "Package.swift"],
    "kotlin":     [".kt", ".kts"],
    "scala":      [".scala", "build.sbt"],
    "elixir":     [".ex", ".exs", "mix.exs"],
    "haskell":    [".hs", "stack.yaml", "cabal.project"],
}

# Directories that are not source code and should be skipped when walking
_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".venv", "venv", "env", ".env",
    "dist", "build", "target", "out", ".cache",
    ".idea", ".vscode", ".eclipse",
    "vendor", "third_party", "deps",
}

# Framework detection: framework name -> list of marker strings to look for
# in file names (relative paths) or package dependency files
_FRAMEWORK_MARKERS: Dict[str, List[str]] = {
    # Python
    "Django":        ["manage.py", "django"],
    "Flask":         ["flask"],
    "FastAPI":       ["fastapi"],
    "SQLAlchemy":    ["sqlalchemy"],
    "Celery":        ["celery"],
    "Pydantic":      ["pydantic"],
    "Pytest":        ["conftest.py", "pytest"],
    # JavaScript / TypeScript
    "React":         ["react", "react-dom"],
    "Vue":           ["vue", "@vue"],
    "Angular":       ["@angular/core"],
    "Next.js":       ["next.config", "next"],
    "Express":       ["express"],
    "NestJS":        ["@nestjs/core"],
    "Fastify":       ["fastify"],
    # Java / Kotlin
    "Spring Boot":   ["spring-boot", "springframework"],
    "Quarkus":       ["quarkus"],
    "Micronaut":     ["micronaut"],
    # Ruby
    "Rails":         ["rails", "actionpack"],
    "Sinatra":       ["sinatra"],
    # Go
    "Gin":           ["github.com/gin-gonic/gin"],
    "Echo":          ["github.com/labstack/echo"],
    "Fiber":         ["github.com/gofiber/fiber"],
    # Rust
    "Actix":         ["actix-web"],
    "Axum":          ["axum"],
    # Infrastructure / cloud
    "Docker":        ["Dockerfile", "docker-compose"],
    "Kubernetes":    ["kubernetes", "k8s", ".yaml"],
    "Terraform":     [".tf", "terraform"],
}

# CI/CD system detection: display name -> marker path or file
_CI_MARKERS = {
    "GitHub Actions":  ".github/workflows",
    "GitLab CI":       ".gitlab-ci.yml",
    "Jenkins":         "Jenkinsfile",
    "CircleCI":        ".circleci/config.yml",
    "Azure Pipelines": "azure-pipelines.yml",
    "Travis CI":       ".travis.yml",
    "Bitbucket Pipelines": "bitbucket-pipelines.yml",
    "Drone CI":        ".drone.yml",
    "Buildkite":       ".buildkite/pipeline.yml",
    "TeamCity":        ".teamcity",
}

# Package manager detection: display name -> marker file
_PKG_MANAGERS = {
    "pip":            "requirements.txt",
    "pipenv":         "Pipfile",
    "poetry":         "pyproject.toml",
    "npm":            "package-lock.json",
    "yarn":           "yarn.lock",
    "pnpm":           "pnpm-lock.yaml",
    "maven":          "pom.xml",
    "gradle":         "build.gradle",
    "cargo":          "Cargo.toml",
    "go modules":     "go.mod",
    "bundler":        "Gemfile.lock",
    "composer":       "composer.json",
    "mix":            "mix.exs",
    "stack":          "stack.yaml",
}


def detect_project(work_dir: str) -> Dict:
    """
    Analyse a project directory and return a structured info dict.

    Returns:
        dict with keys:
            root                - absolute path
            primary_language    - most prevalent language
            languages           - all detected languages, sorted by prevalence
            frameworks          - list of detected frameworks
            package_manager     - detected package manager name
            has_git             - bool
            git_branch          - current branch or None
            git_uncommitted     - number of uncommitted changes
            git_remote          - remote origin URL or None
            has_ci              - bool
            ci_system           - detected CI name or None
            entry_points        - list of likely entry point files
            test_dirs           - list of detected test directories
            source_dirs         - list of likely source directories
            total_files         - total non-skipped file count
            approximate_loc     - rough line count (first 10k files)
    """
    root = Path(work_dir).resolve()

    result: Dict = {
        "root": str(root),
        "primary_language": None,
        "languages": [],
        "frameworks": [],
        "package_manager": None,
        "has_git": False,
        "git_branch": None,
        "git_uncommitted": 0,
        "git_remote": None,
        "has_ci": False,
        "ci_system": None,
        "entry_points": [],
        "test_dirs": [],
        "source_dirs": [],
        "total_files": 0,
        "approximate_loc": 0,
    }

    if not root.is_dir():
        return result

    # ---- Git info --------------------------------------------------------
    if (root / ".git").is_dir():
        result["has_git"] = True
        try:
            result["git_branch"] = _git(
                ["rev-parse", "--abbrev-ref", "HEAD"], root
            )
        except Exception:
            pass
        try:
            changed = _git(["status", "--porcelain"], root)
            result["git_uncommitted"] = len(changed.splitlines()) if changed else 0
        except Exception:
            pass
        try:
            result["git_remote"] = _git(
                ["remote", "get-url", "origin"], root
            )
        except Exception:
            pass

    # ---- Walk the tree ---------------------------------------------------
    ext_counts: Dict[str, int] = {}
    file_count = 0
    loc = 0
    all_rel_paths: List[str] = []

    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            file_count += 1
            ext = Path(fname).suffix.lower()
            if ext:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

            rel = os.path.relpath(os.path.join(dirpath, fname), str(root))
            all_rel_paths.append(rel)

            # Approximate LOC for source files (cap at 10k files for speed)
            if file_count <= 10_000 and ext in (
                ".py", ".js", ".ts", ".java", ".go", ".rs", ".rb",
                ".c", ".cpp", ".cs", ".php", ".swift", ".kt",
            ):
                try:
                    with open(os.path.join(dirpath, fname), errors="replace") as f:
                        loc += sum(1 for _ in f)
                except (PermissionError, OSError):
                    pass

    result["total_files"] = file_count
    result["approximate_loc"] = loc

    # Flatten relative paths to a single string for quick marker scans
    all_paths_str = "\n".join(all_rel_paths)

    # ---- Language detection ----------------------------------------------
    lang_scores: Dict[str, int] = {}
    for lang, indicators in _LANG_INDICATORS.items():
        score = 0
        for indicator in indicators:
            if indicator.startswith("."):
                # File extension: weight by count
                score += ext_counts.get(indicator, 0) * 2
            else:
                # Config file: strong signal
                if (root / indicator).exists():
                    score += 15
                elif indicator in all_paths_str:
                    score += 5
        if score > 0:
            lang_scores[lang] = score

    sorted_langs = sorted(lang_scores.items(), key=lambda x: x[1], reverse=True)
    result["languages"] = [lang for lang, _ in sorted_langs]
    result["primary_language"] = sorted_langs[0][0] if sorted_langs else None

    # ---- Framework detection ---------------------------------------------
    # Read key dependency files for deep framework matching
    dep_content = _read_dep_files(root)
    detected_frameworks = []

    for framework, markers in _FRAMEWORK_MARKERS.items():
        for marker in markers:
            if (root / marker).exists() or marker in all_paths_str or marker in dep_content:
                detected_frameworks.append(framework)
                break

    result["frameworks"] = detected_frameworks

    # ---- Package manager -------------------------------------------------
    for mgr_name, marker_file in _PKG_MANAGERS.items():
        if (root / marker_file).exists():
            result["package_manager"] = mgr_name
            break

    # ---- CI/CD detection -------------------------------------------------
    for ci_name, ci_path in _CI_MARKERS.items():
        if (root / ci_path).exists():
            result["has_ci"] = True
            result["ci_system"] = ci_name
            break

    # ---- Entry points ----------------------------------------------------
    entry_candidates = [
        "main.py", "app.py", "server.py", "wsgi.py", "asgi.py",
        "manage.py", "index.py", "run.py", "start.py",
        "index.js", "app.js", "server.js", "main.js",
        "index.ts", "app.ts", "server.ts", "main.ts",
        "main.go", "cmd/main.go",
        "src/main.rs",
        "Main.java", "Application.java",
        "Program.cs",
    ]
    result["entry_points"] = [ep for ep in entry_candidates if (root / ep).exists()]

    # ---- Test directories ------------------------------------------------
    test_candidates = ["tests", "test", "spec", "__tests__", "test_suite", "integration_tests"]
    result["test_dirs"] = [
        str(Path(d).relative_to(root)) if Path(d).is_absolute() else d
        for d in test_candidates
        if (root / d).is_dir()
    ]

    # ---- Source directories ----------------------------------------------
    src_candidates = ["src", "lib", "app", "pkg", "internal", "core", "source"]
    result["source_dirs"] = [d for d in src_candidates if (root / d).is_dir()]

    return result


def format_project_summary(info: Dict) -> str:
    """
    Format project detection results as a concise human-readable string
    for display in the CLI and for injection into agent prompts.
    """
    lines = []

    if info.get("primary_language"):
        lang_str = info["primary_language"]
        others = [l for l in info.get("languages", []) if l != lang_str]
        if others:
            lang_str += f" (+ {', '.join(others[:3])})"
        lines.append(f"Language: {lang_str}")

    if info.get("frameworks"):
        lines.append(f"Frameworks: {', '.join(info['frameworks'][:5])}")

    if info.get("package_manager"):
        lines.append(f"Package manager: {info['package_manager']}")

    if info.get("has_git"):
        branch = info.get("git_branch") or "unknown"
        git_str = f"Git: branch={branch}"
        n = info.get("git_uncommitted", 0)
        if n:
            git_str += f", {n} uncommitted file(s)"
        remote = info.get("git_remote")
        if remote:
            git_str += f", remote={remote}"
        lines.append(git_str)

    if info.get("has_ci") and info.get("ci_system"):
        lines.append(f"CI/CD: {info['ci_system']}")

    if info.get("test_dirs"):
        lines.append(f"Tests: {', '.join(info['test_dirs'][:4])}")

    if info.get("entry_points"):
        lines.append(f"Entry points: {', '.join(info['entry_points'][:4])}")

    if info.get("source_dirs"):
        lines.append(f"Source dirs: {', '.join(info['source_dirs'][:4])}")

    stats = []
    if info.get("total_files"):
        stats.append(f"{info['total_files']} files")
    if info.get("approximate_loc"):
        loc = info["approximate_loc"]
        stats.append(f"~{loc:,} lines of code")
    if stats:
        lines.append(f"Size: {', '.join(stats)}")

    return "\n".join(lines)


def build_agent_context(info: Dict, extra: str = "") -> str:
    """
    Build a context string suitable for injecting into agent task prompts.
    Provides the agent with project awareness before it starts working.
    """
    parts = []

    summary = format_project_summary(info)
    if summary:
        parts.append(f"=== Project Context ===\nRoot: {info['root']}\n{summary}")

    if extra:
        parts.append(extra)

    return "\n\n".join(parts)


# ---- Helpers -------------------------------------------------------------

def _git(cmd: List[str], cwd: Path) -> str:
    """Run a git command and return stripped stdout. Raises on failure."""
    out = subprocess.check_output(
        ["git"] + cmd,
        cwd=str(cwd),
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    return out.decode("utf-8", errors="replace").strip()


def _read_dep_files(root: Path) -> str:
    """
    Read key dependency manifest files and return their content concatenated.
    Used for framework detection via string matching.
    """
    dep_files = [
        "requirements.txt", "Pipfile", "pyproject.toml", "setup.cfg",
        "package.json", "yarn.lock",
        "pom.xml", "build.gradle",
        "go.mod",
        "Cargo.toml",
        "Gemfile",
        "composer.json",
    ]
    content_parts = []
    for fname in dep_files:
        fpath = root / fname
        if fpath.is_file():
            try:
                # Read first 8 KB — enough for dependency declarations
                content_parts.append(fpath.read_text(errors="replace")[:8192])
            except (PermissionError, OSError):
                pass
    return "\n".join(content_parts)
