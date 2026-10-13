# Contributing to NeuroStage

Thank you for your interest in contributing to **NeuroStage**! We welcome contributions from researchers, machine learning engineers, full-stack developers, and open-source enthusiasts.

Please review this guide before submitting issues or pull requests.

---

## Code of Conduct

All contributors and maintainers are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to `pankajgovekar52@gmail.com`.

---

## Getting Started

### 1. Prerequisites
- **Python**: 3.10 or 3.11+ (tested on 3.11 and 3.13)
- **Node.js**: 20+ and npm 10+
- **Git**: 2.30+
- **Docker** & Docker Compose (optional for containerized testing)

### 2. Fork and Clone
```bash
git clone https://github.com/pankajcseaiml/NeuroStage.git
cd NeuroStage
```

### 3. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install package in editable mode with development dependencies
pip install -e ".[backend,dev]"

# Setup environment file
cp .env.example .env
```

### 4. Frontend Setup
```bash
cd frontend
npm ci
cp .env.example .env.local
cd ..
```

---

## Development Workflow

### Branching Strategy
- **Default branch**: `main` (production-ready, protected branch).
- **Feature branches**: Create descriptive branches from `main`:
  - `feat/depthwise-separable-v2`
  - `fix/eeg-epoch-stride-boundary`
  - `docs/api-specification-update`

### Commit Message Conventions
We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|---|---|
| `feat:` | A new feature or model component |
| `fix:` | A bug fix in ML pipeline, backend, or UI |
| `docs:` | Documentation changes only |
| `test:` | Adding or updating tests |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `ci:` | Changes to CI/CD workflows or configurations |
| `chore:` | Maintenance tasks, dependency bumps |

Example:
```bash
git commit -m "feat(models): add 1D depthwise separable convolution block"
```

---

## Code Quality & Testing Standards

Before opening a pull request, verify that all local checks pass:

### 1. Python Code Quality
```bash
# Run Ruff linter
python -m ruff check neurostage/ backend/ tests/ train.py evaluate.py compress.py

# Run test suite
python -m pytest tests/ -v --tb=short
```

### 2. Frontend Code Quality
```bash
cd frontend
# Run ESLint
npm run lint

# Verify production build succeeds
npm run build
cd ..
```

---

## Pull Request Guidelines

1. **Keep PRs Focused**: Address a single feature or bug per pull request.
2. **Include Tests**: Add unit or integration tests for any new or modified functionality.
3. **No Secrets**: Double-check that no `.env` files, API keys, credentials, or private data are included.
4. **Use PR Template**: Complete the questions and checklist in the [Pull Request Template](.github/pull_request_template.md).
5. **CI Must Pass**: Ensure all automated checks (Backend CI, Frontend CI, Secret Scan) turn green before requesting review.

---

## Security & Sensitive Data Rules

- **Never commit real data or checkpoints**: Large `.pt` files, raw `.edf` recordings, or SQLite databases (`*.db`) belong in `.gitignore`.
- **Synthetic Test Data**: If adding tests, use mock tensors (`torch.zeros(...)`) or generated dummy signals.
- Review our full [Security Policy](SECURITY.md) for vulnerability reporting.
