## Description

Please provide a concise description of the changes introduced in this pull request and the problem they solve.

---

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change adding functionality)
- [ ] 💥 Breaking change (fix or feature that alters existing behavior)
- [ ] 📝 Documentation update
- [ ] 🧪 Tests (adding or updating tests)
- [ ] 🔒 Security hardening
- [ ] 🚀 Performance improvement
- [ ] 🔧 CI/CD or build tooling

---

## Related Issues / Context

Closes # (issue number)

---

## Testing Performed

Please describe the tests executed to verify your changes:

- [ ] Python unit tests passed: `python -m pytest tests/ -v --tb=short`
- [ ] Python linting passed: `python -m ruff check neurostage/ backend/ tests/`
- [ ] Frontend linting passed: `cd frontend && npm run lint`
- [ ] Frontend build passed: `cd frontend && npm run build`

---

## Security & Confidentiality Checklist

- [ ] **NO SECRETS**: I confirm that this PR does NOT contain credentials, API keys, tokens, passwords, `.env` files, or private certificates.
- [ ] **NO MASSIVE DATA**: I confirm that no large binary files (`.pt`, `.edf`, `*.db`) are tracked or added.
- [ ] **INPUT SANITIZATION**: Any newly introduced inputs or uploads have appropriate validation and bounds checking.

---

## Contributor Checklist

- [ ] My code adheres to the project's coding standards and style guidelines.
- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] I have updated the relevant documentation (`README.md`, `docs/`, etc.).
- [ ] I have added tests that prove my fix or feature works as expected.
- [ ] All CI/CD checks are passing.
