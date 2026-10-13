# Security Policy

The NeuroStage engineering and research team takes the security of this system seriously. This document outlines our security model, reporting guidelines, supported versions, and secret handling protocols.

---

## Supported Versions

Only the latest release and the current active development branch receive security updates and patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## Reporting a Vulnerability

> [!CAUTION]
> **DO NOT** publicly disclose security vulnerabilities, active exploits, credentials, or private keys in public GitHub Issues, Discussions, or Pull Requests.

If you discover a security vulnerability in NeuroStage, please report it responsibly using one of the following channels:

1. **GitHub Private Vulnerability Reporting (Preferred)**:
   Navigate to the repository's [Security Advisory tab](https://github.com/pankajcseaiml/NeuroStage/security/advisories/new) and submit a private report.
2. **Direct Maintainer Contact**:
   Email the lead maintainer at `pankajgovekar52@gmail.com` with the subject line `[SECURITY] NeuroStage Vulnerability Report`.

### What to Include in Your Report
To help us triage and resolve the issue quickly, please provide:
- A clear description of the vulnerability and its potential impact.
- Step-by-step reproduction instructions or a minimal Proof-of-Concept (PoC).
- Affected components (e.g., FastAPI backend routes, ML model deserialization, JWT auth, Docker image).
- Any proposed remediation or mitigation.

### Response Timeline
- **Initial Acknowledgment**: Within 48 hours of receiving the report.
- **Triage & Validation**: Within 5 business days.
- **Remediation & Patch**: Target fix released within 14 days, followed by a coordinated security advisory.

---

## Security Model & Zero Trust Architecture

### 1. Secret Management
- **No Hardcoded Secrets**: Secrets (JWT secret keys, database credentials, API tokens) must never be committed to Git.
- **Environment Separation**: Local development utilizes `.env` (copied from `.env.example`). Production deployments must supply credentials via secure environment managers (e.g., Docker secrets, Kubernetes secrets, or GitHub Actions Secrets).
- **Automated Scanning**: The repository executes automated secret leak scans on every pull request and push to `main`.

### 2. Machine Learning & Model Security
- **Artifact Deserialization**: PyTorch model checkpoints are loaded with `map_location='cpu'` and validate state dictionary keys. Unpickling untrusted model files is prohibited.
- **Input Validation**: EEG signal uploads (EDF files) are strictly validated for channel schema, sampling frequency, and maximum file size (default 500 MB) before processing.
- **Resource Constraints**: Epoch preprocessing and model inference execute with bounded batch sizes to prevent memory exhaustion and Denial-of-Service (DoS).

### 3. Backend & API Hardening
- **Authentication**: Stateless JWT authentication with Bcrypt password hashing.
- **Least Privilege Execution**: Docker containers run as non-root users (`appuser` in backend, `nextjs` in frontend).
- **CORS Configuration**: Restrict allowed origins to designated domains in production deployments.

---

## Credential Leak Incident Response Protocol

If credentials, API tokens, or secrets are accidentally committed or leaked:
1. **Immediate Revocation**: Revoke and rotate the exposed token/key immediately at the provider/database.
2. **History Cleansing**: Purge the sensitive commit using `git-filter-repo` or BFG Repo-Cleaner. Never rely solely on a follow-up commit to "delete" a secret.
3. **Audit Access Logs**: Review backend and database logs for unauthorized access during the exposure window.
4. **Notify Maintainers**: Alert repository owners to monitor for anomalous activity.

---

## Disclaimer

NeuroStage is a research and educational prototype developed for automated EEG sleep staging analysis. It is **not clinically certified** as a medical diagnostic device under FDA, CE-MDR, or CDSCO standards. Never make clinical diagnosis or treatment decisions based on NeuroStage outputs.
