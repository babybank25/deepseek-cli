# Security Policy

## Supported Versions

Security fixes are provided for the latest `main` branch only.

## Reporting a Vulnerability

Please do not open public issues for sensitive security reports.

Send a private report with:
- Impact summary
- Reproduction steps
- Affected files/endpoints
- Suggested mitigation (if known)

If you cannot use private channels yet, open a GitHub issue with only
"security report requested" and no exploit details, then wait for maintainer
contact.

## Sensitive Data Handling

This project can handle auth tokens and cookies during discovery/runtime.

Never commit:
- Personal auth tokens
- Browser cookies
- Runtime config/state under local home directories (for example `~/.deepseek_cli/`)
- SSE debug dumps and logs that may contain user prompts or model output

Before pushing:
1. Run tests.
2. Search for accidental secrets in tracked files.
3. Review `git diff` for auth headers, cookies, or bearer tokens.

