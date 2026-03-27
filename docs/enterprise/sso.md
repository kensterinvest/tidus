# Enterprise: SSO / OIDC Integration

*This feature is on the roadmap for the Enterprise tier.*

Contact lapkei01@gmail.com to discuss early access or to share requirements.

---

## Design Intent

Tidus will support SSO via any OIDC-compliant identity provider, allowing enterprises to authenticate Tidus API access using their existing identity infrastructure — no separate Tidus credentials needed.

### Supported Identity Providers (Planned)

| Provider | Protocol | Notes |
|----------|----------|-------|
| Okta | OIDC / OAuth 2.0 | Most common enterprise IdP |
| Azure AD / Entra ID | OIDC / OAuth 2.0 | Microsoft 365 environments |
| Google Workspace | OIDC | Google-native enterprises |
| Auth0 | OIDC | Developer-friendly, multi-tenant |
| Generic OIDC | OIDC | Any provider with a `.well-known/openid-configuration` endpoint |

### How It Will Work

1. Enterprise configures Tidus with their OIDC provider's `issuer_url`, `client_id`, and `client_secret` via environment variables
2. Tidus validates incoming JWTs against the provider's JWKS endpoint
3. The `team_id` and `role` claims are extracted from the token and passed to `get_current_team()` in `deps.py`
4. RBAC enforcement then applies based on the extracted role

### Configuration (Planned)

```env
OIDC_ISSUER_URL=https://your-org.okta.com/oauth2/default
OIDC_CLIENT_ID=0oa...
OIDC_CLIENT_SECRET=...
OIDC_TEAM_CLAIM=tidus_team_id   # JWT claim that maps to Tidus team_id
OIDC_ROLE_CLAIM=tidus_role      # JWT claim that maps to Tidus role
```

---

## Current State

The `get_current_team()` stub in `tidus/api/deps.py` is the designed integration point. SSO replaces the stub with a real JWT validator — no other code changes are required.
