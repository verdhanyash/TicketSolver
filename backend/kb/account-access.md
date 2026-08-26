# NimbusWare Account Access Help

## Password Reset Flow
Use the Forgot Password link on the login page to request a reset email. The reset link is valid for 60 minutes and works only once. If it expires, request a new email; each new request invalidates all earlier links. Reset emails arrive within 5 minutes, so check spam folders before requesting again. Passwords must contain at least 12 characters.

## Account Lockout
An account locks automatically after 5 consecutive failed login attempts and stays locked for a 30-minute cooldown. The cooldown clears itself — no ticket needed — and the failure counter resets after any successful login. Workspace admins can unlock a member immediately from Settings > Members > Unlock instead of waiting out the timer. Repeated lockouts across different sessions may indicate someone is guessing your password; reset it right away.

## MFA Reset
Multi-factor authentication can be reset only through an identity verification ticket, since it bypasses a security control. Open the ticket from the login screen using the Cannot Provide Code option, then supply either a government-issued photo ID or answers to billing verification questions such as the last invoice amount and payment method used. Verified requests are processed within 1 business day. A reset removes the existing authenticator binding and all recovery codes, and the account must re-enroll MFA at next login.

## Admin Roles vs Member Roles
NimbusWare has four roles. Members manage their own tasks and the projects assigned to them. Admins additionally invite and remove members, create and archive projects, configure integrations, and change workspace settings. Billing admins handle invoices, payment methods, and subscription changes without other admin powers. Owners hold every permission plus ownership transfer and workspace deletion, and each workspace has exactly one owner.

## Transferring Workspace Ownership
The current owner starts the transfer from Settings > Members by selecting the target member and choosing Make Owner. The recipient receives a confirmation email and must accept within 72 hours, or the request lapses. Once accepted, the previous owner becomes a regular admin and billing responsibility moves to the new owner starting with the next billing cycle.

## Single Sign-On
SAML and OIDC single sign-on, including SCIM user provisioning and provider-enforced MFA, is available on the Enterprise tier only. Starter, Team, and Business workspaces authenticate with email and password plus MFA. Enterprise admins configure SSO under Settings > Security > Identity Provider and can test the connection before enforcing it for everyone.
