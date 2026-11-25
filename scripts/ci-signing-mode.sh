#!/usr/bin/env bash
# Select CI artifact signing without downgrading a published release.
set -euo pipefail

case "${GITHUB_EVENT_NAME:-}" in
  pull_request|push|workflow_dispatch) ;;
  *) echo 'Unsupported release workflow event' >&2; exit 1 ;;
esac

missing=()
for name in APPLE_CERTIFICATE_BASE64 APPLE_CERTIFICATE_PASSWORD APPLE_TEAM_ID APPLE_ID APPLE_ID_PASSWORD; do
  if [ -z "${!name:-}" ]; then missing+=("$name"); fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  if [ "$GITHUB_EVENT_NAME" != pull_request ] && [ "${DRY_RUN:-false}" != true ]; then
    printf 'Release signing credentials missing: %s\n' "${missing[*]}" >&2
    exit 1
  fi
  printf 'mode=adhoc\nidentity=-\nnotarize=false\n'
else
  printf 'mode=developer-id\nidentity=Developer ID Application: David Dalcu (%s)\n' "$APPLE_TEAM_ID"
  if [ "${DRY_RUN:-false}" = true ]; then
    printf 'notarize=false\n'
  else
    printf 'notarize=true\n'
  fi
fi
