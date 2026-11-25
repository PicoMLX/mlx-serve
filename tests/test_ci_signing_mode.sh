#!/usr/bin/env bash
# Credential-free PR artifacts stay buildable; publishing always requires signing credentials.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import os
import subprocess
import unittest


class SigningModeTests(unittest.TestCase):
    credentials = {
        'APPLE_CERTIFICATE_BASE64': 'certificate-fixture',
        'APPLE_CERTIFICATE_PASSWORD': 'certificate-password-fixture',
        'APPLE_TEAM_ID': 'TEAM123456',
        'APPLE_ID': 'signing@example.test',
        'APPLE_ID_PASSWORD': 'notary-password-fixture',
    }

    def select(self, event, credentials=None, dry_run=False):
        env = {'PATH': os.environ['PATH'], 'GITHUB_EVENT_NAME': event,
               'DRY_RUN': str(dry_run).lower()}
        env.update(credentials or {})
        result = subprocess.run(['bash', 'scripts/ci-signing-mode.sh'], env=env,
                                text=True, capture_output=True)
        outputs = dict(line.split('=', 1) for line in result.stdout.splitlines())
        return result, outputs

    def test_pr_without_credentials_is_ad_hoc(self):
        result, outputs = self.select('pull_request')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs, {'mode': 'adhoc', 'identity': '-', 'notarize': 'false'})

    def test_pr_with_partial_credentials_is_ad_hoc(self):
        result, outputs = self.select('pull_request', {'APPLE_TEAM_ID': 'TEAM123456'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs['mode'], 'adhoc')
        self.assertEqual(outputs['notarize'], 'false')

    def test_real_releases_require_every_credential(self):
        for event in ('push', 'workflow_dispatch'):
            for missing in self.credentials:
                with self.subTest(event=event, missing=missing):
                    credentials = {k: v for k, v in self.credentials.items() if k != missing}
                    result, outputs = self.select(event, credentials)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(missing, result.stderr)
                    self.assertEqual(outputs, {})
                    for name, value in credentials.items():
                        if name != 'APPLE_TEAM_ID':
                            self.assertNotIn(value, result.stderr)

    def test_complete_credentials_keep_signed_notarized_artifacts(self):
        for event in ('pull_request', 'push', 'workflow_dispatch'):
            with self.subTest(event=event):
                result, outputs = self.select(event, self.credentials)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(outputs['mode'], 'developer-id')
                self.assertEqual(outputs['identity'], 'Developer ID Application: David Dalcu (TEAM123456)')
                self.assertEqual(outputs['notarize'], 'true')

    def test_dry_run_never_notarizes(self):
        for credentials, mode in (({}, 'adhoc'), (self.credentials, 'developer-id')):
            with self.subTest(mode=mode):
                result, outputs = self.select('workflow_dispatch', credentials, dry_run=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(outputs['mode'], mode)
                self.assertEqual(outputs['notarize'], 'false')

    def test_unknown_events_fail_closed(self):
        result, outputs = self.select('pull_request_target', self.credentials)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(outputs, {})


unittest.main()
PY
