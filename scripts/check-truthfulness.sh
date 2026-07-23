#!/usr/bin/env bash
# Truthfulness gate: fail when production sources contain simulated behavior
# markers, stub panics, fake machine code, or sleep-based timing.
#
# Scope: src/ and benches/ (production + benchmark code). Test files may
# legitimately mention these patterns when asserting their absence; docs
# describe the history. The allowlist below must stay tiny and reviewed.

set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

check() {
  local pattern="$1"
  local description="$2"
  # Allowlist: lines that assert or document the ban itself.
  local hits
  hits=$(grep -rnE "$pattern" src/ benches/ --include='*.rs' 2>/dev/null \
    | grep -v 'truth-check-allow' || true)
  if [[ -n "$hits" ]]; then
    echo "TRUTHFULNESS FAIL ($description):"
    echo "$hits"
    fail=1
  fi
}

check 'todo!\('             'todo!() in production code'
check 'unimplemented!\('    'unimplemented!() in production code'
check '[Ss]imulated'        'simulated behavior markers'
check 'simulate[ _]speedup' 'fabricated speedups'
check '[Pp]laceholder for'  'placeholder markers'
check 'vec!\[0x90'          'fake NOP machine code'
check 'thread::sleep'       'sleep-based timing'
check 'mock[ _]speedup'     'mock speedups'

# The status file must parse and every "implemented" capability must carry
# tests and verification entries.
python3 - <<'EOF'
import json, sys
d = json.load(open('implementation-status.json'))
bad = []
for c in d['capabilities']:
    if c['status'] == 'implemented':
        if not c.get('correctness_tests'):
            bad.append((c['capability'], 'no correctness_tests'))
        if not c.get('verification'):
            bad.append((c['capability'], 'no verification'))
    if c['status'] not in ('implemented', 'partial', 'absent'):
        bad.append((c['capability'], f"invalid status {c['status']}"))
if bad:
    for capability, reason in bad:
        print(f"TRUTHFULNESS FAIL (status file): {capability}: {reason}")
    sys.exit(1)
print("implementation-status.json: consistent")
EOF

if [[ $fail -ne 0 ]]; then
  echo "Truthfulness check FAILED"
  exit 1
fi
echo "Truthfulness check PASSED"
