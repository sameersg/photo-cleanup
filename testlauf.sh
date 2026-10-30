#!/bin/bash
# Testlauf: richtet alles ein und analysiert die neuesten 500 Fotos.
# Liest die Mediathek nur – keine Alben, nichts wird gelöscht.
set -eo pipefail
cd "$(dirname "$0")"

# Passende Python-Version finden (3.10+)
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "❌ Kein Python 3.10+ gefunden. Bitte zuerst: brew install python@3.12"
  exit 1
fi
echo "✅ Nutze $($PY --version)"

if [ ! -d .venv ]; then
  echo "→ Richte Umgebung ein …"
  "$PY" -m venv .venv
fi
source .venv/bin/activate

if [ ! -f .venv/.installed ]; then
  echo "→ Installiere Pakete (einmalig, ein paar Minuten) …"
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  touch .venv/.installed
fi

echo "→ Starte Testlauf (neueste 500 Fotos, nur Bericht) …"
python photo_cleanup.py scan --limit "${1:-500}" --no-albums 2>&1 | tee testlauf.log
echo
echo "✅ Fertig. Bericht liegt in ~/.photo_cleanup/reports/ – Log in testlauf.log"
