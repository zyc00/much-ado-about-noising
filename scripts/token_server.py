"""Token-gated single-purpose file server for cluster dataset transfer.

Safety properties: serves ONLY files matching a strict whitelist regex,
under a secret 128-bit token path prefix; no directory listings; every
request (including rejects) logged with client IP; supports Range for
parallel chunked pulls. Run it, transfer, then Ctrl-C / kill it.

Usage: python scripts/token_server.py <token> [port]
"""
import http.server
import os
import re
import socketserver
import sys

TOKEN = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8931
ROOT = "/home/jigu/projects/much-ado-about-noising/data/mip_local"
WHITELIST = re.compile(
    r"^(robomimic/[a-z_]+/(ph|mh)/(low_dim(_abs)?|image_abs)\.hdf5"
    r"|groot/[A-Za-z0-9_.-]+\.tar)$")


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parts = self.path.lstrip("/").split("/", 1)
        if (len(parts) != 2 or parts[0] != TOKEN
                or not WHITELIST.match(parts[1])):
            self.log_message("REJECT %s", self.path[:60])
            self.send_error(404)
            return
        p = os.path.join(ROOT, parts[1])
        if not os.path.isfile(p):
            self.send_error(404)
            return
        size = os.path.getsize(p)
        rng = self.headers.get("Range")
        lo, hi = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d+)-(\d*)", rng)
            if m:
                lo = int(m.group(1))
                hi = int(m.group(2)) if m.group(2) else size - 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(hi - lo + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(lo)
            left = hi - lo + 1
            while left > 0:
                b = f.read(min(1 << 20, left))
                if not b:
                    break
                try:
                    self.wfile.write(b)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(b)

    def log_message(self, fmt, *a):
        sys.stderr.write("REQ %s %s\n" % (self.client_address[0], fmt % a))


class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


print(f"serving on 0.0.0.0:{PORT} (whitelist-only, token-gated)",
      flush=True)
S(("0.0.0.0", PORT), H).serve_forever()
