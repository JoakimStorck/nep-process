"""
Bildruteserver — simuleringen sänder, viewers lyssnar.

Servern har ett enda ansvar: att aldrig låta en tittare bromsa
simuleringen. Det styr hela konstruktionen.

  * `publish()` packar bildrutan en gång och lägger den i ett fack. Den
    väntar aldrig på nätet, och den packar inte alls om ingen är ansluten.
  * Varje klient har en egen tråd som skickar den *senaste* bildrutan, inte
    nästa i en kö. En långsam klient hoppar över bildrutor i stället för att
    släpa efter. Det är rätt beteende för en visning: en gammal bild är
    värdelös, en tappad bild är gratis.
  * Kadensen är väggklocka, inte tick. Simuleringen kan gå tusen tick i
    sekunden utan att någon behöver rita tusen bilder.

Att flera klienter kan titta samtidigt följer av att packningen sker en
gång och delas. Kostnaden per extra tittare är en socketskrivning.

Servern binder till 127.0.0.1 som förval. Det är inte en begränsning utan
en säkerhetsposition: nå den utifrån med en SSH-tunnel

    ssh -N -L 8765:localhost:8765 arbetsstationen

så sköter SSH autentisering och kryptering, och simuleringen behöver
varken lösenord eller certifikat. Vill du ändå binda brett, sätt
`--serve-host 0.0.0.0` och var medveten om att vem som helst på nätet då
kan se världen.
"""

from __future__ import annotations

import json
import socket
import threading
import time

from viewframe import PROTOCOL_VERSION, ViewFrame, pack


class _Client:
    __slots__ = ("sock", "addr", "wake", "thread", "alive", "sent", "dropped")

    def __init__(self, sock: socket.socket, addr) -> None:
        self.sock = sock
        self.addr = addr
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.alive = True
        self.sent = 0
        self.dropped = 0


class ViewerServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        fps: float = 10.0,
        compress: bool = True,
        verbose: bool = True,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.min_interval = 1.0 / max(0.1, float(fps))
        self.compress = bool(compress)
        self.verbose = bool(verbose)

        self._latest: bytes | None = None
        self._lock = threading.Lock()
        self._clients: list[_Client] = []
        self._running = True
        self._last_publish = 0.0
        self.frames_packed = 0

        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(8)
        self._srv.settimeout(0.5)

        self._acceptor = threading.Thread(target=self._accept_loop, daemon=True)
        self._acceptor.start()
        self._log(f"lyssnar på {self.host}:{self.port} (protokoll {PROTOCOL_VERSION})")

    # ---------- publikt ----------
    def publish(self, frame: ViewFrame) -> bool:
        """
        Erbjud en bildruta. Returnerar True om den faktiskt packades.

        Anropa varje tick — funktionen avgör själv om det är dags. Utan
        anslutna klienter kostar den ett låsfritt tomhetstest.
        """
        if not self._clients:
            return False
        now = time.monotonic()
        if now - self._last_publish < self.min_interval:
            return False
        self._last_publish = now

        blob = pack(frame, compress=self.compress)
        self.frames_packed += 1
        with self._lock:
            self._latest = blob
            clients = list(self._clients)
        for c in clients:
            if not c.wake.is_set():
                c.wake.set()
            else:
                c.dropped += 1
        return True

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    def close(self) -> None:
        self._running = False
        try:
            self._srv.close()
        except OSError:
            pass
        for c in list(self._clients):
            c.alive = False
            c.wake.set()
            try:
                c.sock.close()
            except OSError:
                pass

    # ---------- internt ----------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[viewer-server] {msg}", flush=True)

    def _accept_loop(self) -> None:
        while self._running:
            try:
                sock, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                sock.settimeout(5.0)
                hello = self._read_line(sock)
                got = int(json.loads(hello).get("protocol", -1))
                if got != PROTOCOL_VERSION:
                    reply = {
                        "ok": False,
                        "error": (
                            f"klienten talar protokoll {got}, servern "
                            f"{PROTOCOL_VERSION} — kör samma commit i båda ändar"
                        ),
                    }
                    sock.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                    sock.close()
                    self._log(f"avvisade {addr}: protokoll {got}")
                    continue
                sock.sendall((json.dumps({"ok": True}) + "\n").encode("utf-8"))
                sock.settimeout(10.0)
            except Exception as exc:
                self._log(f"handskakning misslyckades mot {addr}: {exc}")
                try:
                    sock.close()
                except OSError:
                    pass
                continue

            c = _Client(sock, addr)
            c.thread = threading.Thread(target=self._send_loop, args=(c,), daemon=True)
            with self._lock:
                self._clients.append(c)
            c.thread.start()
            self._log(f"klient ansluten: {addr}  (totalt {len(self._clients)})")

    @staticmethod
    def _read_line(sock: socket.socket, limit: int = 4096) -> str:
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(256)
            if not chunk:
                raise ConnectionError("klienten stängde under handskakning")
            buf += chunk
            if len(buf) > limit:
                raise ValueError("handskakningen är för lång")
        return buf.split(b"\n", 1)[0].decode("utf-8")

    def _send_loop(self, c: _Client) -> None:
        while self._running and c.alive:
            if not c.wake.wait(timeout=0.5):
                continue
            c.wake.clear()
            with self._lock:
                blob = self._latest
            if blob is None:
                continue
            try:
                c.sock.sendall(blob)
                c.sent += 1
            except Exception:
                break

        c.alive = False
        with self._lock:
            if c in self._clients:
                self._clients.remove(c)
        try:
            c.sock.close()
        except OSError:
            pass
        self._log(
            f"klient bortkopplad: {c.addr}  (skickade {c.sent}, hoppade över "
            f"{c.dropped}, kvar {len(self._clients)})"
        )
