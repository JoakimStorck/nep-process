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
    __slots__ = ("sock", "addr", "wake", "thread", "reader", "alive", "sent", "dropped")

    def __init__(self, sock: socket.socket, addr) -> None:
        self.sock = sock
        self.addr = addr
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.reader: threading.Thread | None = None
        self.alive = True
        self.sent = 0
        self.dropped = 0
        # Senast begärda synliga utsnitt: (cx, cy, hw, hh) i världskoordinater,
        # eller None för "inga anspråksrader".
        self.view = None

    @property
    def name(self) -> str:
        return f"{self.addr[0]}:{self.addr[1]}"


class ViewerServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        fps: float = 10.0,
        compress: bool = True,
        verbose: bool = True,
        control: bool = False,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.min_interval = 1.0 / max(0.1, float(fps))
        self.compress = bool(compress)
        self.verbose = bool(verbose)

        self.control = bool(control)
        self.paused = False
        self.paused_by = ""
        self.paused_seconds = 0.0
        self._resume = threading.Event()
        self._resume.set()

        self._latest: bytes | None = None
        self._latest_frame: ViewFrame | None = None
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
        self._log(
            f"lyssnar på {self.host}:{self.port} (protokoll {PROTOCOL_VERSION}, "
            f"styrning {'på' if self.control else 'av'})"
        )

    # ---------- publikt ----------
    def claim_cells(self, grid) -> object:
        """
        Cellerna någon viewer zoomat in tillräckligt på, eller None.

        Unionen över klienterna, inte en per klient: bildrutan packas en gång
        och sänds till alla, och det vanliga fallet är en enda viewer. Två
        viewers på olika ställen får därför varandras celler också, vilket är
        billigare än att packa två rutor.

        `None` betyder inga anspråksrader alls. Det är förvalet tills en klient
        bett om något — en klient som aldrig frågar får floran som
        celltäckning, vilket är det enda som ändå syns utzoomat.
        """
        with self._lock:
            views = [c.view for c in self._clients if getattr(c, "view", None)]
        if not views:
            return None
        out = []
        for v in views:
            try:
                out.append(grid.cells_in_rect(v[0], v[1], v[2], v[3]))
            except Exception:
                continue
        if not out:
            return None
        import numpy as _np
        return _np.unique(_np.concatenate(out)) if len(out) > 1 else out[0]

    def wants_frame(self) -> bool:
        """
        Är det dags att bygga en bildruta?

        Fråga *före* `frame_from_pop`, inte efter. Att bygga bildrutan och
        sedan låta `publish` kasta den kostar ett par millisekunder per tick
        även när ingen tittar — argumentet beräknas ju innan anropet sker.
        """
        if not self._clients:
            return False
        return (time.monotonic() - self._last_publish) >= self.min_interval

    def publish(self, frame: ViewFrame) -> bool:
        """
        Erbjud en bildruta. Returnerar True om den faktiskt packades.

        Kontrollerar samma villkor som `wants_frame` en gång till, så att ett
        anrop utan föregående fråga inte kan slå ut kadensen.
        """
        if not self.wants_frame():
            return False
        self._last_publish = time.monotonic()

        blob = self._stamp_and_pack(frame)
        with self._lock:
            self._latest_frame = frame
            self._latest = blob
            clients = list(self._clients)
        for c in clients:
            if not c.wake.is_set():
                c.wake.set()
            else:
                c.dropped += 1
        return True

    def wait_while_paused(self) -> float:
        """
        Blockera så länge körningen är pausad. Returnerar väntad tid.

        Anropas överst i tickloopen, före steget, så att pausen aldrig
        träffar mitt i en tick. Den returnerade tiden ska dras av från den
        förflutna tiden: väggklockan går under paus, och utan avdraget blir
        varje ms/tick-siffra från en session där någon pausat obrukbar —
        och den ser inte fel ut, den ser bara långsam ut.
        """
        if not self.paused:
            return 0.0
        t0 = time.monotonic()
        while self.paused and self._running:
            self._resume.wait(timeout=0.1)
        dt = time.monotonic() - t0
        self.paused_seconds += dt
        return dt

    def set_paused(self, paused: bool, who: str = "") -> None:
        """Sätt pausläget och underrätta alla klienter omedelbart."""
        paused = bool(paused)
        if paused == self.paused:
            return
        self.paused = paused
        self.paused_by = who if paused else ""
        if paused:
            self._resume.clear()
        else:
            self._resume.set()
        self._log(("pausad av " if paused else "återupptagen av ") + (who or "okänd"))
        self._republish()

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
    def _stamp_and_pack(self, frame: ViewFrame) -> bytes:
        frame.paused = self.paused
        frame.paused_by = self.paused_by
        frame.control_enabled = self.control
        self.frames_packed += 1
        return pack(frame, compress=self.compress)

    def _republish(self) -> None:
        """
        Packa om den senaste bildrutan och skicka den.

        Under paus ändras inte världen, så ingen ny bildruta produceras.
        Utan omstämpling skulle klienten fortsätta visa `paused=False` tills
        någon återupptog körningen — alltså precis tvärtom mot vad som gäller.
        """
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return
        blob = self._stamp_and_pack(frame)
        with self._lock:
            self._latest = blob
            clients = list(self._clients)
        for c in clients:
            c.wake.set()

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
            c.reader = threading.Thread(target=self._recv_loop, args=(c,), daemon=True)
            with self._lock:
                self._clients.append(c)
                has_frame = self._latest is not None
                has_stale = self._latest_frame is not None
            c.thread.start()
            c.reader.start()

            # Väck den nya klienten om det redan finns en bildruta. Utan det
            # får den vänta till nästa publicering — och står körningen still,
            # pausad eller slut, kommer den aldrig. En ansluten klient som
            # visar tom skärm ser ut som ett fel i klienten.
            if has_frame:
                c.wake.set()
            elif has_stale:
                # Det finns en bildruta men ingen packad — den byggdes medan
                # ingen var ansluten, eller körningen har tagit slut. Packa om
                # den så att den första klienten får se världen ändå.
                self._republish()

            self._log(f"klient ansluten: {c.name}  (totalt {len(self._clients)})")

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

    def _recv_loop(self, c: _Client) -> None:
        """
        Ta emot kommandon från en klient. En rad JSON per kommando.

        Styrningen är opt-in på servern. Är den avslagen läses raderna ändå
        och besvaras med ett tydligt nej — annars ser klienten bara en
        tangent som inte gör något.

        Varje ansluten klient får styra. Förtroendegränsen är redan
        socketen: servern binder till localhost och nås utifrån genom en
        SSH-tunnel, så finkornig behörighet innanför den vore teater. Men
        vem som gjorde vad följer med i bildrutan, så att en paus aldrig ser
        ut som en hängning för den andra tittaren.
        """
        buf = b""
        try:
            while self._running and c.alive:
                try:
                    chunk = c.sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        self._handle_command(c, line.decode("utf-8", "replace"))
        except Exception:
            pass
        c.alive = False
        c.wake.set()

    def _handle_command(self, c: _Client, line: str) -> None:
        try:
            msg = json.loads(line)
            cmd = str(msg.get("cmd", ""))
        except Exception:
            return

        # Utsnittet är inte styrning. Det ändrar inte simuleringen, bara vad
        # servern bemödar sig att packa, och ska därför inte kräva
        # `--serve-control`.
        if cmd == "view":
            try:
                if not bool(msg.get("detail", True)):
                    c.view = None
                else:
                    c.view = (float(msg["cx"]), float(msg["cy"]),
                              float(msg["hw"]), float(msg["hh"]))
            except Exception:
                c.view = None
            return

        if not self.control:
            self._log(f"{c.name} bad om {cmd!r} men styrning är avslagen (--serve-control)")
            return

        if cmd == "pause":
            self.set_paused(True, c.name)
        elif cmd == "resume":
            self.set_paused(False, c.name)
        elif cmd == "toggle_pause":
            self.set_paused(not self.paused, c.name)
        else:
            self._log(f"okänt kommando från {c.name}: {cmd!r}")

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
            f"klient bortkopplad: {c.name}  (skickade {c.sent}, hoppade över "
            f"{c.dropped}, kvar {len(self._clients)})"
        )
