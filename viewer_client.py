"""
Viewerklient — kör fönstret här, simuleringen någon annanstans.

Uppsättning på en ny maskin, i sin helhet:

    git clone https://github.com/JoakimStorck/nep-process
    cd nep-process
    pip install -r requirements-viewer.txt
    python viewer_client.py --host arbetsstationen

Beroendena är numpy och pygame. Ingenting mer: `grid.py`, `viewframe.py`
och `viewer_pygame.py` importerar inte simuleringen, och klienten rör
aldrig numba, scipy, pandas eller matplotlib. Fyra filer räcker om du
hellre kopierar än klonar — `viewer_client.py`, `viewer_pygame.py`,
`viewframe.py` och `grid.py`.

Går servern bakom SSH:

    ssh -N -L 8765:localhost:8765 arbetsstationen
    python viewer_client.py            # förvalet är localhost

Klienten är byggd för att stå och vänta. Fönstret öppnas direkt, före
någon anslutning, och den försöker återansluta så länge den är igång.
Det gör det möjligt att ha viewern uppe innan simuleringen startats, och
att starta om servern utan att röra klienten. En avbruten anslutning är
ett normaltillstånd här, inte ett fel.

Arbetsfördelningen är att servern skickar tillstånd och klienten
bestämmer hur det ser ut. Läge, färgaxel, ytfördelning, gamma och zoom
är lokala och kostar ingen rundtur — och ingenting av det belastar
simuleringen.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time

import numpy as np

from viewframe import HEADER_SIZE, PROTOCOL_VERSION, ViewFrame, payload_size, unpack


class FrameReceiver:
    """
    Håller anslutningen och alltid den senaste bildrutan.

    Läsningen ligger i en egen tråd så att fönstret förblir svarande även
    när servern är borta, långsam eller mitt i en omstart. Huvudtråden
    ritar i sin egen takt och tar det som finns.
    """

    def __init__(self, host: str, port: int, retry_s: float = 2.0) -> None:
        self.host, self.port = host, int(port)
        self.retry_s = float(retry_s)
        self._frame: ViewFrame | None = None
        self._lock = threading.Lock()
        self._running = True
        self.status = "startar"
        self.connected = False
        self.frames = 0
        self.bytes_in = 0
        self.last_frame_at = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def latest(self) -> ViewFrame | None:
        with self._lock:
            return self._frame

    def close(self) -> None:
        self._running = False

    # ---------- internt ----------
    def _loop(self) -> None:
        while self._running:
            try:
                self.status = f"ansluter till {self.host}:{self.port}"
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
            except OSError as exc:
                self.status = f"ingen server på {self.host}:{self.port} ({exc.__class__.__name__})"
                self._sleep(self.retry_s)
                continue

            try:
                sock.sendall((json.dumps({"protocol": PROTOCOL_VERSION}) + "\n").encode("utf-8"))
                reply = json.loads(self._read_line(sock))
                if not reply.get("ok"):
                    self.status = str(reply.get("error", "servern avvisade anslutningen"))
                    sock.close()
                    self._sleep(5.0)
                    continue
            except Exception as exc:
                self.status = f"handskakning misslyckades: {exc}"
                try:
                    sock.close()
                except OSError:
                    pass
                self._sleep(self.retry_s)
                continue

            self.connected = True
            self.status = f"ansluten till {self.host}:{self.port}"
            try:
                sock.settimeout(30.0)
                while self._running:
                    head = self._recv_exact(sock, HEADER_SIZE)
                    payload = self._recv_exact(sock, payload_size(head))
                    frame = unpack(head, payload)
                    with self._lock:
                        self._frame = frame
                    self.frames += 1
                    self.bytes_in += HEADER_SIZE + len(payload)
                    self.last_frame_at = time.monotonic()
            except Exception as exc:
                self.status = f"anslutningen bröts: {exc.__class__.__name__} — försöker igen"
            finally:
                self.connected = False
                try:
                    sock.close()
                except OSError:
                    pass
            self._sleep(self.retry_s)

    def _sleep(self, s: float) -> None:
        end = time.monotonic() + s
        while self._running and time.monotonic() < end:
            time.sleep(0.1)

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        parts, got = [], 0
        while got < n:
            chunk = sock.recv(min(1 << 20, n - got))
            if not chunk:
                raise ConnectionError("servern stängde")
            parts.append(chunk)
            got += len(chunk)
        return b"".join(parts)

    @staticmethod
    def _read_line(sock: socket.socket, limit: int = 4096) -> str:
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(256)
            if not chunk:
                raise ConnectionError("servern stängde under handskakning")
            buf += chunk
            if len(buf) > limit:
                raise ValueError("svaret är för långt")
        return buf.split(b"\n", 1)[0].decode("utf-8")


def _draw_waiting(pg, screen, font, lines: list[str]) -> None:
    screen.fill((14, 18, 22))
    for i, text in enumerate(lines):
        surf = font.render(text, True, (210, 220, 230) if i == 0 else (130, 145, 160))
        screen.blit(surf, (24, 30 + i * 24))
    pg.display.flip()


def main() -> int:
    ap = argparse.ArgumentParser(description="PyGame-viewer mot en fjärrsimulering.")
    ap.add_argument("--host", default="127.0.0.1", help="serverns adress (förval: localhost)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--scale", type=int, default=6, help="pixlar per cell")
    ap.add_argument("--mode", default="FLORA", help="CB | B | C | TEMP | FLORA | CLAIM")
    ap.add_argument("--fill", default="wedge", choices=("wedge", "stipple"))
    ap.add_argument("--color-by", default="temp_opt")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--retry", type=float, default=2.0, help="sekunder mellan återförsök")
    a = ap.parse_args()

    import pygame as pg

    from viewer_pygame import ViewerConfig, WorldViewer

    pg.display.init()
    pg.font.init()
    pg.display.set_caption(f"NEP viewer — {a.host}:{a.port}")
    screen = pg.display.set_mode((760, 200))
    font = pg.font.SysFont("Menlo", 15)

    rx = FrameReceiver(a.host, a.port, retry_s=a.retry)
    viewer = WorldViewer(
        ViewerConfig(
            scale=int(a.scale),
            mode=str(a.mode),
            flora_fill=str(a.fill),
            flora_color_by=str(a.color_by),
            fps_cap=int(a.fps),
            render_every=1,
            title=f"NEP viewer — {a.host}:{a.port}",
        )
    )
    viewer._screen = screen

    clock = pg.time.Clock()
    running = True
    seen = 0
    try:
        while running:
            frame = rx.latest()
            if frame is None:
                for ev in pg.event.get():
                    if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key in (pg.K_ESCAPE, pg.K_q)):
                        running = False
                _draw_waiting(
                    pg,
                    screen,
                    font,
                    [
                        "Väntar på simulering",
                        rx.status,
                        "",
                        f"protokoll {PROTOCOL_VERSION}   återförsök var {a.retry:.0f} s",
                        "Q eller Esc avslutar.",
                    ],
                )
                clock.tick(10)
                continue

            # update() pumpar events och returnerar False när fönstret stängs.
            running = viewer.update(frame)
            seen = rx.frames
            clock.tick(int(a.fps))
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
        viewer.close()
        mb = rx.bytes_in / 1e6
        print(f"tog emot {seen} bildrutor, {mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
