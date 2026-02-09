# live_pop_plot.py
import json
import time
import os
import matplotlib.pyplot as plt

POP_FP = "pop.jsonl"
POLL_S = 0.5  # hur ofta vi pollar efter nya rader

plt.ion()
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
ax_pop, ax_state = ax

ts, pop, E, D, births, deaths = [], [], [], [], [], []


def _append_record(obj: dict) -> None:
    # Stöd för både gamla och nya format:
    # 1) {"event":"population","summary":{...}}
    # 2) {"event":"population", ...}  (payload direkt)
    if obj.get("event") != "population":
        return

    s = obj.get("summary")
    if not isinstance(s, dict):
        s = obj

    ts.append(float(s["t"]))
    pop.append(int(s["pop"]))
    E.append(float(s.get("mean_E", float("nan"))))
    D.append(float(s.get("mean_D", float("nan"))))
    births.append(int(s.get("births", 0)))
    deaths.append(int(s.get("deaths", 0)))


def _redraw() -> None:
    ax_pop.clear()
    ax_pop.plot(ts, pop)
    ax_pop.set_ylabel("population")
    ax_pop.set_title("Population")

    ax_state.clear()
    ax_state.plot(ts, E, label="mean_E")
    ax_state.plot(ts, D, label="mean_D")
    ax_state.legend()
    ax_state.set_xlabel("t")
    ax_state.set_ylabel("mean state")
    ax_state.set_title("Mean health")

    fig.canvas.draw()
    fig.canvas.flush_events()


def tail_population(fp: str = POP_FP) -> None:
    # Vänta tills filen finns
    while not os.path.exists(fp):
        time.sleep(POLL_S)

    with open(fp, "r", encoding="utf-8") as f:
        # 1) Läs allt som finns vid start
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                _append_record(json.loads(line))
            except json.JSONDecodeError:
                continue

        _redraw()

        # 2) Tail: läs endast nya rader med readline()
        last_inode = os.fstat(f.fileno()).st_ino
        while True:
            line = f.readline()
            if line:
                line = line.strip()
                if not line:
                    continue
                try:
                    _append_record(json.loads(line))
                    _redraw()
                except json.JSONDecodeError:
                    # om vi råkar läsa mitt i en write: vänta och fortsätt
                    continue
                continue

            # Inget nytt just nu: hantera truncate/rotation
            time.sleep(POLL_S)

            try:
                st = os.stat(fp)
            except FileNotFoundError:
                # filen kan ha bytts bort kort; vänta
                continue

            # Rotation: inode har ändrats -> öppna om och börja taila därifrån (läs från början)
            if st.st_ino != last_inode:
                f.close()
                with open(fp, "r", encoding="utf-8") as f2:
                    f = f2
                    last_inode = os.fstat(f.fileno()).st_ino
                    # vid ny fil: töm arrays om du vill “ny körning = nytt diagram”
                    ts.clear(); pop.clear(); E.clear(); D.clear(); births.clear(); deaths.clear()
                    for line in f.read().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            _append_record(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    _redraw()
                continue

            # Truncate: filstorlek < vår position -> seek(0)
            cur = f.tell()  # OK: vi använder inte iterator/next()
            if st.st_size < cur:
                f.seek(0)
                # valfritt: töm arrays vid truncate
                ts.clear(); pop.clear(); E.clear(); D.clear(); births.clear(); deaths.clear()
                # Läs om hela filen (som nu är ny)
                for line in f.read().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _append_record(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                _redraw()


if __name__ == "__main__":
    tail_population()