import multiprocessing as mp


def _mp_worker(conn, fn, args, kwargs):
    try:
        conn.send(("ok", fn(*args, **kwargs)))
    except Exception as e:                          # noqa: BLE001
        conn.send(("err", repr(e)))
    finally:
        conn.close()


def run_with_timeout(fn, args=(), kwargs=None, timeout=600):
    """Run fn(*args, **kwargs) in a spawned subprocess; SIGKILL it if it has
    not returned within `timeout` seconds, raising TimeoutError.

    Exists because a solver call can hang on a badly ill-conditioned instance
    for hours (observed directly on CUTEst's DUALC1 at kappa~1.1e6 with
    alpha=1.01). A signal-based timeout (SIGALRM) cannot interrupt this:
    Python only checks for pending signals between its own bytecode
    instructions, never while execution is stuck inside a foreign C/native
    call, so the signal would simply never fire until the hang resolves on
    its own -- which is precisely what does not happen. Process isolation is
    the only way to guarantee a hard stop regardless of what the callee is
    doing internally.

    Uses the "spawn" start method deliberately, not "fork": forking a
    process that has already initialised a foreign runtime (threads, GC,
    internal locks) is a known-unsafe pattern -- only the calling thread
    survives fork(), and any lock that runtime held at that instant stays
    held forever in the child. spawn starts a genuinely fresh interpreter, at
    the cost of re-importing in every subprocess -- a few seconds of overhead
    per call, which is a good trade against a repeat of an hours-long hang.

    fn must be a module-level function (picklable by reference), not a
    lambda or closure -- multiprocessing's spawn context pickles the target
    and its arguments to hand them to the child.
    """
    kwargs = kwargs or {}
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(target=_mp_worker, args=(child_conn, fn, args, kwargs))
    p.start()
    if parent_conn.poll(timeout):
        kind, payload = parent_conn.recv()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        if kind == "err":
            raise RuntimeError(payload)
        return payload
    p.kill()
    p.join()
    raise TimeoutError(f"{getattr(fn, '__name__', fn)} exceeded {timeout}s")
