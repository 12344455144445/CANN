"""Minimal Windows input helper. run_ui.ps1 owns the named cross-process mutex."""
import argparse
import ctypes
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwnd", required=True, type=int)
    parser.add_argument("--expected-title", default="Microsoft Edge")
    parser.add_argument("command", choices=["navigate", "newtab", "closetab", "copyall", "replaceall", "copyurl", "click", "refresh"])
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if not hasattr(ctypes, "windll"):
        raise RuntimeError("Desktop automation requires Windows")
    user = ctypes.windll.user32
    hwnd = args.hwnd
    text = ctypes.create_unicode_buffer(4096)
    if not user.IsWindow(hwnd):
        raise RuntimeError("Configured browser HWND no longer exists")
    user.GetWindowTextW(hwnd, text, len(text))
    if args.expected_title.casefold() not in text.value.casefold():
        raise RuntimeError("Configured window is not the expected browser")

    def focus():
        for _ in range(8):
            if user.GetForegroundWindow() == hwnd:
                return
            own = ctypes.windll.kernel32.GetCurrentThreadId()
            target = user.GetWindowThreadProcessId(hwnd, None)
            foreground = user.GetWindowThreadProcessId(user.GetForegroundWindow(), None)
            attached = []
            for thread in {target, foreground}:
                if thread and thread != own and user.AttachThreadInput(own, thread, True):
                    attached.append(thread)
            try:
                user.ShowWindow(hwnd, 9)
                user.BringWindowToTop(hwnd)
                user.SetForegroundWindow(hwnd)
            finally:
                for thread in attached:
                    user.AttachThreadInput(own, thread, False)
            time.sleep(0.08)
        if user.GetForegroundWindow() != hwnd:
            raise RuntimeError("Browser did not receive focus; no input sent")

    def keys(*codes):
        focus()
        for code in codes:
            user.keybd_event(code, 0, 0, 0)
        time.sleep(0.1)
        for code in reversed(codes):
            user.keybd_event(code, 0, 2, 0)

    def unicode(value):
        class Keyboard(ctypes.Structure):
            _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("extra", ctypes.c_size_t)]

        class Mouse(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("extra", ctypes.c_size_t)]

        class Union(ctypes.Union):
            _fields_ = [("keyboard", Keyboard), ("mouse", Mouse)]

        class Input(ctypes.Structure):
            _anonymous_ = ("event",)
            _fields_ = [("type", ctypes.c_ulong), ("event", Union)]

        encoded = value.encode("utf-16-le")
        events = []
        for offset in range(0, len(encoded), 2):
            code = int.from_bytes(encoded[offset:offset + 2], "little")
            for flags in (4, 6):
                events.append(Input(type=1, keyboard=Keyboard(0, code, flags, 0, 0)))
        focus()
        values = (Input * len(events))(*events)
        if user.SendInput(len(events), ctypes.byref(values), ctypes.sizeof(Input)) != len(events):
            raise RuntimeError("Incomplete browser input; do not submit")

    command = args.command
    if command in ("navigate", "newtab"):
        if len(args.arguments) != 1:
            raise ValueError("Navigation requires one URL")
        keys(0x11, 0x54) if command == "newtab" else keys(0x12, 0x44)
        unicode(args.arguments[0])
        keys(0x0D)
        time.sleep(2)
    elif command == "closetab":
        keys(0x11, 0x57)
        time.sleep(0.5)
    elif command in ("copyall", "replaceall"):
        keys(0x11, 0x41)
        keys(0x11, 0x43 if command == "copyall" else 0x56)
        time.sleep(0.8)
    elif command == "copyurl":
        keys(0x12, 0x44)
        keys(0x11, 0x41)
        keys(0x11, 0x43)
        time.sleep(0.4)
        keys(0x1B)
    elif command == "click":
        if len(args.arguments) != 2:
            raise ValueError("Click requires verified x/y coordinates")
        focus()
        user.SetCursorPos(int(args.arguments[0]), int(args.arguments[1]))
        user.mouse_event(2, 0, 0, 0, 0)
        user.mouse_event(4, 0, 0, 0, 0)
        time.sleep(0.8)
    elif command == "refresh":
        keys(0x11, 0x52)
        time.sleep(2)


if __name__ == "__main__":
    main()
