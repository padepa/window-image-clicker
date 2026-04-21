import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import tkinter as tk
from pynput import keyboard
from PIL import ImageGrab, ImageTk
from tkinter import filedialog, messagebox, ttk


pyautogui.FAILSAFE = False

CONFIG_PATH = Path("config.json")
TEMPLATE_DIR = Path("templates")
MAX_POINTS_PER_ACCOUNT = 10


@dataclass
class ClickPoint:
    name: str = "识别点"
    image_path: str = ""
    threshold: float = 0.85
    click_times: int = 1
    interval_ms: int = 800
    enabled: bool = True


@dataclass
class AccountConfig:
    name: str = "账号"
    enabled: bool = True
    window_title: str = ""
    loop_delay_ms: int = 1500
    points: List[ClickPoint] = field(default_factory=list)


class AutoClickerEngine:
    def __init__(self, app: "AutoClickerApp"):
        self.app = app
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._accounts_snapshot: List[AccountConfig] = []

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        if self.is_running:
            self.app.log("任务已经在运行。")
            return

        accounts = self.app.snapshot_enabled_accounts()
        if not accounts:
            messagebox.showwarning("提示", "请至少启用一个账号，并为账号配置识别点。")
            return

        self._accounts_snapshot = accounts
        self._stop_requested.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.app.update_status("运行中")
        self.app.log("任务已启动，按 F8 可停止。")

    def stop(self) -> None:
        if not self.is_running:
            self.app.update_status("已停止")
            return

        self._stop_requested.set()
        self._running.clear()
        self.app.update_status("已停止")
        self.app.log("已请求停止任务。")

    def _run_loop(self) -> None:
        try:
            while not self._stop_requested.is_set():
                if not self._accounts_snapshot:
                    self.app.log("没有可运行的账号，任务自动停止。")
                    break

                for account in self._accounts_snapshot:
                    if self._stop_requested.is_set():
                        break
                    self.app.threadsafe_log(f"处理账号：{account.name}")
                    self._process_account(account)
                    self._sleep_ms(account.loop_delay_ms)
        except Exception as exc:
            self.app.threadsafe_log(f"运行出错：{exc}")
            self.app.after(0, lambda: messagebox.showerror("错误", f"运行出错：{exc}"))
        finally:
            self._running.clear()
            self.app.after(0, lambda: self.app.update_status("已停止"))

    def _process_account(self, account: AccountConfig) -> None:
        for point in account.points:
            if self._stop_requested.is_set():
                return
            if not point.enabled:
                continue
            self._process_point(account.name, account.window_title, point)

    def run_single_point_async(self, account_name: str, window_title: str, point: ClickPoint) -> None:
        worker = threading.Thread(target=self._run_single_point, args=(account_name, window_title, point), daemon=True)
        worker.start()

    def _run_single_point(self, account_name: str, window_title: str, point: ClickPoint) -> None:
        try:
            self.app.threadsafe_log(f"[{account_name}] 开始测试识别点：{point.name}")
            self._process_point(account_name, window_title, point)
        except Exception as exc:
            self.app.threadsafe_log(f"[{account_name}] 测试识别点出错：{exc}")
            self.app.after(0, lambda: messagebox.showerror("错误", f"测试识别点出错：{exc}"))

    def _process_point(self, account_name: str, window_title: str, point: ClickPoint) -> None:
        if not point.image_path:
            self.app.threadsafe_log(f"[{account_name}] {point.name} 未设置模板图片，已跳过。")
            return

        capture_region = self._get_capture_region(window_title)
        screen = self._capture_screen_bgr(capture_region)
        match = self._find_template(screen, point.image_path, point.threshold, capture_region)
        if match is None:
            self.app.threadsafe_log(f"[{account_name}] {point.name} 未识别到目标。")
            return

        x, y = match
        self.app.threadsafe_log(f"[{account_name}] {point.name} 识别成功，点击 ({x}, {y})。")
        for _ in range(max(1, point.click_times)):
            pyautogui.click(x=x, y=y)
            time.sleep(max(point.interval_ms, 0) / 1000)

    @staticmethod
    def _capture_screen_bgr(region: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
        screenshot = ImageGrab.grab(bbox=region, all_screens=True) if region else pyautogui.screenshot()
        rgb = np.array(screenshot.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _find_template(
        screen_bgr: np.ndarray,
        image_path: str,
        threshold: float,
        region: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[tuple[int, int]]:
        template = AutoClickerEngine._read_template_image(image_path)
        if template is None:
            raise FileNotFoundError(f"无法读取模板图片：{image_path}")

        if template.shape[0] > screen_bgr.shape[0] or template.shape[1] > screen_bgr.shape[1]:
            return None

        result = cv2.matchTemplate(screen_bgr, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            return None

        offset_x = region[0] if region else 0
        offset_y = region[1] if region else 0
        center_x = offset_x + max_loc[0] + template.shape[1] // 2
        center_y = offset_y + max_loc[1] + template.shape[0] // 2
        return center_x, center_y

    @staticmethod
    def _read_template_image(image_path: str) -> Optional[np.ndarray]:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    @staticmethod
    def _get_capture_region(window_title: str) -> Optional[tuple[int, int, int, int]]:
        if not window_title:
            return None

        windows = [win for win in gw.getWindowsWithTitle(window_title) if (win.title or "").strip() == window_title]
        if not windows:
            windows = gw.getWindowsWithTitle(window_title)
        if not windows:
            raise RuntimeError(f"未找到窗口：{window_title}")

        win = windows[0]
        if win.isMinimized:
            raise RuntimeError(f"目标窗口已最小化：{window_title}")

        left = int(win.left)
        top = int(win.top)
        right = int(win.right)
        bottom = int(win.bottom)
        if right <= left or bottom <= top:
            raise RuntimeError(f"目标窗口区域无效：{window_title}")
        return (left, top, right, bottom)

    def _sleep_ms(self, milliseconds: int) -> None:
        end_time = time.time() + max(milliseconds, 0) / 1000
        while time.time() < end_time:
            if self._stop_requested.is_set():
                break
            time.sleep(0.05)


class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self.show_tip, add="+")
        self.widget.bind("<Leave>", self.hide_tip, add="+")

    def show_tip(self, _event=None) -> None:
        if self.tip_window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.attributes("-topmost", True)
        self.tip_window.geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip_window,
            text=self.text,
            justify="left",
            background="#fff8d8",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            font=("Microsoft YaHei UI", 9),
        )
        label.pack()

    def hide_tip(self, _event=None) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class AutoClickerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("图像识别自动点击工具")
        self.geometry("1320x820")
        self.minsize(1080, 620)

        self.accounts: List[AccountConfig] = []
        self.selected_account_index: Optional[int] = None
        self.selected_point_index: Optional[int] = None
        self.engine = AutoClickerEngine(self)
        self.hotkey_listener: Optional[keyboard.Listener] = None
        self.capture_window: Optional[tk.Toplevel] = None
        self.capture_canvas: Optional[tk.Canvas] = None
        self.capture_photo: Optional[ImageTk.PhotoImage] = None
        self.capture_image = None
        self.capture_start_x = 0
        self.capture_start_y = 0
        self.capture_rect: Optional[int] = None
        self.capture_dragging = False
        self._normal_alpha = 1.0

        self._build_vars()
        self._build_ui()
        self._load_config()
        self._start_hotkeys()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_vars(self) -> None:
        self.status_var = tk.StringVar(value="已停止")
        self.account_name_var = tk.StringVar()
        self.account_enabled_var = tk.BooleanVar(value=True)
        self.account_window_title_var = tk.StringVar()
        self.account_loop_delay_var = tk.StringVar(value="1500")

        self.point_name_var = tk.StringVar()
        self.point_image_var = tk.StringVar()
        self.point_threshold_var = tk.StringVar(value="0.85")
        self.point_click_times_var = tk.StringVar(value="1")
        self.point_interval_var = tk.StringVar(value="800")
        self.point_enabled_var = tk.BooleanVar(value=True)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="账号列表").grid(row=0, column=0, sticky="w")
        self.account_listbox = tk.Listbox(left, width=28, height=28)
        self.account_listbox.grid(row=1, column=0, sticky="ns")
        self.account_listbox.bind("<<ListboxSelect>>", self.on_account_selected)

        account_btns = ttk.Frame(left, padding=(0, 8, 0, 0))
        account_btns.grid(row=2, column=0, sticky="ew")
        ttk.Button(account_btns, text="新增账号", command=self.add_account).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(account_btns, text="删除账号", command=self.delete_account).grid(row=0, column=1, padx=4)
        ttk.Button(account_btns, text="保存配置", command=self.save_config).grid(row=0, column=2, padx=(4, 0))

        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=3)
        right.rowconfigure(3, weight=0)

        top_bar = ttk.Frame(right)
        top_bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(top_bar, text="状态：").grid(row=0, column=0, sticky="w")
        ttk.Label(top_bar, textvariable=self.status_var, foreground="blue").grid(row=0, column=1, sticky="w", padx=(0, 16))
        self.start_btn = ttk.Button(top_bar, text="开始运行(F9)", command=self.start_run)
        self.start_btn.grid(row=0, column=2, padx=4)
        self.stop_btn = ttk.Button(top_bar, text="停止运行(F8)", command=self.stop_run)
        self.stop_btn.grid(row=0, column=3, padx=4)
        ttk.Label(top_bar, text="说明：每个账号最多 10 个识别点；建议先选目标窗口，再截图模板，再测试。").grid(row=0, column=4, padx=(12, 0), sticky="w")

        account_frame = ttk.LabelFrame(right, text="账号设置", padding=10)
        account_frame.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        for idx in range(6):
            account_frame.columnconfigure(idx, weight=1 if idx in (1, 4) else 0)

        ttk.Label(account_frame, text="账号名称").grid(row=0, column=0, sticky="w")
        ttk.Entry(account_frame, textvariable=self.account_name_var).grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ttk.Checkbutton(account_frame, text="启用账号", variable=self.account_enabled_var).grid(row=0, column=2, sticky="w")
        self.apply_account_btn = ttk.Button(account_frame, text="保存账号设置", command=self.update_account)
        self.apply_account_btn.grid(row=0, column=5, padx=(12, 0), sticky="e")

        self.window_label = ttk.Label(account_frame, text="目标窗口")
        self.window_label.grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.window_combo = ttk.Combobox(account_frame, textvariable=self.account_window_title_var, state="readonly")
        self.window_combo.grid(row=1, column=1, columnspan=4, sticky="ew", padx=(6, 12), pady=(10, 0))
        window_btns = ttk.Frame(account_frame)
        window_btns.grid(row=1, column=5, sticky="e", pady=(10, 0))
        self.refresh_windows_btn = ttk.Button(window_btns, text="刷新窗口列表", command=self.refresh_windows)
        self.refresh_windows_btn.grid(row=0, column=0, padx=(0, 6))
        self.pick_window_btn = ttk.Button(window_btns, text="使用当前窗口", command=self.pick_foreground_window)
        self.pick_window_btn.grid(row=0, column=1)

        self.account_loop_delay_label = ttk.Label(account_frame, text="每轮等待(ms)")
        self.account_loop_delay_label.grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.account_loop_delay_entry = ttk.Entry(account_frame, textvariable=self.account_loop_delay_var, width=12)
        self.account_loop_delay_entry.grid(row=2, column=1, sticky="w", padx=(6, 12), pady=(10, 0))
        ttk.Label(
            account_frame,
            text="一个账号完成一轮识别后，等待多久再进入下一轮。1500 = 1.5 秒，越小执行越快。",
            foreground="#666666",
        ).grid(row=2, column=2, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(
            account_frame,
            text="提示：如果选了目标窗口，截图、测试和自动运行都会只在这个窗口里识别。",
            foreground="#666666",
        ).grid(row=3, column=0, columnspan=6, sticky="w", pady=(8, 0))

        points_frame = ttk.LabelFrame(right, text="识别点列表", padding=10)
        points_frame.grid(row=2, column=0, sticky="nsew")
        points_frame.columnconfigure(0, weight=1)
        points_frame.rowconfigure(0, weight=1)

        self.points_tree = ttk.Treeview(
            points_frame,
            columns=("enabled", "name", "threshold", "times", "interval", "image"),
            show="headings",
            height=10,
        )
        self.points_tree.grid(row=0, column=0, sticky="nsew")
        self.points_tree.bind("<<TreeviewSelect>>", self.on_point_selected)
        columns = {
            "enabled": "启用",
            "name": "名称",
            "threshold": "阈值",
            "times": "点击次数",
            "interval": "间隔(ms)",
            "image": "模板图片",
        }
        for key, title in columns.items():
            self.points_tree.heading(key, text=title)
            self.points_tree.column(key, width=110 if key != "image" else 360, anchor="w")

        point_btns = ttk.Frame(points_frame, padding=(0, 8, 0, 0))
        point_btns.grid(row=1, column=0, sticky="ew")
        ttk.Button(point_btns, text="新增识别点", command=self.add_point).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(point_btns, text="删除识别点", command=self.delete_point).grid(row=0, column=1, padx=4)
        ttk.Button(point_btns, text="刷新列表", command=self.refresh_points_tree).grid(row=0, column=2, padx=4)

        point_frame = ttk.LabelFrame(right, text="识别点设置", padding=10)
        point_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for idx in range(4):
            point_frame.columnconfigure(idx, weight=1)

        ttk.Label(point_frame, text="点位名称").grid(row=0, column=0, sticky="w")
        ttk.Entry(point_frame, textvariable=self.point_name_var).grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ttk.Checkbutton(point_frame, text="启用识别点", variable=self.point_enabled_var).grid(row=0, column=2, sticky="w")

        self.point_image_label = ttk.Label(point_frame, text="模板图片")
        self.point_image_label.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(point_frame, textvariable=self.point_image_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(6, 12), pady=(10, 0))
        image_btns = ttk.Frame(point_frame)
        image_btns.grid(row=1, column=3, sticky="w", pady=(10, 0))
        self.choose_image_btn = ttk.Button(image_btns, text="选择模板图", command=self.choose_image)
        self.choose_image_btn.grid(row=0, column=0, padx=(0, 6))
        self.capture_template_btn = ttk.Button(image_btns, text="截图生成模板", command=self.start_template_capture)
        self.capture_template_btn.grid(row=0, column=1)

        self.point_threshold_label = ttk.Label(point_frame, text="匹配阈值")
        self.point_threshold_label.grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.point_threshold_entry = ttk.Entry(point_frame, textvariable=self.point_threshold_var)
        self.point_threshold_entry.grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=(10, 0))
        self.point_click_times_label = ttk.Label(point_frame, text="点击次数")
        self.point_click_times_label.grid(row=2, column=2, sticky="w", pady=(10, 0))
        self.point_click_times_entry = ttk.Entry(point_frame, textvariable=self.point_click_times_var)
        self.point_click_times_entry.grid(row=2, column=3, sticky="ew", pady=(10, 0))

        self.point_interval_label = ttk.Label(point_frame, text="点击间隔(ms)")
        self.point_interval_label.grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.point_interval_entry = ttk.Entry(point_frame, textvariable=self.point_interval_var)
        self.point_interval_entry.grid(row=3, column=1, sticky="ew", padx=(6, 12), pady=(10, 0))
        point_action_btns = ttk.Frame(point_frame)
        point_action_btns.grid(row=3, column=3, sticky="e", pady=(10, 0))
        self.test_point_btn = ttk.Button(point_action_btns, text="测试当前点", command=self.test_current_point)
        self.test_point_btn.grid(row=0, column=0, padx=(0, 6))
        self.apply_point_btn = ttk.Button(point_action_btns, text="保存识别点设置", command=self.update_point)
        self.apply_point_btn.grid(row=0, column=1)

        ttk.Label(
            point_frame,
            text="提示：阈值越高越严格；建议先截图生成模板，再点测试当前点。",
            foreground="#666666",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))

        log_frame = ttk.LabelFrame(right, text="运行日志", padding=10)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        self.log_text = tk.Text(log_frame, height=8, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self._init_tooltips()

    def _start_hotkeys(self) -> None:
        def on_press(key) -> None:
            if key == keyboard.Key.f8:
                self.after(0, self.stop_run)
            elif key == keyboard.Key.f9:
                self.after(0, self.start_run)

        self.hotkey_listener = keyboard.Listener(on_press=on_press)
        self.hotkey_listener.daemon = True
        self.hotkey_listener.start()
        self.log("热键已注册：F9 启动，F8 停止。")

    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            self.accounts = [AccountConfig(name="账号1")]
            self.refresh_windows()
            self.refresh_account_list()
            self.account_listbox.selection_set(0)
            self.on_account_selected()
            return

        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.accounts = []
        for account_data in raw.get("accounts", []):
            points = [ClickPoint(**point_data) for point_data in account_data.get("points", [])][:MAX_POINTS_PER_ACCOUNT]
            account = AccountConfig(
                name=account_data.get("name", "账号"),
                enabled=account_data.get("enabled", True),
                window_title=account_data.get("window_title", ""),
                loop_delay_ms=account_data.get("loop_delay_ms", 1500),
                points=points,
            )
            self.accounts.append(account)

        if not self.accounts:
            self.accounts = [AccountConfig(name="账号1")]

        self.refresh_windows()
        self.refresh_account_list()
        self.account_listbox.selection_set(0)
        self.on_account_selected()
        self.log("已加载配置文件。")

    def save_config(self) -> None:
        self._sync_current_forms()
        data = {"accounts": [asdict(account) for account in self.accounts]}
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log("配置已保存到 config.json。")
        messagebox.showinfo("提示", "配置已保存。")

    def refresh_account_list(self) -> None:
        self.account_listbox.delete(0, tk.END)
        for idx, account in enumerate(self.accounts, start=1):
            status = "启用" if account.enabled else "停用"
            self.account_listbox.insert(tk.END, f"{idx}. {account.name} [{status}]")

    def refresh_points_tree(self) -> None:
        for item in self.points_tree.get_children():
            self.points_tree.delete(item)

        account = self.get_selected_account()
        if account is None:
            return

        for idx, point in enumerate(account.points):
            self.points_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    "是" if point.enabled else "否",
                    point.name,
                    point.threshold,
                    point.click_times,
                    point.interval_ms,
                    point.image_path,
                ),
            )

    def get_selected_account(self) -> Optional[AccountConfig]:
        if self.selected_account_index is None:
            return None
        if self.selected_account_index >= len(self.accounts):
            return None
        return self.accounts[self.selected_account_index]

    def snapshot_enabled_accounts(self) -> List[AccountConfig]:
        self._sync_current_forms()
        enabled_accounts: List[AccountConfig] = []
        for account in self.accounts:
            if not account.enabled:
                continue
            enabled_points = [ClickPoint(**asdict(point)) for point in account.points if point.enabled]
            if not enabled_points:
                continue
            enabled_accounts.append(
                AccountConfig(
                    name=account.name,
                    enabled=account.enabled,
                    window_title=account.window_title,
                    loop_delay_ms=account.loop_delay_ms,
                    points=enabled_points,
                )
            )
        return enabled_accounts

    def on_account_selected(self, _event=None) -> None:
        selection = self.account_listbox.curselection()
        if not selection:
            self.selected_account_index = None
            return

        self._sync_current_forms()
        self.selected_account_index = selection[0]
        self.selected_point_index = None
        account = self.accounts[self.selected_account_index]
        self.account_name_var.set(account.name)
        self.account_enabled_var.set(account.enabled)
        self.account_window_title_var.set(account.window_title)
        self.account_loop_delay_var.set(str(account.loop_delay_ms))
        self.refresh_points_tree()
        self.clear_point_form()

    def on_point_selected(self, _event=None) -> None:
        account = self.get_selected_account()
        if account is None:
            return

        selection = self.points_tree.selection()
        if not selection:
            self.selected_point_index = None
            return

        idx = int(selection[0])
        self.selected_point_index = idx
        point = account.points[idx]
        self.point_name_var.set(point.name)
        self.point_image_var.set(point.image_path)
        self.point_threshold_var.set(str(point.threshold))
        self.point_click_times_var.set(str(point.click_times))
        self.point_interval_var.set(str(point.interval_ms))
        self.point_enabled_var.set(point.enabled)

    def add_account(self) -> None:
        self._sync_current_forms()
        self.accounts.append(AccountConfig(name=f"账号{len(self.accounts) + 1}"))
        self.refresh_account_list()
        new_index = len(self.accounts) - 1
        self.account_listbox.selection_clear(0, tk.END)
        self.account_listbox.selection_set(new_index)
        self.on_account_selected()
        self.log("已新增账号。")

    def delete_account(self) -> None:
        if self.selected_account_index is None:
            return

        deleted = self.accounts.pop(self.selected_account_index)
        self.refresh_account_list()
        self.selected_account_index = None
        self.selected_point_index = None

        if self.accounts:
            self.account_listbox.selection_set(0)
            self.on_account_selected()
        else:
            self.clear_account_form()
            self.clear_point_form()
            self.refresh_points_tree()

        self.log(f"已删除账号：{deleted.name}")

    def update_account(self) -> None:
        account = self.get_selected_account()
        if account is None:
            return

        account.name = self.account_name_var.get().strip() or "账号"
        account.enabled = self.account_enabled_var.get()
        account.window_title = self.account_window_title_var.get().strip()
        account.loop_delay_ms = self._to_int(self.account_loop_delay_var.get(), 1500, minimum=0)
        self.refresh_account_list()
        self.log(f"账号设置已更新：{account.name}")

    def add_point(self) -> None:
        account = self.get_selected_account()
        if account is None:
            messagebox.showwarning("提示", "请先选择一个账号。")
            return
        if len(account.points) >= MAX_POINTS_PER_ACCOUNT:
            messagebox.showwarning("提示", f"每个账号最多添加 {MAX_POINTS_PER_ACCOUNT} 个识别点。")
            return

        account.points.append(ClickPoint(name=f"识别点{len(account.points) + 1}"))
        self.refresh_points_tree()
        self.log(f"[{account.name}] 已新增识别点。")

    def delete_point(self) -> None:
        account = self.get_selected_account()
        if account is None or self.selected_point_index is None:
            return

        deleted = account.points.pop(self.selected_point_index)
        self.selected_point_index = None
        self.refresh_points_tree()
        self.clear_point_form()
        self.log(f"[{account.name}] 已删除识别点：{deleted.name}")

    def update_point(self) -> None:
        account = self.get_selected_account()
        if account is None or self.selected_point_index is None:
            messagebox.showwarning("提示", "请先在列表中选择一个识别点。")
            return

        point = account.points[self.selected_point_index]
        point.name = self.point_name_var.get().strip() or "识别点"
        point.image_path = self.point_image_var.get().strip()
        point.threshold = self._to_float(self.point_threshold_var.get(), 0.85, minimum=0.1, maximum=1.0)
        point.click_times = self._to_int(self.point_click_times_var.get(), 1, minimum=1)
        point.interval_ms = self._to_int(self.point_interval_var.get(), 800, minimum=0)
        point.enabled = self.point_enabled_var.get()
        self.refresh_points_tree()
        self.log(f"[{account.name}] 已更新识别点：{point.name}")

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择模板图片",
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")],
        )
        if path:
            self.point_image_var.set(path)

    def start_template_capture(self) -> None:
        account = self.get_selected_account()
        if account is None or self.selected_point_index is None:
            messagebox.showwarning("提示", "请先选择一个识别点，再截图取模板。")
            return

        if self.capture_window is not None:
            return

        self.after(50, lambda: self._begin_template_capture(account.window_title))

    def _begin_template_capture(self, window_title: str = "") -> None:
        try:
            self.capture_image = self._capture_screen_image(window_title)
            self.capture_photo = ImageTk.PhotoImage(self.capture_image)
        except Exception as exc:
            self.cancel_template_capture()
            messagebox.showerror("截图失败", f"进入截图模式失败：{exc}")
            self.log(f"进入截图模式失败：{exc}")
            return

        self.capture_window = tk.Toplevel(self)
        self.capture_window.attributes("-topmost", True)
        self.capture_window.title("截图取模板")
        self.capture_window.configure(bg="#1f1f1f")
        self.capture_window.bind("<Escape>", lambda _event: self.cancel_template_capture())

        width, height = self.capture_image.size
        preview_width = min(max(width + 20, 960), 1600)
        preview_height = min(max(height + 90, 700), 1000)
        self.capture_window.geometry(f"{preview_width}x{preview_height}+80+60")
        self.capture_window.minsize(900, 650)

        hint = ttk.Label(
            self.capture_window,
            text="拖动鼠标框选模板区域，按回车保存，按 Esc 取消",
            foreground="#f5f5f5",
            background="#1f1f1f",
            padding=(10, 8),
        )
        hint.pack(anchor="w")

        canvas_wrap = ttk.Frame(self.capture_window)
        canvas_wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        x_scroll = ttk.Scrollbar(canvas_wrap, orient="horizontal")
        y_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical")
        self.capture_canvas = tk.Canvas(
            canvas_wrap,
            width=min(width, preview_width - 40),
            height=min(height, preview_height - 120),
            highlightthickness=0,
            cursor="crosshair",
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
            bg="black",
        )
        x_scroll.config(command=self.capture_canvas.xview)
        y_scroll.config(command=self.capture_canvas.yview)
        self.capture_canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_wrap.columnconfigure(0, weight=1)
        canvas_wrap.rowconfigure(0, weight=1)

        image_item = self.capture_canvas.create_image(0, 0, image=self.capture_photo, anchor="nw", tags=("capture_image",))
        self.capture_canvas.config(scrollregion=(0, 0, width, height))
        self.capture_canvas.bind("<ButtonPress-1>", self.on_capture_press)
        self.capture_canvas.bind("<B1-Motion>", self.on_capture_drag)
        self.capture_canvas.bind("<ButtonRelease-1>", self.on_capture_release)
        self.capture_canvas.tag_bind("capture_image", "<ButtonPress-1>", self.on_capture_press)
        self.capture_canvas.tag_bind("capture_image", "<B1-Motion>", self.on_capture_drag)
        self.capture_canvas.tag_bind("capture_image", "<ButtonRelease-1>", self.on_capture_release)
        self.capture_window.bind("<Return>", lambda _event: self.finish_template_capture())
        self.capture_window.update_idletasks()
        self.capture_window.lift()
        self.capture_window.focus_force()

    @staticmethod
    def _capture_screen_image(window_title: str = ""):
        try:
            region = AutoClickerEngine._get_capture_region(window_title)
            if region:
                return ImageGrab.grab(bbox=region, all_screens=True)
            return pyautogui.screenshot()
        except Exception:
            return ImageGrab.grab(all_screens=True)

    def refresh_windows(self) -> None:
        titles: List[str] = []
        for win in gw.getAllWindows():
            title = (win.title or "").strip()
            if title and title not in titles:
                titles.append(title)
        current = self.account_window_title_var.get().strip()
        if current and current not in titles:
            titles.insert(0, current)
        self.window_combo["values"] = titles

    def pick_foreground_window(self) -> None:
        try:
            active = gw.getActiveWindow()
        except Exception as exc:
            messagebox.showerror("错误", f"获取当前窗口失败：{exc}")
            return

        title = (active.title if active else "").strip()
        if not title:
            messagebox.showwarning("提示", "当前没有可用的前台窗口标题。")
            return

        self.refresh_windows()
        self.account_window_title_var.set(title)
        self.log(f"已选择目标窗口：{title}")

    def _init_tooltips(self) -> None:
        ToolTip(self.start_btn, "开始自动运行全部已启用账号。也可以直接按 F9。")
        ToolTip(self.stop_btn, "停止当前自动运行任务。也可以直接按 F8。")
        ToolTip(self.account_loop_delay_label, "一个账号完成一轮识别后，等待多久再继续下一轮。1500 = 1.5 秒。")
        ToolTip(self.account_loop_delay_entry, "建议 500 到 3000。目标程序反应慢时可以调大。")
        ToolTip(self.apply_account_btn, "保存当前账号名称、目标窗口和轮询间隔到内存中。")
        ToolTip(self.window_label, "选中后，截图、测试和自动运行都会只在这个窗口内识别。")
        ToolTip(self.window_combo, "这里显示可选窗口标题。若列表没有目标程序，先点刷新窗口。")
        ToolTip(self.refresh_windows_btn, "重新读取当前系统里的窗口列表。打开目标程序后点一次。")
        ToolTip(self.pick_window_btn, "把当前最前面的窗口标题自动填进来。先切到目标程序再点它。")
        ToolTip(self.point_image_label, "模板图片就是程序用来识别按钮或图标的小截图。")
        ToolTip(self.choose_image_btn, "手动选择已经存在的模板图片文件。")
        ToolTip(self.capture_template_btn, "从屏幕或目标窗口里框选一块区域，保存成模板图片。")
        ToolTip(self.point_threshold_label, "匹配阈值，越高越严格。识别不到可稍微调低，误识别多可调高。")
        ToolTip(self.point_threshold_entry, "常用范围 0.75 到 0.95，默认 0.85。")
        ToolTip(self.point_click_times_label, "识别成功后连续点击多少次。")
        ToolTip(self.point_click_times_entry, "一般填 1 即可，需要连点再调大。")
        ToolTip(self.point_interval_label, "同一个识别点连续点击时，两次点击之间等待多久。")
        ToolTip(self.point_interval_entry, "单位毫秒。800 = 0.8 秒。")
        ToolTip(self.test_point_btn, "只测试当前选中的这个识别点，不会启动整轮任务。")
        ToolTip(self.apply_point_btn, "保存当前识别点的模板图、阈值、点击次数和间隔。")

    def on_capture_press(self, event) -> None:
        if self.capture_canvas is None:
            return
        self.capture_dragging = True
        self.capture_start_x = int(self.capture_canvas.canvasx(event.x))
        self.capture_start_y = int(self.capture_canvas.canvasy(event.y))
        if self.capture_rect is not None:
            self.capture_canvas.delete(self.capture_rect)
        self.capture_rect = self.capture_canvas.create_rectangle(
            self.capture_start_x,
            self.capture_start_y,
            self.capture_start_x,
            self.capture_start_y,
            outline="#ff3b30",
            width=2,
        )

    def on_capture_drag(self, event) -> None:
        if self.capture_canvas is None or self.capture_rect is None or not self.capture_dragging:
            return
        current_x = int(self.capture_canvas.canvasx(event.x))
        current_y = int(self.capture_canvas.canvasy(event.y))
        self.capture_canvas.coords(self.capture_rect, self.capture_start_x, self.capture_start_y, current_x, current_y)

    def on_capture_release(self, event) -> None:
        if self.capture_canvas is None or self.capture_rect is None:
            return
        self.capture_dragging = False
        current_x = int(self.capture_canvas.canvasx(event.x))
        current_y = int(self.capture_canvas.canvasy(event.y))
        self.capture_canvas.coords(self.capture_rect, self.capture_start_x, self.capture_start_y, current_x, current_y)

    def finish_template_capture(self) -> None:
        if self.capture_image is None or self.capture_rect is None or self.capture_canvas is None:
            self.cancel_template_capture()
            return

        x1, y1, x2, y2 = [int(value) for value in self.capture_canvas.coords(self.capture_rect)]
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left < 5 or bottom - top < 5:
            messagebox.showwarning("提示", "截图区域太小，请重新框选。")
            return

        TEMPLATE_DIR.mkdir(exist_ok=True)
        point_name = self.point_name_var.get().strip() or "识别点"
        safe_name = self._to_safe_filename(point_name)
        file_path = TEMPLATE_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name}.png"
        cropped = self.capture_image.crop((left, top, right, bottom))
        cropped.save(file_path)
        self.point_image_var.set(str(file_path.resolve()))
        self.cancel_template_capture(show_main=False)
        self.lift()
        self.focus_force()
        self.log(f"模板图片已保存：{file_path}")
        messagebox.showinfo("提示", f"模板图片已保存到：\n{file_path}")

    def cancel_template_capture(self, show_main: bool = True) -> None:
        if self.capture_window is not None:
            self.capture_window.destroy()
        self.capture_window = None
        self.capture_canvas = None
        self.capture_photo = None
        self.capture_image = None
        self.capture_rect = None
        self.capture_dragging = False
        if show_main:
            self.lift()
            self.focus_force()

    def start_run(self) -> None:
        self.engine.start()

    def stop_run(self) -> None:
        self.engine.stop()

    def test_current_point(self) -> None:
        account = self.get_selected_account()
        if account is None or self.selected_point_index is None:
            messagebox.showwarning("提示", "请先选择一个识别点。")
            return

        self._sync_current_forms()
        point = account.points[self.selected_point_index]
        point_snapshot = ClickPoint(**asdict(point))
        self.engine.run_single_point_async(account.name, account.window_title, point_snapshot)

    def update_status(self, text: str) -> None:
        self.status_var.set(text)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def threadsafe_log(self, message: str) -> None:
        self.after(0, lambda: self.log(message))

    def clear_account_form(self) -> None:
        self.account_name_var.set("")
        self.account_enabled_var.set(True)
        self.account_window_title_var.set("")
        self.account_loop_delay_var.set("1500")

    def clear_point_form(self) -> None:
        self.point_name_var.set("")
        self.point_image_var.set("")
        self.point_threshold_var.set("0.85")
        self.point_click_times_var.set("1")
        self.point_interval_var.set("800")
        self.point_enabled_var.set(True)

    def _sync_current_forms(self) -> None:
        if self.selected_account_index is not None and self.selected_account_index < len(self.accounts):
            account = self.accounts[self.selected_account_index]
            account.name = self.account_name_var.get().strip() or account.name
            account.enabled = self.account_enabled_var.get()
            account.window_title = self.account_window_title_var.get().strip()
            account.loop_delay_ms = self._to_int(self.account_loop_delay_var.get(), account.loop_delay_ms, minimum=0)

            if self.selected_point_index is not None and self.selected_point_index < len(account.points):
                point = account.points[self.selected_point_index]
                point.name = self.point_name_var.get().strip() or point.name
                point.image_path = self.point_image_var.get().strip()
                point.threshold = self._to_float(self.point_threshold_var.get(), point.threshold, minimum=0.1, maximum=1.0)
                point.click_times = self._to_int(self.point_click_times_var.get(), point.click_times, minimum=1)
                point.interval_ms = self._to_int(self.point_interval_var.get(), point.interval_ms, minimum=0)
                point.enabled = self.point_enabled_var.get()

    @staticmethod
    def _to_int(value: str, default: int, minimum: Optional[int] = None) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if minimum is not None:
            parsed = max(parsed, minimum)
        return parsed

    @staticmethod
    def _to_float(value: str, default: float, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if minimum is not None:
            parsed = max(parsed, minimum)
        if maximum is not None:
            parsed = min(parsed, maximum)
        return parsed

    @staticmethod
    def _to_safe_filename(value: str) -> str:
        ascii_only = value.encode("ascii", errors="ignore").decode("ascii")
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_only).strip("._-")
        return normalized or "template"

    def on_close(self) -> None:
        self.engine.stop()
        if self.hotkey_listener is not None:
            self.hotkey_listener.stop()
        if self.capture_window is not None:
            self.cancel_template_capture(show_main=False)
        self.destroy()


if __name__ == "__main__":
    app = AutoClickerApp()
    app.mainloop()
