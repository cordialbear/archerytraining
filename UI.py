import tkinter as tk
from tkinter import ttk


class ArcheryTrainingApp(tk.Tk):
    """射箭训练辅助主界面（适配树莓派屏幕）。"""

    def __init__(self) -> None:
        super().__init__()

        self.title("射箭训练辅助系统")
        # 树莓派常见屏幕分辨率，后续可在“设置”中调整
        self.geometry("1024x600")

        # 整体风格简单统一
        self._configure_style()

        # 顶部菜单 Tab
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 三个主功能页
        self.train_frame = ttk.Frame(self.notebook)
        self.history_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.train_frame, text="训练辅助")
        self.notebook.add(self.history_frame, text="历史数据")
        self.notebook.add(self.settings_frame, text="设置")

        # 分别构建各个页面
        self._build_train_tab()
        self._build_history_tab()
        self._build_settings_tab()

    # ---------- 样式配置 ----------
    def _configure_style(self) -> None:
        style = ttk.Style()
        # 在树莓派上常用 "clam" 或 "default"
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TNotebook", background="#1e1e1e")
        style.configure("TNotebook.Tab", padding=(20, 8, 20, 8), font=("Microsoft YaHei", 12))
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#2d2d30")],
            foreground=[("selected", "#ffffff")],
        )

    # ---------- 训练辅助 Tab ----------
    def _build_train_tab(self) -> None:
        """
        训练辅助页面：
        左侧：图像显示区（摄像头 / 检测结果）
        右侧：数据评估区（姿态/得分等数值展示）
        """
        # 使用左右两栏布局
        self.train_frame.columnconfigure(0, weight=3, uniform="train")
        self.train_frame.columnconfigure(1, weight=2, uniform="train")
        self.train_frame.rowconfigure(0, weight=1)

        # 图像显示区
        image_area = ttk.LabelFrame(self.train_frame, text="图像显示区")
        image_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        image_area.rowconfigure(0, weight=1)
        image_area.columnconfigure(0, weight=1)

        # 这里先用 Label 占位，后续可以将 OpenCV 图像转换为 PhotoImage 显示
        self.image_label = tk.Label(
            image_area,
            text="等待摄像头画面...",
            bg="black",
            fg="white",
            font=("Microsoft YaHei", 14),
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 数据评估区
        evaluate_area = ttk.LabelFrame(self.train_frame, text="数据评估区")
        evaluate_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        for i in range(6):
            evaluate_area.rowconfigure(i, weight=1)
        evaluate_area.columnconfigure(0, weight=1)
        evaluate_area.columnconfigure(1, weight=1)

        # 一些典型评估指标占位，后续可由姿态检测结果实时刷新
        metrics = [
            "当前射击编号",
            "命中环数",
            "拉弓稳定性",
            "释放稳定性",
            "身体姿态评分",
            "整体建议",
        ]

        self.metric_vars = {}
        for idx, name in enumerate(metrics):
            label = ttk.Label(evaluate_area, text=name + "：", anchor="e")
            label.grid(row=idx, column=0, sticky="e", padx=(10, 5), pady=5)

            var = tk.StringVar(value="--")
            value_label = ttk.Label(evaluate_area, textvariable=var, anchor="w")
            value_label.grid(row=idx, column=1, sticky="w", padx=(5, 10), pady=5)

            self.metric_vars[name] = var

    # ---------- 历史数据 Tab ----------
    def _build_history_tab(self) -> None:
        """历史数据页面：此处先预留列表区域，后续可以接数据库/文件。"""
        self.history_frame.rowconfigure(0, weight=1)
        self.history_frame.columnconfigure(0, weight=1)

        frame = ttk.LabelFrame(self.history_frame, text="训练记录列表")
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        columns = ("time", "session", "avg_score", "note")
        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            height=15,
        )
        tree.heading("time", text="时间")
        tree.heading("session", text="训练场次")
        tree.heading("avg_score", text="平均环数")
        tree.heading("note", text="备注")

        tree.column("time", width=160, anchor="center")
        tree.column("session", width=100, anchor="center")
        tree.column("avg_score", width=100, anchor="center")
        tree.column("note", width=200, anchor="w")

        tree.pack(fill=tk.BOTH, expand=True)

        # 简单占位数据
        tree.insert("", tk.END, values=("--", "--", "--", "暂无历史数据"))

    # ---------- 设置 Tab ----------
    def _build_settings_tab(self) -> None:
        """设置页面：预留基础配置项。"""
        self.settings_frame.columnconfigure(0, weight=1)
        self.settings_frame.columnconfigure(1, weight=1)

        # 摄像头源
        cam_label = ttk.Label(self.settings_frame, text="摄像头源：")
        cam_label.grid(row=0, column=0, sticky="e", padx=10, pady=(20, 10))

        self.camera_source_var = tk.StringVar(value="usb0")
        cam_entry = ttk.Entry(self.settings_frame, textvariable=self.camera_source_var)
        cam_entry.grid(row=0, column=1, sticky="w", padx=10, pady=(20, 10))

        # 分辨率（简单展示，后续可做下拉选择）
        res_label = ttk.Label(self.settings_frame, text="图像分辨率 (宽x高)：")
        res_label.grid(row=1, column=0, sticky="e", padx=10, pady=10)

        self.resolution_var = tk.StringVar(value="640x480")
        res_entry = ttk.Entry(self.settings_frame, textvariable=self.resolution_var)
        res_entry.grid(row=1, column=1, sticky="w", padx=10, pady=10)

        # 占位：保存按钮（具体逻辑后续与 CameraCapture 对接）
        save_btn = ttk.Button(self.settings_frame, text="保存设置", command=self._on_save_settings)
        save_btn.grid(row=2, column=0, columnspan=2, pady=(20, 10))

    def _on_save_settings(self) -> None:
        """
        保存设置的占位回调。
        目前仅打印设置值，后续可写入配置文件或直接应用到 CameraCapture。
        """
        cam = self.camera_source_var.get()
        res = self.resolution_var.get()
        print(f"[设置] 摄像头源: {cam}, 分辨率: {res}")


def main() -> None:
    app = ArcheryTrainingApp()
    app.mainloop()


if __name__ == "__main__":
    main()
