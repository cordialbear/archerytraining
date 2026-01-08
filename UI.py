import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from camera_capture import CameraCapture, PostDetection
from scoreDectect import ScoreDetector


class ArcheryTrainingApp(tk.Tk):
    """射箭训练辅助主界面（适配树莓派屏幕）。"""

    def __init__(self) -> None:
        super().__init__()

        self.title("射箭训练辅助系统")
        # 树莓派常见屏幕分辨率，后续可在“设置”中调整
        self.geometry("1024x600")

        # 摄像头与图像缓存占位
        self._camera: CameraCapture | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self._pose_detector: PostDetection | None = None
        self._score_detector: ScoreDetector | None = None
        self._score_crops: list[tuple[ImageTk.PhotoImage, str]] = []
        self._score_idx: int = 0
        # 保存检测结果，用于在计分区显示所有靶位的成绩
        self._detected_target_results: list = []
        # 模型路径可按需替换为实际文件
        self.target_model_path = "weights/target-yolov8.pt"
        self.arrow_model_path = "weights/arrowV1-model.pt"

        # 整体风格简单统一
        self._configure_style()

        # 顶部菜单 Tab
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 姿态检测开关（在设置页中控制）
        self.pose_detection_enabled = tk.BooleanVar(value=False)

        # 主功能页
        self.train_frame = ttk.Frame(self.notebook)
        self.scoring_frame = ttk.Frame(self.notebook)
        self.history_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.train_frame, text="训练辅助")
        self.notebook.add(self.scoring_frame, text="计分")
        self.notebook.add(self.history_frame, text="历史数据")
        self.notebook.add(self.settings_frame, text="设置")

        # 分别构建各个页面
        self._build_train_tab()
        self._build_scoring_tab()
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

        # 初始化 Picamera2 摄像头
        try:
            # 使用 picamera2 取流：在 CameraCapture 内部根据 source 前缀自动选择
            self._camera = CameraCapture(source="picamera", resolution=(640, 480), fps=30)
            self._camera.startCapture()
            # 启动定时刷新图像
            self._update_camera_frame()
        except Exception as exc:  # 在界面上直接提示错误
            self._camera = None
            self.image_label.config(
                text=f"摄像头初始化失败：{exc}",
                fg="red",
                bg="black",
            )

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

    def _update_camera_frame(self) -> None:
        """从 Picamera2 获取一帧图像，按需做姿态检测，并刷新到图像显示区。"""
        if not self._camera:
            return

        frame = self._camera.getFrame()
        if frame is not None:
            # 如开启姿态检测，则使用 PostDetection 进行推理并绘制关键点
            if self.pose_detection_enabled.get():
                # 延迟初始化 PostDetection，避免程序启动时就加载模型
                if self._pose_detector is None:
                    try:
                        self._pose_detector = PostDetection(self._camera)
                        print("[姿态检测] YOLO 模型已加载")
                    except Exception as exc:
                        print(f"[姿态检测] 初始化失败: {exc}")
                        self._pose_detector = None

                if self._pose_detector is not None:
                    result = self._pose_detector.infer_once()
                    if result is not None:
                        det_frame, keypoints = result
                        # 使用 draw_poses 将关键点画到图像上
                        frame = self._pose_detector.draw_poses(det_frame, keypoints)

            # CameraCapture / PostDetection 输出为 BGR，这里转为 RGB 再交给 Pillow
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # 根据当前 Label 大小缩放图像，避免超出界面
            w = self.image_label.winfo_width() or 640
            h = self.image_label.winfo_height() or 480
            img = img.resize((w, h))

            self._tk_image = ImageTk.PhotoImage(image=img)
            self.image_label.configure(image=self._tk_image, text="")

        # 每 30ms 刷新一次，大约 ~33fps
        self.after(30, self._update_camera_frame)

    # ---------- 计分 Tab ----------
    def _build_scoring_tab(self) -> None:
        """
        计分页面：
        左侧：图像显示区
        右侧：计分区（按靶位展示每轮的箭序号与环数，默认每轮 6 支）
        """
        self.scoring_frame.columnconfigure(0, weight=3, uniform="score")
        self.scoring_frame.columnconfigure(1, weight=2, uniform="score")
        self.scoring_frame.rowconfigure(0, weight=1)

        # 图像显示区（使用 Canvas + Scrollbar 支持滚动）
        image_area = ttk.LabelFrame(self.scoring_frame, text="图像显示区")
        image_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        image_area.rowconfigure(0, weight=1)
        image_area.columnconfigure(0, weight=1)
        image_area.columnconfigure(1, weight=0)

        # Canvas 用于显示所有靶纸图片
        self.score_image_canvas = tk.Canvas(
            image_area,
            bg="black",
            highlightthickness=0,
        )
        self.score_image_canvas.grid(row=0, column=0, sticky="nsew")

        # 垂直滚动条
        score_image_scrollbar = ttk.Scrollbar(
            image_area,
            orient=tk.VERTICAL,
            command=self.score_image_canvas.yview,
        )
        score_image_scrollbar.grid(row=0, column=1, sticky="ns")
        self.score_image_canvas.configure(yscrollcommand=score_image_scrollbar.set)

        # 在 Canvas 上创建可滚动的 Frame
        self.score_image_frame = tk.Frame(self.score_image_canvas, bg="black")
        self.score_image_canvas_window = self.score_image_canvas.create_window(
            (0, 0),
            window=self.score_image_frame,
            anchor="nw",
        )

        # 绑定 Canvas 大小变化事件，更新滚动区域和重新布局
        def _configure_canvas_image(event):
            canvas_width = event.width
            self.score_image_canvas.itemconfig(self.score_image_canvas_window, width=canvas_width)
            # 如果有检测结果，重新布局
            if self._detected_target_results:
                self._display_all_targets(self._detected_target_results)
            else:
                self.score_image_canvas.configure(scrollregion=self.score_image_canvas.bbox("all"))

        self.score_image_canvas.bind("<Configure>", _configure_canvas_image)
        self.score_image_frame.bind("<Configure>", lambda e: self.score_image_canvas.configure(scrollregion=self.score_image_canvas.bbox("all")))

        # 计分区
        scoring_area = ttk.LabelFrame(self.scoring_frame, text="计分区")
        scoring_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        scoring_area.columnconfigure(0, weight=1)

        # 控制区域：箭数输入 + 下一轮按钮
        control_frame = ttk.Frame(scoring_area)
        control_frame.grid(row=0, column=0, sticky="ew", pady=(5, 10), padx=5)
        control_frame.columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="箭数：").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.arrow_count_var = tk.StringVar(value="6")
        arrow_entry = ttk.Entry(control_frame, textvariable=self.arrow_count_var, width=6)
        arrow_entry.grid(row=0, column=1, sticky="w")
        # 箭数变化即重建记录框，保证输入框变动后即时生效
        self.arrow_count_var.trace_add("write", lambda *_: self._render_score_boxes())

        next_round_btn = ttk.Button(control_frame, text="下一轮", command=self._on_next_round)
        next_round_btn.grid(row=0, column=2, sticky="e", padx=(10, 0))

        # 计分检测控制（使用 ScoreDetector）
        detect_btn = ttk.Button(control_frame, text="检测箭靶", command=self._on_detect_targets)
        detect_btn.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        # 计分区（使用 Canvas + Scrollbar 支持滚动）
        score_canvas_frame = ttk.Frame(scoring_area)
        score_canvas_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        scoring_area.rowconfigure(1, weight=1)
        score_canvas_frame.rowconfigure(0, weight=1)
        score_canvas_frame.columnconfigure(0, weight=1)
        score_canvas_frame.columnconfigure(1, weight=0)

        # Canvas 用于显示所有靶位的记录框
        self.score_canvas = tk.Canvas(
            score_canvas_frame,
            highlightthickness=0,
        )
        self.score_canvas.grid(row=0, column=0, sticky="nsew")

        # 垂直滚动条
        score_scrollbar = ttk.Scrollbar(
            score_canvas_frame,
            orient=tk.VERTICAL,
            command=self.score_canvas.yview,
        )
        score_scrollbar.grid(row=0, column=1, sticky="ns")
        self.score_canvas.configure(yscrollcommand=score_scrollbar.set)

        # 在 Canvas 上创建可滚动的 Frame
        self.score_container = ttk.Frame(self.score_canvas)
        self.score_canvas_window = self.score_canvas.create_window(
            (0, 0),
            window=self.score_container,
            anchor="nw",
        )

        # 绑定 Canvas 大小变化事件，更新滚动区域和重新布局
        def _configure_canvas_score(event):
            canvas_width = event.width
            self.score_canvas.itemconfig(self.score_canvas_window, width=canvas_width)
            # 如果有靶位数据，重新布局
            if self.scoring_target_ids:
                self._render_score_boxes()
            else:
                self.score_canvas.configure(scrollregion=self.score_canvas.bbox("all"))

        self.score_canvas.bind("<Configure>", _configure_canvas_score)
        self.score_container.bind("<Configure>", lambda e: self.score_canvas.configure(scrollregion=self.score_canvas.bbox("all")))

        # 默认靶位列表，可按需扩展（如多个靶位）
        self.scoring_target_ids: list[int] = []
        self.arrow_entries: dict[int, list[tk.Entry]] = {}

        self._render_score_boxes()

    def _render_score_boxes(self) -> None:
        """根据箭数和检测到的靶位重建每个靶位的记录框，横向排列，超出宽度时换行。"""
        for child in self.score_container.winfo_children():
            child.destroy()

        # 解析箭数，非法输入则退回默认 6
        try:
            arrow_count = int(self.arrow_count_var.get())
            if arrow_count <= 0:
                raise ValueError
        except Exception:
            arrow_count = 6
            self.arrow_count_var.set(str(arrow_count))

        self.arrow_entries.clear()

        # 如果没有检测到靶位，使用默认靶位号 1
        if not self.scoring_target_ids:
            self.scoring_target_ids = [1]

        # 获取 Canvas 宽度，用于计算每行能放几个记录框
        self.score_canvas.update_idletasks()
        canvas_width = self.score_canvas.winfo_width() or 400
        
        # 每个记录框的宽度（包括边距），假设每个框宽度为 200px，边距 10px
        box_width = 200
        box_padding = 10
        total_box_width = box_width + box_padding * 2
        
        # 计算每行能放几个记录框
        cols_per_row = max(1, int(canvas_width / total_box_width))

        for t_idx, target_id in enumerate(self.scoring_target_ids):
            # 计算行列位置
            row = t_idx // cols_per_row
            col = t_idx % cols_per_row
            
            box = ttk.LabelFrame(self.score_container, text=f"靶位号 {target_id}")
            box.grid(row=row, column=col, sticky="nw", padx=box_padding, pady=box_padding)
            box.columnconfigure(1, weight=1)

            entries: list[tk.Entry] = []
            for i in range(arrow_count):
                ttk.Label(box, text=f"第{i + 1}箭").grid(row=i, column=0, sticky="e", padx=5, pady=2)
                entry = ttk.Entry(box, width=6)
                entry.insert(0, "--")  # 占位
                entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
                entries.append(entry)

            self.arrow_entries[target_id] = entries

        # 更新 Canvas 滚动区域
        self.score_canvas.update_idletasks()
        self.score_canvas.configure(scrollregion=self.score_canvas.bbox("all"))

    def _on_next_round(self) -> None:
        """点击“下一轮”后，按当前箭数重置输入框内容。"""
        self._render_score_boxes()

    # ---------- 计分检测逻辑 ----------
    def _ensure_score_detector(self) -> bool:
        """懒加载 ScoreDetector，加载失败时在图像区提示。"""
        if self._score_detector is not None:
            return True
        try:
            self._score_detector = ScoreDetector(
                target_model_path=self.target_model_path,
                arrow_model_path=self.arrow_model_path,
            )
            return True
        except Exception as exc:
            self._show_error_in_canvas(f"加载计分模型失败：{exc}")
            return False

    def _on_detect_targets(self) -> None:
        """从摄像头抓取一帧，使用 ScoreDetector 检测靶纸与得分，并同时显示所有靶纸图片。"""
        if not self._camera:
            self._show_error_in_canvas("未初始化摄像头")
            return

        if not self._ensure_score_detector():
            return

        frame = self._camera.getFrame()
        if frame is None:
            self._show_error_in_canvas("无法获取摄像头画面")
            return

        try:
            results = self._score_detector.detect(frame)
        except Exception as exc:
            self._show_error_in_canvas(f"计分检测失败：{exc}")
            return

        if not results:
            self._show_error_in_canvas("未检测到靶纸")
            self._detected_target_results = []
            self.scoring_target_ids = []
            self._render_score_boxes()
            return

        # 保存检测结果
        self._detected_target_results = results
        # 更新靶位列表（按检测顺序编号）
        self.scoring_target_ids = list(range(1, len(results) + 1))
        
        # 同时显示所有靶纸图片
        self._display_all_targets(results)
        
        # 更新计分区显示所有靶位的记录框
        self._render_score_boxes()

    def _show_error_in_canvas(self, message: str) -> None:
        """在图像显示区的 Canvas 中显示错误信息。"""
        for widget in self.score_image_frame.winfo_children():
            widget.destroy()
        
        error_label = tk.Label(
            self.score_image_frame,
            text=message,
            bg="black",
            fg="red",
            font=("Microsoft YaHei", 14),
        )
        error_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.score_image_canvas.update_idletasks()
        self.score_image_canvas.configure(scrollregion=self.score_image_canvas.bbox("all"))

    def _display_all_targets(self, results: list) -> None:
        """在图像显示区的 Canvas 中同时显示所有靶纸图片，横向排列，超出宽度时换行。"""
        # 清空之前的显示
        for widget in self.score_image_frame.winfo_children():
            widget.destroy()
        
        if not results:
            return

        # 获取 Canvas 宽度，用于计算图片显示尺寸和每行数量
        self.score_image_canvas.update_idletasks()
        canvas_width = self.score_image_canvas.winfo_width() or 640
        
        # 每张图片的显示宽度（包括边距），假设每张图片宽度为 300px，边距 20px
        item_width = 300
        item_padding = 20
        total_item_width = item_width + item_padding * 2
        
        # 计算每行能放几张图片
        cols_per_row = max(1, int(canvas_width / total_item_width))
        
        # 保持图片宽高比，假设原始图片大致为正方形
        image_width = item_width
        image_height = item_width

        self._score_crops = []  # 保存引用避免被回收

        for idx, target_result in enumerate(results):
            # 将 BGR 裁剪图转为 RGB
            rgb = cv2.cvtColor(target_result.crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            # 保持宽高比缩放
            img.thumbnail((image_width, image_height), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(image=img)
            self._score_crops.append(tk_img)  # 保存引用

            # 计算行列位置
            row = idx // cols_per_row
            col = idx % cols_per_row

            # 创建图片容器 Frame
            img_frame = tk.Frame(self.score_image_frame, bg="black")
            img_frame.grid(row=row, column=col, padx=item_padding, pady=item_padding, sticky="nw")

            # 显示图片
            img_label = tk.Label(
                img_frame,
                image=tk_img,
                bg="black",
            )
            img_label.pack()

            # 在图片下方显示靶纸号和信息
            target_id = idx + 1
            summary = f"靶位号 {target_id} | 总分: {target_result.total_score} | 箭数: {len(target_result.arrows)} | X环: {target_result.x_count}"
            info_label = tk.Label(
                img_frame,
                text=summary,
                bg="black",
                fg="white",
                font=("Microsoft YaHei", 10),
            )
            info_label.pack(pady=(5, 0))

        # 更新 Canvas 滚动区域
        self.score_image_canvas.update_idletasks()
        self.score_image_canvas.configure(scrollregion=self.score_image_canvas.bbox("all"))

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

        # 姿态检测开关
        pose_check = ttk.Checkbutton(
            self.settings_frame,
            text="启用姿态检测 (YOLO-Pose)",
            variable=self.pose_detection_enabled,
        )
        pose_check.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        # 占位：保存按钮（具体逻辑后续与 CameraCapture 对接）
        save_btn = ttk.Button(self.settings_frame, text="保存设置", command=self._on_save_settings)
        save_btn.grid(row=3, column=0, columnspan=2, pady=(20, 10))

    def _on_save_settings(self) -> None:
        """
        保存设置的占位回调。
        目前仅打印设置值，后续可写入配置文件或直接应用到 CameraCapture。
        """
        cam = self.camera_source_var.get()
        res = self.resolution_var.get()
        pose = self.pose_detection_enabled.get()
        print(f"[设置] 摄像头源: {cam}, 分辨率: {res}, 姿态检测开启: {pose}")


def main() -> None:
    app = ArcheryTrainingApp()
    app.mainloop()


if __name__ == "__main__":
    main()
