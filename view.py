"""
View - 视图层
负责所有UI组件的创建和显示更新
"""
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Callable

import cv2
from PIL import Image, ImageTk

from model import ArcheryModel
from scoreDectect import TargetResult


class ArcheryView:
    """射箭训练应用的视图层"""
    
    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root if root is not None else tk.Tk()
        self.root.title("射箭训练辅助系统")
        self.root.geometry("1024x600")
        
        self._configure_style()
        
        # 创建主容器
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各个页面
        self.train_frame = ttk.Frame(self.notebook)
        self.scoring_frame = ttk.Frame(self.notebook)
        self.history_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.train_frame, text="训练辅助")
        self.notebook.add(self.scoring_frame, text="计分")
        self.notebook.add(self.history_frame, text="历史数据")
        self.notebook.add(self.settings_frame, text="设置")
        
        # UI组件引用
        self.train_image_label: Optional[tk.Label] = None
        self.train_metric_vars: Dict[str, tk.StringVar] = {}
        
        self.score_image_canvas: Optional[tk.Canvas] = None
        self.score_image_frame: Optional[tk.Frame] = None
        self.score_canvas: Optional[tk.Canvas] = None
        self.score_container: Optional[ttk.Frame] = None
        self.arrow_count_var: Optional[tk.StringVar] = None
        self.arrow_entries: Dict[int, List[tk.Entry]] = {}
        
        self.settings_vars: Dict[str, tk.Variable] = {}
        
        # 图像缓存
        self._image_cache: List = []
        
        # 构建UI
        self._build_train_tab()
        self._build_scoring_tab()
        self._build_history_tab()
        self._build_settings_tab()
    
    def _configure_style(self):
        """配置UI样式"""
        style = ttk.Style()
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
    def _build_train_tab(self):
        """构建训练辅助页面"""
        self.train_frame.columnconfigure(0, weight=3, uniform="train")
        self.train_frame.columnconfigure(1, weight=2, uniform="train")
        self.train_frame.rowconfigure(0, weight=1)
        
        # 图像显示区
        image_area = ttk.LabelFrame(self.train_frame, text="图像显示区")
        image_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        image_area.rowconfigure(0, weight=1)
        image_area.columnconfigure(0, weight=1)
        
        self.train_image_label = tk.Label(
            image_area,
            text="等待摄像头画面...",
            bg="black",
            fg="white",
            font=("Microsoft YaHei", 14),
        )
        self.train_image_label.pack(fill=tk.BOTH, expand=True)
        
        # 数据评估区
        evaluate_area = ttk.LabelFrame(self.train_frame, text="数据评估区")
        evaluate_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        for i in range(6):
            evaluate_area.rowconfigure(i, weight=1)
        evaluate_area.columnconfigure(0, weight=1)
        evaluate_area.columnconfigure(1, weight=1)
        
        metrics = [
            "当前射击编号",
            "命中环数",
            "拉弓稳定性",
            "释放稳定性",
            "身体姿态评分",
            "整体建议",
        ]
        
        for idx, name in enumerate(metrics):
            label = ttk.Label(evaluate_area, text=name + "：", anchor="e")
            label.grid(row=idx, column=0, sticky="e", padx=(10, 5), pady=5)
            
            var = tk.StringVar(value="--")
            value_label = ttk.Label(evaluate_area, textvariable=var, anchor="w")
            value_label.grid(row=idx, column=1, sticky="w", padx=(5, 10), pady=5)
            
            self.train_metric_vars[name] = var
    
    def update_train_image(self, frame: Optional[cv2.Mat]):
        """更新训练页面的图像显示"""
        if frame is None:
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        w = self.train_image_label.winfo_width() or 640
        h = self.train_image_label.winfo_height() or 480
        img = img.resize((w, h))
        
        tk_img = ImageTk.PhotoImage(image=img)
        self.train_image_label.configure(image=tk_img, text="")
        self.train_image_label.image = tk_img  # 保存引用
    
    def update_train_metrics(self, metrics: Dict[str, str]):
        """更新训练评估指标显示"""
        for name, value in metrics.items():
            if name in self.train_metric_vars:
                self.train_metric_vars[name].set(value)
    
    def show_train_error(self, message: str):
        """显示训练页面错误信息"""
        self.train_image_label.config(
            text=message,
            fg="red",
            bg="black",
            image="",
        )
    
    # ---------- 计分 Tab ----------
    def _build_scoring_tab(self):
        """构建计分页面"""
        self.scoring_frame.columnconfigure(0, weight=3, uniform="score")
        self.scoring_frame.columnconfigure(1, weight=2, uniform="score")
        self.scoring_frame.rowconfigure(0, weight=1)
        
        # 图像显示区
        image_area = ttk.LabelFrame(self.scoring_frame, text="图像显示区")
        image_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        image_area.rowconfigure(0, weight=1)
        image_area.columnconfigure(0, weight=1)
        image_area.columnconfigure(1, weight=0)
        
        self.score_image_canvas = tk.Canvas(
            image_area,
            bg="black",
            highlightthickness=0,
        )
        self.score_image_canvas.grid(row=0, column=0, sticky="nsew")
        
        score_image_scrollbar = ttk.Scrollbar(
            image_area,
            orient=tk.VERTICAL,
            command=self.score_image_canvas.yview,
        )
        score_image_scrollbar.grid(row=0, column=1, sticky="ns")
        self.score_image_canvas.configure(yscrollcommand=score_image_scrollbar.set)
        
        self.score_image_frame = tk.Frame(self.score_image_canvas, bg="black")
        self.score_image_canvas_window = self.score_image_canvas.create_window(
            (0, 0),
            window=self.score_image_frame,
            anchor="nw",
        )
        
        # 计分区
        scoring_area = ttk.LabelFrame(self.scoring_frame, text="计分区")
        scoring_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        scoring_area.columnconfigure(0, weight=1)
        
        # 控制区域
        control_frame = ttk.Frame(scoring_area)
        control_frame.grid(row=0, column=0, sticky="ew", pady=(5, 10), padx=5)
        control_frame.columnconfigure(1, weight=1)
        
        ttk.Label(control_frame, text="箭数：").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.arrow_count_var = tk.StringVar(value="6")
        arrow_entry = ttk.Entry(control_frame, textvariable=self.arrow_count_var, width=6)
        arrow_entry.grid(row=0, column=1, sticky="w")
        
        # 下一轮和检测按钮
        self.next_round_btn = ttk.Button(control_frame, text="下一轮")
        self.next_round_btn.grid(row=0, column=2, sticky="e", padx=(10, 0))
        
        self.detect_btn = ttk.Button(control_frame, text="检测箭靶")
        self.detect_btn.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        
        # 计分区 Canvas
        score_canvas_frame = ttk.Frame(scoring_area)
        score_canvas_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        scoring_area.rowconfigure(1, weight=1)
        score_canvas_frame.rowconfigure(0, weight=1)
        score_canvas_frame.columnconfigure(0, weight=1)
        score_canvas_frame.columnconfigure(1, weight=0)
        
        self.score_canvas = tk.Canvas(
            score_canvas_frame,
            highlightthickness=0,
        )
        self.score_canvas.grid(row=0, column=0, sticky="nsew")
        
        score_scrollbar = ttk.Scrollbar(
            score_canvas_frame,
            orient=tk.VERTICAL,
            command=self.score_canvas.yview,
        )
        score_scrollbar.grid(row=0, column=1, sticky="ns")
        self.score_canvas.configure(yscrollcommand=score_scrollbar.set)
        
        self.score_container = ttk.Frame(self.score_canvas)
        self.score_canvas_window = self.score_canvas.create_window(
            (0, 0),
            window=self.score_container,
            anchor="nw",
        )
    
    def bind_arrow_count_change(self, callback: Callable):
        """绑定箭数变化回调"""
        if self.arrow_count_var:
            self.arrow_count_var.trace_add("write", lambda *_: callback())
    
    def get_arrow_count(self) -> int:
        """获取箭数"""
        try:
            count = int(self.arrow_count_var.get())
            return max(1, count) if count > 0 else 6
        except:
            return 6
    
    def set_arrow_count(self, count: int):
        """设置箭数"""
        if self.arrow_count_var:
            self.arrow_count_var.set(str(count))
    
    def display_target_images(self, target_results: List[TargetResult], on_canvas_resize: Optional[Callable] = None):
        """显示所有靶纸图片"""
        for widget in self.score_image_frame.winfo_children():
            widget.destroy()
        
        if not target_results:
            return
        
        self.score_image_canvas.update_idletasks()
        canvas_width = self.score_image_canvas.winfo_width() or 640
        
        item_width = 300
        item_padding = 20
        total_item_width = item_width + item_padding * 2
        cols_per_row = max(1, int(canvas_width / total_item_width))
        
        self._image_cache.clear()
        
        for idx, target_result in enumerate(target_results):
            rgb = cv2.cvtColor(target_result.crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((item_width, item_width), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(image=img)
            self._image_cache.append(tk_img)
            
            row = idx // cols_per_row
            col = idx % cols_per_row
            
            img_frame = tk.Frame(self.score_image_frame, bg="black")
            img_frame.grid(row=row, column=col, padx=item_padding, pady=item_padding, sticky="nw")
            
            img_label = tk.Label(img_frame, image=tk_img, bg="black")
            img_label.pack()
            
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
        
        self.score_image_canvas.update_idletasks()
        self.score_image_canvas.configure(scrollregion=self.score_image_canvas.bbox("all"))
        
        # 绑定Canvas大小变化事件
        if on_canvas_resize:
            def _configure_canvas(event):
                canvas_width = event.width
                self.score_image_canvas.itemconfig(self.score_image_canvas_window, width=canvas_width)
                if target_results:
                    on_canvas_resize()
                else:
                    self.score_image_canvas.configure(scrollregion=self.score_image_canvas.bbox("all"))
            
            self.score_image_canvas.bind("<Configure>", _configure_canvas)
    
    def show_scoring_error(self, message: str):
        """显示计分页面错误信息"""
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
    
    def render_score_boxes(self, target_ids: List[int], arrow_count: int, on_canvas_resize: Optional[Callable] = None):
        """渲染计分记录框"""
        for child in self.score_container.winfo_children():
            child.destroy()
        
        self.arrow_entries.clear()
        
        if not target_ids:
            target_ids = [1]
        
        self.score_canvas.update_idletasks()
        canvas_width = self.score_canvas.winfo_width() or 400
        
        box_width = 200
        box_padding = 10
        total_box_width = box_width + box_padding * 2
        cols_per_row = max(1, int(canvas_width / total_box_width))
        
        for t_idx, target_id in enumerate(target_ids):
            row = t_idx // cols_per_row
            col = t_idx % cols_per_row
            
            box = ttk.LabelFrame(self.score_container, text=f"靶位号 {target_id}")
            box.grid(row=row, column=col, sticky="nw", padx=box_padding, pady=box_padding)
            box.columnconfigure(1, weight=1)
            
            entries: List[tk.Entry] = []
            for i in range(arrow_count):
                ttk.Label(box, text=f"第{i + 1}箭").grid(row=i, column=0, sticky="e", padx=5, pady=2)
                entry = ttk.Entry(box, width=6)
                entry.insert(0, "--")
                entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
                entries.append(entry)
            
            self.arrow_entries[target_id] = entries
        
        self.score_canvas.update_idletasks()
        self.score_canvas.configure(scrollregion=self.score_canvas.bbox("all"))
        
        # 绑定Canvas大小变化事件
        if on_canvas_resize:
            def _configure_canvas(event):
                canvas_width = event.width
                self.score_canvas.itemconfig(self.score_canvas_window, width=canvas_width)
                if target_ids:
                    on_canvas_resize()
                else:
                    self.score_canvas.configure(scrollregion=self.score_canvas.bbox("all"))
            
            self.score_canvas.bind("<Configure>", _configure_canvas)
    
    def get_score_entry(self, target_id: int, arrow_index: int) -> Optional[tk.Entry]:
        """获取指定靶位和箭支的输入框"""
        if target_id in self.arrow_entries and arrow_index < len(self.arrow_entries[target_id]):
            return self.arrow_entries[target_id][arrow_index]
        return None
    
    # ---------- 历史数据 Tab ----------
    def _build_history_tab(self):
        """构建历史数据页面"""
        self.history_frame.rowconfigure(0, weight=1)
        self.history_frame.columnconfigure(0, weight=1)
        
        frame = ttk.LabelFrame(self.history_frame, text="训练记录列表")
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        columns = ("time", "session", "avg_score", "note")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=15)
        tree.heading("time", text="时间")
        tree.heading("session", text="训练场次")
        tree.heading("avg_score", text="平均环数")
        tree.heading("note", text="备注")
        
        tree.column("time", width=160, anchor="center")
        tree.column("session", width=100, anchor="center")
        tree.column("avg_score", width=100, anchor="center")
        tree.column("note", width=200, anchor="w")
        
        tree.pack(fill=tk.BOTH, expand=True)
        tree.insert("", tk.END, values=("--", "--", "--", "暂无历史数据"))
    
    # ---------- 设置 Tab ----------
    def _build_settings_tab(self):
        """构建设置页面"""
        self.settings_frame.columnconfigure(0, weight=1)
        self.settings_frame.columnconfigure(1, weight=1)
        
        cam_label = ttk.Label(self.settings_frame, text="摄像头源：")
        cam_label.grid(row=0, column=0, sticky="e", padx=10, pady=(20, 10))
        
        self.settings_vars["camera_source"] = tk.StringVar(value="picamera")
        cam_entry = ttk.Entry(self.settings_frame, textvariable=self.settings_vars["camera_source"])
        cam_entry.grid(row=0, column=1, sticky="w", padx=10, pady=(20, 10))
        
        res_label = ttk.Label(self.settings_frame, text="图像分辨率 (宽x高)：")
        res_label.grid(row=1, column=0, sticky="e", padx=10, pady=10)
        
        self.settings_vars["resolution"] = tk.StringVar(value="640x480")
        res_entry = ttk.Entry(self.settings_frame, textvariable=self.settings_vars["resolution"])
        res_entry.grid(row=1, column=1, sticky="w", padx=10, pady=10)
        
        self.settings_vars["pose_detection"] = tk.BooleanVar(value=False)
        pose_check = ttk.Checkbutton(
            self.settings_frame,
            text="启用姿态检测 (YOLO-Pose)",
            variable=self.settings_vars["pose_detection"],
        )
        pose_check.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # 保存设置按钮
        self.save_settings_btn: Optional[ttk.Button] = None
        self.save_settings_btn = ttk.Button(self.settings_frame, text="保存设置")
        self.save_settings_btn.grid(row=3, column=0, columnspan=2, pady=(20, 10))
    
    def get_settings(self) -> Dict[str, any]:
        """获取设置值"""
        return {
            "camera_source": self.settings_vars["camera_source"].get(),
            "resolution": self.settings_vars["resolution"].get(),
            "pose_detection": self.settings_vars["pose_detection"].get(),
        }
    
    def set_settings(self, settings: Dict[str, any]):
        """设置值"""
        if "camera_source" in settings:
            self.settings_vars["camera_source"].set(settings["camera_source"])
        if "resolution" in settings:
            self.settings_vars["resolution"].set(settings["resolution"])
        if "pose_detection" in settings:
            self.settings_vars["pose_detection"].set(settings["pose_detection"])
    
    def load_settings_from_model(self, model_settings):
        """从模型加载设置到视图"""
        self.set_settings({
            "camera_source": model_settings.camera_source,
            "resolution": model_settings.resolution,
            "pose_detection": model_settings.pose_detection_enabled,
        })
    
    def bind_detect_button(self, callback: Callable):
        """绑定检测按钮回调"""
        if self.detect_btn:
            self.detect_btn.configure(command=callback)
    
    def bind_next_round_button(self, callback: Callable):
        """绑定下一轮按钮回调"""
        if self.next_round_btn:
            self.next_round_btn.configure(command=callback)
    
    def bind_save_settings_button(self, callback: Callable):
        """绑定保存设置按钮回调"""
        if self.save_settings_btn:
            self.save_settings_btn.configure(command=callback)
