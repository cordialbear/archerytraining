"""
主应用入口 - 使用MVC架构
"""
from model import ArcheryModel
from view import ArcheryView
from controller import ArcheryController


class ArcheryTrainingApp:
    """射箭训练辅助主应用（MVC架构）"""

    def __init__(self) -> None:
        # 创建 MVC 三层
        self.model = ArcheryModel()
        self.view = ArcheryView()
        self.controller = ArcheryController(self.model, self.view)
        
        # 绑定窗口关闭事件
        self.view.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """窗口关闭事件处理"""
        self.controller.cleanup()
        self.view.root.destroy()
    
    def run(self):
        """运行应用"""
        self.view.root.mainloop()


def main() -> None:
    app = ArcheryTrainingApp()
    app.run()


if __name__ == "__main__":
    main()
