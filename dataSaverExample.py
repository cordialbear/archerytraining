"""dataSaver 模块使用示例。"""

from datetime import datetime

from data import Record, Round, Shoot
from dataSaver import DataSaver, SaveMode, LocalDataSaver, RemoteDataSaver


def example_local_save() -> None:
    """示例1: 本地保存（数据库和txt文件）。"""
    print("=" * 60)
    print("示例1: 本地保存")
    print("=" * 60)

    # 初始化本地保存器
    saver = DataSaver(
        mode=SaveMode.LOCAL,
        db_path="archery_data.db",
        txt_dir="data_records",
        save_to_db=True,
        save_to_txt=True,
    )

    # 创建训练记录
    record = Record(
        time=datetime.now(),
        name="本地训练记录1",
    )

    # 添加成绩
    round1 = Round()
    round1.add_shoot(1, Shoot(score=9, is_x=False))
    round1.add_shoot(2, Shoot(score=10, is_x=True))
    round1.add_shoot(3, Shoot(score=8, is_x=False))
    record.add_round(round1)

    # 保存数据
    success = saver.save(record)
    print(f"保存结果: {'成功' if success else '失败'}")

    # 加载历史记录
    all_records = saver.load()
    print(f"共加载 {len(all_records)} 条记录")


def example_remote_save() -> None:
    """示例2: 远程保存（API）。"""
    print("\n" + "=" * 60)
    print("示例2: 远程保存")
    print("=" * 60)

    # 初始化远程保存器
    saver = DataSaver(
        mode=SaveMode.REMOTE,
        remote_api_url="https://api.example.com/save",
        remote_api_key="your-api-key",
    )

    # 创建训练记录
    record = Record(
        time=datetime.now(),
        name="远程训练记录1",
    )

    # 添加成绩
    round1 = Round()
    round1.add_shoot(1, Shoot(score=10, is_x=True))
    round1.add_shoot(2, Shoot(score=9, is_x=False))
    record.add_round(round1)

    # 保存数据
    success = saver.save(record)
    print(f"保存结果: {'成功' if success else '失败'}")


def example_both_save() -> None:
    """示例3: 同时保存到本地和远程。"""
    print("\n" + "=" * 60)
    print("示例3: 同时保存到本地和远程")
    print("=" * 60)

    # 初始化保存器（同时支持本地和远程）
    saver = DataSaver(
        mode=SaveMode.BOTH,
        db_path="archery_data.db",
        txt_dir="data_records",
        save_to_db=True,
        save_to_txt=True,
        remote_api_url="https://api.example.com/save",
        remote_api_key="your-api-key",
    )

    # 创建训练记录
    record = Record(
        time=datetime.now(),
        name="混合保存记录1",
    )

    # 添加成绩
    round1 = Round()
    round1.add_shoot(1, Shoot(score=9, is_x=False))
    round1.add_shoot(2, Shoot(score=10, is_x=True))
    round1.add_shoot(3, Shoot(score=9, is_x=False))
    record.add_round(round1)

    # 保存数据（会自动保存到本地和远程）
    success = saver.save(record)
    print(f"保存结果: {'成功' if success else '失败'}")


def example_direct_use() -> None:
    """示例4: 直接使用子类。"""
    print("\n" + "=" * 60)
    print("示例4: 直接使用 LocalDataSaver 和 RemoteDataSaver")
    print("=" * 60)

    # 直接使用本地保存器
    local_saver = LocalDataSaver(
        db_path="archery_data.db",
        txt_dir="data_records",
        save_to_db=True,
        save_to_txt=True,
    )

    record = Record(
        time=datetime.now(),
        name="直接使用本地保存器",
    )

    round1 = Round()
    round1.add_shoot(1, Shoot(score=10, is_x=True))
    record.add_round(round1)

    success = local_saver.save(record)
    print(f"本地保存结果: {'成功' if success else '失败'}")

    # 加载记录
    records = local_saver.load()
    print(f"加载了 {len(records)} 条记录")


def main() -> None:
    """主函数，演示各种使用方式。"""
    print("数据保存模块使用示例\n")

    # 示例1: 本地保存
    example_local_save()

    # 示例2: 远程保存（需要有效的API地址，这里只是演示）
    # example_remote_save()

    # 示例3: 同时保存到本地和远程
    # example_both_save()

    # 示例4: 直接使用子类
    example_direct_use()

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
