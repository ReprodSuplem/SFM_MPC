import subprocess
import time
import sys

experiments = [
    "two-path_bottleneck.py",
    "braess_paradox.py",
    "manhattan_grid.py",
    "sioux_falls.py"
]

def run_experiments():
    print("=" * 60)
    print("开始执行仿真实验集")
    print("=" * 60)
    
    total_start_time = time.time()
    
    for script in experiments:
        print(f"\n>>> 正在启动实验: [{script}] ...")
        script_start_time = time.time()
        
        try:
            # 使用 subprocess.run 阻塞执行，确保一个跑完再跑下一个
            # check=True 会在子脚本报错时直接抛出异常，停止后续执行
            result = subprocess.run([sys.executable, script], check=True)
            
            script_duration = time.time() - script_start_time
            print(f"实验 [{script}] 执行完毕! 耗时: {script_duration:.2f} 秒.")
            
        except subprocess.CalledProcessError as e:
            print(f"实验 [{script}] 执行失败，退出码: {e.returncode}")
            print("批处理已终止，请检查上述报错信息。")
            break
        except FileNotFoundError:
            print(f"找不到文件: {script}。请检查文件名是否正确。")
            print("批处理已终止。")
            break

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"所有指定的实验已处理完毕! 总耗时: {total_duration:.2f} 秒.")
    print("=" * 60)

if __name__ == "__main__":
    run_experiments()