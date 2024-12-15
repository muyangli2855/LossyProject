import numpy as np
import compress
import decompress
import computemse

# blocksize, quant_step = 2048, 0.03826 # 74579
# blocksize, quant_step = 1024, 0.03816 # 74397
# blocksize, quant_step = 512, 0.03804 # 74859
# blocksize, quant_step = 768, 0.03816 # 74472
# blocksize, quant_step = 1536, 0.03827 # 74391
# blocksize, quant_step = 1280, 0.03824 # 74291
# blocksize, quant_step = 1152, 0.03826 # 74130
# blocksize, quant_step = 1088, 0.03829 # 74157

def getSizeAndMse(blocksize, quant_step):
    size = compress.compress(blocksize, quant_step)
    decompress.decompress()
    mse = computemse.computemse()
    print(f"blocksize: {blocksize}\tquant_step: {quant_step:.5f}\tsize: {size}\tmse: {mse}")
    return size, mse


def find_best_blocksize_and_quantstep():
    best_blocksize = None
    best_quant_step = None
    min_size = float('inf')  # 用于保存最小的 size
    mse_threshold = 4e-5  # MSE 阈值

    # 遍历 blocksize
    for blocksize in range(1088, 1281):  # 从 1088 到 1280 遍历
        quant_step = 0.03823  # 从 0.03800 开始
        previous_size = None
        previous_quant_step = None

        while True:
            # 获取当前 blocksize 和 quant_step 对应的 size 和 mse
            size, mse = getSizeAndMse(blocksize, quant_step)

            # 检查 MSE 是否超过阈值
            if mse > mse_threshold:
                # 如果 MSE 超过阈值，选择上一次的 quant_step
                if previous_size is not None and previous_size < min_size:
                    best_blocksize = blocksize
                    best_quant_step = previous_quant_step
                    min_size = previous_size
                break

            # 更新最优值
            if size < min_size:
                best_blocksize = blocksize
                best_quant_step = quant_step
                min_size = size

            # 保存当前值
            previous_size = size
            previous_quant_step = quant_step

            # 增加 quant_step
            quant_step += 0.00001

        # 防止循环陷入无限
        quant_step = round(quant_step, 5)  # 避免浮点数误差

    return best_blocksize, best_quant_step, min_size

if __name__ == "__main__":
    # 调用函数并输出结果
    # best_blocksize, best_quant_step, min_size = find_best_blocksize_and_quantstep()
    # print(f"最佳组合: blocksize={best_blocksize}, quant_step={best_quant_step}, size={min_size}")
    # 最佳组合: blocksize=1240, quant_step=0.03823, size=74048

    getSizeAndMse(1240, 0.03823)
