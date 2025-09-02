import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# ==================== 1. 定义无人机参数和模型 ====================
# 我们使用一个简单的双积分器模型作为示例：位置 -> 速度 -> 加速度(控制输入)
# 状态向量 x = [px, py, pz, vx, vy, vz]
# 控制输入 u = [ax, ay, az] (加速度)
# 状态空间方程: 
#   \dot{px} = vx
#   \dot{vx} = ax
#   (同理于y, z)

dt = 0.1  # 控制周期，单位：秒
N = 15    # 预测步长（MPC Horizon）
mass = 1.0  # 无人机质量，kg
g = 9.81   # 重力加速度，m/s^2

# 系统动力学矩阵 (连续时间)
# x_dot = A_c * x + B_c * u
A_c = np.zeros((6, 6))
A_c[0:3, 3:6] = np.eye(3)
B_c = np.zeros((6, 3))
B_c[3:6, 0:3] = np.eye(3)

# 离散化系统（前向欧拉法）
A_d = np.eye(6) + A_c * dt
B_d = B_c * dt

# ==================== 2. 定义MPC优化问题 ====================
# 优化变量
x = cp.Variable((6, N+1))  # 状态序列：6维，N+1个时间点（从0到N）
u = cp.Variable((3, N))    # 控制输入序列：3维，N个时间点（从0到N-1）

# 初始状态参数（在每次求解时更新）
x_init = cp.Parameter(6)

# 参考轨迹参数（在每次求解时更新，这里我们让它跟踪一个点）
reference_trajectory = cp.Parameter((3, N+1))

# 成本函数和约束
cost = 0
constraints = [x[:, 0] == x_init]  # 初始状态约束

for t in range(N):
    # 1. 跟踪误差成本：惩罚状态与参考轨迹的偏差 (这里只跟踪位置x[0:3])
    cost += cp.quad_form(x[0:3, t] - reference_trajectory[:, t], np.eye(3)) * 10.0
    # 2. 控制努力成本：惩罚过大的控制输入
    cost += cp.quad_form(u[:, t], np.eye(3)) * 0.1

    # 系统动力学约束
    constraints += [x[:, t+1] == A_d @ x[:, t] + B_d @ u[:, t]]
    
    # 控制输入约束（假设最大加速度为2 m/s^2）
    constraints += [cp.norm(u[:, t], 'inf') <= 2.0]

# 终端成本：惩罚预测期最后的状态误差
cost += cp.quad_form(x[0:3, N] - reference_trajectory[:, N], np.eye(3)) * 20.0

# 构建优化问题
mpc_problem = cp.Problem(cp.Minimize(cost), constraints)

# ==================== 3. 仿真循环 ====================
# 仿真总时长
sim_time = 20.0
num_steps = int(sim_time / dt)

# 初始化记录数组
log_states = np.zeros((6, num_steps))
log_control = np.zeros((3, num_steps))
log_reference = np.zeros((3, num_steps))

# 初始状态 [px, py, pz, vx, vy, vz]
current_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 生成参考轨迹：一个在X-Y平面的圆，Z轴高度为1
t_range = np.linspace(0, sim_time, num_steps)
ref_radius = 2.0
ref_freq = 0.5
ref_x = ref_radius * np.sin(ref_freq * 2 * np.pi * t_range / sim_time)
ref_y = ref_radius * np.cos(ref_freq * 2 * np.pi * t_range / sim_time)
ref_z = np.ones(num_steps) * 1.0 # 保持高度1米

for i in range(num_steps):
    # 更新参数：当前状态和未来N步的参考轨迹
    x_init.value = current_state
    # 为参考轨迹参数赋值未来N+1个点的值
    ref_indices = np.arange(i, i+N+1) 
    # 防止索引超出范围
    ref_indices = np.clip(ref_indices, 0, num_steps-1) 
    reference_trajectory.value = np.vstack((ref_x[ref_indices], 
                                           ref_y[ref_indices], 
                                           ref_z[ref_indices]))
    
    # 求解MPC问题
    mpc_problem.solve(solver=cp.ECOS, verbose=False)
    
    if mpc_problem.status != cp.OPTIMAL:
        print(f"Step {i}: MPC solver did not find an optimal solution!")
        # 处理求解失败的情况，例如使用上一个控制输入或零输入
        optimal_u = np.zeros(3)
    else:
        # 获取最优控制序列的第一个值
        optimal_u = u.value[:, 0]
    
    # 记录当前状态、控制和参考
    log_states[:, i] = current_state
    log_control[:, i] = optimal_u
    log_reference[:, i] = np.array([ref_x[i], ref_y[i], ref_z[i]])
    
    # 模拟系统演化：将控制输入施加到系统上，得到下一个状态
    # 这里我们使用离散模型进行模拟
    current_state = A_d @ current_state + B_d @ optimal_u

# ==================== 4. 结果可视化 ====================
# 绘制轨迹对比图
fig = plt.figure(figsize=(12, 9))

# 3D轨迹图
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(log_states[0, :], log_states[1, :], log_states[2, :], label='Actual Trajectory')
ax1.plot(log_reference[0, :], log_reference[1, :], log_reference[2, :], 'r--', label='Reference Trajectory')
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_zlabel('Z [m]')
ax1.set_title('3D Trajectory Tracking')
ax1.legend()
ax1.grid(True)

# 位置时间序列
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_range, log_states[0, :], label='X Pos')
ax2.plot(t_range, log_states[1, :], label='Y Pos')
ax2.plot(t_range, log_states[2, :], label='Z Pos')
ax2.plot(t_range, log_reference[0, :], 'r--', label='X Ref')
ax2.plot(t_range, log_reference[1, :], 'r--', label='Y Ref')
ax2.plot(t_range, log_reference[2, :], 'r--', label='Z Ref')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Position [m]')
ax2.set_title('Position vs Time')
ax2.legend()
ax2.grid(True)

# 控制输入时间序列
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t_range[:num_steps-1], log_control[0, :num_steps-1], label='Accel X')
ax3.plot(t_range[:num_steps-1], log_control[1, :num_steps-1], label='Accel Y')
ax3.plot(t_range[:num_steps-1], log_control[2, :num_steps-1], label='Accel Z')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Control Input [m/s^2]')
ax3.set_title('Control Inputs')
ax3.legend()
ax3.grid(True)

# 跟踪误差
tracking_error = np.linalg.norm(log_states[0:3, :] - log_reference, axis=0)
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(t_range, tracking_error)
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Error [m]')
ax4.set_title('Tracking Error (L2 Norm)')
ax4.grid(True)

plt.tight_layout()
plt.show()